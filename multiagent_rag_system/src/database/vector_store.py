"""
Two connection modes (auto-detected from config):
  Server mode — connects to a running Qdrant instance (local Docker or Qdrant Cloud)
  Local mode  — in-process Qdrant stored on disk (no Docker needed, dev/test)

Why Qdrant over FAISS?
  - Full CRUD: delete individual points without rebuilding the entire index
  - Native payload filtering: filter by source, doc_id, content_type at search time
  - Built-in persistence: no manual save/load pickle files
  - Qdrant Cloud: one env-var switch from local to managed cloud
  - REST + gRPC API: queryable from any language / external tool
  - HNSW index with configurable m and ef_construct at collection creation
"""

from __future__ import annotations
import asyncio
import uuid
from typing import Optional

from ..utils.config_loader import get_settings
from ..models.models import DocumentChunk, RetrievedChunk
from ..logger.logger import GLOBAL_LOGGER as logger

settings = get_settings()

class VectorStore:
    """
    Async Qdrant wrapper.

    All heavy Qdrant calls (upsert, search) are dispatched to a thread-pool
    executor because qdrant-client's sync API blocks the calling thread.
    This keeps the asyncio event loop free for other coroutines.

    Collection schema
    -----------------
    Each Qdrant point maps 1:1 to a DocumentChunk:
      id      -> chunk.id (UUID)
      vector  -> normalised float32 embedding
      payload -> all DocumentChunk fields (stored as JSON, filterable)
    """

    def __init__(self, dim: int= settings.embedding_dim) -> None:
        self.dim     = dim
        self._client = None     # initialised in connect()

    async def connect(self) -> None:
        """
        Connect to Qdrant and create the collection if it does not exist.
        Call once at application startup (inside FastAPI lifespan).

        Server mode: settings.qdrant_url is set -> connects to that Qdrant server.
        Local mode:  settings.qdrant_url is ""  -> in-process, persisted to cfg.local_path.
        """
        from qdrant_client import QdrantClient

        loop = asyncio.get_event_loop()

        def _build() -> QdrantClient:
            if settings.qdrant_url:
                return QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    timeout=30,
                )
            return QdrantClient(path=settings.qdrant_index_path)

        self._client = await loop.run_in_executor(None, _build)
        await loop.run_in_executor(None, self._ensure_collection)

        count = await self.count()
        logger.info(
            f"Qdrant ready  collection={settings.collection_name!r}"
            f"points={count}  dim={self.dim}"
        )

    def _ensure_collection(self) -> None:
        """
        Create the Qdrant collection with HNSW params from config.
        Idempotent — skipped if the collection already exists.
        """
        from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

        existing = [c.name for c in self._client.get_collections().collections]
        if settings.collection_name in existing:
            return

        self._client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=self.dim,
                # we use COSINE because the vectors are L2-normalised with sentence-transformers
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=settings.hnsw_m,
                    ef_construct=settings.hnsw_ef_construct,
                    on_disk=False,      # set True to reduce RAM for very large corpora
                ),
            ),
        )
        logger.info(
            f"Collection {settings.collection_name} created"
            f"m={settings.hnsw_m}  ef_construct={settings.hnsw_ef_construct}"
        )

    # Write to QDrant

    async def add_chunks(self, chunks: list[DocumentChunk], embeddings) -> None:
        """
        Upsert a batch of chunks into Qdrant.
        Upsert is idempotent on chunk.id — safe to retry on failure.
        """
        if not chunks:
            logger.info("Chunk is empty!")
            return
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._upsert_sync, chunks, embeddings)

    def _upsert_sync(self, chunks: list[DocumentChunk], embeddings) -> None:
        from qdrant_client.models import PointStruct

        points = []
        for chunk, emb in zip(chunks, embeddings):
            # Hoist doc_id to a top-level payload field so Qdrant can filter
            # on it directly without JSON path nesting.
            payload = {
                "id":           chunk.id,
                "content":      chunk.content,
                "source":       chunk.source,
                "chunk_index":  chunk.chunk_index,
               # "content_type": chunk.content_type.value,
                #"page_number":  chunk.page_number,
                "doc_id":       chunk.doc_id,
                "metadata":     chunk.metadata,
            }
            points.append(
                PointStruct(
                    id=str(uuid.UUID(chunk.id)),     # Qdrant requires valid UUID
                    vector=emb.tolist(),
                    payload=payload,
                )
            )

        # wait=True ensures the points are indexed before we return —
        # important for correctness, not just eventual consistency.
        self._client.upsert(
            collection_name=settings.collection_name,
            points=points,
            wait=True,
        )
        logger.debug(f"Upserted {len(points)} points")

    # Read from QDrant
    async def search(
        self,
        query_vec:  "np.ndarray",
        top_k:      int   = 10,
        threshold:  float = 0.65,
        ef_search:  int   = 128,
        filters:    Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        HNSW approximate nearest-neighbour search.

        ef_search  — query-time beam width:
                     higher  -> better recall, slower  (128-256 for production)
                     lower   -> faster, less accurate  (32-64 for latency-sensitive paths)

        filters    — Qdrant payload filter dict, e.g.
                     {"must": [{"key": "source", "match": {"value": "report.pdf"}}]}
                     Passed to Qdrant's Filter(**filters) constructor.
        """
        if self._client is None:
            logger.info("Qdrant not initialized!")
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._search_sync,
            query_vec.tolist(),
            top_k,
            threshold,
            ef_search,
            filters,
        )

    def _search_sync(
        self,
        query_vec: list,
        top_k:     int,
        threshold: float,
        ef_search: int,
        filters:   Optional[dict],
    ) -> list[RetrievedChunk]:
        from qdrant_client.models import Filter, SearchParams

        qdrant_filter = Filter(**filters) if filters else None

        hits = self._client.query_points(
            collection_name=settings.collection_name,
            query_vector=query_vec,
            limit=top_k,
            score_threshold=threshold,
            query_filter=qdrant_filter,
            search_params=SearchParams(
                hnsw_ef=ef_search,
                exact=False,            # True = brute-force exact search (testing only)
            ),
            with_payload=True,
        )
        return [self._point_to_chunk(hit) for hit in hits]

    @staticmethod
    def _point_to_chunk(hit) -> RetrievedChunk:
        """Reconstruct DocumentChunk from a Qdrant ScoredPoint payload."""
        from src.models.models import ContentType
        p = hit.payload
        chunk = DocumentChunk(
            id=           p["id"],
            content=      p["content"],
            source=       p["source"],
            chunk_index=  p["chunk_index", 0],
            content_type= ContentType(p.get("content_type", "prose")),
            page_number=  p.get("page_number"),
            metadata=     p.get("metadata", {}),
        )
        return RetrievedChunk(chunk=chunk, similarity_score=float(hit.score))

    #Delete from QDrant

    async def delete_document(self, doc_id: str) -> int:
        """
        Delete all points belonging to doc_id using a payload filter.
        No index rebuild — Qdrant handles this natively unlike FAISS.
        Returns number of points deleted.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._delete_sync, doc_id)

    def _delete_sync(self, doc_id: str) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        filt = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )
        # Count before so we can report how many were removed
        before = self._client.count(
            collection_name=settings.collection_name,
            count_filter=filt,
            exact=True,
        ).count

        if before == 0:
            return 0

        self._client.delete(
            collection_name=settings.collection_name,
            points_selector=filt,
            wait=True,
        )
        logger.info(f"Deleted doc_id={doc_id!r}  removed={before}")
        return before

    #Utilities to run the Qdrant_client.count() function as a async function

    async def count(self) -> int:
        if self._client is None:
            return 0
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._client.count(
                collection_name=settings.collection_name, exact=False
            ),
        )
        return result.count

    async def collection_info(self) -> dict:
        """Metadata dict for the /health endpoint."""
        if self._client is None:
            return {"status": "disconnected"}
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(
            None,
            lambda: self._client.get_collection(settings.collection_name),
        )
        return {
            "name":              settings.collection_name,
            "vectors_count":     info.vectors_count,
            "points_count":      info.points_count,
            "status":            str(info.status),
            "hnsw_m":            settings.hnsw_m,
            "hnsw_ef_construct": settings.hnsw_ef_construct,
        }
    
_store: Optional[VectorStore]=None

async def get_vector_store()-> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
        await _store.connect()

    return _store