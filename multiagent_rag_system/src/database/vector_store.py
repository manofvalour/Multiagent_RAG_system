"""
Two connection modes (auto-detected from config):
  Server mode — connects to a running Qdrant instance (local Docker or Qdrant Cloud)
  Local mode  — in-process Qdrant stored on disk (no Docker needed, dev/test)

Why Qdrant over FAISS?
  - Full CRUD: delete individual points without rebuilding the entire index
  - Native payload filtering: filter by sourc, doc_id, content_type at search time
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
from ..models.models import DocumentChunk, RetrievedChunk, PaperMetadata
from ..logger import GLOBAL_LOGGER as logger

settings = get_settings()

class VectorStore:
    """
    Async Qdrant wrapper supporting both general RAG chunks and research papers.

    All heavy Qdrant calls (upsert, search) are dispatched to a thread-pool
    executor because qdrant-client's sync API blocks the calling thread.
    This keeps the asyncio event loop free for other coroutines.

    Collection schema
    -----------------
    Each Qdrant point maps 1:1 to a DocumentChunk:
      id      -> chunk.id (UUID)
      vector  -> normalised float32 embedding
      payload -> all DocumentChunk fields (stored as JSON, filterable)

    For research papers, additional metadata fields are stored:
      paper_id, title, authors, abstract, source, url, pdf_url, topics, citation_count
    """

    RESEARCH_PAPERS_COLLECTION = "research_papers"

    def __init__(
        self,
        dim: int = settings.embeddings.embedding_dim,
        collection_name: Optional[str] = None,
    ) -> None:
        self.config = settings.vector_store
        self.dim = dim
        self._client = None  # initialised in connect()
        # Use specified collection or fall back to config default
        self._collection_name = collection_name or self.config.collection_name

    async def connect(self) -> None:
        """
        Connect to Qdrant and create the collection if it does not exist.
        Call once at application startup (inside FastAPI lifespan).

        Server mode: settings.qdrant_url is set -> connects to that Qdrant server.
        Local mode:  settings.qdrant_url is ""  -> in-process, persisted to cfg.local_path.
        """
        if self._client is not None:
            return

        from qdrant_client import QdrantClient

        loop = asyncio.get_event_loop()

        def _build() -> QdrantClient:
            endpoint = settings.qdrant_endpoint.get_secret_value()
            if endpoint:
                return QdrantClient(
                    url=endpoint,
                    api_key=settings.qdrant_api_key.get_secret_value(),
                    timeout=self.config.timeout,
                )
            return QdrantClient(path=self.config.local_path)

        self._client = await loop.run_in_executor(None, _build)
        await loop.run_in_executor(None, self._ensure_collection)

        count = await self.count()
        logger.info(
            f"Qdrant ready  collection={self._collection_name!r}"
            f"points={count}  dim={self.dim}"
        )

    def _ensure_collection(self) -> None:
        """
        Create the Qdrant collection with HNSW params from config.
        Idempotent — skipped if the collection already exists.
        """
        from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

        if self._client.collection_exists(self._collection_name):
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self.dim,
                # we use COSINE because the vectors are L2-normalised with sentence-transformers
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=self.config.hnsw_m,
                    ef_construct=self.config.hnsw_ef_construct,
                    on_disk=False,      # set True to reduce RAM for very large corpora
                ),
            ),
        )
        logger.info(
            f"Collection {self._collection_name} created"
            f"m={self.config.hnsw_m}  ef_construct={self.config.hnsw_ef_construct}"
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
                "chunk_index":  chunk.chunk_index,
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

        # wait=False: index asynchronously so the HTTP request doesn't time out on large batches
        # Data is searchable shortly after the response returns
        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=False,
            timeout=self.config.timeout,
        )
        logger.info(f"Upserted {len(points)} points")

    # Read from QDrant
    async def search(
        self,
        query_vec: "np.ndarray",
        top_k: int   = 10,
        threshold: float = 0.65,
        ef_search: int   = 128,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        HNSW approximate nearest-neighbour search.
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
        query_vec: list,top_k: int, threshold: float,
        ef_search: int, filters: Optional[dict],
    ) -> list[RetrievedChunk]:
        
        from qdrant_client.models import Filter, SearchParams
        logger.info(f"Raw Filters recieved: {filters}")

        qdrant_filter = None
        if filters:
            allowed_keys = {"must", "should", "must_not", "min_should"}
            sanitized_filters = {
                k: v for k, v in filters.items() 
                if k in allowed_keys and v is not None
            }

            if sanitized_filters:
                try:
                    qdrant_filter = Filter.model_validate(sanitized_filters)
                except Exception as e:
                    logger.warning(f"Filter validation failed: {e}. Proceeding without filters.")
                    qdrant_filter = None
           # qdrant_filter = Filter.model_validate(filters) if filters else None

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vec,
            limit=top_k,
            score_threshold=threshold,
            query_filter=qdrant_filter,
            search_params=SearchParams(
                hnsw_ef=ef_search,
                exact=False,            # True = brute-force exact search (testing only)
            ),
            with_payload=True,
        )

        points = response.points if hasattr(response, "points") else response
        return [self._point_to_chunk(hit) for hit in points]

    @staticmethod
    def _point_to_chunk(hit) -> RetrievedChunk:
        """Reconstruct DocumentChunk from a Qdrant ScoredPoint payload."""
        from ..models.models import ContentType

        if isinstance(hit, tuple):
            # Some qdrant wrappers may return ("points", [...]) when iterated
            _, points = hit
            if not points:
                raise ValueError("no scored points in query result tuple")
            scored_point = points[0]
        else:
            scored_point = hit

        p = scored_point.payload

        chunk = DocumentChunk(
            id=           p["id"],
            content=      p["content"],
            chunk_index=  p.get("chunk_index", 0),
            content_type= ContentType(p.get("content_type", "prose")),
            page_number=  p.get("page_number"),
            doc_id=       p.get("doc_id"),
            metadata=     p.get("metadata", {}),
        )
      #  logger.info(scored_point.score, min(1.0, max(0.0, float(scored_point.score))))
        return RetrievedChunk(chunk=chunk, vector_score=min(1.0, max(0.0, float(scored_point.score))))

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
            collection_name=self._collection_name,
            count_filter=filt,
            exact=True,
        ).count

        if before == 0:
            return 0

        self._client.delete(
            collection_name=self._collection_name,
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
                collection_name=self._collection_name, exact=False
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
            lambda: self._client.get_collection(self._collection_name),
        )
        return {
            "name":              self._collection_name,
            "vectors_count":     getattr(info, "vectors_count", info.points_count),
            "points_count":      info.points_count,
            "status":            str(info.status),
            "hnsw_m":            self.config.hnsw_m,
            "hnsw_ef_construct": self.config.hnsw_ef_construct,
        }

    # Research paper-specific methods

    async def search_paper_by_id(self, paper_id: str) -> list[PaperMetadata]:
        """Search for a paper by its ID in the research_papers collection."""
        if self._client is None:
            return []
        from ..models.models import PaperMetadata as PMPaperMetadata
        loop = asyncio.get_event_loop()

        def _search():
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            filt = Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            )
            response = self._client.query_points(
                collection_name=self.RESEARCH_PAPERS_COLLECTION,
                query=[0.0] * self.dim,
                limit=1,
                query_filter=filt,
                with_payload=True,
            )
            return response.points if hasattr(response, "points") else response

        points = await loop.run_in_executor(None, _search)
        results = []
        for hit in points:
            p = hit.payload
            results.append(PMPaperMetadata(
                paper_id=p.get("paper_id", ""),
                title=p.get("title", ""),
                authors=p.get("authors", []),
                abstract=p.get("abstract", ""),
                source=p.get("source", "arxiv"),
                url=p.get("url", ""),
                pdf_url=p.get("pdf_url"),
                published_date=p.get("published_date"),
                topics=p.get("topics", []),
                citation_count=p.get("citation_count"),
            ))
        return results

    async def get_paper_chunks(self, paper_id: str) -> list[DocumentChunk]:
        """Get all document chunks for a specific paper."""
        if self._client is None:
            return []
        loop = asyncio.get_event_loop()

        def _search():
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            filt = Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            )
            response = self._client.query_points(
                collection_name=self.RESEARCH_PAPERS_COLLECTION,
                query=[0.0] * self.dim,
                limit=100,
                query_filter=filt,
                with_payload=True,
            )
            return response.points if hasattr(response, "points") else response

        points = await loop.run_in_executor(None, _search)
        chunks = []
        for hit in points:
            p = hit.payload
            chunks.append(DocumentChunk(
                id=p.get("id", ""),
                doc_id=p.get("doc_id", ""),
                content=p.get("content", ""),
                metadata=p.get("metadata", {}),
                chunk_index=p.get("chunk_index", 0),
            ))
        return chunks

    async def add_paper_chunks(
        self, chunks: list[DocumentChunk], embeddings, paper: "PaperMetadata"
    ) -> None:
        """Add paper chunks with paper metadata to the research_papers collection."""
        if not chunks:
            return

        loop = asyncio.get_event_loop()

        def _upsert():
            from qdrant_client.models import PointStruct

            points = []
            for chunk, emb in zip(chunks, embeddings):
                payload = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "doc_id": chunk.doc_id,
                    "metadata": chunk.metadata,
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "source": paper.source.value if hasattr(paper.source, 'value') else str(paper.source),
                    "url": paper.url,
                    "pdf_url": paper.pdf_url,
                    "topics": paper.topics,
                }
                points.append(
                    PointStruct(
                        id=str(uuid.UUID(chunk.id)),
                        vector=emb.tolist(),
                        payload=payload,
                    )
                )
            self._client.upsert(
                collection_name=self.RESEARCH_PAPERS_COLLECTION,
                points=points,
                wait=True,
            )

        await loop.run_in_executor(None, _upsert)


_store: Optional[VectorStore] = None


async def get_vector_store(collection_name: Optional[str] = None) -> VectorStore:
    """Singleton for the default vector store (rag_chunks collection)."""
    global _store
    if _store is None:
        _store = VectorStore(collection_name=collection_name)
        await _store.connect()
    return _store


_paper_store: Optional[VectorStore] = None


async def get_paper_vector_store() -> VectorStore:
    """Singleton for research papers vector store."""
    global _paper_store
    if _paper_store is None:
        _paper_store = VectorStore(collection_name=VectorStore.RESEARCH_PAPERS_COLLECTION)
        await _paper_store.connect()
    return _paper_store