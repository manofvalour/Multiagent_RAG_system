from __future__ import annotations
import asyncio
import time
from typing import Optional
import numpy as np
import sys

from multiagent_rag_system.src.utils.config_loader import get_settings
from multiagent_rag_system.src.logger import GLOBAL_LOGGER as logger
from multiagent_rag_system.src.exception.custom_exception import MulitagentragException
from multiagent_rag_system.src.models.models import (
    AgentEvent, AgentStatus,RetrievedChunk)

from multiagent_rag_system.src.utils.general_utils import _timed_event
from multiagent_rag_system.src.database.vector_store import get_vector_store
from multiagent_rag_system.src.embedding.embedding import get_embedder

settings = get_settings()


class ChunkRetrieval:

    NAME = "ChunkRetrieval"

    def __init__(self):
        self.config= settings.retriever
        self.config = settings.retriever
        self._embedder = None
        self._vector_store = None

    async def _ensure_initialized(self):
        """Lazy load dependencies to avoid redundant calls."""
        if self._embedder is None:
            self._embedder = await get_embedder()
        if self._vector_store is None:
            self._vector_store = await get_vector_store()

    async def retrieve(
        self, queries:  list[str],
        filters:  Optional[dict] = None,
    ) -> tuple[list[RetrievedChunk], AgentEvent]:
        """
        queries  -- one or more strings (original + HyDE / multi-query variants)
        filters  -- optional Qdrant payload filter dict, e.g.
                    {"must": [{"key": "source", "match": {"value": "report.pdf"}}]}
        """
        try:
            t0 = time.perf_counter()

            await self._ensure_initialized()

            # Embed all query variants in a single batch
            raw_embedding = await self._embedder.embed(queries)
            embeddings:np.ndarray = np.array(raw_embedding).astype(np.float32)
            
            logger.info("Query variants embedded", n_queries=len(queries))

            # Search concurrently -- one coroutine per query variant
            tasks = [
                self._vector_store.search(
                    query_vec=emb,
                    top_k=self.config.top_k,
                    threshold=self.config.similarity_threshold,
                    ef_search=self.config.hnsw_ef_search,
                    filters=filters,
                )
                for emb in embeddings
            ]
            results_per_query: list[list[RetrievedChunk]] = await asyncio.gather(*tasks)

            # Merge: deduplicate by chunk id, keep the highest similarity score
            best: dict[str, RetrievedChunk] = {}
            for results in results_per_query:
                for r in results:
                    cid = r.chunk.id
                    if cid not in best or r.vector_score > best[cid].vector_score:
                        best[cid] = r

            merged = sorted(best.values(), key=lambda x: x.vector_score, reverse=True)
            event = _timed_event(agent=self.NAME, status=AgentStatus.DONE,
                                    message=f"{len(queries)} Unique Chunks {len(merged)})",
                                    start=t0, kept=len(merged))
                
            logger.info(f"[Retriever] queries={len(queries)}  unique_chunks={len(merged)}")
            return merged, event

        except Exception as e:
            logger.error("Failed to load the chunk retrieval", error=str(e))
            raise MulitagentragException("Failed to Load the Chunk Retrieval agent", error_details=str(e))