from __future__ import annotations
import asyncio
import time
from typing import Optional
import numpy as np

from ..src.utils.config_loader import get_settings
from ..src.logger.logger import GLOBAL_LOGGER as logger
from ..src.exception.custom_exception import MulitagentragException
from ..src.models.models import (
    AgentEvent, AgentStatus,RetrievedChunk)
from ..src.utils.general_utils import _timed_event

from ..src.database.vector_store import get_vector_store
from ..src.embedding.embedding import get_embedder

settings = get_settings()


class ChunkRetrieval:

    NAME = "ChunkRetrieval"

    def __init__(self):
        self.config= settings.retriever

    async def retrieve(
        self, queries:  list[str],
        filters:  Optional[dict] = None,
    ) -> tuple[list[RetrievedChunk], AgentEvent]:
        """
        queries  -- one or more strings (original + HyDE / multi-query variants)
        filters  -- optional Qdrant payload filter dict, e.g.
                    {"must": [{"key": "source", "match": {"value": "report.pdf"}}]}
        """
        t0 = time.perf_counter()

        loop = asyncio.get_event_loop()

        # Embed all query variants in a single batch
        embeddings: np.ndarray = await loop.run_in_executor(
            None,
            lambda: get_embedder.embed(
                queries
            ).astype(np.float32),
        )

        # Search concurrently -- one coroutine per query variant
        tasks = [
            get_vector_store.search(
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
                                message=f"{len(queries)} Retrieved Chunks {len(merged)})",
                                start=t0, kept=len(merged))
            
        logger.info(f"[Retriever] queries={len(queries)}  unique_chunks={len(merged)}")
        return merged, event
