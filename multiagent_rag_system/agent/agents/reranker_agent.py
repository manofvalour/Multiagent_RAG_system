
from __future__ import annotations
import asyncio
import re
import time

from multiagent_rag_system.src.utils.config_loader import get_settings
from multiagent_rag_system.src.logger.logger import GLOBAL_LOGGER as logger
from multiagent_rag_system.src.exception.custom_exception import MulitagentragException
from multiagent_rag_system.src.models.models import (
    AgentEvent, AgentStatus,
    RetrievedChunk, RerankedChunk
    )
from multiagent_rag_system.src.utils.general_utils import _timed_event

settings= get_settings()
        
class RerankerAgent:

    NAME = "RerankerAgent"

    def __init__(self) -> None:
        self._model = None
        self.config = settings.reranker
 
    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.config.model)
            logger.info(f"CrossEncoder loaded: {self.config.model}")
 
    async def rerank(self, query: str, 
                  chunks: list[RetrievedChunk]) -> tuple[list[RerankedChunk], AgentEvent]:
        t0 = time.perf_counter()

        if not self.config.enabled or not chunks:
            reranked = [RerankedChunk(
                    chunk=c.chunk,
                    similarity_score=c.vector_score,
                )
                for c in chunks]
            
            event = _timed_event(agent=self.NAME, status=AgentStatus.DONE,
                            message="No chunk to rerank",
                            start=t0)
            
            return reranked, event
 
        loop = asyncio.get_event_loop()
 
        def _run_cross_encoder():
            self._load()
            pairs  = [(query, c.chunk.content) for c in chunks]
            scores = self._model.predict(pairs).tolist()
            return scores
 
        rerank_scores = await loop.run_in_executor(None, _run_cross_encoder)
 
        reranked = [
            RerankedChunk(
                chunk=c.chunk,
                similarity_score=c.vector_score,
                reranker_score=float(s),
            )
            for c, s in zip(chunks, rerank_scores)
        ]
        reranked.sort(key=lambda x: x.reranker_score, reverse=True)
        reranked = reranked[: self.config.top_n]
        dropped = len(chunks) - len(reranked)
 
        event = _timed_event(agent=self.NAME, status=AgentStatus.DONE,
                            message=f"Validated {len(reranked)}/{len(chunks)} chunks (dropped: {dropped})",
                            start=t0, kept=len(reranked), dropped=dropped,
                            threshold=self.threshold)
        logger.info(
            f"[Reranker] in={len(chunks)}  out={len(reranked)}  "
            f"top_score={reranked[0].reranker_score:.3f}"
        )

        return reranked, event
        