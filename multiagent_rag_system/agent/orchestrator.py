"""
src/agents/orchestrator.py
Wires all agents into a single async pipeline.

Non-streaming flow:
  cache → expand → retrieve → rerank → generate → cache + evaluate

Streaming flow:
  cache → expand → retrieve → rerank → stream tokens → cache + evaluate
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Optional

from ..src.logger.logger import GLOBAL_LOGGER as logger
from src.models.models import QueryRequest, QueryResponse
from ..src.core.observability import traced
from ..src.utils.config_loader import get_settings

settings = get_settings()


class RAGOrchestrator:
    def __init__(
        self,
        expansion,
        retriever,
        reranker,
        generator,
        cache,
        evaluator,
    ) -> None:
        self.expansion = expansion
        self.retriever = retriever
        self.reranker  = reranker
        self.generator = generator
        self.cache     = cache
        self.evaluator = evaluator

    @traced("orchestrator.run")
    async def run(self, query: QueryRequest) -> QueryResponse:
        """Full pipeline, non-streaming. Returns a complete RAGResponse."""
        t0 = time.perf_counter()

        # 1. Cache check — return immediately on hit
        cached = await self.cache.get(query.text)
        if cached:
            cached.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(f"[Orchestrator] cache HIT  latency={cached.latency_ms:.0f}ms")
            return cached

        # 2. Query expansion (HyDE / multi-query)
        expanded_queries, _ = await self.expansion.expand(query)

        # 3. Multi-query retrieval + dedup
        filters = query.filters or None
        retrieved = await self.retriever.retrieve(expanded_queries, filters=filters)
        if not retrieved:
            return QueryResponse(
                query_id=query.id,
                answer="I could not find relevant information to answer your question.",
                sources=[],
                retrieved_chunks=[],
                reranked_chunks=[],
                expanded_queries=expanded_queries,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        # 4. Cross-encoder reranking
        reranked = await self.reranker.rerank(query.text, retrieved)

        # 5. Generation
        answer = await self.generator.generate(query.text, reranked)
        if not isinstance(answer, str):
            answer = "".join([tok async for tok in answer])

        # 6. Build response
        sources  = list(dict.fromkeys(c.chunk.source for c in reranked))
        response = QueryResponse(
            query_id=query.id,
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
            expanded_queries=expanded_queries,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

        # 7. Cache + fire-and-forget RAGAS evaluation
        await self.cache.set(query.text, response)
        asyncio.create_task(
            self.evaluator.evaluate(query.text, answer, reranked)
        )

        logger.info(
            f"[Orchestrator] retrieved={len(retrieved)}  reranked={len(reranked)}  "
            f"expanded={len(expanded_queries)}  latency={response.latency_ms:.0f}ms"
        )
        return response

    async def run_streaming(self, query: RAGQuery) -> AsyncIterator[str]:
        """
        Streaming pipeline — yields SSE-formatted strings.
        Format: "data: <token>\\n\\n"  ...  "data: [DONE]\\n\\n"
        """
        # Cache hit: stream the cached answer word-by-word
        cached = await self.cache.get(query.text)
        if cached:
            for word in cached.answer.split():
                yield f"data: {word} \n\n"
            yield "data: [DONE]\n\n"
            return

        # Expand → retrieve → rerank (same as non-streaming)
        expanded_queries, _ = await self.expansion.expand(query)
        filters = query.filters or None
        retrieved = await self.retriever.retrieve(expanded_queries, filters=filters)

        if not retrieved:
            yield "data: I could not find relevant information.\n\n"
            yield "data: [DONE]\n\n"
            return

        reranked = await self.reranker.rerank(query.text, retrieved)

        # Stream tokens
        token_stream = await self.generator.generate(query.text, reranked)
        full_tokens: list[str] = []

        if isinstance(token_stream, str):
            yield f"data: {token_stream}\n\n"
            full_tokens = [token_stream]
        else:
            async for token in token_stream:
                yield f"data: {token}\n\n"
                full_tokens.append(token)

        yield "data: [DONE]\n\n"

        # Post-stream: cache + evaluate
        answer  = "".join(full_tokens)
        sources = list(dict.fromkeys(c.chunk.source for c in reranked))
        response = QueryResponse(
            query_id=query.id,
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
            expanded_queries=expanded_queries,
        )
        await self.cache.set(query.text, response)
        asyncio.create_task(
            self.evaluator.evaluate(query.text, answer, reranked)
        )