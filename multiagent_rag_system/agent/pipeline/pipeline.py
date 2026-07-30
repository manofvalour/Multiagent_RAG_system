"""
Wires all agents into a single async pipeline.

Non-streaming flow:
  cache -> expand -> retrieve -> rerank -> generate -> cache + evaluate

Streaming flow:
  cache -> expand -> retrieve -> rerank -> stream tokens -> cache + evaluate
"""


from __future__ import annotations
import asyncio
import re
import time
import asyncio
import time
from typing import AsyncIterator, Optional
from multiagent_rag_system.src.observability.observability import traced

from multiagent_rag_system.src.utils.config_loader import get_settings
from multiagent_rag_system.src.logger import GLOBAL_LOGGER as logger
from multiagent_rag_system.src.exception.custom_exception import MulitagentragException
from multiagent_rag_system.src.models.models import (
    AgentEvent, QueryRequest, QueryResponse, 
    HallucinationRisk, ConfidenceBreakdown)
from langsmith import traceable

from multiagent_rag_system.src.cache.cache import SemanticCache

from ..agents.query_expansion import QueryExpansionAgent
from ..agents.evaluator import RAGASEvaluator
from ..agents.confidence_score_agent import ConfidenceScoringAgent
from ..agents.consensus_agent import ConsensusAgent
from ..agents.claim_verification_agent import ClaimVerificationAgent
from ..agents.reranker_agent import RerankerAgent
from ..agents.retrieval_agent import ChunkRetrieval
from ...src.embedding.embedding import EmbeddingProvider

settings = get_settings()


class RAGOrchestrator:
    def __init__(
        self, expansion=QueryExpansionAgent(), retriever= ChunkRetrieval(), 
        reranker=RerankerAgent(), consensus=ConsensusAgent(), cache=SemanticCache(), 
        evaluator= RAGASEvaluator(), confidence_score= ConfidenceScoringAgent(),
        claim_verification= ClaimVerificationAgent()
    ) -> None:
        
        self.expansion = expansion
        self.retriever = retriever
        self.reranker = reranker
        self.consensus = consensus
        self.confidence_scorer = confidence_score
        self.claim_verifier = claim_verification
        self.cache = cache
        #self.evaluator = evaluator

    @traced("orchestrator.run")
    async def run(self, query: QueryRequest) -> QueryResponse:
        """Full pipeline, non-streaming. Returns a complete RAGResponse."""

        t_total = time.perf_counter()
        trace: list[AgentEvent] = []

        #Cache check — return immediately on hit
        cached = await self.cache.get(query.query)
        if cached:
            cached.latency_ms = round((time.perf_counter() - t_total) * 1000, 2)
            logger.info(f"[Orchestrator] cache HIT  latency={cached.latency_ms:.0f}ms")
            return cached

        #Query expansion (HyDE / multi-query)
        expanded_queries, _ = await self.expansion.expand(query)

        #Multi-query retrieval + dedup
        filters = query.filters or None
        retrieved, ev = await self.retriever.retrieve(expanded_queries, filters=filters)
        trace.append(ev)

        if not retrieved:
            return QueryResponse(
                request_id=query.id,
                query=query.query,
                answer="I could not find relevant information to answer your question.",
                claims=[],
                retrieved_chunks=[],
                reranked_chunks=[],
                expanded_queries=expanded_queries,
                confidence=ConfidenceBreakdown(claim_support=0.0, avg_relevance=0.0, source_overlap=0.0, final=0.0),
                hallucination_risk=HallucinationRisk.MEDIUM,
                latency_ms=ev.duration_ms,
            )

        # 4. Cross-encoder reranking
        reranked, ev = await self.reranker.rerank(query.query, retrieved)
        trace.append(ev)

        # 5. Generation
        answer,_, ev = await self.consensus.run(query.query, reranked)
        trace.append(ev)

        if not isinstance(answer, str):
            answer = "".join([tok async for tok in answer])

        # Claim verification and confidence scoring (running claims first, then scoring)
        claims, ev_claims = await self.claim_verifier.run(answer, reranked)
        trace.append(ev_claims)

        confidence_score, hallucination_risk, ev_score = await self.confidence_scorer.run(answer, claims, reranked)
        trace.append(ev_score)

        # 8. Build response
        response = QueryResponse(
            request_id=query.id,
            query=query.query,
            answer=answer,
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
            expanded_queries=expanded_queries,
            confidence=confidence_score,
            claims=claims,
            hallucination_risk=hallucination_risk,
            latency_ms=round((time.perf_counter()- t_total)*1000, 2),
            agent_trace=trace if query.include_trace else [],
        )

        # 7. Cache + fire-and-forget RAGAS evaluation
        await self.cache.set(query.query, response)
       # asyncio.create_task(
     #       self.evaluator.evaluate(query.query, answer, reranked)
     #   )

        logger.info(
            f"[Orchestrator] retrieved= {len(retrieved)} | reranked= {len(reranked)}  "
            f"expanded={len(expanded_queries)} | latency= {response.latency_ms:.0f}ms"
        )
        return response

    async def run_streaming(self, query: QueryRequest) -> AsyncIterator[str]:
        """
        Streaming pipeline — yields SSE-formatted strings.
        Format: "data: <token>\\n\\n"  ...  "data: <json_meta>\\n\\n"  "data: [DONE]\\n\\n"
        The second-to-last event is a JSON blob with full response metadata (claims, confidence, etc).
        """
        t_total = time.perf_counter()
        trace: list[AgentEvent] = []

        # Cache hit: stream the cached answer word-by-word
        cached = await self.cache.get(query.query)
        if cached:
            for word in cached.answer.split():
                yield f"data: {word} \n\n"
            yield f"data: {cached.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Expand → retrieve → rerank (same as non-streaming)
        expanded_queries, _ = await self.expansion.expand(query)
        filters = query.filters or None
        retrieved, ev = await self.retriever.retrieve(expanded_queries, filters=filters)
        trace.append(ev)

        if not retrieved:
            yield "data: I could not find relevant information.\n\n"
            yield "data: [DONE]\n\n"
            return

        reranked, ev = await self.reranker.rerank(query.query, retrieved)
        trace.append(ev)

        # Generate (stream tokens)
        answer, _, ev = await self.consensus.run(query.query, reranked)
        trace.append(ev)

        if not isinstance(answer, str):
            answer = "".join([tok async for tok in answer])

        token_words = answer.split()
        for word in token_words:
            yield f"data: {word} \n\n"

        # Claim verification and confidence scoring
        claims, ev_claims = await self.claim_verifier.run(answer, reranked)
        trace.append(ev_claims)

        confidence_score, hallucination_risk, ev_score = await self.confidence_scorer.run(answer, claims, reranked)
        trace.append(ev_score)

        # Build full response
        response = QueryResponse(
            request_id=query.id,
            query=query.query,
            answer=answer,
            claims=claims,
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
            expanded_queries=expanded_queries,
            confidence=confidence_score,
            hallucination_risk=hallucination_risk,
            latency_ms=round((time.perf_counter() - t_total) * 1000, 2),
            agent_trace=trace if query.include_trace else [],
        )

        # Yield metadata as JSON SSE event, then done
        yield f"data: {response.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

        # Post-stream: cache + fire-and-forget evaluation
        await self.cache.set(query.query, response)
        asyncio.create_task(
            self.evaluator.evaluate(query.query, answer, reranked)
        )

        logger.info(
            f"[Orchestrator] streamed  retrieved={len(retrieved)} | reranked={len(reranked)}  "
            f"expanded={len(expanded_queries)} | latency={response.latency_ms:.0f}ms"
        )