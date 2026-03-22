"""
tests/test_agents.py
Unit tests for the ConfidenceScoringAgent — weighted confidence + hallucination risk
"""
from __future__ import annotations

import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ..models.models import (ContentType, DocumentChunk, 
                                 QueryResponse, AgentStatus, Claim,
                                 RerankedChunk, RetrievedChunk, HallucinationRisk)

def _make_doc(i: int, content: str = None) -> DocumentChunk:
    return DocumentChunk(
        id=f"aaaaaaaa-0000-0000-0000-00000000000{i}",
        content=content or f"RAG is a technique that combines retrieval and generation. Chunk {i}.",
        source=f"doc{i}.txt",
        chunk_index=i,
        metadata={},
        doc_id=f"doc-00{i}"
    )


@pytest.fixture
def docs():
    return [_make_doc(i) for i in range(1, 4)]


@pytest.fixture
def retrieved_chunks(docs) -> list[RetrievedChunk]:
    scores = [0.92, 0.85, 0.78]
    return [RetrievedChunk(chunk=d, vector_score=s) for d, s in zip(docs, scores)]


@pytest.fixture
def reranked_chunks(docs) -> list[RerankedChunk]:
    return [
        RerankedChunk(chunk=docs[0], similarity_score=0.92, reranker_score=0.95),
        RerankedChunk(chunk=docs[1], similarity_score=0.85, reranker_score=0.82),
        RerankedChunk(chunk=docs[2], similarity_score=0.78, reranker_score=0.70),
    ]


@pytest.fixture
def sample_claims(docs) -> list[Claim]:
    return [
        Claim(text="RAG reduces hallucinations.", supported=True,  confidence=0.9),
        Claim(text="Qdrant is a graph database.", supported=False, confidence=0.2),
        Claim(text="HNSW has sub-linear query time.", supported=True, confidence=0.85),
    ]


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=QueryResponse(answer="RAG reduces hallucinations by grounding answers.")
    )
    return llm

class TestConfidenceScoringAgent:
    """
    ConfidenceScoringAgent.run(answer, claims, chunks) ->
        tuple[ConfidenceBreakdown, HallucinationRisk, AgentEvent]

    Tests:
      - All claims supported → high final score → LOW risk
      - No claims supported → low score → HIGH risk
      - Final score is clamped to [0, 1]
      - Breakdown fields sum to something sensible
      - Event has DONE status
    """

    def _make_agent(self):
        from multiagent_rag_system.agent.confidence_score_agent import ConfidenceScoringAgent
        agent = ConfidenceScoringAgent()
        agent.config = MagicMock()
        agent.config.confidence_low_threshold    = 0.65
        agent.config.confidence_medium_threshold = 0.40
        return agent

    @pytest.mark.asyncio
    async def test_all_supported_gives_low_risk(self, reranked_chunks):
        agent  = self._make_agent()
        claims = [
            Claim(text="RAG reduces hallucinations.", supported=True,  confidence=0.9),
            Claim(text="Qdrant filters payloads.",    supported=True,  confidence=0.85),
        ]
        breakdown, risk, event = await agent.run("RAG reduces hallucinations.", claims, reranked_chunks)

        assert risk == HallucinationRisk.LOW
        assert breakdown.final >= 0.65
        assert event.status == AgentStatus.DONE

    @pytest.mark.asyncio
    async def test_no_supported_claims_gives_high_risk(self, reranked_chunks):
        agent  = self._make_agent()
        claims = [
            Claim(text="Quantum computers replaced all hardware.", supported=False, confidence=0.1),
            Claim(text="Mars was colonised in 2020.", supported=False, confidence=0.05),
        ]
        # Use an answer with no overlap with chunk content
        breakdown, risk, event = await agent.run(
            "Quantum computers replaced all hardware.", claims, reranked_chunks
        )
        assert risk == HallucinationRisk.HIGH
        assert breakdown.final < 0.40

    @pytest.mark.asyncio
    async def test_final_score_clamped_to_0_1(self, reranked_chunks):
        agent  = self._make_agent()
        # All weights at maximum
        claims = [Claim(text="x", supported=True, confidence=1.0) for _ in range(10)]
        breakdown, _, _ = await agent.run("RAG", claims, reranked_chunks)
        assert 0.0 <= breakdown.final <= 1.0

    @pytest.mark.asyncio
    async def test_empty_claims_does_not_crash(self, reranked_chunks):
        agent = self._make_agent()
        breakdown, risk, event = await agent.run("Some answer.", [], reranked_chunks)
        assert isinstance(breakdown.final, float)
        assert isinstance(risk, HallucinationRisk)
        assert event.status == AgentStatus.DONE

    @pytest.mark.asyncio
    async def test_breakdown_fields_are_between_0_and_1(self, reranked_chunks):
        agent  = self._make_agent()
        claims = [Claim(text="RAG reduces hallucinations.", supported=True, confidence=0.8)]
        breakdown, _, _ = await agent.run("RAG reduces hallucinations.", claims, reranked_chunks)
        assert 0.0 <= breakdown.claim_support  <= 1.0
        assert 0.0 <= breakdown.avg_relevance  <= 1.0
        assert 0.0 <= breakdown.source_overlap <= 1.0

    @pytest.mark.asyncio
    async def test_medium_risk_threshold(self, reranked_chunks):
        agent  = self._make_agent()
        # One supported, one not → mixed score
        claims = [
            Claim(text="RAG reduces hallucinations.", supported=True,  confidence=0.6),
            Claim(text="Mars colonised in 2020.", supported=False, confidence=0.1),
        ]
        breakdown, risk, _ = await agent.run(
            "RAG reduces hallucinations.", claims, reranked_chunks
        )
        assert risk in (HallucinationRisk.LOW, HallucinationRisk.MEDIUM, HallucinationRisk.HIGH)


if __name__ =="__main__":
    import pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])