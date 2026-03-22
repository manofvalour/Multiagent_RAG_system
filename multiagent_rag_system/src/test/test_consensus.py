

"""
tests/test_agents.py
Unit tests for ConsensusAgent — N-way parallel generation + majority vote
"""
from __future__ import annotations

import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ..models.models import (ContentType, DocumentChunk, 
                            QueryResponse, AgentStatus, 
                            Claim, RerankedChunk, RetrievedChunk)


def _make_doc(i: int, content: str = None) -> DocumentChunk:
    return DocumentChunk(
        id=f"aaaaaaaa-0000-0000-0000-00000000000{i}",
        content=content or f"RAG is a technique that combines retrieval and generation. Chunk {i}.",
        source=f"doc{i}.txt",
        chunk_index=i,

        metadata={},
        doc_id = f"doc-00{i}"
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
        RerankedChunk(chunk=docs[2], similarity_score=0.78, reranker_score=0.70),]

@pytest.fixture
def sample_claims(docs) -> list[Claim]:
    return [
        Claim(text="RAG reduces hallucinations.", supported=True,  confidence=0.9),
        Claim(text="Qdrant is a graph database.", supported=False, confidence=0.2),
        Claim(text="HNSW has sub-linear query time.", supported=True, confidence=0.85),]

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=QueryResponse(answer="RAG reduces hallucinations by grounding answers.")
    )
    return llm


class TestConsensusAgent:
    """
    ConsensusAgent.run(query, chunks) ->
        tuple[str, list[str], AgentEvent]

    Tests:
      - Returns best answer, all candidates, and DONE event
      - Handles partial generator failures gracefully
      - All generators failing returns FAILED event (no raise)
      - best answer is one of the candidates
    """

    def _make_agent(self, answers: list[str], n: int = None):
        from multiagent_rag_system.agent.consensus_agent import ConsensusAgent

        n = n or len(answers)
        # Build mock generators that return preset answers
        generators = []
        for i, ans in enumerate(answers):
            g = MagicMock()
            g.run = AsyncMock(return_value=(ans, MagicMock()))
            generators.append(g)

        agent = ConsensusAgent.__new__(ConsensusAgent)
        agent.generators = generators
        return agent

    @pytest.mark.asyncio
    async def test_returns_best_answer_and_all_candidates(self, reranked_chunks):
        answers = [
            "RAG reduces hallucinations by grounding answers in retrieved documents.",
            "RAG reduces hallucinations by grounding answers in retrieved documents.",
            "RAG is a retrieval system.",
        ]
        agent = self._make_agent(answers)
        best, candidates, event = await agent.run("What is RAG?", reranked_chunks)

        # The majority answer (appears twice) should win
        assert best == answers[0]
        assert len(candidates) == 3
        assert event.status == AgentStatus.DONE

    @pytest.mark.asyncio
    async def test_best_is_one_of_candidates(self, reranked_chunks):
        answers = ["Answer A", "Answer B", "Answer A"]
        agent = self._make_agent(answers)
        best, candidates, _ = await agent.run("q", reranked_chunks)
        assert best in candidates

    @pytest.mark.asyncio
    async def test_partial_failure_still_returns_answer(self, reranked_chunks):
        """Two generators succeed, one raises — should return DONE with 2 candidates."""
        from multiagent_rag_system.agent.consensus_agent import ConsensusAgent

        agent = ConsensusAgent.__new__(ConsensusAgent)
        g_ok1 = MagicMock(); g_ok1.run = AsyncMock(return_value=("Good answer.", MagicMock()))
        g_ok2 = MagicMock(); g_ok2.run = AsyncMock(return_value=("Good answer.", MagicMock()))
        g_bad = MagicMock(); g_bad.run = AsyncMock(side_effect=Exception("LLM down"))
        agent.generators = [g_ok1, g_ok2, g_bad]

        best, candidates, event = await agent.run("q", reranked_chunks)
        assert best == "Good answer."
        assert len(candidates) == 2
        assert event.status == AgentStatus.DONE

    @pytest.mark.asyncio
    async def test_all_generators_fail_returns_failed_event(self, reranked_chunks):
        from multiagent_rag_system.agent.consensus_agent import ConsensusAgent

        agent = ConsensusAgent.__new__(ConsensusAgent)
        agent.generators = [
            MagicMock(run=AsyncMock(side_effect=Exception("fail")))
            for _ in range(3)
        ]
        best, candidates, event = await agent.run("q", reranked_chunks)
        assert event.status == AgentStatus.FAILED
        assert len(candidates) == 0

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])