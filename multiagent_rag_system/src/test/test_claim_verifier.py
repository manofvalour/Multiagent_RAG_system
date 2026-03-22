"""
tests/test_agents.py
Unit tests for the ClaimVerificationAgent — LLM + lexical claim verification
"""
from __future__ import annotations

import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ..models.models import (ContentType, DocumentChunk, 
                                 QueryResponse,
                                 AgentStatus, Claim,
                                 RerankedChunk, RetrievedChunk
)
from ..llm.llm import LLMResponse

def _make_doc(i: int, content: str = None) -> DocumentChunk:
    return DocumentChunk(
        id=f"aaaaaaaa-0000-0000-0000-00000000000{i}",
        content=content or f"RAG is a technique that combines retrieval and generation. Chunk {i}.",
        source=f"doc{i}.txt",
        chunk_index=i,
        doc_id=f"doc-00{i}",
        metadata={},
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

class TestClaimVerificationAgent:
    """
    ClaimVerificationAgent.run(answer, chunks) ->
        tuple[list[Claim], AgentEvent]

    Tests:
      - Splits answer into claims and verifies each
      - LLM path: supported=True when LLM returns {"supported": true}
      - LLM path: supported=False when LLM returns {"supported": false}
      - Fallback to lexical when LLM JSON is invalid
      - Empty answer returns empty claims list
      - Short sentences (< 15 chars) are filtered out
    """

    def _make_agent(self, llm=None, use_llm: bool = True):
        from multiagent_rag_system.agent.claim_verification_agent import ClaimVerificationAgent
        agent = ClaimVerificationAgent(llm=llm or MagicMock(), use_llm=use_llm)
        agent.config = MagicMock()
        agent.config.claim_support_threshold = 0.20
        return agent

    @pytest.mark.asyncio
    async def test_llm_supported_claim(self, reranked_chunks):
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=LLMResponse(
            text=json.dumps({"supported": True, "confidence": 0.95, "reason": "Found in source."})
        ))
        agent = self._make_agent(llm, use_llm=True)
        answer = "RAG reduces hallucinations by grounding answers in retrieved documents."

        claims, event = await agent.run(answer, reranked_chunks)
        assert len(claims) >= 1
        assert claims[0].supported is True
        assert event.status == AgentStatus.DONE

    @pytest.mark.asyncio
    async def test_llm_unsupported_claim(self, reranked_chunks):
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=LLMResponse(
            text=json.dumps({"supported": False, "confidence": 0.1, "reason": "Not in sources."})
        ))
        agent = self._make_agent(llm, use_llm=True)
        answer = "Quantum computers have replaced all classical computers globally."

        claims, event = await agent.run(answer, reranked_chunks)
        assert claims[0].supported is False

    @pytest.mark.asyncio
    async def test_falls_back_to_lexical_on_bad_json(self, reranked_chunks):
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=LLMResponse(text="not json at all"))
        agent = self._make_agent(llm, use_llm=True)
        answer = "RAG reduces hallucinations by grounding answers in retrieved documents."

        # Should not raise — lexical fallback handles the JSON parse error
        claims, event = await agent.run(answer, reranked_chunks)
        assert isinstance(claims, list)
        assert event.status == AgentStatus.DONE

    @pytest.mark.asyncio
    async def test_empty_answer_returns_empty_claims(self, reranked_chunks):
        agent = self._make_agent()
        claims, event = await agent.run("", reranked_chunks)
        assert claims == []
        assert event.status == AgentStatus.DONE

    @pytest.mark.asyncio
    async def test_short_sentences_filtered_out(self, reranked_chunks):
        """Sentences shorter than 15 chars should be dropped."""
        agent = self._make_agent(use_llm=False)
        # "Yes." and "OK." are shorter than 15 chars — should produce 0 or 1 real claim
        answer = "Yes. OK. RAG reduces hallucinations by grounding answers in retrieved documents."
        claims, _ = await agent.run(answer, reranked_chunks)
        # Only the long sentence should produce a claim
        for c in claims:
            assert len(c.text) > 15

    @pytest.mark.asyncio
    async def test_lexical_mode_no_llm_call(self, reranked_chunks):
        llm = MagicMock()
        llm.complete = AsyncMock()
        agent = self._make_agent(llm, use_llm=False)
        answer = "RAG reduces hallucinations by grounding answers in retrieved documents."

        await agent.run(answer, reranked_chunks)
        llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_n_supported_in_event_metadata(self, reranked_chunks):
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=LLMResponse(
            text=json.dumps({"supported": True, "confidence": 0.9, "reason": "ok"})
        ))
        agent = self._make_agent(llm, use_llm=True)
        answer = "RAG reduces hallucinations. Qdrant is a vector database."

        claims, event = await agent.run(answer, reranked_chunks)
        n_supported = sum(1 for c in claims if c.supported)
        # Event message should mention supported count
        assert str(n_supported) in event.message



if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])