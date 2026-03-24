"""
tests/test_agents.py
Unit tests for the AnswerGeneratorAgent — LLM-based answer generation
 """
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ..models.models import (DocumentChunk, 
                                 AgentStatus, Claim,
                                 RerankedChunk, RetrievedChunk)
from ..llm.llms import LLMResponse

#Shared fixtures
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
        return_value=LLMResponse(text="RAG reduces hallucinations by grounding answers.")
    )
    return llm


# AnswerGeneratorAgent
class TestAnswerGeneratorAgent:
    """
    AnswerGeneratorAgent.run(query, chunks, temperature) ->
        tuple[str, AgentEvent]

    Tests:
      - Happy path returns answer string + AgentEvent with DONE status
      - Empty chunks returns a graceful fallback message without LLM call
      - Temperature is offset per agent_id for ensemble diversity
      - LLM error raises MulitagentragException
    """

    def _make_agent(self, mock_llm, agent_id: int = 0):
        from multiagent_rag_system.agent.agents.answer_generator_agent import AnswerGeneratorAgent
        agent = AnswerGeneratorAgent(llm=mock_llm, agent_id=agent_id)
        
        # Override config.top_n so we don't need real settings
        agent.config = MagicMock()
        agent.config.top_n = 3
        return agent

    @pytest.mark.asyncio
    async def test_returns_answer_and_event(self, mock_llm, reranked_chunks):
        agent = self._make_agent(mock_llm)
        answer, event = await agent.run("What is RAG?", reranked_chunks)

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert event.status == AgentStatus.DONE
        mock_llm.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_fallback(self, mock_llm):
        agent = self._make_agent(mock_llm)
        answer, event = await agent.run("What is RAG?", [])

        assert "not contain" in answer.lower() or "sources do not" in answer.lower()
        assert event.status == AgentStatus.DONE
        mock_llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_temperature_offset_per_agent_id(self, reranked_chunks):
        """Agent 0 uses base temp, agent 1 uses base + 0.05, etc."""
        captured = []

        async def fake_complete(system, user, temperature=None):
            captured.append(temperature)
            return LLMResponse(text="answer")

        llm0 = MagicMock(); llm0.complete = fake_complete
        llm1 = MagicMock(); llm1.complete = fake_complete

        a0 = self._make_agent(llm0, agent_id=0)
        a1 = self._make_agent(llm1, agent_id=1)

        await a0.run("q", reranked_chunks, temperature=0.5)
        await a1.run("q", reranked_chunks, temperature=0.5)

        assert captured[0] == pytest.approx(0.50)
        assert captured[1] == pytest.approx(0.55)

    @pytest.mark.asyncio
    async def test_llm_error_raises_exception(self, reranked_chunks):
        from ..exception.custom_exception import MulitagentragException

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=Exception("API timeout"))
        agent = self._make_agent(llm)

        with pytest.raises(MulitagentragException):
            await agent.run("What is RAG?", reranked_chunks)

    @pytest.mark.asyncio
    async def test_answer_is_stripped(self, mock_llm, reranked_chunks):
        """Leading/trailing whitespace from LLM must be stripped."""
        mock_llm.complete = AsyncMock(
            return_value=LLMResponse(text="  answer with spaces  ")
        )
        agent = self._make_agent(mock_llm)
        answer, _ = await agent.run("q", reranked_chunks)
        assert not answer.startswith(" ")
        assert not answer.endswith(" ")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])