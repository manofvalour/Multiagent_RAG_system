"""
tests/test_agents.py
Unit tests for the QueryExpansionAgent    — HyDE + multi-query via Groq
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ..models.models import (ContentType, DocumentChunk, 
                                  QueryRequest, QueryResponse,
                                 Claim, RerankedChunk, RetrievedChunk)

def _make_doc(i: int, content: str = None) -> DocumentChunk:
    return DocumentChunk(
        id=f"aaaaaaaa-0000-0000-0000-00000000000{i}",
        content=content or f"RAG is a technique that combines retrieval and generation. Chunk {i}.",
        source=f"doc{i}.txt",
        chunk_index=i,

        metadata={},
        doc_id =f"doc-00{i}"
    )


@pytest.fixture
def docs():
    return [_make_doc(i) for i in range(1, 4)]

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


class TestQueryExpansionAgent:
    """
    QueryExpansionAgent.expand(query) ->
        tuple[list[str], Optional[str]]

    Tests:
      - Disabled expansion returns only original query
      - hyde strategy returns [original, hyde_doc] + hyde_doc as second element
      - multi_query strategy returns [original] + N variants, None as second element
      - both strategy returns all combined, hyde_doc as second element
      - original query is always first in the list
    """

    def _make_agent(self, strategy: str = "hyde", enabled: bool = True):
        from multiagent_rag_system.agent.query_expansion import QueryExpansionAgent

        agent = QueryExpansionAgent.__new__(QueryExpansionAgent)
        agent.api_key  = "test-key"
        agent._groqai  = None

        config_mock = MagicMock()
        config_mock.enabled = enabled
        config_mock.strategy = strategy
        config_mock.hyde_temperature = 0.7
        config_mock.num_queries = 3

        model_config_mock = MagicMock()
        model_config_mock.model_name = "test-model"

        agent.config = config_mock
        agent.model_config = model_config_mock

        return agent

    def _mock_groq_response(self, text: str):
        resp = MagicMock()
        resp.choices[0].message.content = text
        return resp

    @pytest.mark.asyncio
    async def test_disabled_returns_only_original(self):
        agent = self._make_agent(enabled=False)
        q = QueryRequest(query="What is RAG?")
        queries, hyde = await agent.expand(q)
        assert queries == ["What is RAG?"]
        assert hyde is None

    @pytest.mark.asyncio
    async def test_hyde_strategy(self):
        agent = self._make_agent(strategy="hyde")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=self._mock_groq_response("RAG retrieves documents and uses them.")
        )
        agent._groqai = mock_client

        q = QueryRequest(query="What is RAG?")
        queries, hyde = await agent.expand(q)

        assert queries[0] == "What is RAG?"
        assert "RAG retrieves" in queries[1]
        assert hyde == queries[1]
        assert len(queries) == 2

    @pytest.mark.asyncio
    async def test_multi_query_strategy(self):
        agent = self._make_agent(strategy="multi_query")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=self._mock_groq_response(
                "How does RAG work?\nExplain retrieval augmented generation\nWhat makes RAG useful?"
            )
        )
        agent._groqai = mock_client

        q = QueryRequest(query="What is RAG?")
        queries, hyde = await agent.expand(q)

        assert queries[0] == "What is RAG?"
        assert hyde is None
        assert len(queries) == 4   # original + 3 rephrasings

    @pytest.mark.asyncio
    async def test_both_strategy_combines_results(self):
        agent = self._make_agent(strategy="both")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            self._mock_groq_response("RAG retrieves and generates."),  # HyDE call
            self._mock_groq_response("How does RAG work?\nExplain RAG\nWhat is RAG used for?"),
        ])
        agent._groqai = mock_client

        q = QueryRequest(query="What is RAG?")
        queries, hyde = await agent.expand(q)

        assert queries[0] == "What is RAG?"
        assert hyde is not None
        assert hyde in queries
        assert len(queries) >= 4  # original + hyde + multi-query variants

    @pytest.mark.asyncio
    async def test_original_always_first(self):
        for strategy in ("hyde", "multi_query", "both"):
            agent = self._make_agent(strategy=strategy)
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=self._mock_groq_response("some content\nmore content\neven more")
            )
            agent._groqai = mock_client
            q = QueryRequest(query="test query")
            queries, _ = await agent.expand(q)
            assert queries[0] == "test query", f"Failed for strategy={strategy}"
 
if __name__=="__main__":
    import pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
