import asyncio
import pytest

from unittest.mock import AsyncMock, MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ..src.models.models import (
    DocumentChunk, QueryRequest, RetrievedChunk, HallucinationRisk
)

from ..agent.pipeline import (
    RetrievalValidationAgent, AnswerGeneratorAgent, ConsensusAgent,
    ClaimVerificationAgent, ConfidenceScoringAgent, MultiAgentRAGPipeline,
    _overlap_ratio, _word_text
)


def make_chunk(content: str, score: float = 0.8, source: str = "test") -> RetrievedChunk:
    chunk = DocumentChunk(
        id="test-id", doc_id="doc-1", content=content,
        source=source, embedding=[0.1] * 16,
    )
    return RetrievedChunk(chunk=chunk, vector_score=score, relevance_score=score)


CHUNKS = [
    make_chunk("RAG combines retrieval with generation to reduce hallucination in LLMs.", 0.9),
    make_chunk("Async Python enables parallel agent execution using asyncio.gather.", 0.8),
    make_chunk("Claim verification checks each assertion against source documents.", 0.75),
]


# Unit tests: helpers

def test_word_set_normalises():
    assert "rag" in _word_text("RAG systems are powerful!")
    assert "are" in _word_text("RAG systems are powerful!")

def test_overlap_ratio_identical():
    assert _overlap_ratio("hello world", "hello world") == pytest.approx(1.0)

def test_overlap_ratio_no_overlap():
    assert _overlap_ratio("hello world", "foo bar baz") == pytest.approx(0.0)

def test_overlap_ratio_partial():
    score = _overlap_ratio("hello world foo", "hello bar baz")
    assert 0.0 < score < 1.0


#Retrieval Validation Agent 
@pytest.mark.asyncio
async def test_retrieval_validation_keeps_relevant():
    agent = RetrievalValidationAgent(threshold=0.05)
    query = "RAG hallucination reduction"
    validated, event = await agent.run(query, CHUNKS[:])
    assert len(validated) > 0
    assert all(rc.relevance_score >= 0.05 for rc in validated)
    assert event.agent == "RetrievalValidation"

@pytest.mark.asyncio
async def test_retrieval_validation_drops_all_irrelevant():
    agent = RetrievalValidationAgent(threshold=0.99)  # impossibly high
    validated, _ = await agent.run("RAG", CHUNKS[:])
    assert validated == []

@pytest.mark.asyncio
async def test_retrieval_validation_sorts_by_relevance():
    agent = RetrievalValidationAgent(threshold=0.0)
    validated, _ = await agent.run("RAG hallucination", CHUNKS[:])
    scores = [rc.relevance_score for rc in validated]
    assert scores == sorted(scores, reverse=True)

@pytest.mark.asyncio
async def test_retrieval_validation_empty_input():
    agent = RetrievalValidationAgent()
    validated, event = await agent.run("query", [])
    assert validated == []
    assert event.status.value == "done"


#Answer Generator Agent
@pytest.mark.asyncio
async def test_answer_generator_uses_context():
    from ..src.llm.llm import SimulatedLLMClient
    agent = AnswerGeneratorAgent(llm=SimulatedLLMClient(), agent_id=0)
    answer, event = await agent.run("What is RAG?", CHUNKS)
    assert len(answer) > 20
    assert event.agent == "AnswerGenerator-0"

@pytest.mark.asyncio
async def test_answer_generator_refuses_empty_context():
    from ..src.llm.llm import SimulatedLLMClient
    agent = AnswerGeneratorAgent(llm=SimulatedLLMClient())
    answer, _ = await agent.run("What is RAG?", [])
    assert "not contain" in answer.lower() or "insufficient" in answer.lower()


#Consensus Agent
@pytest.mark.asyncio
async def test_consensus_returns_string():
    from ..src.llm.llm import SimulatedLLMClient
    # Patch generators to use simulated LLM
    with patch("multiagent_rag_system.agent.pipeline.AnswerGeneratorAgent", lambda agent_id=0: AnswerGeneratorAgent(SimulatedLLMClient(), agent_id)):
        agent = ConsensusAgent(n=3)
        for g in agent.generators:
            g.llm = SimulatedLLMClient()
        answer, candidates, event = await agent.run("What is RAG?", CHUNKS)
        assert isinstance(answer, str) and len(answer) > 0
        assert len(candidates) == 3
        assert event.metadata["n_candidates"] == 3

@pytest.mark.asyncio
async def test_consensus_single_agent():
    from ..src.llm.llm import SimulatedLLMClient
    agent = ConsensusAgent(n=1)
    agent.generators[0].llm = SimulatedLLMClient()
    answer, candidates, _ = await agent.run("test query", CHUNKS)
    assert len(candidates) == 1


#Claim Verification Agent 
@pytest.mark.asyncio
async def test_claim_verification_supports_grounded_claim():
    agent = ClaimVerificationAgent(use_llm=False)
    answer = "RAG combines retrieval with generation. Async Python enables parallel execution."
    claims, event = await agent.run(answer, CHUNKS)
    assert len(claims) >= 1
    assert any(c.supported for c in claims)

@pytest.mark.asyncio
async def test_claim_verification_flags_unsupported():
    agent = ClaimVerificationAgent(use_llm=False)
    # Completely unrelated claim
    answer = "The capital of France is Paris and the Eiffel Tower was built in 1889."
    claims, _ = await agent.run(answer, CHUNKS)
    assert any(not c.supported for c in claims)

@pytest.mark.asyncio
async def test_claim_verification_empty_answer():
    agent = ClaimVerificationAgent(use_llm=False)
    claims, event = await agent.run("", CHUNKS)
    assert claims == []

@pytest.mark.asyncio
async def test_claim_confidence_range():
    agent = ClaimVerificationAgent(use_llm=False)
    claims, _ = await agent.run("RAG combines retrieval with language models.", CHUNKS)
    for claim in claims:
        assert 0.0 <= claim.confidence <= 1.0


#Confidence Scoring Agent
@pytest.mark.asyncio
async def test_confidence_low_risk_with_supported_claims():
    agent = ConfidenceScoringAgent()
    from ..src.models.models import Claim
    claims = [
        Claim(text="RAG reduces hallucination", supported=True, confidence=0.9),
        Claim(text="Async Python helps parallelism", supported=True, confidence=0.8),
    ]
    breakdown, risk, event = await agent.run("RAG reduces hallucination with async Python.", claims, CHUNKS)
    assert 0.0 <= breakdown.final <= 1.0
    assert risk in [HallucinationRisk.LOW, HallucinationRisk.MEDIUM, HallucinationRisk.HIGH]

@pytest.mark.asyncio
async def test_confidence_high_risk_unsupported_claims():
    agent = ConfidenceScoringAgent()
    from ..src.models.models import Claim
    claims = [
        Claim(text="xyz", supported=False, confidence=0.0),
        Claim(text="abc", supported=False, confidence=0.0),
    ]
    breakdown, risk, _ = await agent.run("completely unrelated content", claims, [])
    assert risk == HallucinationRisk.HIGH

@pytest.mark.asyncio
async def test_confidence_breakdown_fields():
    agent = ConfidenceScoringAgent()
    from ..src.models.models import Claim
    claims = [Claim(text="test claim", supported=True, confidence=0.7)]
    breakdown, _, _ = await agent.run("test answer", claims, CHUNKS)
    assert hasattr(breakdown, "claim_support")
    assert hasattr(breakdown, "avg_relevance")
    assert hasattr(breakdown, "source_overlap")
    assert hasattr(breakdown, "final")


#Full pipeline integration
@pytest.mark.asyncio
async def test_full_pipeline_end_to_end():
    from ..src.llm.llm import SimulatedLLMClient

    pipeline = MultiAgentRAGPipeline()
    # Inject simulated LLM into all generators
    for g in pipeline.consensus.generators:
        g.llm = SimulatedLLMClient()
    pipeline.verifier.llm = SimulatedLLMClient()
    pipeline.verifier.use_llm = False  # use lexical for speed

    req = QueryRequest(query="How does RAG reduce hallucination?", top_k=5)
    result = await pipeline.run(req, CHUNKS)

    assert isinstance(result.answer, str) and len(result.answer) > 0
    assert 0.0 <= result.confidence.final <= 1.0
    assert result.hallucination_risk in list(HallucinationRisk)
    assert result.latency_ms > 0
    assert len(result.agent_trace) >= 4

@pytest.mark.asyncio
async def test_pipeline_with_empty_retrieval():
    from ..src.llm.llm import SimulatedLLMClient
    pipeline = MultiAgentRAGPipeline()
    for g in pipeline.consensus.generators:
        g.llm = SimulatedLLMClient()
    pipeline.verifier.use_llm = False

    req = QueryRequest(query="completely unknown topic", top_k=5)
    result = await pipeline.run(req, [])
    assert result.answer  # should gracefully refuse, not crash
    assert result.hallucination_risk == HallucinationRisk.HIGH

@pytest.mark.asyncio
async def test_pipeline_trace_disabled():
    from src.llm.llm import SimulatedLLMClient
    pipeline = MultiAgentRAGPipeline()
    for g in pipeline.consensus.generators:
        g.llm = SimulatedLLMClient()
    pipeline.verifier.use_llm = False

    req = QueryRequest(query="RAG test", top_k=3, include_trace=False)
    result = await pipeline.run(req, CHUNKS)
    assert result.agent_trace == []


#Ingestion tests
#@pytest.mark.asyncio
#async def test_ingestion_sentence_chunking():
 #   from ..agent.ingestion import DocumentIngestionPipeline
  #  pipeline = DocumentIngestionPipeline()
   # chunks = pipeline._chunk_sentences(
    #    "This is the first sentence. This is the second sentence. And the third one here.",
     #   max_words=20, overlap_words=5
    #)
   # assert len(chunks) >= 1
   # assert all(isinstance(c, str) and len(c) > 0 for c in chunks)

#def test_ingestion_fixed_chunking():
 #   from ..agent.ingestion import DocumentIngestionPipeline
  #  pipeline = DocumentIngestionPipeline()
   # text = " ".join([f"word{i}" for i in range(100)])
    #chunks = pipeline._chunk_fixed(text, size=20, overlap=5)
   # assert len(chunks) > 1
    # Each chunk should have ~20 words
   # for chunk in chunks:
    #    assert len(chunk.split()) <= 20

def test_ingestion_chunk_id_deterministic():
    from ..agent.ingestion import DocumentIngestionPipeline
    id1 = DocumentIngestionPipeline._chunk_id("doc1", 0, "same content")
    id2 = DocumentIngestionPipeline._chunk_id("doc1", 0, "same content")
    assert id1 == id2

def test_ingestion_chunk_id_unique():
    from ..agent.ingestion import DocumentIngestionPipeline
    id1 = DocumentIngestionPipeline._chunk_id("doc1", 0, "content a")
    id2 = DocumentIngestionPipeline._chunk_id("doc1", 1, "content b")
    assert id1 != id2


# ─── Performance tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_completes_under_5s():
    from ..src.llm.llm import SimulatedLLMClient
    import time
    pipeline = MultiAgentRAGPipeline()
    for g in pipeline.consensus.generators:
        g.llm = SimulatedLLMClient()
    pipeline.verifier.use_llm = False

    req = QueryRequest(query="How does RAG reduce hallucination?", top_k=5)
    t0 = time.perf_counter()
    result = await pipeline.run(req, CHUNKS * 3)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"Pipeline took {elapsed:.2f}s, expected < 5s"

@pytest.mark.asyncio
async def test_concurrent_queries():
    from ..src.llm.llm import SimulatedLLMClient
    pipeline = MultiAgentRAGPipeline()
    for g in pipeline.consensus.generators:
        g.llm = SimulatedLLMClient()
    pipeline.verifier.use_llm = False

    reqs = [QueryRequest(query=f"query {i}", top_k=3) for i in range(5)]
    results = await asyncio.gather(*[pipeline.run(r, CHUNKS) for r in reqs])
    assert len(results) == 5
    assert all(r.answer for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
    #test_retrieval_validation_sorts_by_relevance()