from __future__ import annotations
import asyncio
import re
import time
from collections import Counter
from typing import Optional
import numpy as np

from ..src.utils.config_loader import get_settings
from ..src.logger.logger import GLOBAL_LOGGER as logger
from ..src.exception.custom_exception import MulitagentragException
from ..src.models.models import (
    AgentEvent, AgentStatus, Claim, ConfidenceBreakdown,
    HallucinationRisk, QueryRequest, QueryResponse,
    RetrievedChunk, RerankedChunk
    )
from query_expansion import QueryExpansionAgent
from ..src.llm.llm import BaseLLMClient, LLMResponse, get_llm_client
from ..src.database.vector_store import get_vector_store
from ..src.embedding.embedding import get_embedder

""" Things to buid
        - Retrieval Validation Agent (Done)
        - Reranker Agent (Done)
        - Answer Generator Agent (Done)
        - Consensus Agent (Done)
        - Claim Verification Agent (Done)
        - Confidence Score Agent (Done)
"""
settings = get_settings()

def _word_text(text:str)-> set[str]:
    """cleaning the dataset. removing puntuation, 
    changing the case, and splitting per word"""

    return set(re.sub(r"[^\w\s]", "", text.lower()).split())

def _overlap_ratio(a: str, b:str)-> float:
    """measuring the textual similarities"""
    wa,wb = _word_text(a), _word_text(b)
    if not wa:
        return 0.0
    
    return len(wa & wb)/len(wa)

def _timed_event(agent:str, status: AgentStatus,
                 message:str, start:float, **meta)->AgentEvent:
    
    return AgentEvent(
        agent=agent,
        status =status,
        message =message,
        duration_ms = round((time.perf_counter()-start) *1000,2),
        metadata=meta,
    )


class ChunkRetrieval:

    NAME = "ChunkRetrieval"

    def __init__(self):
        pass

    async def run(
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
                top_k=settings.top_k_retrieval,
                threshold=settings.retrieval_relevance_threshold,
                ef_search=settings.hnsw_ef,
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

        
class RerankerAgent:

    NAME = "RerankerAgent"

    def __init__(self) -> None:
        self._model = None
        self.reranker = settings.reranker['reranker']
 
    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.reranker.model)
            logger.info(f"CrossEncoder loaded: {self.reranker.model}")
 
    async def run(self, query: str, 
                  chunks: list[RetrievedChunk]) -> tuple[list[RerankedChunk], AgentEvent]:
        t0 = time.perf_counter()

        if not self.reranker.enabled or not chunks:
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
        reranked = reranked[: self.reranker.top_n]
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
        

class AnswerGeneratorAgent:
    """
    Generates answer based on the validated retrieved. A single LLM call with a
    structured system prompt emphasising grounded, citation_based, and refusal to 
    speculate beyond provided sources

    """

    NAME = "AnswerGenerator"
    SYSTEM_PROMPT = """You are a precise, citation-grounded 
                question-answering assistant
                
                Rules you MUST follow:
                1. Answer only from the provided context passages.
                2. If the answer is not in the context, say: "The provided sources do not contain enough information to answer this question."
                3. Do NOT invent facts, statistics, names, or dates not present in the sources.
                4. Reference source numbers (e.g. [1], [2]) inline when using that source.
                5. Keep the answer concise: 2-4 sentences for simple questions, structured paragraphs for complex ones.
                6. Never Start with "Based on the context" - be direct.
                """
    
    def __init__(self, llm: Optional[BaseLLMClient]=None, agent_id:int=0):
        self.llm = llm or get_llm_client()
        self.agent_id = agent_id
        self.agent_name = f"{self.NAME}-{agent_id}"

    async def run(self,
            query:str, chunks:list[RetrievedChunk], 
            temperature: Optional[float]=None,
            )-> tuple[str, AgentEvent]:
        
        try:
            t0= time.perf_counter()

            if not chunks:
                msg= "The Provided sources do not contain enough information to answer this question."
                return msg, _timed_event(agent=self.agent_name, status=AgentStatus.DONE, message=msg,
                                         start=t0,)
            
            context_parts =[]
            for i, rc in enumerate(chunks[: settings.top_k_after_rerank]):
                context_parts.append(
                f"[{i}] Source:{rc.chunk.source}/n{rc.chunk.content.strip()}")

            # Vary the temperature per agent for ensemble when a base temperature is provided.
            # If no temperature is provided, let the LLM client use its default.
            temp = (temperature + self.agent_id * 0.05) if temperature is not None else None

            resp: LLMResponse = await self.llm.complete(
                system=self.SYSTEM_PROMPT, user = query, 
                temperature=temp,
            )
            msg = f"Generated answer ({resp.output_tokens} tokens)"
            event = _timed_event(
                agent=self.agent_name, status=AgentStatus.DONE, message=msg,
                start= t0, tokens = resp.total_tokens, llm_latency_ms=resp.latency_ms,
            )
            logger.info("Answer Generated by the LLM")
            return resp.text.strip(), event

        except Exception as e:
            logger.error("Answer Generation failed", agent=self.NAME, error=str(e))
            raise MulitagentragException(f"Unable to generate answer due LLM error", error_details=str(e))

    
class ConsensusAgent:
    """
    Runs N-numbers of AnswerGeneratorAgents in Parallel, 
    then select the best anser by a word-freq majority vote.
    """
    NAME = "Conssensus Agent"

    def __init__(self, n:int= settings.consensus_n_agents):
        self.generators = [AnswerGeneratorAgent(agent_id=i) for i in range(n)]

    async def run(self,
                query:str, chunks: list[RetrievedChunk]
              )-> tuple[str, list[str], AgentEvent]:
        try:
            t0 = time.perf_counter()

            results = await asyncio.gather(
                *[g.run(query,chunks) for g in self.generators], 
                return_exceptions=True,
            )
            answers = [r[0] for r in results if isinstance(r, tuple)]

            if not answers:
                return "Unable to generate consensus answer.", [], \
                _timed_event(agent = self.NAME, status=AgentStatus.FAILED, message="All generators failed", start=t0)
            
            word_freq: Counter = Counter()
            for ans in answers:
                word_freq.update(_word_text(ans))

            best= max(answers, key=lambda a: sum(word_freq[w] for w in _word_text(a)))
            event = _timed_event(
                agent = self.NAME, status= AgentStatus.DONE,
                message = f"Consensus from {len(answers)} candidates",
                start=t0, n_candidates = len(answers), best_len=len(best),
            )
            logger.info('Consensus_complete', **event.metadata)
            return best, answers, event
        
        except Exception as e:
            logger.error(f"Consensus agent failed", error=str(e))
            raise MulitagentragException("Consensus Agent failed! ", error_details=str(e))
        
class ClaimVerificationAgent:
    """
    Splits the answer into atomic claims (sentences) and verifies each against retrieved chunks via
    LLM-assisted entailment check. Fails back to lexical overlap
    when LLM is unavailable.
    """
    NAME = "ClaimVerification"

    VERIFY_SYSTEM = """You are a fact_checking assistant. Given a CLAIM and a list of SOURCE passages,
    determine if the CLAIM is directly supported by the sources,
    Respond with ONLY valid JSON: {"supported": true/false, "confidence": 0.0 - 1.0, "reason":"one Sentence"}
    No extra text, no markdown.
    """

    def __init__(self, llm: Optional[BaseLLMClient]=None, use_llm:bool=True):
        self.llm=llm or get_llm_client()
        self.use_llm = use_llm

    def _split_claims(self, text: str)-> list[str]:
        """ Split text into atomic sentence-level claims."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip())> 15]
    
    async def _verify_one_llm(self, claim: str, chunks: list[RetrievedChunk])->tuple[bool, float]:
        sources = "\n".join(
            f"[{i+1}] {rc.chunk.content[:200]}" for i,rc in enumerate(chunks[:4])
        )
        user_msg =f"CLAIM: {claim}\n\nSOURCES:\n{sources}"
        try:
            resp = await self.llm.complete(system=self.VERIFY_SYSTEM, user=user_msg, temperature=0.0)
            import json as _json
            data = _json.loads(resp.text.strip())
            return bool(data.get("supported")), float(data.get("confidence", 0.5))
        
        except Exception:
            return self._verify_lexical(claim,chunks)
        
    def _verify_lexical(self, claim: str, chunks: list[RetrievedChunk])->tuple[bool, float]:
        best = max(
            (_overlap_ratio(claim, rc.chunk.content) for rc in chunks),
            default=0.0,
        )
        supported = best>= settings.claim_support_threshold
        return supported, min(1.0, best*2)

    async def _lexical_async(self, claim: str, chunks: list[RetrievedChunk]):
        return self._verify_lexical(claim, chunks)
    
    async def run(
            self, answer: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[Claim], AgentEvent]:
        t0 = time.perf_counter()
        sentences = self._split_claims(answer)

        if not sentences:
            return [], _timed_event(agent=self.NAME, status=AgentStatus.DONE, message="No claims extracted", start=t0)
        
        ## verify all claims concurrently
        verify_tasks = [
            self._verify_one_llm(s, chunks) if self.use_llm
            else self._lexical_async(s, chunks)
            for s in sentences
        ]
        results = await asyncio.gather(*verify_tasks, return_exceptions=True)

        claims: list[Claim] = []
        for sentence, result in zip(sentences, results):
            if isinstance(result, Exception):
                supported, confidence = self._verify_lexical(sentence,chunks)
            else:
                supported, confidence = result
            supporting = [rc for rc in chunks if _overlap_ratio(sentence, rc.chunk.content)> 0.15]
            claims.append(Claim(
                text=sentence, supported = supported,
                confidence=round(confidence, 3), supporting_chunks = supporting[:2],
            ))
        
        n_supported = sum(1 for c in claims if c.supported)
        event = _timed_event(
            agent=self.NAME, status=AgentStatus.DONE,
            message=f"{n_supported}/{len(claims)} claims supported",
            start=t0, total=len(claims), supported = n_supported,
            unsupported=len(claims) - n_supported,
        )
        logger.info("Claim_verification", **event.metadata)
        return claims, event
    
class ConfidenceScoringAgent:
    """
    Computes a final [0,1] confidence score and assigns a hallucination risk
    level using a wighted combination fo three orthogonal signals
    """

    NAME = "ConfidenceScorer"
    WEIGHTS = {"claim_support": 0.50, "avg_relevance":0.30, "source_overlap":0.20}

    async def run(
            self, answer: str, claims: list[Claim], chunks: list[RetrievedChunk]
    ) -> tuple[ConfidenceBreakdown, HallucinationRisk, AgentEvent]:
        t0 = time.perf_counter()

        claim_support = (
            sum(1 for c in claims if c.supported)/max(len(claims), 1)
        )
        avg_relevance = (
            sum(rc.relevance_score for rc in chunks)/ max(len(chunks),1)
        )

        src_text = " ".join(rc.chunk.content for rc in chunks)
        source_overlap = _overlap_ratio(answer, src_text)

        final = (
            self.WEIGHTS["claim_support"] * claim_support
            +self.WEIGHTS['avg_relevance'] * avg_relevance
            + self.WEIGHTS['source_overlap']* source_overlap
        )
        final = round(min(1.0, max(0.0, final)),4)

        if final >= settings.confidence_low_threshold:
            risk = HallucinationRisk.LOW
        elif final>= settings.confidence_medium_threshold:
            risk = HallucinationRisk.MEDIUM
        else:
            risk = HallucinationRisk.HIGH

        breakdown = ConfidenceBreakdown(
            claim_support=round(claim_support, 4),
            avg_relevance = round(avg_relevance, 4),
            source_overlap=round(source_overlap, 4),
            final= final,
        )
        event = _timed_event(
            agent=self.NAME, status=AgentStatus.DONE,
            message=f"Confidence={final:.3f}, Risk={risk.value}",
            start=t0, **breakdown.model_dump(),
        )
        logger.info("confidence_score", **event.metadata)
        return breakdown, risk, event

class MultiAgentRAGPipeline:
    """
    Orchestrates the full 5-agent pipeline.
    Agents 4+5 run concurrently after generation.
    """
    def __init__(self):
        self.query_expander = QueryExpansionAgent()
        self.retrieval = ChunkRetrieval()
        self.validator = RetrievalValidationAgent()
        self.reranker = RerankerAgent()
        self.consensus = ConsensusAgent()
        self.verifier = ClaimVerificationAgent()
        self.scorer = ConfidenceScoringAgent()

    async def run(
            self, request: QueryRequest,
    )-> QueryResponse:
        t_total = time.perf_counter()
        trace: list[AgentEvent] = []

        #Query expander
        expand_query, _ = await self.query_expander(request.query)

        #Chunk Retrieval
        retrieved_chunk, ev = await self.retrieval.run(expand_query)
        trace.append(ev)

        ## Reranking with an LLM
        reranked, ev = await self.reranker.run(request.query, retrieved_chunk)
        trace.append(ev)

        # Consensus Generation
        answer, all_answers, ev = await self.consensus.run(request.query, reranked)
        trace.append(ev)

        # Claim verification and confidence scoring (running concurrently)
        (claims, ev_claims), (breakdown, risk, ev_score) = await asyncio.gather(
            self.verifier.run(answer, reranked),
            self.scorer.run(answer, [], reranked),
        )
        trace.extend([ev_claims, ev_score])

        breakdown, risk, ev_final = await self.scorer.run(answer, claims, reranked)
        trace.append(ev_final)

        return QueryResponse(
            query=request.query,
            answer=answer,
            claims = claims,
            retrieved_chunks=reranked,
            confidence = breakdown,
            hallucination_risk = risk,
            latency_ms=round((time.perf_counter()- t_total)*1000, 2),
            agent_trace=trace if request.include_trace else [],
        )

if __name__=="__main__":
    h = MultiAgentRAGPipeline()
    print('correct')