from __future__ import annotations
import asyncio
import re
import time
from collections import Counter
from typing import Optional

from ..src.utils.config_loader import get_settings
from ..src.logger.logger import GLOBAL_LOGGER as logger
from ..src.exception.custom_exception import MulitagentragException
from ..src.models.models import (
    AgentEvent, AgentStatus, Claim, ConfidenceBreakdown,
    HallucinationRisk, QueryRequest, QueryResponse,
    RetrievedChunk,
    )
from ..src.llm.llm import BaseLLMClient, LLMResponse, get_llm_client

""" Things to buid
        - Retrieval Validation Agent (Done)
        - Answer Generator Agent (Done)
        - Consensus Agent
        - Claim Verification Agent
        - Confidence Score Agent
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


class RetrievalValidationAgent:
    """
    Combines vector similarity with keyword overlap for a blended relevance score.
    Drops chunks below threshold to prevent context poisoning
    """

    NAME = "RetrievalValidation"

    def _init_(self, threshold:float= settings.retrieval_relevance_threshold):
        self.threshold= threshold

    async def run(self, query:str, chunks: list[RetrievedChunk])-> tuple[list[RetrievedChunk], AgentEvent]:
        try:

            t0 = time.perf_counter()

            if not chunks:
                logger.info("Chunk is empty!")
                return [], _timed_event(self.NAME, AgentStatus.DONE, "No chunks to Validate", t0)
            
            
            for rc in chunks:
                overlap = _overlap_ratio(query, rc.chunk.content)
                rc.relevance_score= round(0.55 * rc.vector_score + 0.45*overlap,4)

            validated = [rc for rc in chunks if rc.relevance_score >=self.threshold]
            validated.sort(key=lambda x:x.relevance_score, reverse=True) ## soritng the validated chunk

            dropped = len(chunks) - len(validated)
            event = _timed_event(self.NAME, AgentStatus.DONE,
                                f"Validated {len(validated)}/{len(chunks)} chunks (dropped: {dropped})",
                                t0, kept=len(validated), dropped=dropped,
                                threshold=self.threshold)
            
            logger.info("retrieved_validation", **event.metadata)
            return validated,event

        except Exception as e:
            logger.error(f"Retrieval Validaton failed", error= str(e))
            raise MulitagentragException("Retrieval Validation failed", error_details= str(e))
        

class AnswerGeneratorAgent:
    """
    Generates answer based on the validated and retrieved. A single LLM call with a
    structured system pompt emphasising grounded, citation_based, and refusal to 
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
        self.llm = llm
        self.agent_id = agent_id

    async def run(self,
            query:str, chunks:list[RetrievedChunk], 
            temperature: Optional[float]=None,
            )-> tuple[str, AgentEvent]:
        
        try:
            t0= time.perf_counter()

            if not chunks:
                msg= "The Provided sources do not contain enough information to answer this question."
                return msg, _timed_event(self.NAME, AgentStatus.DONE, msg,
                                         t0,)
            
            context_parts =[]
            for i, rc in enumerate(chunks[: settings.top_k_after_rerank]):
                context_parts.append(
                f"[{i}] Source:{rc.chunk.source}/n{rc.chunk.content.strip()}")

            ## varying the temperature per agent for ensemble
            temp = (temperature or settings.llm_temperature) + self.agent_id *0.05

            resp: LLMResponse = await self.llm.complete(
                system=self.SYSTEM_PROMPT, user = query, 
                temperature=temp,
            )
            msg = f"Generated answer ({resp.output_tokens} tokens)"
            event = _timed_event(
                self.NAME, AgentStatus.DONE, msg,
                start= t0, tokens = resp.total_tokens, llm_latency_ms=resp.latency_ms,
            )
            logger.info("Answer Generated by the LLM")
            return resp.text.strip(), event

        except Exception as e:
            logger.error("Answer Generation failed", agent=self.NAME, error=str(e))
            MulitagentragException(f"Unable to generate answer due LLM error", error_details=str(e))

    
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
                _timed_event(self.NAME, AgentStatus.FAILED, "All generators failed", t0)
            
            word_freq: Counter = Counter()
            for ans in answers:
                word_freq.update(_word_text(ans))

            best= max(answers, key=lambda a: sum(word_freq[w] for w in _word_text(a)))
            event = _timed_event(
                self.NAME, AgentStatus.DONE,
                f"Consensus from {len(answers)} candidates",
                t0, n_candidates = len(answers), best_len=len(best),
            )
            logger.info('Consensus_complete', **event.metadata)
            return best, answers, event
        
        except Exception as e:
            logger.error(f"Consensus agent failed", error=str(e))
            raise MulitagentragException("Consus Agent failed! ", error_details=str(e))
        
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
    
    async def run(
            self, answer: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[Claim], AgentEvent]:
        t0 = time.perf_counter()
        sentences = self._split_claims(answer)

        if not sentences:
            return [], _timed_event(self.NAME, AgentStatus.DONE, "No claims extracted", t0)
        
        ## verify all claims concurrently
        verify_tasks = [
            self._verify_one_llm(s,chunks) if self.use_llm
            else asyncio.coroutine(lambda s=s: self._verify_lexical(s, chunks))()
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
                confidenc=round(confidence, 3), supporting_chunks = supporting[:2],
            ))
        
        n_supported = sum(1 for c in claims if c.supported)
        event = _timed_event(
            self.NAME, AgentStatus.DONE,
            f"{n_supported}/{len(claims)} claims supported",
            t0, total=len(claims), supported = n_supported,
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
            self.NAME, AgentStatus.DONE,
            f"Confidence={final:.3f}, Risk={risk.value}",
            t0, **breakdown.model_dump(),
        )
        logger.info("confidence_score", **event.metadata)
        return breakdown, risk, event

class MultiAgentRAGPipeline:
    """
    Orchestrates the full 5-agent pipeline.
    Agents 4+5 run concurrently after generation.
    """
    def __init__(self):
        self.validator = RetrievalValidationAgent()
        self.consensus = ConsensusAgent()
        self.verifier = ClaimVerificationAgent()
        self.scorer = ConfidenceScoringAgent()

    async def run(
            self, request: QueryRequest,
            raw_chunks: list[RetrievedChunk],
    )-> QueryResponse:
        t_total = time.perf_counter()
        trace: list[AgentEvent] = []

        # Retrieval validation
        validated,ev = await self.validator.run(request.query, raw_chunks)
        trace.append(ev)

        top_chunks = validated[: settings.top_k_after_rerank]

        # Consensus Generation
        answer, all_answers, ev = await self.consensus.run(request.query, top_chunks)
        trace.append(ev)

        # Claim verification and confidence scoring (running concurrently)
        (claims, ev_claims), (breakdown, risk, ev_score) = await asyncio.gather(
            self.verifier.run(answer, top_chunks),
            self.scorer.run(answer, [], top_chunks),
        )
        trace.extend([ev_claims, ev_score])

        breakdown, risk, ev_final = await self.scorer.run(answer, claims, top_chunks)
        trace.append(ev_final)

        return QueryResponse(
            query=request.query,
            answer=answer,
            claims = claims,
            retrieved_chunks=top_chunks,
            confidence = breakdown,
            hallucination_risk = risk,
            latency_ms=round((time.perf_counter()- t_total)*1000, 2),
            agent_trace=trace if request.include_trace else [],
        )




if __name__=="__main__":
    print('correct')