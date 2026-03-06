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
        
class ClaimmVerificationAgent:
    pass

class ConfidenceScoringAgent:
    pass

class MultiAgentRAGPipeline:
    pass




if __name__=="__main__":
    print('correct')