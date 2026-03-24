from __future__ import annotations
import asyncio
import re
import time
from typing import Optional

from multiagent_rag_system.src.utils.config_loader import get_settings
from multiagent_rag_system.src.logger.logger import GLOBAL_LOGGER as logger
from multiagent_rag_system.src.exception.custom_exception import MulitagentragException
from multiagent_rag_system.src.models.models import (
    AgentEvent, AgentStatus, Claim, RerankedChunk, RetrievedChunk)
from multiagent_rag_system.src.utils.general_utils import _overlap_ratio, _timed_event
from multiagent_rag_system.src.llm.llms import BaseLLMClient, get_llm_client

settings = get_settings()

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
        self.config = settings.agents

    def _split_claims(self, text: str)-> list[str]:
        """ Split text into atomic sentence-level claims."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip())> 15]
    
    async def _verify_one_llm(self, claim: str, chunks: list[RerankedChunk])->tuple[bool, float]:
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
        
    def _verify_lexical(self, claim: str, chunks: list[RerankedChunk])->tuple[bool, float]:
        best = max(
            (_overlap_ratio(claim, rc.chunk.content) for rc in chunks),
            default=0.0,
        )
        supported = best>= self.config.claim_support_threshold
        return supported, min(1.0, best*2)

    async def _lexical_async(self, claim: str, chunks: list[RerankedChunk]):
        return self._verify_lexical(claim, chunks)
    
    async def run(
            self, answer: str, chunks: list[RerankedChunk]
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
