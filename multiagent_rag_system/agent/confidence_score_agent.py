from __future__ import annotations
import asyncio
import re
import time
from typing import Optional

from ..src.utils.config_loader import get_settings
from ..src.logger.logger import GLOBAL_LOGGER as logger
from ..src.exception.custom_exception import MulitagentragException
from ..src.models.models import (
    AgentEvent, AgentStatus, Claim, ConfidenceBreakdown,
    HallucinationRisk, RerankedChunk,
    )
from ..src.utils.general_utils import _timed_event, _overlap_ratio

settings = get_settings()
class ConfidenceScoringAgent:
    """
    Computes a final [0,1] confidence score and assigns a hallucination risk
    level using a wighted combination fo three orthogonal signals
    """

    NAME = "ConfidenceScorer"
    WEIGHTS = {"claim_support": 0.50, "avg_relevance":0.30, "source_overlap":0.20}

    def __init__(self):
        self.config = settings.agents

    async def run(
            self, answer: str, claims: list[Claim], chunks: list[RerankedChunk]
    ) -> tuple[ConfidenceBreakdown, HallucinationRisk, AgentEvent]:
        t0 = time.perf_counter()

        claim_support = (
            sum(1 for c in claims if c.supported)/max(len(claims), 1)
        )
        avg_relevance = (
            sum(rc.reranker_score for rc in chunks)/ max(len(chunks),1)
        )

        src_text = " ".join(rc.chunk.content for rc in chunks)
        source_overlap = _overlap_ratio(answer, src_text)

        final = (
            self.WEIGHTS["claim_support"] * claim_support
            +self.WEIGHTS['avg_relevance'] * avg_relevance
            + self.WEIGHTS['source_overlap']* source_overlap
        )
        final = round(min(1.0, max(0.0, final)),4)

        if final >= self.config.confidence_low_threshold:
            risk = HallucinationRisk.LOW
        elif final>= self.config.confidence_medium_threshold:
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