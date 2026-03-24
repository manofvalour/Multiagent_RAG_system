from __future__ import annotations
import asyncio
import re
import time
from collections import Counter
from typing import Optional
import numpy as np

from multiagent_rag_system.src.utils.config_loader import get_settings
from multiagent_rag_system.src.logger.logger import GLOBAL_LOGGER as logger
from multiagent_rag_system.src.exception.custom_exception import MulitagentragException
from multiagent_rag_system.src.models.models import (
    AgentEvent, AgentStatus, RetrievedChunk
    )
from multiagent_rag_system.agent.agents.answer_generator_agent import AnswerGeneratorAgent
from multiagent_rag_system.src.utils.general_utils import _timed_event, _word_text

settings = get_settings()

class ConsensusAgent:
    """
    Runs N-numbers of AnswerGeneratorAgents in Parallel, 
    then select the best anser by a word-freq majority vote.
    """
    NAME = "Conssensus Agent"

    def __init__(self, n:int= settings.agents.consensus_n_agents):
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
        
    
