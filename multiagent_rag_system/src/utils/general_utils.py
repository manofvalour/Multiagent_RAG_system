from __future__ import annotations
import re
import time


from ..utils.config_loader import get_settings
from ..models.models import (AgentEvent, AgentStatus)

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