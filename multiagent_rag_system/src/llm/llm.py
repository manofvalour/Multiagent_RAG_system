from __future__ import annotations
import asyncio
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..utils.config_loader import get_settings
from ..logger import GLOBAL_LOGGER as logger

settings = get_settings()
cfg = settings.active_llm

class LLMResponse:
    def __init__(self, text:str, input_tokens:int =0, output_tokens:int=0,
                 latency_ms: float=0):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms

    @property
    def total_tokens(self)-> int:
        return self.input_tokens+self.output_tokens
    
class BaseLLMClient(ABC):
    @abstractmethod
    async def complete(self, system:str,
                       user:str, temperature:Optional[float]=None)->LLMResponse: ...
    @abstractmethod
    async def health_check(self) -> bool: ...

class AnthropicClient(BaseLLMClient):
    BASE_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self):
        key = settings.anthropic_api_key.get_secret_value()
        self._headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        self._client = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)

    @retry(
        stop= stop_after_attempt(3),
        wait = wait_exponential(multiplier=1, min=1, max=10),
        retry = retry_if_exception_type(httpx.ConnectError, httpx.TimeoutException),
        reraise=True
    )

    async def complete(self, system:str, user:str, temperature: Optional[float]=None)-> LLMResponse:
        t0 =time.perf_counter()
        payload = {
            "model": cfg.model_name,
            "max_tokens": cfg.max_output_tokens,
            "temperature": temperature if temperature is not None else cfg.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

        resp = await self._client.post(self.BASE_URL, headers=self._headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        return LLMResponse(
            text= text,
            input_tokens = usage.get("input_tokens", 0),
            output_tokens = usage.get("output_tokens", 0),
            latency_ms= (time.perf_counter() - t0)*1000,
        )
    
    async def health_check(self):
        try:
            resp = await self._client.get("https://api.anthropic.com", timeout=5)
            return resp.status_code < 500
        
        except Exception:
            return False
        

class OpenAIClient(BaseLLMClient):
    BASE_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self):
        key = settings.openai_api_key.get_secret_value()
        self._headers= {
            "Authorization":f"Bearer {key}",
            "Content-Type": 'application/json'
        }

        self._client= httpx.AsyncClient(timeout= settings.llm_timeout_seconds)

    @retry(
        stop = stop_after_attempt(3),
        wait = wait_exponential(multiplier=1, min=1, max=10),
        retry= retry_if_exception_type(httpx.ConnectError, httpx.TimeoutException),
        reraise=True
    )

    async def complete(self,system: str, user:str, temperature: Optional[float]=None)-> LLMResponse:
        t0 = time.perf_counter()
        payload = {
            "model": cfg.model_name,
            'max_tokens': cfg.max_output_tokens,
            "messages": [{'role': 'system', "content": system}, {"role": "user", "content": user}]
        }

        resp = await self._client.post(self.BASE_URL, headers = self._headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data['choices'][0]["message"]['content']
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens= usage.get("prompt_tokens",0),
            output_tokens = usage.get("completion_tokens", 0),
            latency_ms = (time.perf_counter()- t0) *1000,
        )
    
    async def health_check(self)-> bool:
        try:
            resp = await self._client.get("https://api.openai.com", timeout =5)
            return resp.status_code<500
        
        except Exception:
            return False
        

class SimulatedLLMClient(BaseLLMClient):
    """
    Deterministic simulated LLM for development and testing.
    """

    async def complete(self,system:str, user:str, temperature: Optional[float]= None)->LLMResponse:
        await asyncio.sleep(0.03)

        lines = [l.strip() for l in user.split("\n") if l.strip() and not l.startswith("Q:")]
        context = " ".join(lines[:6])
        query_line = next((l for l in user.split("\n") if l.startswith("Q:")), "")
        query =query_line.replace("Q:", "").strip() or "this topic"

        answer = (
            f"Based on the provider sources, {query.lower().rstrip('?')}can be understood as follows: "
            f"{context[:300].rstrip('.')}. "
            f"These findings are grounded in the retrieved doumentation and reflect the current state of knowledge."
        )

        return LLMResponse(text=answer, input_tokens=len(user.split()), output_tokens = len(answer.split()), latency_ms =30)
    
    async def health_check(self) -> bool:
        return True
    

_client_instance:Optional[BaseLLMClient]=None

def get_llm_client()-> BaseLLMClient:
    global _client_instance
    if _client_instance is not None:
        return _client_instance
    
    provider =settings.llm_providers
    if provider == "anthropic" and settings.anthropic_api_key.get_secret_value():
        _client_instance= AnthropicClient()
    elif provider == "openai" and settings.openai_api_key.get_secret_value():
        _client_instance = OpenAIClient()

    else:
        logger.warning("llm_no_api_key", provider=provider, fallback='simulated')
        _client_instance = SimulatedLLMClient()

    logger.info("llm_client_created", provider=type(_client_instance).__name__)
    return _client_instance