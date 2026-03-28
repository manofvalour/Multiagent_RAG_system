"""
core/llm.py — Async LLM client abstraction
Supports Anthropic, OpenAI, and Ollama.
Implements: retry with exponential backoff, token counting, streaming stubs.
"""
from __future__ import annotations
import asyncio
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import sys

from ..utils.config_loader import get_settings
from ..logger import GLOBAL_LOGGER as logger
from ..exception.custom_exception import MulitagentragException

settings = get_settings()

class LLMResponse:
    def __init__(self, text: str, 
                 input_tokens: int = 0, 
                 output_tokens: int = 0, 
                 latency_ms: float = 0):
        
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseLLMClient(ABC):
    @abstractmethod
    async def complete(self, system: str, user: str, temperature: Optional[float] = None) -> LLMResponse: ...

    @abstractmethod
    async def health_check(self) -> bool: ...


#Anthropic
class AnthropicClient(BaseLLMClient):
    def __init__(self):
        key = settings.anthropic_api_key.get_secret_value()
        self.llm_config = settings.llm_providers['anthropic']
        self._headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        self._client = httpx.AsyncClient(timeout=self.llm_config.timeout_seconds)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        reraise=True,
    )
    async def complete(self, system: str, user: str, temperature: Optional[float] = None) -> LLMResponse:
        try:
            t0 = time.perf_counter()
            payload = {
                "model": self.llm_config.model_name,
                "max_tokens": self.llm_config.max_output_tokens,
                "temperature": temperature if temperature is not None else self.llm_config.temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            resp = await self._client.post(self.llm_config.base_url, headers=self._headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"]
            usage = data.get("usage", {})

            response =  LLMResponse(
                text=text,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

            logger.info("LLM generated successfully")
            return response
        
        except Exception as e:
            logger.error("Failed to complete the Model generation operation with Anthropic", error=str(e))
            raise MulitagentragException(e,sys)

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("https://api.anthropic.com", timeout=5)
            return resp.status_code < 500
        except Exception:
            return False


#GroqAPI
class GroqClient(BaseLLMClient):
    def __init__(self):
        key = settings.groq_api_key.get_secret_value()
        self.llm_config = settings.llm_providers['groq']
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self._client = httpx.AsyncClient(timeout=self.llm_config.timeout_seconds)

    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, 
                                 min=1, max=10), 
                                 reraise=True)
    
    async def complete(self, system: str, user: str,
                       temperature: Optional[float] = None) -> LLMResponse:
        try:
            t0 = time.perf_counter()
            payload = {
                "model": self.llm_config.model_name,
                "max_tokens": self.llm_config.max_output_tokens,
                "temperature": temperature if temperature is not None else self.llm_config.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }

            resp = await self._client.post(self.llm_config.base_url, headers=self._headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            response = LLMResponse(
                text=text,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            logger.info("LLM generation completed")
            return response
        
        except Exception as e:
            logger.error("Failed to complete the Model generation operation with Groq", error=str(e))
            raise MulitagentragException(e,sys)

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("https://api.groq.com", timeout=5)
            return resp.status_code < 500
        except Exception:
            return False


_client_instance: Optional[BaseLLMClient] = None


def get_llm_client() -> BaseLLMClient:
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    provider = settings.active_provider
    if provider == "anthropic" and settings.anthropic_api_key.get_secret_value():
        _client_instance = AnthropicClient()
    elif provider == "groq" and settings.groq_api_key.get_secret_value():
        _client_instance = GroqClient()
    
    logger.info("llm_client_created", provider=type(_client_instance).__name__)
    return _client_instance
