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

from ..utils.config_loader import get_settings
from ..logger.logger import GLOBAL_LOGGER as logger
from ..exception.custom_exception import MulitagentragException

settings = get_settings()


class LLMResponse:
    def __init__(self, text: str, input_tokens: int = 0, output_tokens: int = 0, latency_ms: float = 0):
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
    BASE_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self):
        key = settings.anthropic_api_key.get_secret_value()
        self.llm_config = settings.llm_providers[settings.active_provider]
        self.base_url = self.llm_config.base_url
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
        t0 = time.perf_counter()
        payload = {
            "model": self.llm_config.model_name,
            "max_tokens": self.llm_config.max_output_tokens,
            "temperature": temperature if temperature is not None else self.llm_config.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        resp = await self._client.post(self.BASE_URL, headers=self._headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("https://api.anthropic.com", timeout=5)
            return resp.status_code < 500
        except Exception:
            return False


#GroqAPI
class GroqClient(BaseLLMClient):
    BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self):
        key = settings.groq_api_key.get_secret_value()
        self.llm_config = settings.llm_providers[settings.active_provider]
        self.base_url = self.llm_config.base_url
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self._client = httpx.AsyncClient(timeout=self.llm_config.timeout_seconds)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    async def complete(self, system: str, user: str, temperature: Optional[float] = None) -> LLMResponse:
        t0 = time.perf_counter()
        payload = {
            "model": self.llm_config.model_name,
            "max_tokens": self.llm_config.max_output_tokens,
            "temperature": temperature if temperature is not None else self.llm_config.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        resp = await self._client.post(self.BASE_URL, headers=self._headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("https://api.openai.com", timeout=5)
            return resp.status_code < 500
        except Exception:
            return False


# ─── Simulated (no API key required — for dev/demo) ──────────────────────────

class SimulatedLLMClient(BaseLLMClient):
    """
    Deterministic simulated LLM for development and testing.
    Generates grounded answers from the context passed in the user message.
    """

    async def complete(self, system: str, user: str, temperature: Optional[float] = None) -> LLMResponse:
        await asyncio.sleep(0.03)  # simulate network
        # Extract context blocks from user message
        lines = [l.strip() for l in user.split("\n") if l.strip() and not l.startswith("Q:")]
        context = " ".join(lines[:6])
        query_line = next((l for l in user.split("\n") if l.startswith("Q:")), "")
        query = query_line.replace("Q:", "").strip() or "this topic"
        answer = (
            f"Based on the provided sources, {query.lower().rstrip('?')} can be understood as follows: "
            f"{context[:300].rstrip('.')}. "
            f"These findings are grounded in the retrieved documentation and reflect the current state of knowledge."
        )
        return LLMResponse(text=answer, input_tokens=len(user.split()), output_tokens=len(answer.split()), latency_ms=30)

    async def health_check(self) -> bool:
        return True


# ─── Factory ─────────────────────────────────────────────────────────────────

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
    else:
        logger.info("llm_no_api_key", provider=provider, fallback="simulated")
        _client_instance = SimulatedLLMClient()

    logger.info("llm_client_created", provider=type(_client_instance).__name__)
    return _client_instance


#if __name__ == "__main__":
 #       model = get_llm_client()
       # model.health_check()

       # print(llm_config.model_json_schema)
   #     print("we're live")
  #      logger.info("ConfigLoader test run completed succesfully")
    #except Exception as e:
     #   raise MulitagentragException("failed to load the config file")
    
