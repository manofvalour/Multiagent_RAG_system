

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


class LLMLoader(BaseLLMClient):
    """
    Loading the LLM needed for the project
    """
    def __init__(self):
        self.llm_config = settings.llm_providers[settings.active_provider]
        self.base_url = self.llm_config.base_url

        if settings.active_llm =='anthropic':
            self._headers = {
                "x-api-key": settings.anthropic_api_key.get_secret_value(),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

        elif settings.active_llm =='groq':
            self._headers= {
                "Authorization":f"Bearer {settings.groq_api_key.get_secret_value}",
                "Content-Type": 'application/json'
            }

        self.model: httpx.AsyncClient  = None

    def _client(self):
        if not self._client:
            return self.model
        self.model = httpx.AsyncClient(timeout=self.llm_config.timeout_seconds)
        return self.model


    @retry(
        stop= stop_after_attempt(3),
        wait = wait_exponential(multiplier=1, min=1, max=10),
        retry = retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        reraise=True
    )

    async def complete(self, system:str, user:str, temperature: Optional[float]=None)-> LLMResponse:
        try:

            t0 =time.perf_counter()
            config = self.llm_config
            payload = {
                "model": config.model_name,
                "max_tokens": config.max_output_tokens,
                "temperature": temperature if temperature is not None else config.temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }

            resp = await self._client.post(self.base_url, headers=self._headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            model_name = config.model_name=='groq'
            text =  data['choices'][0]["message"]["content"] if model_name else data["content"][0]["text"]
            usage = data.get("usage", {})

            logger.info("Model generation completed")


            return LLMResponse(
                text= text,
                input_tokens = usage.get("input_tokens", 0),
                output_tokens = usage.get("output_tokens", 0),
                latency_ms= (time.perf_counter() - t0)*1000,
            )
        
        except Exception as e:
            logger.error("Unable to generate content", error= str(e))
            raise MulitagentragException("Unable to generate content", e)
    
    async def health_check(self):
        try:
            model_name = self.llm_config.model_name=='groq'
            resp = await self._client.get("https://api.groq.com" if model_name else "https://api.anthropic.com", timeout=5)
            return resp.status_code < 500
        
        except Exception as e:
            logger.error("Failed health check", error=str(e))
            raise MulitagentragException("Failed health check", e)
        
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
    
    
    else:
        provider = settings.active_provider
        try:
            _client_instance = LLMLoader()
            logger.info("successfully loaded the model")

        except Exception as e:
            logger.warning("llm_model not available", provider = provider, fallback = "simulated")
            _client_instance = SimulatedLLMClient()

    logger.info("llm_client_created", provider=type(_client_instance).__name__)
    return _client_instance



if __name__ == "__main__":
    try:
        model = get_llm_client()

        #print(config.model_json_schema)
        print("we're live")
        logger.info("ConfigLoader test run completed succesfully")
    except Exception as e:
        raise MulitagentragException("failed to load the config file")
    
