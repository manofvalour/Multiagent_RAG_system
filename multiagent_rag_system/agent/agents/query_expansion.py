"""
Rewrites the user query into variants that maximise retrieval recall.

HyDE — generate a hypothetical answer, embed it instead of the question
Multi_query — rephrase the question N ways to hit different chunks
Both — run both concurrently, combine results
"""
from __future__ import annotations

import asyncio
from typing import Optional
from langsmith import traceable

from multiagent_rag_system.src.utils.config_loader import get_settings
from multiagent_rag_system.src.logger import GLOBAL_LOGGER as logger
from multiagent_rag_system.src.exception.custom_exception import MulitagentragException
from multiagent_rag_system.src.models.models import QueryRequest
from multiagent_rag_system.src.llm.llms import BaseLLMClient, get_llm_client

settings = get_settings()

class QueryExpansionAgent:
    def __init__(
        self) -> None:
        self._llm: Optional[BaseLLMClient] = None
        self.config = settings.query_expansion
        self.model_config = settings.llm_providers[settings.active_provider]

    def _client(self) -> BaseLLMClient:
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    @traceable(name="Query Expansion Agent")
    async def expand(self, query: QueryRequest) -> tuple[list[str], Optional[str]]:
        """
        Returns:
          expanded_queries — list of strings for the retriever
          hyde_doc — hypothetical answer text (or None)
        
        If expansion fails, gracefully returns the original query.
        """
        try:
            if not self.config.enabled:
                return [query.query], None

            if self.config.strategy == "hyde":
                hyde_doc = await self._hyde(query.query)
                return [query.query, hyde_doc], hyde_doc

            elif self.config.strategy == "multi_query":
                variants = await self._multi_query(query.query)
                return [query.query] + variants, None

            else:   # both
                hyde_doc, variants = await asyncio.gather(
                    self._hyde(query.query),
                    self._multi_query(query.query),
                )
                return [query.query, hyde_doc] + variants, hyde_doc
            
        except asyncio.TimeoutError as e:
            logger.info(
                f"Query expansion timed out after {self.config.timeout_seconds}s, "
                "using original query",
                error=str(e)
            )
            return [query.query], None
        except Exception as e:
            logger.info(
                f"Query expansion failed, using original query as fallback",
                error=str(e)
            )
            return [query.query], None
        
    async def _hyde(self, query: str) -> str:
        """
        Generate a hypothetical ideal-answer paragraph.
        Embedding the answer instead of the question bridges the vocabulary gap
        between queries ("What is X?") and passages ("X is a technique that…").
        """
        prompt = (
            f"Write a concise, factual paragraph that perfectly answers:\n"
            f"Question: {query}\n\n"
            f"Paragraph (3-5 sentences, use domain-specific terminology):"
        )
        resp = await self._client().complete(
            system="You are a precise, factual assistant.",
            user=prompt,
            temperature=self.config.hyde_temperature,
        )
        result = resp.text.strip()
        logger.debug(f"HyDE doc: {result[:80]}…")
        return result

    async def _multi_query(self, query: str) -> list[str]:
        """Generate N rephrased versions of the query."""
        prompt = (
            f"Generate {self.config.num_queries} different ways to ask the following question.\n"
            f"Each rephrasing should approach it from a different angle.\n"
            f"Output one rephrasing per line, no numbering or bullets.\n\n"
            f"Original: {query}\n\nRephrasings:"
        )
        resp = await self._client().complete(
            system="You are a helpful assistant.",
            user=prompt,
            temperature=0.8,
        )
        lines = resp.text.strip().splitlines()
        return [l.strip() for l in lines if l.strip()][: self.config.num_queries]
