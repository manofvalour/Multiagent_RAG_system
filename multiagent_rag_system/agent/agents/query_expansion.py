"""
Rewrites the user query into variants that maximise retrieval recall.

HyDE — generate a hypothetical answer, embed it instead of the question
Multi_query — rephrase the question N ways to hit different chunks
Both — run both concurrently, combine results
"""
from __future__ import annotations

import asyncio
from typing import Optional
from groq import AsyncGroq

from multiagent_rag_system.src.utils.config_loader import get_settings
from multiagent_rag_system.src.logger import GLOBAL_LOGGER as logger
from multiagent_rag_system.src.exception.custom_exception import MulitagentragException
from multiagent_rag_system.src.models.models import QueryRequest
settings = get_settings()

class QueryExpansionAgent:
    def __init__(
        self) -> None:

        self._groqai: Optional[AsyncGroq] = None
        self.api_key = settings.groq_api_key.get_secret_value()
        self.config = settings.query_expansion
        self.model_config = settings.llm_providers[settings.active_provider]

    def _client(self) -> AsyncGroq:
        if self._groqai is None:
            self._groqai = AsyncGroq(api_key =self.api_key)
        return self._groqai

    async def expand(self, query: QueryRequest) -> tuple[list[str], Optional[str]]:
        """
        Returns:
          expanded_queries — list of strings for the retriever
          hyde_doc — hypothetical answer text (or None)
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
            
        except Exception as e:
            logger.error("Failed to load the expansion model", error = str(e))
            raise MulitagentragException("failed to load teh expansion model", error_details = str(e))
        
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
        resp = await self._client().chat.completions.create(
            model=self.model_config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature = self.config.hyde_temperature,
            max_tokens=256,
        )
        result = resp.choices[0].message.content.strip()
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
        resp = await self._client().chat.completions.create(
            model=self.model_config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=256,
        )
        lines = resp.choices[0].message.content.strip().splitlines()
        return [l.strip() for l in lines if l.strip()][: self.config.num_queries]