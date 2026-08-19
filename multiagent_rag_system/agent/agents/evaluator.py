"""
RAGAS-based quality evaluation, run asynchronously on a sampled fraction
of queries so it never adds latency to the live response path.
"""
from __future__ import annotations

import asyncio
import random
from typing import Optional
import os
import warnings

from multiagent_rag_system.src.utils.config_loader import get_settings
from multiagent_rag_system.src.models.models import RAGASScores, RerankedChunk
from multiagent_rag_system.src.logger import GLOBAL_LOGGER as logger

from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

from groq import Groq
from ragas.llms import llm_factory
from google import genai
from openai import OpenAI, AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings


settings = get_settings()

class RAGASEvaluator:
    """
    Wraps RAGAS evaluation behind two guards
   """

    def __init__(self, provider:str="groq") -> None:
        self.cfg      = settings.evaluation
        self.settings = settings
      #  self.evaluate = self.cfg.enabled = enable
        if provider == 'groq':
            client = AsyncOpenAI(api_key = settings.ragas_groq_api_key.get_secret_value(),
                                base_url="https://api.groq.com/openai/v1",)
            model = 'openai/gpt-oss-120b'
        elif provider == 'gemini':
            client = AsyncOpenAI(
                api_key= settings.ragas_gemini_api_key.get_secret_value(),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            model = 'gemini-3.5-flash-lite'

        self.llm = llm_factory(model=model,
                        client=client)

        EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
        self.embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL_NAME)

    async def evaluate(self, query:str,
        answer:str, chunks:list[RerankedChunk],
      #  ground_truth:Optional[str] = None,
    ) -> Optional[RAGASScores]:
        """
        Returns RAGASScores if evaluation ran, None if skipped or failed.
        """

      #  if not self.cfg.enabled or random.random() > self.cfg.sample_rate:
       #     return None
        try:
            scores = await asyncio.wait_for(
                self._run(query, answer, chunks, 
                ), timeout=120000)
        
        except asyncio.TimeoutError:
            logger.warning("RAGAS evaluation timed out after 60s")
            return None

        if scores:
            logger.info("[RAGAS] faithfulness={faithfulness} "
                        "relevancy={relevancy} "
                        "precision={precision}"
                        "recall={recall}".format(
                    faithfulness=scores.faithfulness,
                    relevancy=scores.answer_relevancy,
                    precision=scores.context_precision,
                    recall = scores.context_recall
                )
            )

        return scores

    async def _run(self, query: str, answer: str,
        chunks: list[RerankedChunk],# ground_truth: Optional[str],
    ) -> Optional[RAGASScores]:
        """
        Synchronous RAGAS execution — always called inside run_in_executor.
        """
        try:
            
            faithfulness_metric = Faithfulness(llm=self.llm)
            relevancy_metric = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
            precision_metric = ContextPrecision(llm=self.llm)
            recall_metric = ContextRecall(llm=self.llm)


            # Extract plain text from RerankedChunk objects.
            retrieved_contexts = [c.chunk.content for c in chunks]

            faithfulness_score, relevancy_score, precision_score, recall_score = await asyncio.gather(
            faithfulness_metric.ascore(
                    user_input=query, response=answer, retrieved_contexts=retrieved_contexts),
            relevancy_metric.ascore(
                    user_input=query, response=answer),
            precision_metric.ascore(
                    user_input=query, reference=answer, retrieved_contexts=retrieved_contexts),
            recall_metric.ascore(
                    user_input=query, reference=answer, retrieved_contexts= retrieved_contexts),
            )
          
            return RAGASScores(
                faithfulness=faithfulness_score.value,
                answer_relevancy=relevancy_score.value,
                context_precision=precision_score.value,
                context_recall=recall_score.value
            )

        except ImportError as ie:
            logger.warning(f"ragas or dependencies not installed — skipping evaluation: {ie}")
            return None
        except Exception as exc:
            logger.error(f"RAGAS evaluation failed: {exc}")
            return None