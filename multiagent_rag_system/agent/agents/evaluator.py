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

settings = get_settings()


class RAGASEvaluator:
    """
    Wraps RAGAS evaluation behind two guards
   """

    def __init__(self) -> None:
        self.cfg      = settings.evaluation
        self.settings = settings

    async def evaluate(self, query:str,
        answer:str, chunks:list[RerankedChunk],
        ground_truth:Optional[str] = None,
    ) -> Optional[RAGASScores]:
        """
        Returns RAGASScores if evaluation ran, None if skipped or failed.
        """

        if not self.cfg.enabled or random.random() > self.cfg.sample_rate:
            return None

        loop   = asyncio.get_event_loop()
        try:
            scores = await asyncio.wait_for(
                loop.run_in_executor(
                None, self._run, query, answer, chunks, 
                ground_truth), timeout=60.0)
        
        except asyncio.TimeoutError:
            logger.warning("RAGAS evaluation timed out after 60s")
            return None

        if scores:
            logger.info("[RAGAS] faithfulness={faithfulness} "
                        "relevancy={relevancy} "
                        "precision={precision}".format(
                    faithfulness=scores.faithfulness,
                    relevancy=scores.answer_relevancy,
                    precision=scores.context_precision,
                )
            )

        return scores

    def _run(self, query: str, answer: str,
        chunks: list[RerankedChunk], ground_truth: Optional[str],
    ) -> Optional[RAGASScores]:
        """
        Synchronous RAGAS execution — always called inside run_in_executor.
        """
        try:
            warnings.filterwarnings("ignore", message="resource_tracker.*")
            # RAGAS requires OpenAI API key for evaluation models
            openai_key = self.settings.openai_api_key.get_secret_value()
            if not openai_key:
                logger.warning("RAGAS evaluation skipped: OPENAI_API_KEY not configured")
                return None
            os.environ["OPENAI_API_KEY"] = openai_key

            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevance,
                context_precision,
                context_recall,
                faithfulness,
            )

            # Extract plain text from RerankedChunk objects.
            contexts = [c.chunk.content for c in chunks]

            # RAGAS Dataset format: each key is a column, each value is a list.
            data: dict = {
                "question": [query],
                "answer":   [answer],
                "contexts": [contexts],  # list of lists
            }

            # context_recall requires a reference answer to compare against.
            # Only add it when ground_truth was provided by the caller.
            metrics = [faithfulness, answer_relevance, context_precision]
            if ground_truth:
                data["ground_truth"] = [ground_truth]
                metrics.append(context_recall)

            result = evaluate(Dataset.from_dict(data), metrics=metrics)
            df     = result.to_pandas()

            # Handle column name variations across RAGAS versions
            def safe_get(col_name):
                if col_name in df.columns:
                    val = df[col_name].iloc[0]
                    return float(val) if val is not None else None
                return None

            return RAGASScores(
                faithfulness=safe_get("faithfulness"),
                answer_relevancy=safe_get("answer_relevancy") or safe_get("answer_relevance"),
                context_precision=safe_get("context_precision"),
                context_recall=safe_get("context_recall"),
            )

        except ImportError as ie:
            logger.warning(f"ragas or dependencies not installed — skipping evaluation: {ie}")
            return None
        except Exception as exc:
            # Log specific error types for debugging
            err_msg = str(exc)
            if "OpenAI" in err_msg or "API" in err_msg.upper():
                logger.warning(f"RAGAS evaluation skipped due to OpenAI API issue: {exc}")
            elif "timeout" in err_msg.lower():
                logger.warning(f"RAGAS evaluation timed out: {exc}")
            elif "rate limit" in err_msg.lower():
                logger.warning(f"RAGAS evaluation rate limited: {exc}")
            else:
                logger.error(f"RAGAS evaluation failed: {exc}")
            return None