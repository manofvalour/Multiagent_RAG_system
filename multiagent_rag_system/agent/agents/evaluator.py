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
            os.environ["OPENAI_API_KEY"] = self.settings.active_api_key

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

            result = evaluate(Dataset.from_dict(data), metrics=metrics,
                              run_config={"max_workers":1})
            df     = result.to_pandas()

            return RAGASScores(
                faithfulness= float(df["faithfulness"].iloc[0]) if "faithfulness" in df.columns else None,
                answer_relevancy= float(df["answer_relevancy"].iloc[0]) if "answer_relevancy" in df.columns else None,
                context_precision= float(df["context_precision"].iloc[0]) if "context_precision" in df.columns else None,
                context_recall= float(df["context_recall"].iloc[0]) if "context_recall" in df.columns else None,
            )

        except ImportError:
            logger.warning("ragas not installed — skipping evaluation")
            return None
        except Exception as exc:
            logger.error(f"RAGAS evaluation failed: {exc}")
            return None