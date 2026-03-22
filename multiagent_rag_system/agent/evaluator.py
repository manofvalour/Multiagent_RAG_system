"""
src/agents/evaluator.py
RAGAS-based quality evaluation, run asynchronously on a sampled fraction
of queries so it never adds latency to the live response path.
"""
from __future__ import annotations

import asyncio
import random
from typing import Optional

from ..src.utils.config_loader import get_settings
from ..src.models.models import RAGASScores, RerankedChunk
from ..src.logger.logger import GLOBAL_LOGGER as logger

settings = get_settings()


class RAGASEvaluator:
    """
    Wraps RAGAS evaluation behind two guards:

    1. Enabled flag  — evaluation.enabled in config.yaml
    2. Sample gate   — only runs on (sample_rate * 100)% of calls,
                       e.g. sample_rate=0.1 means 10% of queries are evaluated
    """

    def __init__(self) -> None:
        # Pull evaluation sub-config from the merged Settings object.
        # settings.evaluation is an EvaluationConfig with .enabled and .sample_rate
        self.cfg      = settings.evaluation
        self.settings = settings

    async def evaluate(
        self,
        query:str,
        answer:str,
        chunks:list[RerankedChunk],
        ground_truth:Optional[str] = None,
    ) -> Optional[RAGASScores]:
        """
        Sample-gated: skips (1 - sample_rate) fraction of calls.
        Runs the blocking RAGAS evaluate() in a thread pool so it
        never blocks the asyncio event loop.

        Returns RAGASScores if evaluation ran, None if skipped or failed.
        """
        # Guard 1: feature flag
        # Guard 2: probabilistic sampling — random.random() is in [0, 1),
        #          so random.random() > 0.1 is True 90% of the time when
        #          sample_rate=0.1, meaning evaluation is skipped 90% of calls
        if not self.cfg.enabled or random.random() > self.cfg.sample_rate:
            return None

        loop   = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None, self._run, query, answer, chunks, ground_truth
        )

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

        RAGAS makes its own LLM calls internally. It reads OPENAI_API_KEY from
        the environment by default, but we use the active provider's key instead
        so the evaluator respects whichever LLM is configured in config.yaml.
        """
        try:
            import os
            # settings.active_api_key resolves the correct key for the active
            # provider (groq / anthropic / openai) and unwraps the SecretStr.
            # We set OPENAI_API_KEY because RAGAS uses the OpenAI SDK internally
            # regardless of which provider generated the answer.
            os.environ["OPENAI_API_KEY"] = self.settings.active_api_key

            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            # Extract plain text from RerankedChunk objects.
            # RAGAS only needs the content strings, not scores or metadata.
            contexts = [c.chunk.content for c in chunks]

            # RAGAS Dataset format: each key is a column, each value is a list.
            # One row per query/answer pair — we evaluate one at a time.
            data: dict = {
                "question": [query],
                "answer":   [answer],
                "contexts": [contexts],  # list of lists: [[passage1, passage2, ...]]
            }

            # context_recall requires a reference answer to compare against.
            # Only add it when ground_truth was provided by the caller.
            metrics = [faithfulness, answer_relevancy, context_precision]
            if ground_truth:
                data["ground_truth"] = [ground_truth]
                metrics.append(context_recall)

            result = evaluate(Dataset.from_dict(data), metrics=metrics)
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