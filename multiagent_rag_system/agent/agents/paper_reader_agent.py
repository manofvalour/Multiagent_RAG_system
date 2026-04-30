"""
agent/agents/paper_reader_agent.py
Analyzes research papers with AI-generated intuition, math breakdowns,
architecture diagrams, and code reimplementation snippets.
"""
from __future__ import annotations
import asyncio
import json
import re
import time
import sys
from typing import Optional

from langsmith import traceable

from multiagent_rag_system.src.utils.config_loader import get_settings
from multiagent_rag_system.src.logger import GLOBAL_LOGGER as logger
from multiagent_rag_system.src.exception.custom_exception import MulitagentragException
from multiagent_rag_system.src.models.models import (
    AgentEvent, AgentStatus, PaperMetadata, PaperAnalysis,
    CodeSnippet, DocumentChunk
)
from multiagent_rag_system.src.utils.general_utils import _timed_event
from multiagent_rag_system.src.llm.llms import get_llm_client, LLMResponse

settings = get_settings()


class PaperReaderAgent:
    """
    Analyzes a research paper using specialized LLM prompts for:
    - Summary + key findings
    - Plain-English intuition
    - Mathematical concept breakdown (LaTeX)
    - Code snippet extraction and explanation
    - Architecture description
    """
    NAME = "PaperReaderAgent"

    SUMMARY_PROMPT = """You are a research paper analyst. 
    Given the paper below, provide:

        1. A 3-5 sentence TLDR summary
        2. A bullet list of 3-6 key findings

        Respond ONLY with valid JSON:
        {{
        "summary": "3-5 sentence summary",
        "key_findings": ["finding 1", "finding 2", ...]
        }}

        PAPER:
        Title: {title}
        Authors: {authors}
        Abstract: {abstract}
        """

    INTUITION_PROMPT = """Explain this paper's core contribution 
    in plain English as if to a CS undergraduate who has taken 
    algorithms and basic ML courses. Focus on the "why" and 
    the intuition, not technical details.

        Title: {title}
        Abstract: {abstract}
        {math_preview}
        """

    MATH_PROMPT = """Identify and explain the key mathematical 
    concepts in this paper. Present LaTeX equations with 
    clear explanations of each variable and what the 
    equation means intuitively.

        Title: {title}
        Abstract: {abstract}
        Key Sections: {sections}
        """

    CODE_PROMPT = """Extract and explain any code snippets 
    or algorithms described in this paper. Show how to 
    reimplement the core algorithm. If the paper describes 
    a model architecture, provide pseudocode or actual 
    Python-like code.

        Title: {title}
        Abstract: {abstract}
        Sections: {sections}
        """

    ARCHITECTURE_PROMPT = """Describe the model or system 
    architecture presented in this paper. Explain the data 
    flow, key components, and how different parts interact. 
    Provide diagram descriptions where applicable.

        Title: {title}
        Abstract: {abstract}
        Sections: {sections}
        """

    LIMITATIONS_PROMPT = """Briefly describe the limitations of 
    this paper's approach and potential future work directions.

        Title: {title}
        Abstract: {abstract}
        """

    RELATED_PROMPT = """Based on this paper's references and 
    related work section, list up to 5 related paper titles 
    or arXiv IDs that are most relevant.

        Title: {title}
        Abstract: {abstract}
        """

    def __init__(self, llm=None):
        self.llm = llm or get_llm_client()
        self.config = settings.paper_reader

    def _build_context(self, paper: PaperMetadata, chunks: list[DocumentChunk]) -> dict:
        """Build context dict for prompts from paper metadata and chunks."""
        sections = "\n\n".join(
            f"[Section {i+1}]: {c.content[:500]}"
            for i, c in enumerate(chunks[:5])
        )
        return {
            "title": paper.title,
            "authors": ", ".join(paper.authors),
            "abstract": paper.abstract,
            "sections": sections,
            "math_preview": "The paper contains mathematical notation." if self.config.include_math else "",
        }

    async def _call_llm(self, system: str, user: str, temperature: float = 0.3) -> str:
        """Call LLM with retry."""
        try:
            resp = await self.llm.complete(
                system=system,
                user=user,
                temperature=temperature,
            )
            return resp.text.strip()
        
        except Exception as e:
            logger.error("paper_reader_llm_failed", error=str(e))
            raise MulitagentragException(f"LLM call failed: {e}", error_details=str(sys.exc_info()))

    async def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text.strip())
        
        except json.JSONDecodeError:
            logger.warning("json_parse_failed", text=text[:200])
            return {}

    @traceable(name="Paper Reader Agent")
    async def run(
        self,
        paper: PaperMetadata,
        chunks: list[DocumentChunk],
        include_math: bool = True,
        include_code: bool = True,
    ) -> tuple[PaperAnalysis, AgentEvent]:
        """
        Run full paper analysis across all dimensions.
        Returns (PaperAnalysis, AgentEvent).
        """
        t0 = time.perf_counter()
        ctx = self._build_context(paper, chunks)

        try:
            # Run all analysis tasks concurrently
            tasks = {
                "summary": self._analyze_summary(ctx),
                "intuition": self._analyze_intuition(ctx),
                "architecture": self._analyze_architecture(ctx),
                "limitations": self._analyze_limitations(ctx),
                "related": self._analyze_related(ctx),
            }

            if include_math:
                tasks["math"] = self._analyze_math(ctx)

            if include_code:
                tasks["code"] = self._analyze_code(ctx)

            results = await asyncio.gather(
                tasks.values(),
                return_exceptions=True,
            )

            task_names = list(tasks.keys())
            result_dict = {}
            for name, result in zip(task_names, results):
                if isinstance(result, Exception):
                    logger.warning(f"task_{name}_failed", error=str(result))
                    result_dict[name] = {} if name in ("summary",) else ""
                else:
                    result_dict[name] = result

            # Extract structured results
            summary_data = result_dict.get("summary", {})
            summary = summary_data.get("summary", "")
            key_findings = summary_data.get("key_findings", [])

            intuition = result_dict.get("intuition", "")
            math_breakdown = result_dict.get("math", "") if include_math else ""
            code_snippets = result_dict.get("code", []) if include_code else []
            architecture_notes = result_dict.get("architecture", "")
            limitations = result_dict.get("limitations", "")
            related_papers = result_dict.get("related", [])

            analysis = PaperAnalysis(
                paper_id=paper.paper_id,
                summary=summary,
                key_findings=key_findings,
                intuition=intuition,
                math_breakdown=math_breakdown,
                code_snippets=code_snippets if isinstance(code_snippets, list) else [],
                architecture_notes=architecture_notes,
                limitations=limitations,
                related_papers=related_papers if isinstance(related_papers, list) else [],
                relevance_score=0.8,
            )

            event = _timed_event(
                agent=self.NAME,
                status=AgentStatus.DONE,
                message=f"Paper analysis complete for {paper.paper_id}",
                start=t0,
                has_math=include_math,
                has_code=include_code,
            )
            logger.info("paper_analysis_complete", paper_id=paper.paper_id, **event.metadata)
            return analysis, event

        except Exception as e:
            logger.error("paper_reader_failed", paper_id=paper.paper_id, error=str(e))
            raise MulitagentragException(f"PaperReaderAgent failed: {e}", error_details=str(sys.exc_info()))

    async def _analyze_summary(self, ctx: dict) -> dict:
        text = await self._call_llm(
            system="You are a research paper analyst. Respond with valid JSON only.",
            user=self.SUMMARY_PROMPT.format(**ctx),
        )
        return await self._parse_json_response(text)

    async def _analyze_intuition(self, ctx: dict) -> str:
        return await self._call_llm(
            system="You are a helpful research assistant specializing in explaining ML/AI papers to students.",
            user=self.INTUITION_PROMPT.format(**ctx),
            temperature=0.5,
        )

    async def _analyze_math(self, ctx: dict) -> str:
        return await self._call_llm(
            system="You are a mathematical writing assistant. Present equations in LaTeX format with clear explanations.",
            user=self.MATH_PROMPT.format(**ctx),
            temperature=0.2,
        )

    async def _analyze_code(self, ctx: dict) -> list[CodeSnippet]:
        text = await self._call_llm(
            system="You are a code extraction assistant. Extract and explain code snippets. Respond with valid JSON only.",
            user=self.CODE_PROMPT.format(**ctx),
            temperature=0.2,
        )
        try:
            data = await self._parse_json_response(text)
            snippets = []
            for item in data.get("code_snippets", data.get("snippets", [])):
                if isinstance(item, dict):
                    snippets.append(CodeSnippet(
                        language=item.get("language", "python"),
                        code=item.get("code", item.get("snippet", "")),
                        description=item.get("description", ""),
                        file_path=item.get("file_path"),
                    ))
            return snippets
        except Exception:
            return []

    async def _analyze_architecture(self, ctx: dict) -> str:
        return await self._call_llm(
            system="You are a systems architecture analyst. Describe architectures clearly with data flow diagrams described in text.",
            user=self.ARCHITECTURE_PROMPT.format(**ctx),
            temperature=0.3,
        )

    async def _analyze_limitations(self, ctx: dict) -> str:
        return await self._call_llm(
            system="You are a critical research analyst. Be objective about limitations.",
            user=self.LIMITATIONS_PROMPT.format(**ctx),
            temperature=0.3,
        )

    async def _analyze_related(self, ctx: dict) -> list[str]:
        text = await self._call_llm(
            system="You are a research paper analyst. Respond with valid JSON only.",
            user=self.RELATED_PROMPT.format(**ctx),
            temperature=0.2,
        )
        try:
            data = await self._parse_json_response(text)
            return data.get("related_papers", data.get("related", []))
        except Exception:
            return []
