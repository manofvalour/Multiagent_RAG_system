"""
src/fetcher/paper_fetcher_service.py
Multi-source research paper fetcher for arXiv, 
Papers with Code, and HuggingFace.
"""
from __future__ import annotations

import asyncio
import httpx
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
import arxiv

from multiagent_rag_system.src.models.models import (
    PaperMetadata, PaperSource, CodeSnippet, ScrapeResult
)
from multiagent_rag_system.src.utils.config_loader import get_settings
from multiagent_rag_system.src.logger import GLOBAL_LOGGER as logger

settings = get_settings()


class PaperFetcherService:
    """Fetches research papers from multiple sources concurrently."""

    def __init__(self) -> None:
        self.sources = settings.research_sources

    async def fetch_arxiv_papers(
        self, query: str, max_results: int = 10
    ) -> list[PaperMetadata]:
        """Fetch papers from arXiv using the arxiv package."""
        if not self.sources.arxiv.enabled:
            return []

        try:
            loop = asyncio.get_event_loop()
            papers = await loop.run_in_executor(
                None, self._fetch_arxiv_sync, query, max_results
            )
            
            return papers
        
        except Exception as e:
            logger.error("arxiv_fetch_failed", error=str(e))
            return []

    def _fetch_arxiv_sync(self, query: str, max_results: int) -> list[PaperMetadata]:
        """Fetch paper metadata from arXiv (no PDF scraping)."""
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(client.results(search))

        papers = []
        for result in results:
            papers.append(PaperMetadata(
                paper_id=f"arxiv:{result.entry_id.split('/')[-1]}",
                title=result.title,
                authors=[str(a) for a in result.authors],
                abstract=result.summary,
                source=PaperSource.ARXIV,
                content=None,
                url=result.entry_id,
                pdf_url=result.pdf_url,
                published_date=result.published,
                topics=list(result.categories),
                citation_count=None,
            ))
        return papers

    async def fetch_all(
        self,
        query: str,
        sources: list[PaperSource],
        max_results: int = 10,
    ) -> list[PaperMetadata]:
        """Fetch from all specified sources concurrently."""
        tasks = []
        source_map = {
            PaperSource.ARXIV: self.fetch_arxiv_papers,
        }

        for source in sources:
            fetcher = source_map.get(source)
            if fetcher:
                tasks.append(fetcher(query, max_results))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_papers = []
        for result in results:
            if isinstance(result, list):
                all_papers.extend(result)
            elif isinstance(result, Exception):
                logger.error("source_fetch_error", error=str(result))

        return all_papers

    async def scrape_paper_pdf(self, pdf_url: str) -> Optional[str]:
        """Fetch and scrape a single paper PDF, returning markdown content."""
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
            from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy

            pdf_crawler_strategy = PDFCrawlerStrategy()
            pdf_scraping_strategy = PDFContentScrapingStrategy()
            run_config = CrawlerRunConfig(scraping_strategy=pdf_scraping_strategy)

            async with AsyncWebCrawler(crawler_strategy=pdf_crawler_strategy) as crawler:
                result = await crawler.arun(url=pdf_url, config=run_config)
                if result and result.markdown:
                    logger.info("pdf_scrape_success", url=pdf_url)
                    return result.markdown
                return None
        except Exception as e:
            logger.error("pdf_scrape_failed", url=pdf_url, error=str(e))
            return None

    async def fetch_recent(
        self, source: PaperSource, days: int = 1
    ) -> list[PaperMetadata]:
        """Fetch recent papers from a source."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        query = f" submittedDate:>{cutoff.strftime('%Y%m%d')}"

        if source == PaperSource.ARXIV:
            return await self.fetch_arxiv_papers(query, max_results=10)
        return []
