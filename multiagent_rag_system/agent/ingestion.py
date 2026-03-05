"""
agents/ingestion.py — Document ingestion pipeline
Supports fixed-size, sentence, and semantic chunking strategies.
Handles deduplication, embedding, and indexing.
"""
from __future__ import annotations
import hashlib
import re
import time
import uuid
from typing import Optional

from core.config import get_settings
from src.logger.logger import GLOBAL_LOGGER as logger
from core.models import ChunkStrategy, DocumentChunk, IngestRequest, IngestResponse
from core.vector_store import get_embedder, get_vector_store

logger = get_logger(__name__)
settings = get_settings()


class DocumentIngestionPipeline:

    async def ingest(self, req: IngestRequest) -> IngestResponse:
        t0 = time.perf_counter()
        doc_id = str(uuid.uuid4())

        # 1. Chunk the document
        raw_chunks = self._chunk(req.content, req.chunk_strategy, req.chunk_size, req.chunk_overlap)
        logger.info("document_chunked", doc_id=doc_id, n_chunks=len(raw_chunks), strategy=req.chunk_strategy)

        # 2. Build DocumentChunk objects
        chunks = [
            DocumentChunk(
                id=self._chunk_id(doc_id, i, text),
                doc_id=doc_id,
                content=text,
                source=req.source,
                metadata={**req.metadata, "chunk_index": i, "strategy": req.chunk_strategy},
                chunk_index=i,
            )
            for i, text in enumerate(raw_chunks)
        ]

        # 3. Embed
        embedder = await get_embedder()
        texts = [c.content for c in chunks]
        embeddings = await embedder.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 4. Index
        store = await get_vector_store()
        await store.add_chunks(chunks)

        ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("ingestion_complete", doc_id=doc_id, chunks=len(chunks), ms=ms)
        return IngestResponse(document_id=doc_id, chunks_created=len(chunks), processing_ms=ms)

    # ── Chunking strategies ──────────────────────────────────────────────────

    def _chunk(self, text: str, strategy: ChunkStrategy, size: int, overlap: int) -> list[str]:
        if strategy == ChunkStrategy.FIXED:
            return self._chunk_fixed(text, size, overlap)
        elif strategy == ChunkStrategy.SENTENCE:
            return self._chunk_sentences(text, size, overlap)
        else:
            return self._chunk_semantic(text, size, overlap)

    def _chunk_fixed(self, text: str, size: int, overlap: int) -> list[str]:
        words = text.split()
        if not words:
            return []
        chunks, start = [], 0
        while start < len(words):
            end = min(start + size, len(words))
            chunk = " ".join(words[start:end])
            if len(chunk.strip()) > 0:
                chunks.append(chunk.strip())
            start += size - overlap
        return chunks

    def _chunk_sentences(self, text: str, max_words: int, overlap_words: int) -> list[str]:
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        chunks, current_words, current_sents = [], 0, []
        for sent in sentences:
            n = len(sent.split())
            if current_words + n > max_words and current_sents:
                chunks.append(" ".join(current_sents))
                # Overlap: keep last few sentences
                overlap_buf, overlap_count = [], 0
                for s in reversed(current_sents):
                    wc = len(s.split())
                    if overlap_count + wc > overlap_words:
                        break
                    overlap_buf.insert(0, s)
                    overlap_count += wc
                current_sents = overlap_buf
                current_words = overlap_count
            current_sents.append(sent)
            current_words += n

        if current_sents:
            chunks.append(" ".join(current_sents))
        return [c for c in chunks if c.strip()]

    def _chunk_semantic(self, text: str, max_words: int, overlap_words: int) -> list[str]:
        """
        Paragraph-aware chunking: split on double newlines first,
        then fall back to sentence chunking within large paragraphs.
        """
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        result = []
        for para in paragraphs:
            if len(para.split()) <= max_words:
                result.append(para)
            else:
                result.extend(self._chunk_sentences(para, max_words, overlap_words))
        return result

    @staticmethod
    def _chunk_id(doc_id: str, index: int, content: str) -> str:
        h = hashlib.sha256(f"{doc_id}:{index}:{content[:50]}".encode()).hexdigest()[:12]
        return f"chunk-{h}"