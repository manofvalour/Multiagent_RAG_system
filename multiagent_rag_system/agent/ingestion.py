"""
agents/ingestion.py — Document ingestion pipeline
Supports fixed-size, sentence, and semantic chunking strategies.
Handles deduplication, embedding, and indexing.
"""
from __future__ import annotations
import hashlib
import time
import uuid
import re
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..src.utils.config_loader import get_settings
from ..src.logger.logger import GLOBAL_LOGGER as logger
from ..src.models.models import DocumentChunk, IngestRequest, IngestResponse
from ..src.database.vector_store import get_embedder, get_vector_store


settings = get_settings()


class DocumentIngestionPipeline:


    async def ingest(self, req: IngestRequest) -> IngestResponse:
        t0 = time.perf_counter()
        doc_id = str(uuid.uuid4())

        # Chunk the document
        raw_chunks = self._chunk(req.content, req.chunk_size, req.chunk_overlap)
        logger.info("document_chunked", doc_id=doc_id, n_chunks=len(raw_chunks))

        # 2. Build DocumentChunk objects
        chunks = [
            DocumentChunk(
                id=self._chunk_id(doc_id, i, text),
                doc_id=doc_id,
                content=text,
                source=req.source,
                metadata={**req.metadata, "chunk_index": i},
                chunk_index=i,
            )
            for i, text in enumerate(raw_chunks)
        ]

        # Embed the chunked document
        embedder = await get_embedder()
        texts = [c.content for c in chunks]
        embeddings = await embedder.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        #Index the embeded chunks and add to vector database
        store = await get_vector_store()
        await store.add_chunks(chunks)

        ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("ingestion_complete", doc_id=doc_id, chunks=len(chunks), ms=ms)
        return IngestResponse(document_id=doc_id, chunks_created=len(chunks), processing_ms=ms)


    # Chunking
    def _chunk(self, text: str, size: int, overlap: int) -> list[str]:
        if not text:
            logger.info("input is empty!")
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, 
            chunk_overlap=overlap, 
            length_function=len,   
            add_start_index=True,  
        )

        chunks = splitter.split_text(text)

        return chunks

    def _chunk_fixed(self, text: str, size: int, overlap: int) -> list[str]:
        """Fixed-size chunking (same as _chunk) for test compatibility."""
        return self._chunk(text, size, overlap)

    def _chunk_sentences(self, text: str, max_words: int, overlap_words: int) -> list[str]:
        """Chunk by sentence boundaries while obeying max word count and overlap."""
        if not text:
            return []

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        current_words: list[str] = []

        def flush_current():
            if current_words:
                chunks.append(" ".join(current_words).strip())

        for sentence in sentences:
            words = sentence.split()
            if len(current_words) + len(words) <= max_words:
                current_words.extend(words)
                continue

            # flush what we have and start a new chunk, including overlap
            flush_current()
            overlap = current_words[-overlap_words:] if overlap_words > 0 else []
            current_words = overlap + words

            # if a single sentence is larger than max_words, still keep it
            if len(current_words) > max_words:
                flush_current()
                current_words = []

        flush_current()
        return [c for c in chunks if c]

    @staticmethod
    def _chunk_id(doc_id: str, index: int, content: str) -> str:
        h = hashlib.sha256(f"{doc_id}:{index}:{content[:50]}".encode()).hexdigest()[:12]
        return f"chunk-{h}"
    

if __name__== '__main__':
    pass