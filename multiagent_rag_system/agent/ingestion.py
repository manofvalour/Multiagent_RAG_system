"""
agents/ingestion.py — Document ingestion pipeline
Supports fixed-size, sentence, and semantic chunking strategies.
Handles deduplication, embedding, and indexing.
"""
from __future__ import annotations
import hashlib
import time
import uuid
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

    @staticmethod
    def _chunk_id(doc_id: str, index: int, content: str) -> str:
        h = hashlib.sha256(f"{doc_id}:{index}:{content[:50]}".encode()).hexdigest()[:12]
        return f"chunk-{h}"
    

if __name__== '__main__':
    pass