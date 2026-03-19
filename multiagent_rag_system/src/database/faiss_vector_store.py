from __future__ import annotations
import asyncio
import json
import os
import pickle
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import sentence_transformers
import hashlib

from ..utils.config_loader import get_settings
from ..logger import GLOBAL_LOGGER as logger
from ..exception.custom_exception import MulitagentragException
from ..models.models import DocumentChunk,RetrievedChunk

settings = get_settings()

class VectorStoreBase(ABC):
    @abstractmethod
    async def add_chunks(self, chunks: list[DocumentChunk])->None: ...

    @abstractmethod
    async def search(self, query_embedding: list[float], top_k:int)-> list[DocumentChunk]: ...

    @abstractmethod
    async def delete_document(self, doc_id: str)-> int: ...

    @abstractmethod
    async def count(self)-> int: ...

class FAISSVectorStore(VectorStoreBase):
    """
    In-pocess FAISS index with an append_only JSON Wal for persistence.
    """
    def __init__(self):
        self._index = None
        self._chunks: list[DocumentChunk]=[]
        self._id_to_pos: dict[str, int]= {}
        self._lock = asyncio.Lock()
        self._wal_path = settings.faiss_index_path + ".wal"
        os.makedirs(os.path.dirname(settings.faiss_index_path) or ".", exist_ok=True)

    async def initialize(self)-> None:
        """
        Load or create FAISS index, replay WAL.
        """
        async with self._lock:
            try:
                import faiss
                dim = settings.embedding_dim
                if os.path.exists(settings.faiss_index_path):
                    loop =asyncio.get_event_loop()
                    self._index = await loop.run_in_executor(
                        None, faiss.read_index, settings.faiss_index_path
                    )

                    with open(settings.faiss_index_path + ".meta", 'rb') as f:
                        self._chunks = pickle.load(f)

                    self._id_to_pos = {c.id: i for i, c in enumerate(self._chunks)}
                    logger.info("faiss_index_loaded", vectors = len(self._chunks))

                else:
                    self._index = faiss.IndexFlatIP(dim)
                    logger.info("faiss_index_created", dim=dim)

            except ImportError:
                logger.warning('Faiss_unavailable', fallback = "numpy_brute_force")
                self._index = None

    async def add_chunks(self, chunks: list[DocumentChunk])-> None:
        if not chunks:
            return
        
        async with self._lock:
            embeddings = np.array([c.embedding for c in chunks], dtype=np.float32)
            if self._index is not None:
                self._index.add(embeddings)

            for chunk in chunks:
                self._id_to_pos[chunk.id]= len(self._chunks)
                self._chunks.append(chunk)

            await self._persist()
        logger.info("chunks_added", count=len(chunks))

    async def search(self, query_embedding: list[float], top_k:int)-> list[RetrievedChunk]:
        async with self._lock:
            if not self._chunks:
                return []
            
            q = np.array([query_embedding], dtype=np.float32)
            if self._index is not None:
                k = min(top_k * 2, len(self._chunks))
                scores, indices = self._index.search(q,k)
                hits = [(float(scores[0][i]), self._chunks[int(indices[0][i])])
                        for i in range(k) if indices[0][i]>=0]
                
            else:
                matrix = np.array([c.embedding for c in self._chunks], dtype=np.float32)
                sims = (matrix @ q.T).flatten()
                top_idx = np.argsort(sims)[::-1][: top_k * 2]
                hits = [(float(sims[i]), self._chunks[i]) for i in top_idx]

            hits.sort(key=lambda x: x[0], reverse=True)
            return [
                RetrievedChunk(chunk=chunk, vector_score =max(0.0, min(1.0, score)))
                for score, chunk in hits[:top_k]
            ]
        
    async def delete_document(self, doc_id: str) -> int:
        async with self._lock:
            before = len(self._chunks)
            self._chunks = [c for c in self._chunks if c.doc_id != doc_id]
            removed = before - len(self._chunks)
            if removed:
                await self._rebuild_index()
            return removed
        
    async def count(self) -> int:
        return len(self._chunks)
    
    async def _persist(self) -> None:
        if self._index is None:
            return 
        
        try: 
            import faiss
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, faiss.write_index, self._index, settings.faiss_index_path)
            with open(settings.faiss_index_path + ".meta", 'wb') as f:
                pickle.dump(self._chunks, f)

        except Exception as e:
            logger.error("faiss_persist_failed", error=str(e))
            raise MulitagentragException("FAISS persist failed")
        
    async def _rebuild_index(self)-> None:
        try:
            import faiss
            dim = settings.embedding_dim
            self._index = faiss.IndexFlatIP(dim)
            if self._chunks:
                emb=np.array([c.embedding for c in self._chunks], dtype=np.float32)
                self._index.add(emb)
            self._id_to_pos = {c.id:i for i, c in enumerate(self._chunks)}
            await self._persist()

        except Exception as e:
            logger.error("faiss_rebuild_failed", error= str(e))
            raise MulitagentragException("Failed to Rebuild FAISS", e)
            
_store: Optional[FAISSVectorStore]=None
_embedder: Optional[EmbeddingProvider]= None

async def get_vector_store()-> FAISSVectorStore:
    global _store
    if _store is None:
        _store = FAISSVectorStore()
        await _store.initialize()

    return _store

async def get_embedder()-> EmbeddingProvider:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingProvider()
        await _embedder._load()

    return _embedder

