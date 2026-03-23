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

class EmbeddingProvider:
    """
    Wraps a sentence-transformers model with bathing and caching.
    """

    def __init__(self):
        self._model = None
        self._lock = asyncio.Lock()
       # self._cache: dict[str, list[float]]= {}
        self.config = settings.embeddings

    async def _load(self):
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            
            try:
                from sentence_transformers import SentenceTransformer
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None, SentenceTransformer, self.config.model
                )
                logger.info("embedding_model_loaded", model=self.config.model)

            except Exception as e:
                logger.warning("embedding_model_unavailable", error=str(e),
                               fallback = "hash_embedding")
                #raise MulitagentragException("Unable to load embedding model")
            
    async def embed(self, texts: list[str]) -> list[list[float]]:
        await self._load()
        results = []
        to_encode: list[tuple[int, str]]=[]

        for i, text in enumerate(texts):
            #key = text[:200]
            #if key in self._cache:
             ##   results.append((i, self._cache[key]))

            #else:
            to_encode.append((i, text))

        if self._model is not None:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    [t for _, t in to_encode],
                    batch_size = self.config.batch_size,
                    normalize_embeddings = True,
                    show_progress_bar = False
                ).tolist(),
            )

            for (i, text), vec in zip(to_encode, raw):
               # self._cache[text[:200]]= vec
                results.append((i, vec))

        elif to_encode:
            for i, text in to_encode:
                vec = self._hash_embed(text, self.config.embedding_dim)
                results.append((i,vec))

        results.sort(key=lambda x:x[0])
        return [v for _, v in results]
    
    def _hash_embed(self,text:str, dim:int)-> list[float]:
        """
        Deterministic fallback embedding
        """

        h = hashlib.sha256(text.encode()).digit()
        raw = [(b/ 255.0)*2-1 for b in (h*((dim//len(h)) +1))[:dim]]
        norm = max(float(np.linalg.norm(raw)), 1e-9)
        return [v/norm for v in raw]
    

_embed: Optional[EmbeddingProvider]=None

async def get_embedder()-> EmbeddingProvider:
    global _embed
    if _embed is None:
        _embed = EmbeddingProvider()
        await _embed._load()

    return _embed