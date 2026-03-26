from __future__ import annotations
import asyncio
import sys
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional

from ..utils.config_loader import get_settings
from ..logger import GLOBAL_LOGGER as logger
from ..exception.custom_exception import MulitagentragException

settings = get_settings()

class EmbeddingProvider:
    """
    Wraps a sentence-transformers model with bathing and caching.
    """

    def __init__(self):
        self._model = None
        self._lock = asyncio.Lock()
        self.config = settings.embeddings

    async def _load(self):
        try:
            if self._model is not None:
                return
            async with self._lock:
                if self._model is not None:
                    return
                
                from sentence_transformers import SentenceTransformer
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None, SentenceTransformer, self.config.model
                )
                logger.info("embedding_model_loaded", model=self.config.model)

        except Exception as e:
            logger.error("Unable to load the embedding Model", error=str(e))
            raise MulitagentragException("Unable to load teh embedding model", e)
            
    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            await self._load()
            results = []
            to_encode: list[tuple[int, str]]=[]

            for i, text in enumerate(texts):
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
                    ).tolist(),)

                for (i, text), vec in zip(to_encode, raw):
                    results.append((i, vec))

            results.sort(key=lambda x:x[0])
            embedded_tokens = [v for _, v in results]
            logger.info("Tokens embedded successfully")

            return embedded_tokens

        except Exception as e:
            logger.error("Failed to embed Query/Document", error= str(e))
            raise MulitagentragException(e, sys)
    

_embed: Optional[EmbeddingProvider]=None

async def get_embedder()-> EmbeddingProvider:
    global _embed
    if _embed is None:
        _embed = EmbeddingProvider()
        await _embed._load()

    return _embed