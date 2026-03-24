"""
Two cache classes that work together:

  SemanticCache   — embedding-based similarity lookup for RAG responses.
                    Stores (query_embedding, RAGResponse) pairs in Redis.
                    Returns a cached answer when a new query is >= similarity_threshold
                    similar to a previously seen one — handles paraphrases automatically.

  CacheClient     — general-purpose async Redis wrapper used by the API layer.
                    Handles: query result caching (hash-keyed), rate limiting
                    (sliding-window sorted set), query history (bounded list),
                    and arbitrary key/value get/set/delete.

Both classes share the same underlying Redis connection via get_redis().
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Optional

import numpy as np

from ..utils.config_loader import get_settings
from ..models.models import QueryResponse
from ..logger.logger import GLOBAL_LOGGER as logger
from ..embedding.embedding import get_embedder

settings = get_settings()

#Module-level Redis connection (shared singleton)

_redis = None


async def get_redis():
    """
    Lazily initialise the Redis connection once and reuse it.
    """
    global _redis

    if _redis is None:
        try:
            import redis.asyncio as aioredis
            _redis = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await _redis.ping()
            logger.info("redis_connected", url=settings.redis_url)
        except Exception as e:
            logger.warning("redis_unavailable", error=str(e))
            _redis = None

    return _redis


#Key builders
def _query_cache_key(query: str, top_k: int) -> str:
    """SHA-256 hash of (query, top_k) → short, collision-resistant cache key."""
    digest = hashlib.sha256(f"{query}:{top_k}".encode()).hexdigest()[:16]
    return f"rag:query:{digest}"


def _rate_limit_key(identifier: str) -> str:
    return f"rag:ratelimit:{identifier}"


#SemanticCache
class SemanticCache:
    """
    Embedding-based similarity cache for RAG query/response pairs.

    Stores each cached entry as two Redis keys:
      cache:{query_id}:emb -> JSON float list  (normalised query embedding)
      cache:{query_id}:resp -> JSON RAGResponse
      cache:index -> Redis Set of all cached query_ids

    On lookup, computes cosine similarity between the incoming query embedding
    and all stored embeddings. Returns the stored response if the best score
    >= similarity_threshold — this handles paraphrased queries automatically.

    Falls back silently to no-op when Redis is unavailable (self._client is None).
    """

    def __init__(self, config=None, redis_url: str = None, embed_model=None) -> None:
        self.config = config or settings.cache #CacheConfig
        self.embed_model = embed_model or get_embedder()
        self._r = None #set in connect()

    #Lifecycle
    async def _client(self):
        """Connect to Redis. Call once at application startup."""

        if self._r is None:
            self._r = await get_redis()

        return self._r

    async def close(self) -> None:
        r = await self._client()
        if r:
            await r.aclose()

    #Similarity lookup
    async def get(self, query: str) -> Optional[QueryResponse]:
        """Return a cached RAGResponse if a similar-enough query exists."""
        
        r = await self._client()
        if not r or not self.config.enabled:
            return None

        q_emb = self.embed_model.embed(query)
        ids = await r.smembers("cache:index")

        best_score, best_id = 0.0, None
        for cid in ids:
            raw = await r.get(f"cache:{cid}:emb")
            if not raw:
                continue
            cached_emb = np.array(json.loads(raw), dtype=np.float32)

            # np.dot on two L2-normalised vectors equals cosine similarity
            score = float(np.dot(q_emb, cached_emb))
            if score > best_score:
                best_score, best_id = score, cid

        if best_score >= self.config.similarity_threshold and best_id:
            raw_resp = await r.get(f"cache:{best_id}:resp")
            if raw_resp:
                resp = QueryResponse(**json.loads(raw_resp))
                resp.cached = True
                return resp

        return None

    async def set(self, query: str, response: QueryResponse) -> None:
        """Store a query embedding and its QueryResponse with TTL."""
        
        r = await self._client()
        if not r or not self.config.enabled:
            return

        qid = response.request_id
        emb = self.embed_model.embed(query)

        pipe = r.pipeline()
        # Convert numpy array to list for JSON serialization
        emb_list = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
        pipe.set(f"cache:{qid}:emb", json.dumps(emb_list), ex=self.config.ttl_seconds)
        pipe.set(f"cache:{qid}:resp", response.model_dump_json(), ex=self.config.ttl_seconds)
        pipe.sadd("cache:index", qid)
        await pipe.execute()

    # Rate limiter (sorted-set sliding window)
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: int,
    ) -> tuple[bool, int]:
        """
        Sliding-window rate limiter using a Redis sorted set.
        Each request is stored as a member with its timestamp as the score.
        Expired entries (outside the window) are removed on every check.

        Returns (allowed, remaining_requests).
        Falls back to (True, limit) when Redis is unavailable.
        """
        r = await self._client()
        if not r:
            return True, limit

        now = time.time()
        key = f"ratelimit:{identifier}"
        window_s = now - window

        pipe = r.pipeline()
        pipe.zremrangebyscore(key, 0, window_s) #remove entries older than window
        pipe.zadd(key, {str(now): now}) # record this request
        pipe.zcard(key) # count requests in window
        pipe.expire(key, window) # auto-expire the key itself
        results = await pipe.execute()

        count = results[2]
        allowed = count <= limit
        remaining = max(0, limit - count)
        return allowed, remaining
    

#CacheClient for the API layer
class CacheClient:
    """
    General-purpose async Redis wrapper for the API layer.

    Responsibilities:
      - get / set / delete arbitrary JSON values by key
      - get_query_result / set_query_result  (hash-keyed, for the /query endpoint)
      - check_rate_limit  (incr-based counter, simpler than sorted-set)
      - lpush_bounded / lrange  (bounded list for query history / analytics)
      - ping  (for /health endpoint latency check)

    Uses get_redis() which falls back to _InMemoryCache if Redis is down.
    """

    def __init__(self) -> None:
        self._r = None
        self.cfg = settings.cache

    async def _client(self):
        """Lazily resolve the Redis (or fallback) client."""
        if self._r is None:
            self._r = await get_redis()
        return self._r

    #Basic key/value
    async def get(self, key: str) -> Optional[Any]:
        r = await self._client()
        try:
            raw = await r.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.error("cache_get_failed", key=key, error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = None,
    ) -> None:
        r = await self._client()
        ttl = ttl if ttl is not None else self.cfg.ttl_seconds
        try:
            await r.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.error("cache_set_failed", key=key, error=str(e))

    async def delete(self, key: str) -> None:
        r = await self._client()
        try:
            await r.delete(key)
        except Exception:
            pass  # deletion failure is non-fatal


    #Rate limiter (incr-based counter)
    async def check_rate_limit(self, identifier: str) -> tuple[bool, int]:
        """Returns (allowed, remaining). Config comes from settings.rate_limit."""
        r   = await self._client()
        key = _rate_limit_key(identifier)
        try:
            pipe = r.pipeline()
            await pipe.incr(key)
            await pipe.expire(key, self.cfg.window_seconds)
            results = await pipe.execute()
            count = results[0]
            remaining = max(0, self.cfg.requests_per_minute - count)
            allowed = count <= self.cfg.requests_per_minute
            return allowed, remaining
        
        except Exception:
            # Redis error — fail open (allow the request)
            return True, self.cfg.requests_per_minute

    #Bounded list (query history / analytics)
    async def lpush_bounded(
        self, key: str, value: Any, max_len: int = 1000
    ) -> None:
        """Push value to the front of a list and trim to max_len entries."""
        r = await self._client()
        try:
            serialised = json.dumps(value, default=str)
            pipe = r.pipeline()
            await pipe.lpush(key, serialised)
            await pipe.ltrim(key, 0, max_len - 1)
            await pipe.execute()
        except Exception as e:
            logger.error("cache_lpush_failed", error=str(e))

    async def lrange(
        self, key: str, start: int = 0, stop: int = -1
    ) -> list[Any]:
        """Return a slice of a list, deserialising each entry from JSON."""
        r = await self._client()
        try:
            raw = await r.lrange(key, start, stop)
            return [json.loads(item) for item in raw]
        except Exception:
            return []

    #Health

    async def ping(self) -> float:
        """Ping Redis and return round-trip latency in milliseconds."""
        r  = await self._client()
        try:
            t0 = time.perf_counter()
            await r.ping()
            return (time.perf_counter() - t0) * 1000
        except Exception as e:
            logger.error("Redis not available", error = str(e))
            raise