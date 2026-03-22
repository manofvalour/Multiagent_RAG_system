"""
src/core/cache.py

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
If Redis is unavailable at startup, _InMemoryCache silently takes over so
the service keeps running without a cache — useful in dev and CI.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional

import numpy as np

from ..utils.config_loader import get_settings
from ..models.models import QueryResponse
from ..logger.logger import GLOBAL_LOGGER as logger
from ..embedding.embedding import get_embedder

settings = get_settings()

# ── Module-level Redis connection (shared singleton) ──────────────────────────

_redis = None


async def get_redis():
    """
    Lazily initialise the Redis connection once and reuse it.
    Falls back to _InMemoryCache if Redis is unreachable — this means
    the service degrades gracefully (no caching) rather than crashing.
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
            logger.warning("redis_unavailable", error=str(e), fallback="in_memory")
            _redis = _InMemoryCache()

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
      cache:{query_id}:emb   -> JSON float list  (normalised query embedding)
      cache:{query_id}:resp  -> JSON RAGResponse
      cache:index            -> Redis Set of all cached query_ids

    On lookup, computes cosine similarity between the incoming query embedding
    and all stored embeddings. Returns the stored response if the best score
    >= similarity_threshold — this handles paraphrased queries automatically.

    Falls back silently to no-op when Redis is unavailable (self._client is None).
    """

    def __init__(self, redis_url: str) -> None:
        self.cfg = settings.cache           # CacheConfig
        self.redis_url = redis_url
        self.embed_model = get_embedder()
        self._client = None          # set in connect()

    #Lifecycle
    async def connect(self) -> None:
        """Connect to Redis. Call once at application startup."""
        try:
            import redis.asyncio as aioredis
            self._client = await aioredis.from_url(
                self.redis_url, encoding="utf-8", decode_responses=True
            )
            await self._client.ping()
            logger.info("SemanticCache connected", url=self.redis_url)
        except Exception as e:
            logger.warning("SemanticCache unavailable", error=str(e))
            self._client = None

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    #Similarity lookup
    async def get(self, query: str) -> Optional[QueryResponse]:
        """Return a cached RAGResponse if a similar-enough query exists."""
        if not self._client or not self.cfg.enabled:
            return None

        q_emb = self._embed(query)
        ids   = await self._client.smembers("cache:index")

        best_score, best_id = 0.0, None
        for cid in ids:
            raw = await self._client.get(f"cache:{cid}:emb")
            if not raw:
                continue
            cached_emb = np.array(json.loads(raw), dtype=np.float32)
            # np.dot on two L2-normalised vectors equals cosine similarity
            score = float(np.dot(q_emb, cached_emb))
            if score > best_score:
                best_score, best_id = score, cid

        if best_score >= self.cfg.similarity_threshold and best_id:
            raw_resp = await self._client.get(f"cache:{best_id}:resp")
            if raw_resp:
                resp = QueryResponse(**json.loads(raw_resp))
                resp.cached = True
                return resp

        return None

    async def set(self, query: str, response: QueryResponse) -> None:
        """Store a query embedding and its QueryResponse with TTL."""
        if not self._client or not self.cfg.enabled:
            return

        qid = response.id
        emb = self._embed(query).tolist()

        pipe = self._client.pipeline()
        pipe.set(f"cache:{qid}:emb", json.dumps(emb), ex=self.cfg.ttl_seconds)
        pipe.set(f"cache:{qid}:resp", response.model_dump_json(), ex=self.cfg.ttl_seconds)
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
        if not self._client:
            return True, limit

        now      = time.time()
        key      = f"ratelimit:{identifier}"
        window_s = now - window

        pipe = self._client.pipeline()
        pipe.zremrangebyscore(key, 0, window_s)  # remove entries older than window
        pipe.zadd(key, {str(now): now})           # record this request
        pipe.zcard(key)                           # count requests in window
        pipe.expire(key, window)                  # auto-expire the key itself
        results = await pipe.execute()

        count     = results[2]
        allowed   = count <= limit
        remaining = max(0, limit - count)
        return allowed, remaining

    # ── Helper ────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single string and return a normalised float32 vector."""
        return self.embed_model.embed(
            [text])[0].astype(np.float32)


# ── CacheClient ───────────────────────────────────────────────────────────────

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
        self._r    = None
        self.cfg   = settings.cache
        self.rl    = settings.rate_limit

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
        key:   str,
        value: Any,
        ttl:   int = None,
    ) -> None:
        r   = await self._client()
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

    # ── Query result cache ─────────────────────────────────────────────────
    # Keyed by SHA-256 hash of (query, top_k) — deterministic, short, safe

    async def get_query_result(self, query: str, top_k: int) -> Optional[dict]:
        return await self.get(_query_cache_key(query, top_k))

    async def set_query_result(
        self, query: str, top_k: int, result: dict
    ) -> None:
        await self.set(_query_cache_key(query, top_k), result)

    # ── Rate limiter (incr-based counter) ─────────────────────────────────
    # Simpler than the sorted-set approach in SemanticCache — uses a single
    # counter that resets when the key expires. Suitable for per-endpoint
    # limits where millisecond precision isn't needed.

    async def check_rate_limit(self, identifier: str) -> tuple[bool, int]:
        """Returns (allowed, remaining). Config comes from settings.rate_limit."""
        r   = await self._client()
        key = _rate_limit_key(identifier)
        try:
            pipe = r.pipeline()
            await pipe.incr(key)
            await pipe.expire(key, self.rl.window_seconds)
            results = await pipe.execute()
            count     = results[0]
            remaining = max(0, self.rl.requests_per_minute - count)
            allowed   = count <= self.rl.requests_per_minute
            return allowed, remaining
        except Exception:
            # Redis error — fail open (allow the request)
            return True, self.rl.requests_per_minute

    # ── Bounded list (query history / analytics) ──────────────────────────

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
        t0 = time.perf_counter()
        await r.ping()
        return (time.perf_counter() - t0) * 1000


#In-memory fallback
class _InMemoryCache:
    """
    Dict-based Redis stub. Used when Redis is unreachable so the service
    starts and runs without caching rather than refusing to boot.

    Implements the same async surface as aioredis.Redis for the methods
    used by CacheClient and SemanticCache.
    """

    def __init__(self) -> None:
        self._store: dict[str, str]       = {}
        self._lists: dict[str, list[str]] = {}
        self._zsets: dict[str, dict]      = {}

    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        # TTL is ignored in the in-memory fallback
        self._store[key] = value

    async def set(self, key: str, value: str, ex: int = None) -> None:
        self._store[key] = value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def incr(self, key: str) -> int:
        val = int(self._store.get(key, "0")) + 1
        self._store[key] = str(val)
        return val

    async def expire(self, key: str, ttl: int) -> None:
        pass  # no TTL support in memory fallback

    async def smembers(self, key: str) -> set:
        return set(self._store.get(key, "").split(",")) - {""}

    async def sadd(self, key: str, *values) -> None:
        existing = set(self._store.get(key, "").split(",")) - {""}
        existing.update(values)
        self._store[key] = ",".join(existing)

    async def lpush(self, key: str, *values) -> None:
        lst = self._lists.setdefault(key, [])
        for v in values:
            lst.insert(0, v)

    async def ltrim(self, key: str, start: int, stop: int) -> None:
        lst = self._lists.get(key, [])
        self._lists[key] = lst[start: stop + 1 if stop >= 0 else None]

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        lst = self._lists.get(key, [])
        return lst[start: stop + 1 if stop >= 0 else None]

    async def zremrangebyscore(self, key: str, mn: float, mx: float) -> None:
        zset = self._zsets.get(key, {})
        self._zsets[key] = {k: v for k, v in zset.items() if not (mn <= v <= mx)}

    async def zadd(self, key: str, mapping: dict) -> None:
        self._zsets.setdefault(key, {}).update(mapping)

    async def zcard(self, key: str) -> int:
        return len(self._zsets.get(key, {}))

    def pipeline(self) -> "_FakePipeline":
        return _FakePipeline(self)


class _FakePipeline:
    """
    Minimal pipeline stub for _InMemoryCache.
    Queues operations and replays them on execute().
    """

    def __init__(self, cache: _InMemoryCache) -> None:
        self._cache = cache
        self._ops:  list = []

    async def incr(self, key: str):
        self._ops.append(("incr", key))
        return self

    async def expire(self, key: str, ttl: int):
        self._ops.append(("expire", key, ttl))
        return self

    async def set(self, key: str, value: str, ex: int = None):
        self._ops.append(("set", key, value))
        return self

    async def setex(self, key: str, ttl: int, value: str):
        self._ops.append(("setex", key, ttl, value))
        return self

    async def sadd(self, key: str, *values):
        self._ops.append(("sadd", key, *values))
        return self

    async def lpush(self, key: str, *values):
        self._ops.append(("lpush", key, *values))
        return self

    async def ltrim(self, key: str, start: int, stop: int):
        self._ops.append(("ltrim", key, start, stop))
        return self

    async def zremrangebyscore(self, key: str, mn: float, mx: float):
        self._ops.append(("zremrangebyscore", key, mn, mx))
        return self

    async def zadd(self, key: str, mapping: dict):
        self._ops.append(("zadd", key, mapping))
        return self

    async def zcard(self, key: str):
        self._ops.append(("zcard", key))
        return self

    async def execute(self) -> list:
        results = []
        for op in self._ops:
            name = op[0]
            if name == "incr":
                results.append(await self._cache.incr(op[1]))
            elif name == "expire":
                await self._cache.expire(op[1], op[2])
                results.append(True)
            elif name == "set":
                await self._cache.set(op[1], op[2])
                results.append(True)
            elif name == "setex":
                await self._cache.setex(op[1], op[2], op[3])
                results.append(True)
            elif name == "sadd":
                await self._cache.sadd(op[1], *op[2:])
                results.append(1)
            elif name == "lpush":
                await self._cache.lpush(op[1], *op[2:])
                results.append(1)
            elif name == "ltrim":
                await self._cache.ltrim(op[1], op[2], op[3])
                results.append(True)
            elif name == "zremrangebyscore":
                await self._cache.zremrangebyscore(op[1], op[2], op[3])
                results.append(0)
            elif name == "zadd":
                await self._cache.zadd(op[1], op[2])
                results.append(1)
            elif name == "zcard":
                results.append(await self._cache.zcard(op[1]))
            else:
                results.append(None)
        return results