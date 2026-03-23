"""
tests/test_cache.py
Unit tests for SemanticCache, CacheClient, _InMemoryCache, and _FakePipeline.

Coverage map:
  SemanticCache   — get (miss / hit / disabled), set, check_rate_limit (sorted-set)
  CacheClient     — get, set, delete, get/set_query_result, check_rate_limit (incr),
                    lpush_bounded, lrange, ping
  _InMemoryCache  — all async methods (used as the fallback when Redis is down)
  _FakePipeline   — execute() with every op type
"""
from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from ..cache.cache import (
    CacheClient,
    SemanticCache,
    _query_cache_key,
    _rate_limit_key,
)
from ..utils.config_loader import CacheConfig
from ..models.models import QueryResponse


#Shared helpers

def _unit_vec(dim: int = 4) -> np.ndarray:
    """Return a normalised float32 vector of given dimension."""
    v = np.ones(dim, dtype=np.float32)
    return v / np.linalg.norm(v)


def _make_response(qid: str = "test-qid", answer: str = "Test answer") -> QueryResponse:
    return QueryResponse(
        request_id=qid,
        answer=answer,
        claims=["doc.txt"],
        retrieved_chunks=[],
        reranked_chunks=[],
        expanded_queries=[],
        confidence=[],
        hallucination_risk=0.0,
        agent_trace=[],
    )


def _make_embed_model(vec: np.ndarray = None) -> MagicMock:
    """Mock embed model whose encode() always returns the given vector."""
    vec = vec if vec is not None else _unit_vec()
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([vec], dtype=np.float32))
    return model


#Key builder tests

class TestKeyBuilders:
    def test_query_cache_key_is_deterministic(self):
        k1 = _query_cache_key("What is RAG?", 5)
        k2 = _query_cache_key("What is RAG?", 5)
        assert k1 == k2

    def test_query_cache_key_differs_by_top_k(self):
        assert _query_cache_key("query", 5) != _query_cache_key("query", 10)

    def test_query_cache_key_differs_by_query(self):
        assert _query_cache_key("query A", 5) != _query_cache_key("query B", 5)

    def test_query_cache_key_has_prefix(self):
        assert _query_cache_key("q", 1).startswith("rag:query:")

    def test_rate_limit_key_has_prefix(self):
        assert _rate_limit_key("user-1").startswith("rag:ratelimit:")


#SemanticCache tests

@pytest.fixture
def sem_cfg():
    return CacheConfig(enabled=True, ttl_seconds=300, similarity_threshold=0.90)


@pytest.fixture
def sem_cache(sem_cfg):
    embed = _make_embed_model(_unit_vec())
    cache = SemanticCache(sem_cfg, "redis://localhost:6379/0", embed)
    # Inject a mock Redis client directly — no network needed
    mock_r = MagicMock()
    mock_r.smembers  = AsyncMock(return_value=set())
    mock_r.get       = AsyncMock(return_value=None)
    mock_r.pipeline  = MagicMock(return_value=MagicMock(
        set=MagicMock(),
        sadd=MagicMock(),
        execute=AsyncMock(return_value=[True, True, 1]),
    ))
    mock_r.ping = AsyncMock(return_value=True)
    cache._client = mock_r
    return cache


class TestSemanticCacheGet:
    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        cfg   = CacheConfig(enabled=False)
        cache = SemanticCache(cfg, "", _make_embed_model())
        cache._client = MagicMock()     # should never be touched
        result = await cache.get("any query")
        assert result is None
        cache._client.smembers.assert_not_called() if hasattr(cache._client, "smembers") else None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_client(self):
        cfg   = CacheConfig(enabled=True)
        cache = SemanticCache(cfg, "", _make_embed_model())
        # _client is None by default — simulates Redis unavailable
        result = await cache.get("any query")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_cache_empty(self, sem_cache):
        sem_cache._client.smembers = AsyncMock(return_value=set())
        result = await sem_cache.get("What is RAG?")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_cached_response_on_high_similarity(self, sem_cache):
        """Same embedding vector should score 1.0 similarity → cache hit."""
        stored_emb  = _unit_vec().tolist()
        response    = _make_response()

        sem_cache._client.smembers = AsyncMock(return_value={"test-qid"})
        sem_cache._client.get      = AsyncMock(side_effect=[
            json.dumps(stored_emb),         # first call: embedding
            response.model_dump_json(),     # second call: response body
        ])

        result = await sem_cache.get("What is RAG?")
        assert result is not None
        assert result.cached is True
        assert result.answer == "Test answer"

    @pytest.mark.asyncio
    async def test_returns_none_on_low_similarity(self, sem_cache):
        """Orthogonal vector → similarity ≈ 0 → below threshold → miss."""
        # _unit_vec() is [0.5, 0.5, 0.5, 0.5]. Orthogonal vector is [1,-1,0,0]/√2
        orthogonal = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        orthogonal /= np.linalg.norm(orthogonal)

        sem_cache._client.smembers = AsyncMock(return_value={"old-qid"})
        sem_cache._client.get      = AsyncMock(return_value=json.dumps(orthogonal.tolist()))

        result = await sem_cache.get("completely different query")
        assert result is None


class TestSemanticCacheSet:
    @pytest.mark.asyncio
    async def test_set_calls_pipeline(self, sem_cache):
        response = _make_response("qid-001")
        await sem_cache.set("What is RAG?", response)
        sem_cache._client.pipeline.assert_called()

    @pytest.mark.asyncio
    async def test_set_noop_when_disabled(self):
        cfg   = CacheConfig(enabled=False)
        cache = SemanticCache(cfg, "", _make_embed_model())
        cache._client = MagicMock()
        await cache.set("query", _make_response())
        cache._client.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_noop_when_no_client(self):
        cfg   = CacheConfig(enabled=True)
        cache = SemanticCache(cfg, "", _make_embed_model())
        # No exception should be raised
        await cache.set("query", _make_response())


class TestSemanticCacheRateLimit:
    @pytest.mark.asyncio
    async def test_allows_within_limit(self, sem_cache):
        mock_pipe = MagicMock()
        mock_pipe.zremrangebyscore = MagicMock()
        mock_pipe.zadd             = MagicMock()
        mock_pipe.zcard            = MagicMock()
        mock_pipe.expire           = MagicMock()
        mock_pipe.execute          = AsyncMock(return_value=[0, 1, 5, True])
        sem_cache._client.pipeline = MagicMock(return_value=mock_pipe)

        allowed, remaining = await sem_cache.check_rate_limit("user-1", limit=60, window=60)
        assert allowed is True
        assert remaining == 55

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self, sem_cache):
        mock_pipe = MagicMock()
        mock_pipe.zremrangebyscore = MagicMock()
        mock_pipe.zadd  = MagicMock()
        mock_pipe.zcard = MagicMock()
        mock_pipe.expire = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[0, 1, 61, True])
        sem_cache._client.pipeline = MagicMock(return_value=mock_pipe)

        allowed, remaining = await sem_cache.check_rate_limit("user-1", limit=60, window=60)
        assert allowed is False
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_falls_back_when_no_client(self):
        cfg   = CacheConfig(enabled=True)
        cache = SemanticCache(cfg, "", _make_embed_model())
        # No client — should return (True, limit) without raising
        allowed, remaining = await cache.check_rate_limit("user-1", limit=60, window=60)
        assert allowed is True
        assert remaining == 60


#CacheClient tests
@pytest.fixture
def mock_redis():
    """A MagicMock that looks like an aioredis.Redis client."""
    r = MagicMock()
    r.get    = AsyncMock(return_value=None)
    r.setex  = AsyncMock()
    r.delete = AsyncMock()
    r.ping   = AsyncMock(return_value=True)
    r.lrange = AsyncMock(return_value=[])

    pipe = MagicMock()
    pipe.incr    = AsyncMock(return_value=pipe)
    pipe.expire  = AsyncMock(return_value=pipe)
    pipe.lpush   = AsyncMock(return_value=pipe)
    pipe.ltrim   = AsyncMock(return_value=pipe)
    pipe.execute = AsyncMock(return_value=[1, True])
    r.pipeline   = MagicMock(return_value=pipe)

    return r


@pytest.fixture
def cache_client(mock_redis):
    client    = CacheClient()
    client._r = mock_redis   # bypass get_redis()
    return client


class TestCacheClientGetSet:
    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing_key(self, cache_client, mock_redis):
        mock_redis.get = AsyncMock(return_value=None)
        result = await cache_client.get("missing:key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_deserialises_json(self, cache_client, mock_redis):
        mock_redis.get = AsyncMock(return_value=json.dumps({"foo": "bar"}))
        result = await cache_client.get("some:key")
        assert result == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_get_returns_none_on_redis_error(self, cache_client, mock_redis):
        mock_redis.get = AsyncMock(side_effect=Exception("connection lost"))
        result = await cache_client.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_serialises_to_json(self, cache_client, mock_redis):
        await cache_client.set("my:key", {"value": 42}, ttl=60)
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[0] == "my:key"
        assert args[1] == 60
        assert json.loads(args[2]) == {"value": 42}

    @pytest.mark.asyncio
    async def test_set_uses_default_ttl(self, cache_client, mock_redis):
        await cache_client.set("key", "value")
        args = mock_redis.setex.call_args[0]
        # default TTL comes from settings.cache.ttl_seconds
        assert args[1] > 0

    @pytest.mark.asyncio
    async def test_delete_calls_redis(self, cache_client, mock_redis):
        await cache_client.delete("stale:key")
        mock_redis.delete.assert_called_once_with("stale:key")

    @pytest.mark.asyncio
    async def test_delete_silently_ignores_errors(self, cache_client, mock_redis):
        mock_redis.delete = AsyncMock(side_effect=Exception("gone"))
        # Should not raise
        await cache_client.delete("key")


class TestCacheClientQueryResult:
    @pytest.mark.asyncio
    async def test_get_query_result_miss(self, cache_client, mock_redis):
        mock_redis.get = AsyncMock(return_value=None)
        result = await cache_client.get_query_result("What is RAG?", 5)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_query_result(self, cache_client, mock_redis):
        payload = {"answer": "RAG is great", "sources": ["doc.txt"]}
        mock_redis.get = AsyncMock(return_value=json.dumps(payload))
        result = await cache_client.get_query_result("What is RAG?", 5)
        assert result["answer"] == "RAG is great"

    @pytest.mark.asyncio
    async def test_set_query_result_calls_setex(self, cache_client, mock_redis):
        await cache_client.set_query_result("query", 5, {"answer": "yes"})
        mock_redis.setex.assert_called_once()


class TestCacheClientRateLimit:
    @pytest.mark.asyncio
    async def test_allows_within_limit(self, cache_client, mock_redis):
        pipe = mock_redis.pipeline()
        pipe.execute = AsyncMock(return_value=[5, True])
        allowed, remaining = await cache_client.check_rate_limit("user-1")
        assert allowed is True
        assert remaining >= 0

    @pytest.mark.asyncio
    async def test_blocks_when_count_exceeds_limit(self, cache_client, mock_redis):
        pipe = mock_redis.pipeline()
        # Simulate 100 requests already in the window (limit is 60)
        pipe.execute = AsyncMock(return_value=[100, True])
        allowed, remaining = await cache_client.check_rate_limit("user-heavy")
        assert allowed is False
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_fails_open_on_redis_error(self, cache_client, mock_redis):
        mock_redis.pipeline = MagicMock(side_effect=Exception("redis down"))
        allowed, remaining = await cache_client.check_rate_limit("user-1")
        assert allowed is True


class TestCacheClientList:
    @pytest.mark.asyncio
    async def test_lpush_bounded_calls_pipeline(self, cache_client, mock_redis):
        await cache_client.lpush_bounded("rag:history", {"query": "test"})
        pipe = mock_redis.pipeline()
        pipe.execute.assert_called()

    @pytest.mark.asyncio
    async def test_lrange_returns_deserialised_list(self, cache_client, mock_redis):
        items = [json.dumps({"q": f"query {i}"}) for i in range(3)]
        mock_redis.lrange = AsyncMock(return_value=items)
        result = await cache_client.lrange("rag:history", 0, 2)
        assert len(result) == 3
        assert result[0]["q"] == "query 0"

    @pytest.mark.asyncio
    async def test_lrange_returns_empty_on_error(self, cache_client, mock_redis):
        mock_redis.lrange = AsyncMock(side_effect=Exception("gone"))
        result = await cache_client.lrange("missing:key")
        assert result == []


class TestCacheClientPing:
    @pytest.mark.asyncio
    async def test_ping_returns_positive_latency(self, cache_client, mock_redis):
        mock_redis.ping = AsyncMock(return_value=True)
        latency = await cache_client.ping()
        assert latency >= 0.0

    @pytest.mark.asyncio
    async def test_ping_raises_on_failure(self, cache_client, mock_redis):
        mock_redis.ping = AsyncMock(side_effect=Exception("timeout"))
        with pytest.raises(Exception):
            await cache_client.ping()


# ── _InMemoryCache tests ──────────────────────────────────────────────────────

class TestInMemoryCache:
    @pytest.mark.asyncio
    async def test_ping_returns_true(self):
        c = _InMemoryCache()
        assert await c.ping() is True

    @pytest.mark.asyncio
    async def test_get_missing_key_returns_none(self):
        c = _InMemoryCache()
        assert await c.get("missing") is None

    @pytest.mark.asyncio
    async def test_setex_and_get(self):
        c = _InMemoryCache()
        await c.setex("k", 60, "hello")
        assert await c.get("k") == "hello"

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        c = _InMemoryCache()
        await c.set("k", "world")
        assert await c.get("k") == "world"

    @pytest.mark.asyncio
    async def test_delete_removes_key(self):
        c = _InMemoryCache()
        await c.setex("k", 60, "v")
        await c.delete("k")
        assert await c.get("k") is None

    @pytest.mark.asyncio
    async def test_incr_increments(self):
        c = _InMemoryCache()
        assert await c.incr("counter") == 1
        assert await c.incr("counter") == 2
        assert await c.incr("counter") == 3

    @pytest.mark.asyncio
    async def test_sadd_and_smembers(self):
        c = _InMemoryCache()
        await c.sadd("myset", "a", "b", "c")
        members = await c.smembers("myset")
        assert members == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_lpush_and_lrange(self):
        c = _InMemoryCache()
        await c.lpush("mylist", "first")
        await c.lpush("mylist", "second")
        # lpush inserts at front, so second is at index 0
        result = await c.lrange("mylist", 0, -1)
        assert result[0] == "second"
        assert result[1] == "first"

    @pytest.mark.asyncio
    async def test_ltrim_limits_list(self):
        c = _InMemoryCache()
        for i in range(5):
            await c.lpush("mylist", str(i))
        await c.ltrim("mylist", 0, 2)
        result = await c.lrange("mylist", 0, -1)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_zset_operations(self):
        c = _InMemoryCache()
        now = time.time()
        await c.zadd("zk", {"req1": now - 120, "req2": now})
        assert await c.zcard("zk") == 2
        # Remove entries older than 60 seconds
        await c.zremrangebyscore("zk", 0, now - 60)
        assert await c.zcard("zk") == 1


# ── _FakePipeline tests ───────────────────────────────────────────────────────

class TestFakePipeline:
    @pytest.mark.asyncio
    async def test_incr_and_expire_execute(self):
        c    = _InMemoryCache()
        pipe = _FakePipeline(c)
        await pipe.incr("counter")
        await pipe.expire("counter", 60)
        results = await pipe.execute()
        assert results[0] == 1   # incr returned 1
        assert results[1] is True

    @pytest.mark.asyncio
    async def test_set_execute(self):
        c    = _InMemoryCache()
        pipe = _FakePipeline(c)
        await pipe.set("k", "v")
        await pipe.execute()
        assert await c.get("k") == "v"

    @pytest.mark.asyncio
    async def test_sadd_execute(self):
        c    = _InMemoryCache()
        pipe = _FakePipeline(c)
        await pipe.sadd("s", "a", "b")
        await pipe.execute()
        members = await c.smembers("s")
        assert "a" in members

    @pytest.mark.asyncio
    async def test_lpush_ltrim_execute(self):
        c    = _InMemoryCache()
        pipe = _FakePipeline(c)
        await pipe.lpush("lst", "x")
        await pipe.ltrim("lst", 0, 0)
        await pipe.execute()
        result = await c.lrange("lst", 0, -1)
        assert result == ["x"]

    @pytest.mark.asyncio
    async def test_zset_pipeline_execute(self):
        c    = _InMemoryCache()
        pipe = _FakePipeline(c)
        now  = time.time()
        await pipe.zremrangebyscore("zk", 0, now - 60)
        await pipe.zadd("zk", {str(now): now})
        await pipe.zcard("zk")
        results = await pipe.execute()
        assert results[2] == 1   # zcard returned 1

    @pytest.mark.asyncio
    async def test_unknown_op_returns_none(self):
        c    = _InMemoryCache()
        pipe = _FakePipeline(c)
        pipe._ops.append(("unknown_op", "key"))
        results = await pipe.execute()
        assert results[0] is None




if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
    #test_retrieval_validation_sorts_by_relevance()