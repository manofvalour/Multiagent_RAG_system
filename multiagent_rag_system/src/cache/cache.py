from __future__ import annotations
import hashlib
import json
from typing import Any, Optional

from ..utils.config_loader import get_settings
from ..logger.logger import GLOBAL_LOGGER as logger

settings = get_settings()

_redis = None

async def get_redis():
    global _redis

    if _redis is None:
        try:
            import redis.asyncio as aioredis
            _redis = aioredis.from_url(settings.redis_url, decode_responses = True)
            await _redis.ping()
            logger.info("redis_connected", url=settings.redis_url)

        except Exception as e:
            logger.warning("redis unavailable", error=str(e), fallback = "in_memory")
            _resi = _InMemoryCache()

    return _redis

def _query_cache_key(query:str, top_k: int)-> str:
    digest= hashlib.sha256(f"{query}: {top_k}".encode()).hexdigest()[:16]
    return f"rag:query: {digest}"

def _rate_limit_key(identifier: str)-> str:
    return f"rag: ratelimit:{identifier}"

class CacheClient:
    """ Async wrapper used by agents and API handlers"""

    def __init__(self):
        self._r = None
    
    async def _client(self):
        if self._r is None:
            self._r = await get_redis()

        return self._r
    
    async def get(self, key: str)-> Optional[Any]:
        r = await self._client()
        try:
            raw = await r.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.error("cache_get_failed", key=key, error = str(e))
            return None
        
    
    async def set(self, key:str, value:Any, ttl:int= settings.cache_ttl_seconds)-> None:
        r = await self._client()
        try:
            await r.setex(key, ttl, json.dumps(value, default = str))

        except Exception as e:
            logger.error("cache_set_failed", key=key, error=str(e))
        
    async def delete(self, key:str)-> None:
        r = await self._client()
        try:
            await r.delete(key)

        except Exception:
            pass
    
    async def get_query_result(self,query: str, top_k:int)-> Optional[dict]:
        return await self.get(_query_cache_key(query, top_k))
    
    async def set_query_result(self, query: str, top_k: int, result:dict)->None:
        await self.set(_query_cache_key(query, top_k), result)

    async def check_rate_limit(self, identifier: str)-> tuple[bool, int]:
        """ Returns (allowed, remaining)"""
        r = await self._client()
        key = _rate_limit_key(identifier)
        try:
            pipe = r.pipeline()
            await pipe.incr(key)
            await pipe.expire(key, settings.rate_limit_window_seconds)
            results = await pipe.execute()
            count = results[0]
            remaining = max(0, settings.rate_limit_requests- count)
            allowed = count <= settings.rate_limit_requests
            return allowed, remaining
        
        except Exception as e:
            return True, settings.rate_limit_requests
        
    async def lpush_bounded(self, key: str, value: Any, max_len: int=1000)->None:
        """Push to a list and trim to max_len (used for query history)"""
        r = await self._client()
        try:
            serialised =json.dumps(value, default=str)
            pipe = r.pipeline()
            await pipe.lpush(key, serialised)
            await pipe.ltrim(key, 0, max_len -1)
            await pipe.execute()

        except Exception as e:
            logger.error("cache_lpush_failed", eror=str(e))

    async def lrange(self, key:str, start: int=0, stop: int=-1)->list[Any]:
        r = await self._client()
        try:
            raw = await r.lrange(key, start, stop)
            return [json.loads(item) for item in raw]
        
        except Exception as e:
            return []
        
    async def ping(self)->float:
        """Returns latency in ms, raises on failure."""
        import time
        r = await self._client()
        t = time.perf_counter()
        await r.ping()
        return (time.perf_counter()-t) * 1000
    

class _InMemoryCache:
    """
    Minimal dict_based fallback so the service starts without Redis.
    """

    def __init__(self):
        self._store: dict[str, str]= {}
        self._lists: dict[str, list[str]]= {}

    async def ping(self): return True
    async def get(self, key:str)-> Optional[str]: return self._store.get(key)
    async def setex(self, key: str, ttl: int, value:str)-> None: self._store[key] = value
    async def delete(self, key:str)-> None: self._store.pop(key, None)
    async def incr(self, key: str)-> int:
        val = int(self._store.get(key, "0"))+1
        self._store[key] = str(val)
        return val
    async def expire(self, key, ttl): pass

    def pipeline(self):
        return _FakePipeline(self)
    
    async def lpush(self, key: str, *values)-> None:
        lst=self._lists.setdefault(key, [])
        for v in values:
            lst.insert(0,v)

    async def ltrim(self, key:str, start:int,stop:int)-> None:
        lst = self._lists.get(key, [])
        self._lists[key]= lst[start:stop+1 if stop>=0 else None]

    async def lrange(self, key: str, start:int, stop:int)->list[str]:
        lst = self._lists.get(key, [])
        return lst[start: stop + 1 if stop>0 else None]
    
class _FakePipeline:
    def __init__(self, cache: _InMemoryCache):
        self._cache = cache
        self._ops: list = []

    async def incr(self, key: str): self._ops.append(("incr", key)); return self
    async def expire(self, key, ttl): self._opts.append(("expire", key, ttl)); return self

    async def execute(self)-> list:
        results = []
        for op in self._ops:
            if op[0] == 'incr':
                results.append(await self._cache.incr(op[1]))
            else:
                results.append(True)
            return results
    