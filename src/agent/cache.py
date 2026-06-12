"""LLM 响应缓存 — 减少重复调用，节省 token 成本"""

from __future__ import annotations
import asyncio
import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Optional

from src.observability.metrics import cache_hits_total, cache_misses_total


class LLMCache:
    """简单的 TTL 内存缓存，用于 LLM 响应去重

    命中策略：相同的 (model, messages, temperature) 组合返回缓存结果。
    默认 TTL: 1 小时，最大条目 500。
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(model: str, messages: list[dict], temperature: float) -> str:
        """根据请求参数生成缓存键"""
        raw = json.dumps({"model": model, "messages": messages, "temperature": temperature}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model: str, messages: list[dict], temperature: float = 0.7) -> Optional[str]:
        """获取缓存命中，过期返回 None"""
        key = self._make_key(model, messages, temperature)
        with self._lock:
            if key in self._cache:
                expires_at, value = self._cache[key]
                if time.time() < expires_at:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    cache_hits_total.labels(cache_name="llm_cache").inc()
                    return value
                del self._cache[key]
            self._misses += 1
            cache_misses_total.labels(cache_name="llm_cache").inc()
            return None

    def set(self, model: str, messages: list[dict], temperature: float, response: str) -> None:
        """写入缓存"""
        key = self._make_key(model, messages, temperature)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time() + self._ttl, response)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 3),
        }


# 模块级单例
llm_cache = LLMCache()


class AgentCache:
    """Agent 执行结果缓存

    缓存相同输入的 Agent 执行结果，避免重复 LLM 调用。
    基于 (agent_type, input_hash) 的缓存，带 TTL 和 LRU 淘汰。

    线程安全（get/set 使用 asyncio.Lock，invalidate/stats 使用 threading.Lock）。
    """

    def __init__(self, default_ttl: int = 3600, max_entries: int = 500):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, agent_type: str, task_input: dict) -> str:
        """生成缓存键 — 使用 json.dumps(default=str) 以 C 速度处理不可序列化对象。

        相比递归的 _sanitize_for_json 方法（Python 级遍历 + try/except 每元素），
        json.dumps(default=str) 在内层的 C json encoder 中处理，快 10-100 倍。
        """
        try:
            normalized = json.dumps(task_input, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            # 极端情况回退：某些对象 str() 后仍含有不可序列化内容
            normalized = json.dumps(str(task_input), sort_keys=True, ensure_ascii=False)
        input_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"{agent_type}:{input_hash}"

    async def get(self, agent_type: str, task_input: dict) -> Optional[dict]:
        """获取缓存命中，过期返回 None"""
        key = self._make_key(agent_type, task_input)
        async with self._async_lock:
            if key in self._cache:
                expires_at, result = self._cache[key]
                if time.time() < expires_at:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    cache_hits_total.labels(cache_name="agent_cache").inc()
                    return result
                del self._cache[key]
            self._misses += 1
            cache_misses_total.labels(cache_name="agent_cache").inc()
            return None

    async def set(self, agent_type: str, task_input: dict, result: dict, ttl: Optional[int] = None) -> None:
        """写入缓存"""
        if result.get("status") == "failed":
            return
        key = self._make_key(agent_type, task_input)
        async with self._async_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time() + (ttl or self._default_ttl), result)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)

    def invalidate(self, agent_type: Optional[str] = None) -> None:
        """清空缓存（同步方法，使用 threading.Lock）"""
        with self._sync_lock:
            if agent_type is None:
                self._cache.clear()
            else:
                keys = [k for k in self._cache if k.startswith(f"{agent_type}:")]
                for k in keys:
                    del self._cache[k]
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 3),
        }


# 全局 Agent 缓存实例
agent_cache = AgentCache()
