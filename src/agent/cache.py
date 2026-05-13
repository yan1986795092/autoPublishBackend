"""LLM 响应缓存 — 减少重复调用，节省 token 成本"""

from __future__ import annotations
import hashlib
import json
import time
import threading
from collections import OrderedDict
from typing import Optional


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
                    return value
                del self._cache[key]
            self._misses += 1
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
