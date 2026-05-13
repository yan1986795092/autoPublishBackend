"""性能监控 — Token 消耗、响应时间、缓存命中率统计"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MonitorMetrics:
    total_requests: int = 0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    by_agent: dict = field(default_factory=lambda: defaultdict(lambda: {"calls": 0, "tokens": 0, "time_ms": 0.0}))

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / max(self.total_requests, 1)

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / max(total, 1)

    def record(self, agent: str, tokens: int, elapsed_ms: float, cache_hit: bool = False):
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_time_ms += elapsed_ms
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        self.by_agent[agent]["calls"] += 1
        self.by_agent[agent]["tokens"] += tokens
        self.by_agent[agent]["time_ms"] += elapsed_ms

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "avg_time_ms": round(self.avg_time_ms, 1),
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "errors": self.errors,
            "by_agent": dict(self.by_agent),
        }


_monitor: MonitorMetrics | None = None


def get_monitor() -> MonitorMetrics:
    global _monitor
    if _monitor is None:
        _monitor = MonitorMetrics()
    return _monitor
