"""Cache monitoring — bridged to OTel Prometheus metrics.

Tracks cache hits and misses via both in-memory counters (for instant summary)
and OpenTelemetry counters (for Prometheus dashboards).

The OTel counters are registered in ``src.observability.metrics._init_agent_metrics``
as ``agent.cache.hit.total`` and ``agent.cache.miss.total``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MonitorMetrics:
    """Cache hit/miss tracker, bridged to OTel Prometheus counters.

    Usage::

        monitor = get_monitor()
        monitor.record_cache_hit()
        monitor.record_cache_miss()
        summary = monitor.summary()   # {"cache_hits": 5, "cache_misses": 3, "cache_hit_rate": 0.625}
    """

    cache_hits: int = 0
    cache_misses: int = 0

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / max(total, 1)

    # ── Recording helpers ───────────────────────────────────────────────────

    def record_cache_hit(self) -> None:
        """Record a cache hit (in-memory + OTel counter)."""
        self.cache_hits += 1
        _otel_cache_hit()

    def record_cache_miss(self) -> None:
        """Record a cache miss (in-memory + OTel counter)."""
        self.cache_misses += 1
        _otel_cache_miss()

    def summary(self) -> dict:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 3),
        }


# ── Singleton access ────────────────────────────────────────────────────────

_monitor: MonitorMetrics | None = None


def get_monitor() -> MonitorMetrics:
    global _monitor
    if _monitor is None:
        _monitor = MonitorMetrics()
    return _monitor


# ── OTel bridge helpers (lazy import to avoid circular deps at module level) ─

def _otel_cache_hit() -> None:
    try:
        from src.observability.metrics import agent_cache_hit
        if agent_cache_hit is not None:
            agent_cache_hit.add(1)
    except Exception:
        pass


def _otel_cache_miss() -> None:
    try:
        from src.observability.metrics import agent_cache_miss
        if agent_cache_miss is not None:
            agent_cache_miss.add(1)
    except Exception:
        pass
