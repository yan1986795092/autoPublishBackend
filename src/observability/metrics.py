"""Agent metrics — Prometheus counters, histograms & in-memory StatsCollector.

Usage::

    from src.observability.metrics import stats_collector

    # Record events (dual-writes to Prometheus + in-memory)
    stats_collector.record_agent_run(status="success", skill_name="writing", duration_s=1.5)
    stats_collector.record_llm_call(model="gpt-4o", provider="openai", status="success",
                                    prompt_tokens=100, completion_tokens=50, duration_s=2.0)
    stats_collector.record_tool_call(tool_name="web_search", status="success", duration_s=0.8)

    # Get snapshot for the metrics API
    snapshot = stats_collector.get_stats()
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from prometheus_client import Counter, Histogram, Gauge


# ── Agent-level counters ──────────────────────────────────────────────

agent_runs_total = Counter(
    "agent_runs_total",
    "Total number of agent runs, by status and skill",
    labelnames=["status", "skill_name"],
)

agent_run_duration_seconds = Histogram(
    "agent_run_duration_seconds",
    "Agent run duration in seconds",
    labelnames=["skill_name"],
    buckets=(1, 2.5, 5, 10, 15, 20, 30, 45, 60, 90, 120, float("inf")),
)

agent_tasks_total = Counter(
    "agent_tasks_total",
    "Total number of agent sub-tasks, by agent_type and status",
    labelnames=["agent_type", "status"],
)

# ── LLM counters ─────────────────────────────────────────────────────

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed, by model, provider, and token type",
    labelnames=["model", "provider", "type"],  # type: prompt | completion
)

llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests, by model and provider",
    labelnames=["model", "provider", "status"],  # status: success | error
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration in seconds",
    labelnames=["model", "provider"],
    buckets=(0.1, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, float("inf")),
)

# ── Tool call counters ────────────────────────────────────────────────

tool_calls_total = Counter(
    "tool_calls_total",
    "Total tool calls, by tool_name and status",
    labelnames=["tool_name", "status"],  # status: success | error
)

tool_call_duration_seconds = Histogram(
    "tool_call_duration_seconds",
    "Tool call duration in seconds",
    labelnames=["tool_name"],
    buckets=(0.1, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, float("inf")),
)

# ── Cache counters ────────────────────────────────────────────────────

cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
    labelnames=["cache_name"],
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
    labelnames=["cache_name"],
)

# ── System gauges ─────────────────────────────────────────────────────

up_gauge = Gauge(
    "autopublish_up",
    "Whether the autopublish-api service is up (1) or down (0)",
)

info_gauge = Gauge(
    "autopublish_info",
    "A metric with a single 'version' label for service info",
    labelnames=["version"],
)


# ── StatsCollector: in-memory aggregation + Prometheus bridge ─────────


class StatsCollector:
    """Lightweight in-memory stats collector that also writes to Prometheus.

    Accumulates counters that are:
    1. Exposed as Prometheus metrics via ``/metrics``
    2. Queryable via ``get_stats()`` for the ``/agent/metrics`` API
    """

    def __init__(self) -> None:
        self._agent_runs: dict[str, int] = defaultdict(int)
        self._agent_run_durations: list[float] = []

        self._llm_calls: dict[str, int] = defaultdict(int)
        self._llm_errors: int = 0
        self._llm_prompt_tokens: int = 0
        self._llm_completion_tokens: int = 0
        self._llm_durations: list[float] = []

        self._tool_calls: dict[str, dict[str, int]] = {}  # tool_name -> {total, errors}

    # ── Agent runs ─────────────────────────────────────────────────

    def record_agent_run(self, status: str = "", skill_name: str = "", duration_s: float = 0) -> None:
        """Record an agent run completion."""
        self._agent_runs["total"] += 1
        self._agent_runs[status] += 1
        if duration_s > 0:
            self._agent_run_durations.append(duration_s)
            self._agent_run_durations = self._agent_run_durations[-1000:]  # sliding window
        try:
            agent_runs_total.labels(status=status, skill_name=skill_name).inc()
            if duration_s > 0:
                agent_run_duration_seconds.labels(skill_name=skill_name).observe(duration_s)
        except Exception:
            pass

    # ── LLM calls ──────────────────────────────────────────────────

    def record_llm_call(
        self,
        model: str = "",
        provider: str = "",
        status: str = "success",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        duration_s: float = 0,
    ) -> None:
        """Record an LLM call."""
        self._llm_calls["total"] += 1
        if status != "success":
            self._llm_errors += 1
            self._llm_calls["error"] += 1
        self._llm_prompt_tokens += prompt_tokens
        self._llm_completion_tokens += completion_tokens
        if duration_s > 0:
            self._llm_durations.append(duration_s)
            self._llm_durations = self._llm_durations[-1000:]
        try:
            llm_requests_total.labels(model=model, provider=provider, status=status).inc()
            if prompt_tokens:
                llm_tokens_total.labels(model=model, provider=provider, type="prompt").inc(prompt_tokens)
            if completion_tokens:
                llm_tokens_total.labels(model=model, provider=provider, type="completion").inc(completion_tokens)
            if duration_s > 0:
                llm_request_duration_seconds.labels(model=model, provider=provider).observe(duration_s)
        except Exception:
            pass

    # ── Tool calls ─────────────────────────────────────────────────

    def record_tool_call(self, tool_name: str = "", status: str = "success", duration_s: float = 0) -> None:
        """Record a tool call."""
        if tool_name not in self._tool_calls:
            self._tool_calls[tool_name] = {"total": 0, "errors": 0}
        self._tool_calls[tool_name]["total"] += 1
        if status != "success":
            self._tool_calls[tool_name]["errors"] += 1
        try:
            tool_calls_total.labels(tool_name=tool_name, status=status).inc()
            if duration_s > 0:
                tool_call_duration_seconds.labels(tool_name=tool_name).observe(duration_s)
        except Exception:
            pass

    # ── Stats snapshot ─────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return a snapshot of all accumulated stats for the metrics API."""
        total_runs = self._agent_runs.get("total", 0)
        total_llm = self._llm_calls.get("total", 0)

        def _p95(values: list[float]) -> float:
            if not values:
                return 0.0
            sorted_vals = sorted(values)
            idx = max(0, int(len(sorted_vals) * 0.95) - 1)
            return round(sorted_vals[idx], 2)

        return {
            "agent_runs": {
                "total": total_runs,
                "success": self._agent_runs.get("success", 0),
                "failed": self._agent_runs.get("failed", 0),
                "timeout": self._agent_runs.get("timeout", 0),
                "success_rate": round(self._agent_runs.get("success", 0) / max(total_runs, 1), 4),
                "p95_duration_ms": _p95(self._agent_run_durations) * 1000 if self._agent_run_durations else 0,
            },
            "llm_calls": {
                "total": total_llm,
                "errors": self._llm_errors,
                "prompt_tokens": self._llm_prompt_tokens,
                "completion_tokens": self._llm_completion_tokens,
                "total_tokens": self._llm_prompt_tokens + self._llm_completion_tokens,
                "p95_duration_ms": _p95(self._llm_durations) * 1000 if self._llm_durations else 0,
            },
            "tool_calls": {
                name: dict(data) for name, data in sorted(self._tool_calls.items())
            },
        }


# Module-level singleton
stats_collector = StatsCollector()


# ── Backward-compatible bridge functions ──────────────────────────────


def record_agent_run(status: str = "", skill_name: str = "", duration_s: float = 0) -> None:
    """Backward-compatible bridge. Delegates to stats_collector."""
    stats_collector.record_agent_run(status=status, skill_name=skill_name, duration_s=duration_s)


def record_agent_task(agent_type: str = "", task_type: str = "", status: str = "", duration_s: float = 0) -> None:
    """Backward-compatible bridge. Records sub-task completion."""
    try:
        agent_tasks_total.labels(agent_type=agent_type, status=status).inc()
    except Exception:
        pass
