"""Agent 指标 —— Prometheus 计数器、直方图和内存 StatsCollector。

用法::

    from src.observability.metrics import stats_collector

    # 记录事件（双写：Prometheus + 内存）
    stats_collector.record_agent_run(status="success", skill_name="writing", duration_s=1.5)
    stats_collector.record_llm_call(model="gpt-4o", provider="openai", status="success",
                                    prompt_tokens=100, completion_tokens=50, duration_s=2.0)
    stats_collector.record_tool_call(tool_name="web_search", status="success", duration_s=0.8)

    # 获取 metrics API 的快照
    snapshot = stats_collector.get_stats()
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from prometheus_client import Counter, Histogram, Gauge


# ── Agent 级别计数器 ────────────────────────────────────────────────────

agent_runs_total = Counter(
    "agent_runs_total",
    "Agent 运行总次数，按状态和技能统计",
    labelnames=["status", "skill_name"],
)

agent_run_duration_seconds = Histogram(
    "agent_run_duration_seconds",
    "Agent 运行持续时间（秒）",
    labelnames=["skill_name"],
    buckets=(1, 2.5, 5, 10, 15, 20, 30, 45, 60, 90, 120, float("inf")),
)

agent_tasks_total = Counter(
    "agent_tasks_total",
    "Agent 子任务总次数，按 agent 类型和状态统计",
    labelnames=["agent_type", "status"],
)

# ── LLM 计数器 ─────────────────────────────────────────────────────────

llm_tokens_total = Counter(
    "llm_tokens_total",
    "LLM 消耗的总 token 数，按模型、提供者和 token 类型统计",
    labelnames=["model", "provider", "type"],  # type: prompt | completion
)

llm_requests_total = Counter(
    "llm_requests_total",
    "LLM 请求总次数，按模型和提供者统计",
    labelnames=["model", "provider", "status"],  # status: success | error
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM 请求持续时间（秒）",
    labelnames=["model", "provider"],
    buckets=(0.1, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, float("inf")),
)

# ── 工具调用计数器 ──────────────────────────────────────────────────────

tool_calls_total = Counter(
    "tool_calls_total",
    "工具调用总次数，按工具名称和状态统计",
    labelnames=["tool_name", "status"],  # status: success | error
)

tool_call_duration_seconds = Histogram(
    "tool_call_duration_seconds",
    "工具调用持续时间（秒）",
    labelnames=["tool_name"],
    buckets=(0.1, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, float("inf")),
)

# ── 缓存计数器 ──────────────────────────────────────────────────────────

cache_hits_total = Counter(
    "cache_hits_total",
    "缓存命中总次数",
    labelnames=["cache_name"],
)

cache_misses_total = Counter(
    "cache_misses_total",
    "缓存未命中总次数",
    labelnames=["cache_name"],
)

# ── 系统仪表指标 ────────────────────────────────────────────────────────

up_gauge = Gauge(
    "autopublish_up",
    "表示 autopublish-api 服务是否运行（1）或停止（0）",
)

info_gauge = Gauge(
    "autopublish_info",
    "包含单个 'version' 标签的指标，用于服务信息",
    labelnames=["version"],
)


# ── StatsCollector：内存聚合 + Prometheus 桥接 ─────────────────────────


class StatsCollector:
    """轻量级内存统计收集器，同时写入 Prometheus。

    累积计数器的用途：
    1. 通过 ``/metrics`` 暴露为 Prometheus 指标
    2. 可通过 ``get_stats()`` 查询，用于 ``/agent/metrics`` API
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

    # ── Agent 运行 ──────────────────────────────────────────────────

    def record_agent_run(self, status: str = "", skill_name: str = "", duration_s: float = 0) -> None:
        """记录一次 agent 运行完成。"""
        self._agent_runs["total"] += 1
        self._agent_runs[status] += 1
        if duration_s > 0:
            self._agent_run_durations.append(duration_s)
            self._agent_run_durations = self._agent_run_durations[-1000:]  # 滑动窗口
        try:
            agent_runs_total.labels(status=status, skill_name=skill_name).inc()
            if duration_s > 0:
                agent_run_duration_seconds.labels(skill_name=skill_name).observe(duration_s)
        except Exception:
            pass

    # ── LLM 调用 ────────────────────────────────────────────────────

    def record_llm_call(
        self,
        model: str = "",
        provider: str = "",
        status: str = "success",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        duration_s: float = 0,
    ) -> None:
        """记录一次 LLM 调用。"""
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

    # ── 工具调用 ────────────────────────────────────────────────────

    def record_tool_call(self, tool_name: str = "", status: str = "success", duration_s: float = 0) -> None:
        """记录一次工具调用。"""
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

    # ── 统计快照 ────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """返回所有累积统计信息的快照，用于 metrics API。"""
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


# 模块级单例
stats_collector = StatsCollector()


# ── 向后兼容桥接函数 ──────────────────────────────────────────────────


def record_agent_run(status: str = "", skill_name: str = "", duration_s: float = 0) -> None:
    """向后兼容桥接。委托给 stats_collector。"""
    stats_collector.record_agent_run(status=status, skill_name=skill_name, duration_s=duration_s)


def record_agent_task(agent_type: str = "", task_type: str = "", status: str = "", duration_s: float = 0) -> None:
    """向后兼容桥接。记录子任务完成。"""
    try:
        agent_tasks_total.labels(agent_type=agent_type, status=status).inc()
    except Exception:
        pass
