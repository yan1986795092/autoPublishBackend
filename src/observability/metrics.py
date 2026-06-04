"""Observability: Prometheus metrics definitions.

All custom business metrics are defined here and registered on the global meter.
Use `create_meters()` to initialize. Import `agent_meter`, `llm_meter`, `task_meter`
anywhere you need to record measurements.
"""

from __future__ import annotations

from typing import Optional

from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, Meter

# ── Meter handles (set by create_meters) ──────────────────────────────────────
agent_meter: Optional[Meter] = None
llm_meter: Optional[Meter] = None
task_meter: Optional[Meter] = None

# ── Agent metrics ─────────────────────────────────────────────────────────────
agent_run_counter: Optional[Counter] = None
agent_run_duration: Optional[Histogram] = None
agent_task_counter: Optional[Counter] = None
agent_task_duration: Optional[Histogram] = None
agent_think_duration: Optional[Histogram] = None
memory_read_duration: Optional[Histogram] = None
memory_write_duration: Optional[Histogram] = None
agent_cache_hit: Optional[Counter] = None
agent_cache_miss: Optional[Counter] = None

# ── Tool metrics ──────────────────────────────────────────────────────────────
tool_call_counter: Optional[Counter] = None
tool_call_duration: Optional[Histogram] = None


def _init_agent_metrics(meter: Meter) -> None:
    global agent_run_counter, agent_run_duration
    global agent_task_counter, agent_task_duration, agent_think_duration
    global memory_read_duration, memory_write_duration
    global agent_cache_hit, agent_cache_miss

    agent_run_counter = meter.create_counter(
        name="agent.run.total",
        description="Total number of agent orchestration runs",
        unit="1",
    )
    agent_run_duration = meter.create_histogram(
        name="agent.run.duration",
        description="Duration of agent orchestration runs in seconds",
        unit="s",
    )
    agent_task_counter = meter.create_counter(
        name="agent.task.total",
        description="Total number of agent tasks executed",
        unit="1",
    )
    agent_task_duration = meter.create_histogram(
        name="agent.task.duration",
        description="Duration of individual agent tasks in seconds",
        unit="s",
    )
    agent_think_duration = meter.create_histogram(
        name="agent.think.duration",
        description="Duration of agent think (LLM call + tool calls) in milliseconds",
        unit="ms",
    )
    memory_read_duration = meter.create_histogram(
        name="memory.read.duration",
        description="Duration of memory read operations in milliseconds",
        unit="ms",
    )
    memory_write_duration = meter.create_histogram(
        name="memory.write.duration",
        description="Duration of memory write operations in milliseconds",
        unit="ms",
    )
    agent_cache_hit = meter.create_counter(
        name="agent.cache.hit.total",
        description="Total number of agent cache hits",
        unit="1",
    )
    agent_cache_miss = meter.create_counter(
        name="agent.cache.miss.total",
        description="Total number of agent cache misses",
        unit="1",
    )


def _init_task_metrics(meter: Meter) -> None:
    global tool_call_counter, tool_call_duration

    tool_call_counter = meter.create_counter(
        name="agent.tool.call.total",
        description="Total number of tool calls",
        unit="1",
    )
    tool_call_duration = meter.create_histogram(
        name="agent.tool.call.duration",
        description="Duration of tool calls in milliseconds",
        unit="ms",
    )


def create_meters() -> tuple[Meter, Meter, Meter]:
    """Create and register all Prometheus metrics on three meters.

    Returns (agent_meter, llm_meter, task_meter).
    """
    global agent_meter, llm_meter, task_meter
    global agent_run_counter, agent_task_counter, agent_think_duration
    global memory_read_duration, memory_write_duration
    global agent_cache_hit, agent_cache_miss
    global tool_call_counter, tool_call_duration

    agent_meter = metrics.get_meter("agent", "1.0.0")
    llm_meter = metrics.get_meter("llm", "1.0.0")
    task_meter = metrics.get_meter("task", "1.0.0")

    _init_agent_metrics(agent_meter)
    _init_task_metrics(task_meter)

    return agent_meter, llm_meter, task_meter


# ── Convenience recording helpers ─────────────────────────────────────────────

def record_agent_run(status: str, skill_name: str, duration_s: float) -> None:
    if agent_run_counter:
        agent_run_counter.add(1, {"status": status, "skill": skill_name})
    if agent_run_duration:
        agent_run_duration.record(duration_s, {"skill": skill_name})


def record_agent_task(agent_type: str, task_type: str, status: str, duration_s: float) -> None:
    if agent_task_counter:
        agent_task_counter.add(1, {"agent_type": agent_type, "task_type": task_type, "status": status})
    if agent_task_duration:
        agent_task_duration.record(duration_s, {"agent_type": agent_type})


def record_agent_think(agent_type: str, llm_call_count: int, tool_call_count: int, duration_ms: float) -> None:
    if agent_think_duration:
        agent_think_duration.record(duration_ms, {"agent_type": agent_type})


def record_tool_call(tool_name: str, tool_category: str, status: str, duration_ms: float) -> None:
    if tool_call_counter:
        tool_call_counter.add(1, {"tool_name": tool_name, "category": tool_category, "status": status})
    if tool_call_duration:
        tool_call_duration.record(duration_ms, {"tool_name": tool_name})


def record_memory_operation(layer: str, operation: str, count: int, duration_ms: float) -> None:
    if layer == "read" and memory_read_duration:
        memory_read_duration.record(duration_ms, {"layer": layer})
    elif layer == "write" and memory_write_duration:
        memory_write_duration.record(duration_ms, {"layer": layer})


def record_cache_hit() -> None:
    if agent_cache_hit:
        agent_cache_hit.add(1)


def record_cache_miss() -> None:
    if agent_cache_miss:
        agent_cache_miss.add(1)
