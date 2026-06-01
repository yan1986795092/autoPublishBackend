"""Observability: Prometheus metrics definitions.

All custom business metrics are defined here and registered on the global meter.
Use `create_meters()` to initialize. Import `agent_meter`, `gateway_meter`, `task_meter`
anywhere you need to record measurements.
"""

from __future__ import annotations

from typing import Any, Optional

from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, ObservableGauge, Meter

# ── Meter handles (set by create_meters) ──────────────────────────────────────
agent_meter: Optional[Meter] = None
gateway_meter: Optional[Meter] = None
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

# ── Gateway metrics ────────────────────────────────────────────────────────────
gateway_request_counter: Optional[Counter] = None
gateway_request_duration: Optional[Histogram] = None
gateway_tokens_total: Optional[Counter] = None
gateway_cost_total: Optional[Counter] = None
gateway_provider_errors: Optional[Counter] = None
gateway_circuit_breaker_state: Optional[ObservableGauge] = None
gateway_rate_limit_wait: Optional[Histogram] = None

# ── Tool metrics ──────────────────────────────────────────────────────────────
tool_call_counter: Optional[Counter] = None
tool_call_duration: Optional[Histogram] = None

# ── Circuit breaker reference for ObservableGauge callback ────────────────────
_circuit_breaker_ref: Any = None


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


def _init_gateway_metrics(meter: Meter) -> None:
    global gateway_request_counter, gateway_request_duration
    global gateway_tokens_total, gateway_cost_total
    global gateway_provider_errors, gateway_circuit_breaker_state, gateway_rate_limit_wait

    gateway_request_counter = meter.create_counter(
        name="gateway.requests.total",
        description="Total number of gateway LLM requests",
        unit="1",
    )
    gateway_request_duration = meter.create_histogram(
        name="gateway.request.duration",
        description="Duration of gateway LLM requests in seconds",
        unit="s",
    )
    gateway_tokens_total = meter.create_counter(
        name="gateway.tokens.total",
        description="Total number of tokens processed",
        unit="1",
    )
    gateway_cost_total = meter.create_counter(
        name="gateway.cost.total",
        description="Total estimated LLM cost in USD",
        unit="USD",
    )
    gateway_provider_errors = meter.create_counter(
        name="gateway.provider.errors.total",
        description="Total number of LLM provider errors",
        unit="1",
    )
    gateway_circuit_breaker_state = meter.create_observable_gauge(
        name="gateway.circuit.breaker.state",
        description="Current circuit breaker state (0=closed, 1=open, 2=half_open)",
        unit="1",
        callbacks=[_read_circuit_breaker_state],
    )
    gateway_rate_limit_wait = meter.create_histogram(
        name="gateway.rate_limit.wait",
        description="Time spent waiting for rate limit in seconds",
        unit="s",
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


def set_circuit_breaker(cb) -> None:
    """Register a CircuitBreaker instance for the ObservableGauge callback.

    Called during GatewayContainer startup (after CircuitBreaker is created).
    The callback reads state on each OTel collection interval.
    """
    global _circuit_breaker_ref
    _circuit_breaker_ref = cb


def _read_circuit_breaker_state(options) -> list:
    """OTel ObservableGauge callback — reads current circuit breaker state."""
    if _circuit_breaker_ref is None:
        return []
    results = []
    try:
        for model_key in getattr(_circuit_breaker_ref, "_snapshots", {}):
            state = _circuit_breaker_ref.get_state(model_key)
            state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state, 0)
            results.append(metrics.Measurement(state_value, {"model": model_key}))
    except Exception:
        pass  # best-effort: don't break the metrics pipeline
    return results


def create_meters() -> tuple[Meter, Meter, Meter]:
    """Create and register all Prometheus metrics on three meters.

    Returns (agent_meter, gateway_meter, task_meter).
    """
    global agent_meter, gateway_meter, task_meter
    global agent_run_counter, agent_task_counter, agent_think_duration
    global memory_read_duration, memory_write_duration
    global agent_cache_hit, agent_cache_miss
    global gateway_request_counter, gateway_request_duration
    global gateway_tokens_total, gateway_cost_total, gateway_provider_errors, gateway_rate_limit_wait
    global tool_call_counter, tool_call_duration

    agent_meter = metrics.get_meter("agent", "1.0.0")
    gateway_meter = metrics.get_meter("gateway", "1.0.0")
    task_meter = metrics.get_meter("task", "1.0.0")

    _init_agent_metrics(agent_meter)
    _init_gateway_metrics(gateway_meter)
    _init_task_metrics(task_meter)

    return agent_meter, gateway_meter, task_meter


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


def record_gateway_request(
    source_app: str,
    provider: str,
    model: str,
    status: str,
    status_code: int,
    duration_s: float,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
) -> None:
    attrs = {"source_app": source_app, "provider": provider, "model": model, "status": status}
    if gateway_request_counter:
        gateway_request_counter.add(1, attrs)
    if gateway_request_duration:
        gateway_request_duration.record(duration_s, attrs)
    if gateway_tokens_total:
        gateway_tokens_total.add(prompt_tokens, {**attrs, "type": "prompt"})
        gateway_tokens_total.add(completion_tokens, {**attrs, "type": "completion"})
    if gateway_cost_total:
        gateway_cost_total.add(cost_usd, attrs)


def record_gateway_error(provider: str, model: str, error_type: str) -> None:
    attrs = {"provider": provider, "model": model, "error_type": error_type}
    if gateway_provider_errors:
        gateway_provider_errors.add(1, attrs)


def record_cache_hit() -> None:
    """Increment the cache hit counter."""
    if agent_cache_hit:
        agent_cache_hit.add(1)


def record_cache_miss() -> None:
    """Increment the cache miss counter."""
    if agent_cache_miss:
        agent_cache_miss.add(1)
