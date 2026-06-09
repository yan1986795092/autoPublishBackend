"""Observability API: query traces and logs from Elasticsearch + Prometheus metrics.

All queries go to Elasticsearch (traces, spans, aggregated stats).
Prometheus is used for real-time metrics.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import httpx

from .es_client import (
    search_spans,
    get_trace_by_id,
    get_trace_spans,
    search_logs,
    get_trace_stats,
    search_agent_events,
    get_es,
)

router = APIRouter(prefix="/api/v1/observability", tags=["observability"])

PROMETHEUS_HOST = os.getenv("PROMETHEUS_HOST", "http://localhost:9090")
JAEGER_UI = os.getenv("JAEGER_UI", "http://localhost:16686")


# ── Request / Response Models ──────────────────────────────────────────────────

class SpanRow(BaseModel):
    span_id: str
    trace_id: str
    parent_span_id: str
    operation_name: str
    service_name: str
    start_time: str
    duration_ms: float
    is_error: bool
    kind: str
    tags: dict[str, Any]
    logs: list[dict]
    score: float


class TraceStatsResponse(BaseModel):
    total_traces: int
    total_spans: int
    error_count: int
    error_rate: float
    by_service: list[dict]
    by_operation: list[dict]
    p50_duration_ms: Optional[float] = None
    p95_duration_ms: Optional[float] = None
    p99_duration_ms: Optional[float] = None


class TraceSearchResponse(BaseModel):
    total: int
    page: int
    page_size: int
    spans: list[SpanRow]


class TraceDetailResponse(BaseModel):
    trace_id: str
    total_duration_ms: float
    start_time: str
    end_time: str
    span_count: int
    error_count: int
    spans: list[SpanRow]


class AgentEventRow(BaseModel):
    timestamp: str
    session_id: str = ""
    run_id: str = ""
    step_index: int = 0
    parent_span_id: str = ""
    agent_name: str = ""
    event_type: str = ""
    level: str = "INFO"
    message: str = ""
    thought: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict[str, Any]] = None
    tool_output: Optional[str] = None
    duration_ms: Optional[float] = None
    token_usage: Optional[dict[str, Any]] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_stack: Optional[str] = None
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None
    service: str = ""
    trace_id: str = ""
    index: str = ""
    score: float = 1.0


class AgentEventSearchResponse(BaseModel):
    total: int
    page: int
    page_size: int
    events: list[AgentEventRow]


class AgentEventDiagnosticsResponse(BaseModel):
    total: int
    latest_event: Optional[AgentEventRow] = None
    services: list[dict[str, Any]]
    agent_names: list[dict[str, Any]]
    event_types: list[dict[str, Any]]


# ── Time utilities ───────────────────────────────────────────────────────────────

def _parse_time(time_str: str | None, default_delta_hours: int = 1) -> int:
    """Parse time string to Unix ms. Supports: '1h', '24h', '7d', '2024-01-01', ISO8601."""
    if not time_str:
        delta = timedelta(hours=default_delta_hours)
        return int((datetime.now(tz=timezone.utc) - delta).timestamp() * 1000)

    # Relative: "1h", "24h", "7d"
    unit_map = {"h": 3600, "m": 60, "d": 86400, "w": 604800}
    for unit, seconds in unit_map.items():
        if time_str.endswith(unit):
            try:
                value = int(time_str[:-1])
                return int((datetime.now(tz=timezone.utc) - timedelta(seconds=value * seconds)).timestamp() * 1000)
            except ValueError:
                pass

    # ISO / date string
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except ValueError:
        pass

    # Default
    delta = timedelta(hours=default_delta_hours)
    return int((datetime.now(tz=timezone.utc) - delta).timestamp() * 1000)


# ── Trace Search ───────────────────────────────────────────────────────────────

@router.get("/traces", response_model=TraceSearchResponse)
async def search_traces(
    service: str = Query("", description="Service name filter"),
    operation: str = Query("", description="Operation/span name filter"),
    trace_id: str = Query("", description="Trace ID filter"),
    keyword: str = Query("", description="Full-text keyword search in tags/logs"),
    span_type: str = Query("", description="Span type filter (e.g., agent_think, tool_call, model_request)"),
    status: str = Query("all", description="Status filter: error | ok | all"),
    time: str = Query("1h", description="Time range: 1h, 24h, 7d, or ISO8601"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """Search traces/spans from Elasticsearch.

    Examples:
    - GET /api/v1/observability/traces?service=autopublish-api
    - GET /api/v1/observability/traces?status=error&time=24h
    - GET /api/v1/observability/traces?keyword=memory_search&span_type=tool_call
    """
    start_ms = _parse_time(time)
    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    from_ = (page - 1) * page_size

    result = await search_spans(
        service=service or None,
        operation=operation or None,
        trace_id=trace_id or None,
        keyword=keyword or None,
        span_type=span_type or None,
        status=status if status in ("error", "ok") else None,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        size=page_size,
        from_=from_,
    )

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    spans = [
        SpanRow(
            span_id=s["span_id"],
            trace_id=s["trace_id"],
            parent_span_id=s.get("parent_span_id", ""),
            operation_name=s["operation_name"],
            service_name=s.get("service_name", ""),
            start_time=s["start_time"],
            duration_ms=s["duration_ms"],
            is_error=s.get("is_error", False),
            kind=s.get("kind", "INTERNAL"),
            tags=s.get("tags", {}),
            logs=s.get("logs", []),
            score=s.get("score", 1.0),
        )
        for s in result["hits"]
    ]

    return TraceSearchResponse(
        total=result["total"],
        page=page,
        page_size=page_size,
        spans=spans,
    )


@router.get("/traces/{trace_id}", response_model=TraceDetailResponse)
async def get_trace(trace_id: str):
    """Get a complete trace with all spans organized as a tree."""
    trace = await get_trace_by_id(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    spans = [
        SpanRow(
            span_id=s["span_id"],
            trace_id=s["trace_id"],
            parent_span_id=s.get("parent_span_id", ""),
            operation_name=s["operation_name"],
            service_name=s.get("service_name", ""),
            start_time=s["start_time"],
            duration_ms=s["duration_ms"],
            is_error=s.get("is_error", False),
            kind=s.get("kind", "INTERNAL"),
            tags=s.get("tags", {}),
            logs=s.get("logs", []),
            score=s.get("score", 1.0),
        )
        for s in trace["spans"]
    ]

    return TraceDetailResponse(
        trace_id=trace_id,
        total_duration_ms=trace["total_duration_ms"],
        start_time=trace["start_time"],
        end_time=trace["end_time"],
        span_count=trace["span_count"],
        error_count=trace["error_count"],
        spans=spans,
    )


@router.get("/traces/{trace_id}/spans/{span_id}")
async def get_span_detail(trace_id: str, span_id: str):
    """Get full details of a specific span."""
    spans = await get_trace_spans(trace_id)
    for s in spans:
        if s["span_id"] == span_id:
            return {
                **s,
                "trace_url": f"{JAEGER_UI}/trace/{trace_id}",
            }
    raise HTTPException(status_code=404, detail="Span not found")


# ── Aggregated Stats ────────────────────────────────────────────────────────────

@router.get("/stats", response_model=TraceStatsResponse)
async def get_stats_route(
    service: str = Query("", description="Service name filter"),
    time: str = Query("1h", description="Time range: 1h, 24h, 7d, or ISO8601"),
):
    """Get aggregated trace statistics (counts, error rates, latency percentiles)."""
    start_ms = _parse_time(time)
    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    stats = await get_trace_stats(
        service=service or None,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
    )

    if "error" in stats:
        raise HTTPException(status_code=503, detail=stats["error"])

    # 确保所有字段都有默认值，防止 ES 不可用时 Pydantic 校验失败
    return TraceStatsResponse(
        total_traces=stats.get("total_traces", 0),
        total_spans=stats.get("total_spans", 0),
        error_count=stats.get("error_count", 0),
        error_rate=stats.get("error_rate", 0),
        by_service=stats.get("by_service", []),
        by_operation=stats.get("by_operation", []),
        p50_duration_ms=stats.get("p50_duration_ms"),
        p95_duration_ms=stats.get("p95_duration_ms"),
        p99_duration_ms=stats.get("p99_duration_ms"),
    )


@router.get("/agent-events", response_model=AgentEventSearchResponse)
async def get_agent_events(
    service: str = Query("", description="Service name filter"),
    session_id: str = Query("", description="Session ID filter"),
    run_id: str = Query("", description="Run ID filter"),
    agent_name: str = Query("", description="Agent name filter"),
    event_type: str = Query("", description="Event type filter"),
    keyword: str = Query("", description="Keyword search in message/tool/model/error fields"),
    level: str = Query("", description="Log level: DEBUG, INFO, WARNING, ERROR"),
    time: str = Query("1h", description="Time range: 1h, 24h, 7d, or ISO8601"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    start_ms = _parse_time(time)
    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    from_ = (page - 1) * page_size

    result = await search_agent_events(
        service=service or None,
        session_id=session_id or None,
        run_id=run_id or None,
        agent_name=agent_name or None,
        event_type=event_type or None,
        keyword=keyword or None,
        level=level or None,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        size=page_size,
        from_=from_,
    )

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    events = [AgentEventRow(**item) for item in result["hits"]]
    return AgentEventSearchResponse(
        total=result["total"],
        page=page,
        page_size=page_size,
        events=events,
    )


@router.get("/agent-events/diagnostics", response_model=AgentEventDiagnosticsResponse)
async def get_agent_events_diagnostics():
    es = await get_es()
    if not es:
        raise HTTPException(status_code=503, detail="Elasticsearch not available")

    try:
        resp = await es.search(
            index="agent-events-*",
            query={"match_all": {}},
            sort=[
                {"@timestamp": {"order": "desc", "unmapped_type": "date", "missing": "_last"}},
                {"timestamp": {"order": "desc", "unmapped_type": "date", "missing": "_last"}},
            ],
            size=1,
            track_total_hits=True,
            aggs={
                "services": {"terms": {"field": "service", "size": 10}},
                "agent_names": {"terms": {"field": "agent_name", "size": 10}},
                "event_types": {"terms": {"field": "event_type", "size": 10}},
            },
            ignore_unavailable=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Diagnostics query failed: {exc}")

    hits = (resp.get("hits", {}) or {}).get("hits", [])
    latest_event = AgentEventRow(**hits[0]["_source"], index=hits[0].get("_index", ""), score=hits[0].get("_score") if hits[0].get("_score") is not None else 1.0) if hits else None
    total = (resp.get("hits", {}) or {}).get("total", {}).get("value", 0)
    aggs = resp.get("aggregations", {}) or {}

    return AgentEventDiagnosticsResponse(
        total=total,
        latest_event=latest_event,
        services=[{"key": b.get("key", ""), "count": b.get("doc_count", 0)} for b in (aggs.get("services", {}) or {}).get("buckets", [])],
        agent_names=[{"key": b.get("key", ""), "count": b.get("doc_count", 0)} for b in (aggs.get("agent_names", {}) or {}).get("buckets", [])],
        event_types=[{"key": b.get("key", ""), "count": b.get("doc_count", 0)} for b in (aggs.get("event_types", {}) or {}).get("buckets", [])],
    )


# ── Structured Log Search ───────────────────────────────────────────────────────

@router.get("/logs")
async def search_structured_logs(
    keyword: str = Query(..., description="Search keyword (required)"),
    service: str = Query("", description="Service filter"),
    level: str = Query("", description="Log level: DEBUG, INFO, WARNING, ERROR"),
    time: str = Query("1h", description="Time range"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """Search structured logs from Elasticsearch.

    Searches both dedicated "logs-*" index and span log events.
    """
    start_ms = _parse_time(time)
    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    from_ = (page - 1) * page_size

    result = await search_logs(
        keyword=keyword,
        service=service or None,
        level=level or None,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        size=page_size,
        from_=from_,
    )

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return {
        "total": result["total"],
        "page": page,
        "page_size": page_size,
        "hits": result["hits"],
    }


# ── Prometheus Metrics ─────────────────────────────────────────────────────────

@router.get("/metrics")
async def query_metrics(
    query: str = Query(..., description="PromQL query"),
):
    """Execute a PromQL query against Prometheus.

    Examples:
    - ?query=sum(rate(agent_run_total[5m])) by (status)
    - ?query=histogram_quantile(0.95, sum(rate(agent_task_duration_seconds_bucket[5m])) by (le))
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                f"{PROMETHEUS_HOST}/api/v1/query",
                params={"query": query},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Prometheus unavailable: {e}")


@router.get("/metrics/range")
async def query_metrics_range(
    query: str = Query(..., description="PromQL range query"),
    start: str = Query(..., description="Start time (Unix s or ISO8601)"),
    end: str = Query(..., description="End time (Unix s or ISO8601)"),
    step: str = Query("15s", description="Query resolution step (e.g., 15s, 1m, 5m)"),
):
    """Execute a PromQL range query."""
    start_s = _parse_time(start) // 1000
    end_s = _parse_time(end) // 1000

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                f"{PROMETHEUS_HOST}/api/v1/query_range",
                params={"query": query, "start": start_s, "end": end_s, "step": step},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Prometheus unavailable: {e}")


@router.get("/metrics/series")
async def query_metrics_series(
    match: str = Query(..., description="Series selector (e.g., {__name__=~\"agent_.*\"})"),
    start: str = Query("", description="Start time (Unix s or ISO8601)"),
    end: str = Query("", description="End time (Unix s or ISO8601)"),
):
    """List all series matching a selector."""
    params = {"match[]": match}
    if start:
        params["start"] = _parse_time(start) // 1000
    if end:
        params["end"] = _parse_time(end) // 1000

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                f"{PROMETHEUS_HOST}/api/v1/series",
                params=params,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Prometheus unavailable: {e}")


# ── Quick Dashboard Data ────────────────────────────────────────────────────────

@router.get("/dashboard/overview")
async def dashboard_overview():
    """Get all data needed for the observability overview page in one call."""
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    hour_ago = now_ms - 3600 * 1000
    day_ago = now_ms - 86400 * 1000

    stats_hour = await get_trace_stats(start_time_ms=hour_ago, end_time_ms=now_ms)
    stats_day = await get_trace_stats(start_time_ms=day_ago, end_time_ms=now_ms)

    # 移除可能的 error 字段，避免前端收到脏数据
    if "error" in stats_hour:
        stats_hour = {k: v for k, v in stats_hour.items() if k != "error"}
    if "error" in stats_day:
        stats_day = {k: v for k, v in stats_day.items() if k != "error"}

    # Key metric queries via Prometheus
    async with httpx.AsyncClient(timeout=10.0) as client:
        metric_queries = [
            ("success_rate_1h", '1 - sum(rate(agent_run_total{status="failed"}[1h])) / sum(rate(agent_run_total[1h]))'),
            ("runs_1h", "sum(increase(agent_run_total[1h]))"),
            ("tasks_1h", "sum(increase(agent_task_total[1h]))"),
            ("errors_1h", "sum(increase(agent_run_total{status=\"failed\"}[1h]))"),
            ("p95_latency_1h", "histogram_quantile(0.95, sum(rate(agent_run_duration_seconds_bucket[1h])) by (le)) * 1000"),
        ]

        metrics = {}
        for name, promql in metric_queries:
            try:
                resp = await client.get(
                    f"{PROMETHEUS_HOST}/api/v1/query",
                    params={"query": promql},
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", {})
                    results = data.get("result", [])
                    if results:
                        value = results[0]["value"][1]
                        metrics[name] = round(float(value), 4) if value != "NaN" else None
                    else:
                        metrics[name] = None
                else:
                    metrics[name] = None
            except Exception:
                metrics[name] = None

    return {
        "trace_stats_hour": stats_hour,
        "trace_stats_day": stats_day,
        "metrics": metrics,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
