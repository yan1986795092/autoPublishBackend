"""Elasticsearch client for querying traces, structured logs, and agent events."""

from __future__ import annotations

import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_es_client: Optional[Any] = None  # AsyncElasticsearch (lazy import)


def get_es_url() -> str:
    return os.getenv(
        "ELASTICSEARCH_URL",
        os.getenv("OTEL_ELASTICSEARCH_URL", "http://localhost:9200"),
    )


async def get_es() -> Optional[Any]:
    """Get or create the Elasticsearch client."""
    global _es_client
    if _es_client is None:
        from elasticsearch import AsyncElasticsearch  # lazy import
        url = get_es_url()
        try:
            _es_client = AsyncElasticsearch(
                hosts=[url],
                request_timeout=30,
                retry_on_timeout=True,
                max_retries=3,
            )
            info = await _es_client.info()
            logger.info("Connected to Elasticsearch %s", info["version"]["number"])
        except Exception as e:
            logger.warning("Failed to connect to Elasticsearch at %s: %s", url, e)
            return None
    return _es_client


async def close_es() -> None:
    global _es_client
    if _es_client:
        await _es_client.close()
        _es_client = None


# ── Index patterns ────────────────────────────────────────────────────────────
_TRACE_INDEX = "traces-{date}"
_TRACE_INDEX_PATTERN = "traces-*"
_AGENT_EVENTS_INDEX_PATTERN = "agent-events-*"

# Trace field mappings from Jaeger ES schema
_TRACE_ID = "traceID"
_SPAN_ID = "spanID"
_PARENT_SPAN_ID = "parentSpanID"
_TRACE_ID_HR = "traceId"
_SPAN_ID_HR = "spanId"
_PARENT_SPAN_ID_HR = "parentSpanId"

_OPERATION_NAME = "operationName"
_START_TIME_MS = "startTimeMillis"
_START_TIME = "startTime"
_DURATION_MS = "durationMicros"   # microseconds in Jaeger ES schema
_DURATION_MS_HR = "duration"      # milliseconds in OTel format
_SERVICE_NAME = "process.serviceName"
_SERVICE_NAME_ROOT = "serviceName"
_SERVICE_NAME_DOTTED = "service.name"
_PROCESS = "process"
_TAGS = "tags"
_LOGS = "logs"
_REFERENCES = "references"
_KIND = "kind"
_STATUS_CODE = "status.code"
_STATUS_MSG = "status.message"
_ERROR = "error"


def _keyword_field(field: str) -> str:
    """Return the keyword subfield commonly used for exact-match aggregations."""
    return f"{field}.keyword"


def _extract_tag_value(tag: dict) -> Any:
    """Extract Jaeger/OTel tag values across possible serialized shapes."""
    if not isinstance(tag, dict):
        return None
    for key in ("value", "vStr", "vInt64", "vInt", "vFloat64", "vFloat", "vBool", "vBinary"):
        if key in tag and tag.get(key) is not None:
            return tag.get(key)
    return None


def _normalize_tags(tags_list: Any) -> dict[str, Any]:
    """Flatten Jaeger tag arrays into a simple key/value dict."""
    if not isinstance(tags_list, list):
        return {}
    return {
        str(tag.get("key", "")): _extract_tag_value(tag)
        for tag in tags_list
        if isinstance(tag, dict) and tag.get("key")
    }


def _extract_service_name(src: dict, tags: dict[str, Any]) -> str:
    """Resolve service name across Jaeger and OTel span document layouts."""
    process = src.get(_PROCESS, {})
    if isinstance(process, dict):
        service = (
            process.get("serviceName")
            or process.get(_SERVICE_NAME)
            or process.get(_SERVICE_NAME_ROOT)
            or process.get(_SERVICE_NAME_DOTTED)
        )
        if service:
            return str(service)

    for key in (_SERVICE_NAME, _SERVICE_NAME_ROOT, _SERVICE_NAME_DOTTED):
        value = tags.get(key) or src.get(key)
        if value:
            return str(value)

    return ""


def _service_filter_clause(service: str) -> dict:
    """Match service name across trace schemas used by Jaeger/OTel exports."""
    return {
        "bool": {
            "should": [
                {"term": {_keyword_field(_SERVICE_NAME): service}},
                {"term": {_keyword_field(_SERVICE_NAME_ROOT): service}},
                {"term": {_keyword_field(_SERVICE_NAME_DOTTED): service}},
                {"term": {_SERVICE_NAME: service}},
                {"term": {_SERVICE_NAME_ROOT: service}},
                {"term": {_SERVICE_NAME_DOTTED: service}},
            ],
            "minimum_should_match": 1,
        }
    }


def _safe_trace_sort(order: str) -> list[dict]:
    """Sort traces without failing on mixed index mappings."""
    return [
        {
            _START_TIME_MS: {
                "order": order,
                "unmapped_type": "long",
                "missing": "_last" if order == "desc" else "_first",
            }
        },
        {
            _START_TIME: {
                "order": order,
                "unmapped_type": "long",
                "missing": "_last" if order == "desc" else "_first",
            }
        },
        {"@timestamp": {"order": order, "unmapped_type": "date"}},
    ]

_TAG_KEYS = [
    "agent.system", "agent.node", "agent.run.id", "agent.skill.name",
    "agent.user_request.preview", "agent.complexity", "agent.status",
    "agent.task.id", "task.agent_type", "batch.size", "custom.run_id",
    "custom.task_id", "gen_ai.system", "gen_ai.request.model",
    "gen_ai.usage.prompt_tokens", "gen_ai.usage.completion_tokens",
    "gen_ai.response.latency_ms", "llm.model", "llm.status",
    "tool.name", "tool.category", "tool.status",
]


def _normalize_span(hit: dict) -> dict:
    """Normalize Jaeger ES span document to a consistent format."""
    src = hit["_source"]
    fields = src.get("fields", {})

    # Jaeger stores spanID/traceID in "fields" and "tag" format
    span_id = (
        fields.get(_SPAN_ID, [None])[0]
        if isinstance(fields.get(_SPAN_ID), list) else fields.get(_SPAN_ID)
    ) or src.get(_SPAN_ID)

    trace_id = (
        fields.get(_TRACE_ID, [None])[0]
        if isinstance(fields.get(_TRACE_ID), list) else fields.get(_TRACE_ID)
    ) or src.get(_TRACE_ID)

    # Start time (Jaeger stores microseconds)
    start_us = src.get(_START_TIME_MS) or int(src.get(_START_TIME, 0))
    start_ms = start_us // 1000 if start_us > 1e15 else start_us

    # Duration (Jaeger stores microseconds, normalize to ms)
    dur_us = src.get(_DURATION_MS, 0)
    dur_ms = dur_us / 1000 if dur_us else 0

    # Tags → flat dict
    tags_list = src.get(_TAGS, [])
    tags = _normalize_tags(tags_list)

    # Service name
    process = src.get(_PROCESS, {})
    service = _extract_service_name(src, tags)

    # Status
    status_obj = src.get("status", {})
    status_code = status_obj.get("code", 0)
    is_error = status_code == 2 or tags.get(_ERROR) is not None

    # Log events
    logs = src.get(_LOGS, [])

    # References (parent-child)
    refs = src.get(_REFERENCES, [])

    return {
        "span_id": span_id or "",
        "trace_id": trace_id or "",
        "parent_span_id": _get_parent_span_id(src),
        "operation_name": src.get(_OPERATION_NAME, ""),
        "service_name": service,
        "start_time": _ms_to_iso(start_ms),
        "duration_ms": round(dur_ms, 2),
        "duration_us": dur_us,
        "kind": tags.get(_KIND, "INTERNAL"),
        "status_code": status_code,
        "is_error": is_error,
        "tags": tags,
        "logs": [
            {"timestamp": _ms_to_iso(l.get("timestamp", 0) // 1000), "fields": {f.get("key"): f.get("value") for f in l.get("fields", [])}}
            for l in logs
        ],
        "references": [
            {"ref_type": r.get("refType", ""), "span_id": r.get("spanID", ""), "trace_id": r.get("traceID", "")}
            for r in refs
        ],
        "process": {
            "service_name": service,
            "tags": {t.get("key"): t.get("value") for t in process.get("tags", [])},
        },
        "index": hit["_index"],
        "score": hit.get("_score") if hit.get("_score") is not None else 1.0,
    }


def _get_parent_span_id(src: dict) -> str:
    """Extract parent span ID from span references."""
    refs = src.get(_REFERENCES, [])
    for r in refs:
        if r.get("refType") == "CHILD_OF":
            return r.get("spanID", "")
    # Also check tags for parentSpanID
    for t in src.get(_TAGS, []):
        if t.get("key") == "parentSpanId":
            return str(t.get("value", ""))
    return ""


def _ms_to_iso(ms: int) -> str:
    from datetime import datetime, timezone
    if not ms or ms < 0:
        return ""
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
    except (ValueError, OSError):
        return ""


# ── Query builders ───────────────────────────────────────────────────────────

async def search_spans(
    service: str | None = None,
    operation: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    keyword: str | None = None,
    span_type: str | None = None,
    status: str | None = None,  # "error" | "ok" | "all"
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    size: int = 50,
    from_: int = 0,
) -> dict:
    """Search spans in Elasticsearch (Jaeger index).

    Args:
        service: Filter by service name
        operation: Filter by operation (span) name
        trace_id: Filter by trace ID
        span_id: Filter by span ID
        keyword: Full-text search in tag values
        span_type: Filter by span type (e.g., agent_think, tool_call, model_request)
        status: "error" | "ok" | "all"
        start_time_ms: Start of time range (Unix ms)
        end_time_ms: End of time range (Unix ms)
        size: Max results
        from_: Pagination offset

    Returns:
        {"total": int, "hits": [normalized_span, ...]}
    """
    es = await get_es()
    if not es:
        return {"total": 0, "hits": [], "error": "Elasticsearch not available"}

    must_clauses = []
    filter_clauses = []

    # Time range (required for performance)
    if start_time_ms or end_time_ms:
        range_query = {}
        if start_time_ms:
            range_query["gte"] = start_time_ms
        if end_time_ms:
            range_query["lte"] = end_time_ms
        filter_clauses.append({"range": {_START_TIME_MS: range_query}})

    if service:
        filter_clauses.append(_service_filter_clause(service))

    if operation:
        filter_clauses.append({"wildcard": {_OPERATION_NAME: f"*{operation}*"}})

    if trace_id:
        filter_clauses.append({"term": {_TRACE_ID: trace_id}})

    if span_id:
        filter_clauses.append({"term": {_SPAN_ID: span_id}})

    if span_type:
        filter_clauses.append({"wildcard": {_OPERATION_NAME: f"*{span_type}*"}})

    if status == "error":
        filter_clauses.append({"term": {"tag.status.code": 2}})
    elif status == "ok":
        filter_clauses.append({"term": {"tag.status.code": 1}})

    if keyword:
        must_clauses.append({
            "query_string": {
                "query": keyword,
                "fields": ["tag.*", "operationName", "process.serviceName", "logs.fields.*"],
                "default_operator": "AND",
            }
        })

    query = {
        "bool": {
            "filter": filter_clauses,
            "must": must_clauses if must_clauses else [{"match_all": {}}],
        }
    }

    try:
        resp = await es.search(
            index=_TRACE_INDEX_PATTERN,
            query=query,
            sort=_safe_trace_sort("desc"),
            from_=from_,
            size=size,
            track_total_hits=True,
            ignore_unavailable=True,
        )

        total = resp["hits"]["total"]["value"]
        hits = [_normalize_span(hit) for hit in resp["hits"]["hits"]]

        return {"total": total, "hits": hits}

    except Exception as e:
        logger.error("Elasticsearch search failed: %s", e)
        return {"total": 0, "hits": [], "error": str(e)}


async def get_trace_spans(trace_id: str) -> list[dict]:
    """Get all spans for a given trace ID."""
    es = await get_es()
    if not es:
        return []

    try:
        resp = await es.search(
            index=_TRACE_INDEX_PATTERN,
            query={"term": {_TRACE_ID: trace_id}},
            sort=_safe_trace_sort("asc"),
            size=1000,
            ignore_unavailable=True,
        )
        return [_normalize_span(hit) for hit in resp["hits"]["hits"]]

    except Exception as e:
        logger.error("Failed to get trace spans for %s: %s", trace_id, e)
        return []


async def get_trace_by_id(trace_id: str) -> dict | None:
    """Get a complete trace with all spans, organized."""
    spans = await get_trace_spans(trace_id)
    if not spans:
        return None

    # Build tree structure
    span_map = {s["span_id"]: s for s in spans}
    roots = []

    for span in spans:
        parent_id = span.get("parent_span_id", "")
        if parent_id and parent_id in span_map:
            parent = span_map[parent_id]
            if "children" not in parent:
                parent["children"] = []
            parent["children"].append(span)
        else:
            roots.append(span)

    # Compute relative timestamps
    if roots:
        min_start = min(_parse_ts(s["start_time"]) for s in spans if s["start_time"])
        for span in spans:
            span["relative_start_ms"] = max(0, _parse_ts(span["start_time"]) - min_start)

    # Trace-level stats
    start_times = [_parse_ts(s["start_time"]) for s in spans if s["start_time"]]
    end_times = [(_parse_ts(s["start_time"]) + s["duration_us"] / 1000) for s in spans if s["start_time"]]
    total_duration = max(end_times) - min(start_times) if start_times and end_times else 0

    return {
        "trace_id": trace_id,
        "spans": spans,
        "roots": roots,
        "span_map": span_map,
        "total_duration_ms": round(total_duration, 2),
        "start_time": _ms_to_iso(int(min(start_times))) if start_times else "",
        "end_time": _ms_to_iso(int(max(end_times))) if end_times else "",
        "span_count": len(spans),
        "error_count": sum(1 for s in spans if s.get("is_error")),
    }


async def search_logs(
    keyword: str,
    service: str | None = None,
    level: str | None = None,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    size: int = 100,
    from_: int = 0,
) -> dict:
    """Search structured logs from stdout/stderr forwarded via Filebeat/OTel.

    Note: Currently searches Jaeger spans (which contain log events).
    For full log search, route structlog JSON to a dedicated "logs-*" ES index.
    """
    es = await get_es()
    if not es:
        return {"total": 0, "hits": [], "error": "Elasticsearch not available"}

    must_clauses = []
    filter_clauses = []

    if keyword:
        must_clauses.append({
            "multi_match": {
                "query": keyword,
                "fields": ["message", "event", "logger", "level", "tags.*"],
                "type": "best_fields",
            }
        })

    if level:
        filter_clauses.append({"term": {"level": level.upper()}})

    if service:
        filter_clauses.append({"term": {"service.name": service}})

    if start_time_ms or end_time_ms:
        range_query = {}
        if start_time_ms:
            range_query["gte"] = start_time_ms
        if end_time_ms:
            range_query["lte"] = end_time_ms
        filter_clauses.append({"range": {"@timestamp": range_query}})

    query = {
        "bool": {
            "filter": filter_clauses,
            "must": must_clauses if must_clauses else [{"match_all": {}}],
        }
    }

    try:
        # Search in traces (which have log events) and a logs index
        resp = await es.search(
            index="logs-*,traces-*",
            query=query,
            sort=[{"@timestamp": "desc"}, {"startTimeMillis": "desc"}],
            from_=from_,
            size=size,
            track_total_hits=True,
            ignore_unavailable=True,
        )

        total = resp["hits"]["total"]["value"]
        hits = [
            {
                **(_normalize_span(hit) if "fields" in hit["_source"] else {"_source": hit["_source"]}),
                "index": hit["_index"],
                "score": hit.get("_score") if hit.get("_score") is not None else 1.0,
            }
            for hit in resp["hits"]["hits"]
        ]

        return {"total": total, "hits": hits}

    except Exception as e:
        logger.error("Elasticsearch logs search failed: %s", e)
        return {"total": 0, "hits": [], "error": str(e)}


def _parse_ts(ts: str) -> int:
    from datetime import datetime, timezone
    if not ts:
        return 0
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return int(datetime.fromisoformat(ts).timestamp() * 1000)
    except (ValueError, OSError):
        return 0


def _normalize_agent_event(hit: dict) -> dict:
    src = hit.get("_source", {})
    return {
        "timestamp": src.get("timestamp") or src.get("@timestamp", ""),
        "session_id": src.get("session_id", ""),
        "run_id": src.get("run_id", ""),
        "step_index": src.get("step_index", 0),
        "parent_span_id": src.get("parent_span_id", ""),
        "agent_name": src.get("agent_name", ""),
        "event_type": src.get("event_type", ""),
        "level": src.get("level", "INFO"),
        "message": src.get("message", ""),
        "thought": src.get("thought"),
        "tool_name": src.get("tool_name"),
        "tool_input": src.get("tool_input"),
        "tool_output": src.get("tool_output"),
        "duration_ms": src.get("duration_ms"),
        "token_usage": src.get("token_usage"),
        "error_type": src.get("error_type"),
        "error_message": src.get("error_message"),
        "error_stack": src.get("error_stack"),
        "llm_model": src.get("llm_model"),
        "llm_provider": src.get("llm_provider"),
        "service": src.get("service", ""),
        "trace_id": src.get("trace_id", ""),
        "index": hit.get("_index", ""),
        "score": hit.get("_score") if hit.get("_score") is not None else 1.0,
    }


def _agent_event_term_clause(field: str, value: str) -> dict:
    """Match agent-event exact fields with or without keyword subfields."""
    return {
        "bool": {
            "should": [
                {"term": {f"{field}.keyword": value}},
                {"term": {field: value}},
            ],
            "minimum_should_match": 1,
        }
    }


def _safe_agent_event_sort() -> list[dict]:
    """Sort agent events across layouts that may use timestamp or @timestamp."""
    return [
        {"@timestamp": {"order": "desc", "unmapped_type": "date", "missing": "_last"}},
        {"timestamp": {"order": "desc", "unmapped_type": "date", "missing": "_last"}},
    ]


def _agent_event_time_filter_clause(start_time_ms: int | None, end_time_ms: int | None) -> dict | None:
    """Match agent event time range across possible timestamp fields."""
    if not start_time_ms and not end_time_ms:
        return None

    range_query: dict[str, int] = {}
    if start_time_ms:
        range_query["gte"] = start_time_ms
    if end_time_ms:
        range_query["lte"] = end_time_ms

    return {
        "bool": {
            "should": [
                {"range": {"@timestamp": range_query}},
                {"range": {"timestamp": range_query}},
            ],
            "minimum_should_match": 1,
        }
    }


async def search_agent_events(
    service: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    agent_name: str | None = None,
    event_type: str | None = None,
    keyword: str | None = None,
    level: str | None = None,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    size: int = 50,
    from_: int = 0,
) -> dict:
    es = await get_es()
    if not es:
        return {"total": 0, "hits": [], "error": "Elasticsearch not available"}

    must_clauses = []
    filter_clauses = []

    time_filter = _agent_event_time_filter_clause(start_time_ms, end_time_ms)
    if time_filter:
        filter_clauses.append(time_filter)

    if service:
        filter_clauses.append(_agent_event_term_clause("service", service))
    if session_id:
        filter_clauses.append(_agent_event_term_clause("session_id", session_id))
    if run_id:
        filter_clauses.append(_agent_event_term_clause("run_id", run_id))
    if agent_name:
        filter_clauses.append(_agent_event_term_clause("agent_name", agent_name))
    if event_type:
        filter_clauses.append(_agent_event_term_clause("event_type", event_type))
    if level:
        filter_clauses.append(_agent_event_term_clause("level", level.upper()))

    if keyword:
        must_clauses.append({
            "query_string": {
                "query": keyword,
                "fields": [
                    "message",
                    "thought",
                    "tool_name",
                    "tool_output",
                    "error_message",
                    "llm_model",
                    "llm_provider",
                ],
                "default_operator": "AND",
            }
        })

    query = {
        "bool": {
            "must": must_clauses or [{"match_all": {}}],
            "filter": filter_clauses,
        }
    }

    try:
        resp = await es.search(
            index=_AGENT_EVENTS_INDEX_PATTERN,
            query=query,
            sort=_safe_agent_event_sort(),
            from_=from_,
            size=size,
            track_total_hits=True,
            ignore_unavailable=True,
        )
        hits = [_normalize_agent_event(h) for h in resp["hits"]["hits"]]
        total_obj = resp["hits"].get("total", 0)
        total = total_obj.get("value", 0) if isinstance(total_obj, dict) else int(total_obj)
        return {"total": total, "hits": hits}
    except Exception as e:
        logger.error("Elasticsearch agent event search failed: %s", e)
        return {"total": 0, "hits": [], "error": str(e)}


# ── Aggregation helpers ────────────────────────────────────────────────────────

async def get_trace_stats(
    service: str | None = None,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
) -> dict:
    """Get aggregated stats for traces."""
    es = await get_es()
    default_stats = {
        "total_traces": 0,
        "total_spans": 0,
        "error_count": 0,
        "error_rate": 0,
        "by_service": [],
        "by_operation": [],
        "p50_duration_ms": None,
        "p95_duration_ms": None,
        "p99_duration_ms": None,
    }
    if not es:
        return default_stats

    filter_clauses = []
    if service:
        filter_clauses.append(_service_filter_clause(service))
    if start_time_ms or end_time_ms:
        range_query = {}
        if start_time_ms:
            range_query["gte"] = start_time_ms
        if end_time_ms:
            range_query["lte"] = end_time_ms
        filter_clauses.append({"range": {_START_TIME_MS: range_query}})

    query = {"bool": {"filter": filter_clauses}} if filter_clauses else {"match_all": {}}

    aggs = {
        "error_spans": {
            "filter": {"term": {"tag.status.code": 2}}
        },
        "by_service": {
            "terms": {"field": _keyword_field(_SERVICE_NAME), "size": 20}
        },
        "by_operation": {
            "terms": {"field": _keyword_field(_OPERATION_NAME), "size": 20}
        },
        "p50_duration": {
            "percentiles": {"field": _DURATION_MS, "percents": [50, 95, 99]}
        },
    }

    try:
        resp = await es.search(
            index=_TRACE_INDEX_PATTERN,
            query=query,
            aggs=aggs,
            size=0,
            ignore_unavailable=True,
            track_total_hits=True,
        )

        aggs_result = resp.get("aggregations", {})
        total_spans = (resp.get("hits", {}) or {}).get("total", {}).get("value", 0)
        error_spans = (aggs_result.get("error_spans", {}) or {}).get("doc_count", 0)
        total_traces = total_spans

        by_service = [
            {"service": b["key"], "count": b["doc_count"]}
            for b in (aggs_result.get("by_service", {}) or {}).get("buckets", [])
        ]
        by_operation = [
            {"operation": b["key"], "count": b["doc_count"]}
            for b in (aggs_result.get("by_operation", {}) or {}).get("buckets", [])
        ]
        p50 = (aggs_result.get("p50_duration", {}) or {}).get("values", {}).get("50.0")
        p95 = (aggs_result.get("p50_duration", {}) or {}).get("values", {}).get("95.0")
        p99 = (aggs_result.get("p50_duration", {}) or {}).get("values", {}).get("99.0")

        return {
            "total_traces": total_traces,
            "total_spans": total_spans,
            "error_count": error_spans,
            "error_rate": round(error_spans / total_spans * 100, 2) if total_spans else 0,
            "by_service": by_service,
            "by_operation": by_operation,
            "p50_duration_ms": p50,
            "p95_duration_ms": p95,
            "p99_duration_ms": p99,
        }

    except Exception as e:
        logger.warning("Elasticsearch stats aggregation degraded, falling back to totals only: %s", e)
        try:
            resp = await es.search(
                index=_TRACE_INDEX_PATTERN,
                query=query,
                size=0,
                ignore_unavailable=True,
                track_total_hits=True,
            )
            total_spans = (resp.get("hits", {}) or {}).get("total", {}).get("value", 0)
            return {
                **default_stats,
                "total_traces": total_spans,
                "total_spans": total_spans,
            }
        except Exception as fallback_error:
            logger.error("Elasticsearch stats fallback failed: %s", fallback_error)
            return {**default_stats, "error": str(fallback_error)}
