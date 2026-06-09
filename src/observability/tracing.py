"""Observability: lightweight tracing setup (no OTLP Collector needed).

This module provides the bare minimum OTel setup so agent code can use
``get_tracer()`` and ``start_as_current_span()`` without changes.

HTTP spans are handled by ``LoggingMiddleware`` which writes directly to
ES traces-* in Jaeger format — no OTel pipeline involved.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import (
    Resource,
    SERVICE_NAME,
    SERVICE_VERSION,
    DEPLOYMENT_ENVIRONMENT,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.trace import Tracer
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.propagate import set_global_textmap

logger = logging.getLogger(__name__)

_tracer: Optional[Tracer] = None
_meter: Optional[Meter] = None
_initialized = False


def setup_observability(
    service_name: str = "autopublish-api",
    service_version: str = "2.0.0",
    settings=None,
) -> tuple[Tracer, Meter]:
    """Initialize OTel tracer/meter and structlog.

    Note: No OTLP exporter is configured here. HTTP spans are written by
    ``LoggingMiddleware`` directly to ES traces-*. OTel spans from agent code
    are generated but not exported (they are for in-process context only).
    """
    global _tracer, _meter, _initialized
    if _initialized:
        return _tracer or trace.get_tracer(service_name), _meter or metrics.get_meter(service_name)

    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        DEPLOYMENT_ENVIRONMENT: os.getenv("DEPLOYMENT_ENVIRONMENT", "development"),
    })

    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    set_global_textmap(TraceContextTextMapPropagator())

    # In-memory metrics (queried by in-process observability endpoints)
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    _tracer = trace.get_tracer(service_name, service_version)
    _meter = metrics.get_meter(service_name, service_version)
    _initialized = True

    logger.info(
        "Observability initialized (direct-to-ES tracing, no OTLP Collector) service=%s version=%s",
        service_name,
        service_version,
    )
    return _tracer, _meter


def get_tracer(name: str = "autopublish") -> Tracer:
    """Get a tracer instance. Initializes defaults if not yet set."""
    global _tracer
    if _tracer is None:
        return trace.get_tracer(name)
    return _tracer


def get_meter(name: str = "autopublish") -> Meter:
    """Get a meter instance. Initializes defaults if not yet set."""
    global _meter
    if _meter is None:
        return metrics.get_meter(name)
    return _meter


def setup_local_dev(settings=None) -> tuple[Tracer, Meter]:
    """Local dev mode: file-based tracing (no ES, no Collector needed).

    Spans are saved to logs/traces.jsonl. HTTP spans are written by
    ``LoggingMiddleware`` to logs/traces.jsonl as well.
    """
    import json
    from pathlib import Path
    from opentelemetry.sdk.trace.export import SpanExporter

    logs_dir = Path(__file__).resolve().parents[3] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    class JsonlSpanExporter(SpanExporter):
        def export(self, spans, timeout_millis=30000):
            with open(logs_dir / "traces.jsonl", "a", encoding="utf-8") as f:
                for span in spans:
                    data = json.loads(span.to_json())
                    f.write(json.dumps({
                        "name": data.get("name", ""),
                        "trace_id": data.get("context", {}).get("trace_id", "")[:16],
                        "status": data.get("status", {}).get("status_code", ""),
                    }) + "\n")
            return False

        def shutdown(self):
            pass

    if settings is not None:
        from src.core.logging import setup_logging
        setup_logging(settings)
    else:
        from src.config import Settings
        from src.core.logging import setup_logging
        setup_logging(Settings(log_level="DEBUG", log_json_format=False))

    resource = Resource.create({SERVICE_NAME: "autopublish-api-dev"})

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        __import__("opentelemetry.sdk.trace.export").sdk.trace.export.BatchSpanProcessor(
            JsonlSpanExporter()
        )
    )
    trace.set_tracer_provider(tracer_provider)
    set_global_textmap(TraceContextTextMapPropagator())

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    global _tracer, _meter, _initialized
    _tracer = trace.get_tracer("autopublish-api-dev")
    _meter = metrics.get_meter("autopublish-api-dev")
    _initialized = True

    return _tracer, _meter
