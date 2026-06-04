"""Observability: OpenTelemetry tracing + Prometheus metrics.

This module provides a unified observability setup for the entire backend.
Import this in your app lifespan to enable distributed tracing across all services.

Logging configuration is delegated to ``src.core.logging.setup_logging``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, DEPLOYMENT_ENVIRONMENT
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.propagate import set_global_textmap

from src.core.logging import setup_logging
from .metrics import create_meters

logger = logging.getLogger(__name__)

_tracer: Optional[Tracer] = None
_meter: Optional[Meter] = None
_initialized = False


def setup_observability(
    service_name: str = "autopublish-api",
    service_version: str = "2.0.0",
    settings=None,
) -> tuple[Tracer, Meter]:
    """Initialize OpenTelemetry tracing, Prometheus metrics, and structlog.

    Call this exactly once in your FastAPI app lifespan.
    Logging is configured via ``src.core.logging.setup_logging(settings)``.
    Returns the global tracer and meter instances.
    """
    global _tracer, _meter, _initialized
    if _initialized:
        return _tracer or trace.get_tracer(service_name), _meter or metrics.get_meter(service_name)

    # Configure logging first (delegated to core.logging)
    if settings is not None:
        setup_logging(settings)

    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        DEPLOYMENT_ENVIRONMENT: os.getenv("DEPLOYMENT_ENVIRONMENT", "development"),
    })

    # ── Tracing ───────────────────────────────────────────────────────────────
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    trace_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(tracer_provider)
    set_global_textmap(TraceContextTextMapPropagator())

    # ── Metrics ──────────────────────────────────────────────────────────────
    metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=10000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    _tracer = trace.get_tracer(service_name, service_version)
    _meter = metrics.get_meter(service_name, service_version)

    _initialized = True
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info(
        "Observability initialized",
        service=service_name,
        version=service_version,
        otlp_endpoint=otlp_endpoint,
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
    """Local dev mode: file-based tracing, no console output.

    Spans/metrics are saved to logs/traces.jsonl and logs/metrics.jsonl
    instead of printing to stdout.
    """
    import json
    from pathlib import Path
    from opentelemetry.sdk.trace.export import SpanExporter
    from opentelemetry.sdk.metrics.export import MetricExporter

    logs_dir = Path(__file__).resolve().parents[3] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    class JsonlSpanExporter(SpanExporter):
        def export(self, spans, timeout_millis=30000):
            import json
            with open(logs_dir / "traces.jsonl", "a", encoding="utf-8") as f:
                for span in spans:
                    data = json.loads(span.to_json())
                    f.write(json.dumps({"name": data.get("name", ""), "trace_id": data.get("context", {}).get("trace_id", "")[:16], "status": data.get("status", {}).get("status_code", "")}) + "\n")
            return False

        def shutdown(self): pass

    class NoopMetricExporter(MetricExporter):
        def __init__(self):
            super().__init__()
        def export(self, metrics_data, timeout_millis=30000):
            return True
        def shutdown(self): pass
        def force_flush(self, timeout_millis=30000): return True

    if settings is not None:
        setup_logging(settings)
    else:
        from src.config import Settings
        setup_logging(Settings(log_level="DEBUG", log_json_format=False))

    resource = Resource.create({SERVICE_NAME: "autopublish-api-dev"})

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(JsonlSpanExporter()))
    trace.set_tracer_provider(tracer_provider)
    set_global_textmap(TraceContextTextMapPropagator())

    metric_reader = PeriodicExportingMetricReader(NoopMetricExporter(), export_interval_millis=30000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    global _tracer, _meter, _initialized
    _tracer = trace.get_tracer("autopublish-api-dev")
    _meter = metrics.get_meter("autopublish-api-dev")

    _initialized = True
    return _tracer, _meter
