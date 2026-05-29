"""Observability: OpenTelemetry tracing + Prometheus metrics + structured logging.

This module provides a unified observability setup for the entire backend.
Import this in your app lifespan to enable distributed tracing across all services.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import structlog
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

from .metrics import create_meters

logger = logging.getLogger(__name__)

_tracer: Optional[Tracer] = None
_meter: Optional[Meter] = None
_initialized = False


def setup_observability(
    service_name: str = "autopublish-api",
    service_version: str = "2.0.0",
    log_json: bool = True,
    debug: bool = False,
) -> tuple[Tracer, Meter]:
    """Initialize OpenTelemetry tracing, Prometheus metrics, and structlog.

    Call this exactly once in your FastAPI app lifespan.
    Returns the global tracer and meter instances.
    """
    global _tracer, _meter, _initialized
    if _initialized:
        return _tracer or trace.get_tracer(service_name), _meter or metrics.get_meter(service_name)

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

    # ── Structured Logging ───────────────────────────────────────────────────
    _setup_structlog(log_json=log_json, debug=debug)

    _initialized = True
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info(
        "Observability initialized",
        service=service_name,
        version=service_version,
        otlp_endpoint=otlp_endpoint,
        log_json=log_json,
    )

    return _tracer, _meter


def _setup_structlog(log_json: bool, debug: bool) -> None:
    """Configure structlog with JSON or console output."""
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if log_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route standard logging through structlog (only for named loggers to avoid cycles)
    for _name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        _l = logging.getLogger(_name)
        # Only inject if no handler is set (avoid double-output when running under uvicorn)
        if not _l.handlers:
            _h = logging.StreamHandler(sys.stderr)
            _h.setFormatter(logging.Formatter("%(message)s"))
            _l.addHandler(_h)
        _l.setLevel(logging.DEBUG if debug else logging.INFO)


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


def setup_local_dev() -> tuple[Tracer, Meter]:
    """Local dev mode: console-only tracing, no OTLP export.

    Useful when running without Docker (e.g., IDE debugger).
    """
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader

    resource = Resource.create({SERVICE_NAME: "autopublish-api-dev"})

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tracer_provider)
    set_global_textmap(TraceContextTextMapPropagator())

    metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=10000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    global _tracer, _meter, _initialized
    _tracer = trace.get_tracer("autopublish-api-dev")
    _meter = metrics.get_meter("autopublish-api-dev")

    _setup_structlog(log_json=False, debug=True)

    _initialized = True
    return _tracer, _meter
