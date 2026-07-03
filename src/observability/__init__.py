"""可观测性模块：轻量级设置。

HTTP 追踪 span 由 LoggingMiddleware 通过 UnifiedEsWriter 直接写入 ES traces-* 索引。
不使用 OTel SDK。
"""

from .tracing import (
    setup_observability,
    setup_local_dev,
    get_tracer,
)
from . import metrics

__all__ = [
    "setup_observability",
    "setup_local_dev",
    "get_tracer",
    "metrics",
]
