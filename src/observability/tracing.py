"""可观测性：轻量级追踪设置（无 OTel）。

Agent 追踪由 `AgentTracer` 处理（参见 agent_tracer.py），它将 trace
直接以 Jaeger 兼容的 trace 格式写入 UnifiedEsWriter —— 无需 Jaeger 服务器。

HTTP 追踪由 LoggingMiddleware → UnifiedEsWriter → ES 处理。

本模块仅提供 `setup_observability()` 用于服务初始化
和 `get_tracer()` 用于向后兼容。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.logging_shared.unified_writer import UnifiedEsWriter

logger = logging.getLogger(__name__)


def setup_observability(
    service_name: str = "autopublish-api",
    service_version: str = "2.0.0",
    settings=None,
) -> tuple[None, None]:
    """初始化可观测性基础设施。

    - 确保 UnifiedEsWriter 单例已启动（traces/events/logs → ES）
    - 配置 structlog（如果提供了 settings）

    返回 (None, None) —— 无 OTel tracer；Agent span 请使用 AgentTracer。
    """
    if settings is not None:
        from src.logging_shared.config import setup as setup_logging
        setup_logging(settings)
    else:
        from src.config import Settings
        from src.logging_shared.config import setup as setup_logging
        setup_logging(Settings(log_level="DEBUG", log_json_format=False))

    # UnifiedEsWriter 在 main.py 的 lifespan 中启动 —— 此处仅确保已初始化
    UnifiedEsWriter.get_instance()

    logger.info(
        "可观测性已初始化（trace writer: UnifiedEsWriter）service=%s",
        service_name,
    )
    return None, None


def setup_local_dev(settings=None) -> tuple[None, None]:
    """本地开发模式：与 setup_observability 相同，无 OTel。"""
    if settings is not None:
        from src.logging_shared.config import setup as setup_logging
        setup_logging(settings)
    else:
        from src.config import Settings
        from src.logging_shared.config import setup as setup_logging
        setup_logging(Settings(log_level="DEBUG", log_json_format=False))

    UnifiedEsWriter.get_instance()
    logger.info("本地开发可观测性已初始化")
    return None, None


def get_tracer(name: str = "autopublish"):
    """向后兼容的桩函数。实际追踪请使用 AgentTracer。

    Args:
        name: 已忽略（为保持 API 兼容性保留）。
    Returns: None（Agent 追踪直接使用 AgentTracer）。
    """
    from src.observability.agent_tracer import get_agent_tracer
    return get_agent_tracer()


def get_meter(name: str = "autopublish"):
    """桩函数：未配置 meter。"""
    return None
