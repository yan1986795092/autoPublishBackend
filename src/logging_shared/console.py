"""访问日志风格的终端输出 — 每行一个 HTTP 请求。

用法:
    from .console import log_http_line
    log_http_line("INFO", "POST", "/api/chat", 200, 152.3, "abc123")
    # 输出: INFO:  POST /api/chat 200 152ms [trace:abc123]
"""

from __future__ import annotations

import logging
import sys

_console_handler: logging.Handler | None = None
_console_logger: logging.Logger | None = None


def _ensure_logger() -> logging.Logger:
    """延迟初始化访问日志记录器，使用简洁的格式。"""
    global _console_logger, _console_handler
    if _console_logger is not None:
        return _console_logger

    _console_logger = logging.getLogger("access")
    _console_logger.setLevel(logging.INFO)
    _console_logger.propagate = False

    if not _console_logger.handlers:
        _console_handler = logging.StreamHandler(sys.stderr)
        _console_handler.setFormatter(
            logging.Formatter("%(levelname)s:  %(message)s")
        )
        _console_logger.addHandler(_console_handler)

    return _console_logger


def log_http_line(
    level: str,
    method: str,
    path: str,
    status: int,
    duration_ms: float,
    trace_id: str = "",
    error: str = "",
) -> None:
    """每收到一个 HTTP 请求打印一行访问日志。

    格式: {level}:  {method} {path} {status} {duration}ms [trace:{id}]
    错误信息会追加到耗时之后。
    """
    parts = [f"{method} {path} {status} {duration_ms:.0f}ms"]
    if trace_id:
        parts.append(f"[trace:{trace_id[:12]}]")
    if error:
        parts.append(error)

    line = " ".join(parts)
    logger = _ensure_logger()
    if level == "ERROR":
        logger.error(line)
    else:
        logger.info(line)


def log_agent_line(event_type: str, agent_name: str = "", **details) -> None:
    """打印简洁的单行 agent 事件摘要（开发环境使用）。

    格式: agent_start  WritingAgent  run=abc12345
            tool_call    read_file     45ms
            agent_finish WritingAgent  12.5s  tokens=1500  cost=$0.0006
    """
    parts = [event_type]
    if agent_name:
        parts.append(agent_name)
    for key, value in details.items():
        if value is not None:
            parts.append(f"{key}={value}")

    logger = _ensure_logger()
    logger.info("  ".join(parts))
