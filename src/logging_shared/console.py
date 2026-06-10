"""Access-log style console output — one line per HTTP request.

Usage:
    from .console import log_http_line
    log_http_line("INFO", "POST", "/api/chat", 200, 152.3, "abc123")
    # Output: INFO:  POST /api/chat 200 152ms [trace:abc123]
"""

from __future__ import annotations

import logging
import sys

_console_handler: logging.Handler | None = None
_console_logger: logging.Logger | None = None


def _ensure_logger() -> logging.Logger:
    """Lazy-init the access-log logger with a clean format."""
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
    """Print one access-log-style line per HTTP request.

    Format: {level}:  {method} {path} {status} {duration}ms [trace:{id}]
    Errors append error info after duration.
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
    """Print a compact one-line agent event summary (dev environment).

    Format: agent_start  WritingAgent  run=abc12345
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
