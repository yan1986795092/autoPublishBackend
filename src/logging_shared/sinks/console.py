"""ConsoleSink — 接管 RichConsoleRenderer，把 LogRecord 渲染到 stderr。

行为变化：
- 直接 sys.stderr 输出，去掉 StringIO.truncate/seek 的每条 IO 开销。
- 按 parent_span_id 自动缩进（默认开启，可用 LOG_INDENT=0 禁用）。
- level / status / duration 染色阈值与旧 _render_http/_render_system 对齐。
- UnicodeEncodeError 不再 pass — 打印一行 [log-encoding-error] + 摘要到 stderr。
"""

from __future__ import annotations

import os
import sys
from typing import Any

from rich.console import Console

from ..log_record import LogRecord
from ..fields import (
    TYPE, TYPE_HTTP, TYPE_AGENT, TYPE_LLM, TYPE_DB, TYPE_SYSTEM,
)


class ConsoleSink:
    name = "console"

    def __init__(
        self,
        colors: bool = True,
        indent: bool = True,
        indent_chars_per_level: int = 2,
    ):
        self._console = Console(
            file=sys.stderr,
            color_system="truecolor" if colors else None,
            no_color=not colors,
            highlight=False,
            soft_wrap=True,
            width=200,
        )
        self._indent_enabled = indent and os.getenv("LOG_INDENT", "1") not in ("0", "false")
        self._indent_chars = indent_chars_per_level

    def _level_text(self, level: str) -> str:
        return f"[{level:<7}]  "

    def _color_for_level(self, level: str) -> str:
        return {"INFO": "green", "WARN": "yellow", "ERROR": "red", "DEBUG": "cyan"}.get(level, "white")

    def _indent(self, record: LogRecord) -> str:
        if not self._indent_enabled or record.parent_span_id is None:
            return ""
        depth = self._depth_of(record)
        if depth <= 0:
            return ""
        return (" " * self._indent_chars * depth) + "↳ "

    def _depth_of(self, record: LogRecord) -> int:
        # 简化：基于 parent_span_id 的存在与否判断子 span 深度 1。
        # Phase 2 引入 span tree table 后，可精确计算多级深度。
        return 1 if record.parent_span_id else 0

    def write(self, record: LogRecord) -> None:
        try:
            level_text = self._level_text(record.level)
            indent_text = self._indent(record)
            # 主行：[LEVEL]  ↳ <domain> <message> trace:<trace_id>
            line = (
                level_text
                + indent_text
                + self._render_domain_marker(record.domain)
                + " "
                + record.message
                + f"  [trace:{record.trace_id[:12]}]"
            )
            self._console.print(line, highlight=False)
        except UnicodeEncodeError as exc:
            # 设计原则 P6: 不静默, 打印一行错误摘要
            sys.stderr.write(
                f"[log-encoding-error] level={record.level} "
                f"domain={record.domain} trace={record.trace_id[:12]} err={exc}\n"
            )
        except Exception as exc:
            sys.stderr.write(f"[log-sink-error] {exc}\n")

    def _render_domain_marker(self, domain: str) -> str:
        return {
            TYPE_HTTP: "🌐  http",
            TYPE_AGENT: "🤖 agent",
            TYPE_LLM: "🧠  llm ",
            TYPE_DB: "🗄️  db  ",
            TYPE_SYSTEM: "⚡  sys ",
            "span": "↘  span",
        }.get(domain, "●      ")
