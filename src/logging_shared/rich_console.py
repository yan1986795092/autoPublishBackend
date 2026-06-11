"""Rich-based console renderer for structlog — colorful per-type output.

Supports 5 log categories, each with distinct icon, color, and layout:

    🌐 HTTP    — cyan/green/yellow/red by status
    🤖 Agent   — indented step chain (start → thought → tool → finish)
    🧠 LLM     — magenta, token breakdown, latency, cost
    🗄️  DB     — yellow, SQL summary, rows, duration
    ⚡ System  — green, startup/shutdown/lifecycle events

Usage (via structlog config):

    processors.append(RichConsoleRenderer(colors=True))

Each log event must carry a ``_type`` key in its event_dict matching one of the
5 categories above. Untyped events fall back to structlog.dev.ConsoleRenderer.
"""

from __future__ import annotations

import io
import time
from typing import Any

import structlog
from rich.console import Console as RichConsole
from rich.style import Style
from rich.text import Text


class RichConsoleRenderer:
    """structlog processor: renders event_dict as Rich-colored output by ``_type``.

    Handles 5 log categories (http, agent, llm, db, system).
    Falls back to structlog.dev.ConsoleRenderer for untyped events.

    The renderer maintains one shared ``rich.Console`` connected to a reusable
    ``StringIO`` buffer so that every call avoids fresh allocations.
    """

    def __init__(self, colors: bool = True):
        self._buffer = io.StringIO()
        self._console = RichConsole(
            file=self._buffer,
            color_system="truecolor" if colors else None,
            no_color=not colors,
            highlight=False,
            soft_wrap=True,
            width=200,
        )
        self._fallback = structlog.dev.ConsoleRenderer(colors=colors)

    # ── Public API (structlog processor contract) ────────────────────────

    def __call__(self, logger: Any, name: str, event_dict: dict) -> str:
        event_type = event_dict.pop("_type", None)
        handler = getattr(self, f"_render_{event_type}", None) if event_type else None
        if handler is not None:
            # Prepend colored level indicator — the root handler no longer adds [LEVEL]
            level = event_dict.pop("level", "info").upper()
            text = self._render_level(level)
            text.append(handler(event_dict))
            return self._capture(text)
        return self._fallback(logger, name, event_dict)

    # ── Internal helpers ────────────────────────────────────────

    @staticmethod
    def _render_level(level: str) -> Text:
        """Render a colored level prefix: ``[INFO]  `` / ``[ERROR] `` etc."""
        level_colors = {
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "DEBUG": "cyan",
            "CRITICAL": "red",
        }
        color = level_colors.get(level, "white")
        bold = level in ("ERROR", "WARNING", "CRITICAL")
        return Text(f"[{level:<7}]  ", Style(color=color, bold=bold))

    def _capture(self, text: Text) -> str:
        """Render a Rich ``Text`` to an ANSI-coloured string via the shared buffer."""
        self._buffer.truncate(0)
        self._buffer.seek(0)
        self._console.print(text, end="")
        return self._buffer.getvalue()

    @staticmethod
    def _pop(event_dict: dict, key: str, default: Any = "") -> Any:
        return event_dict.pop(key, default)

    @staticmethod
    def _fmt_ms(ms: float) -> str:
        if ms >= 1000:
            return f"{ms / 1000:.1f}s"
        if ms >= 1:
            return f"{ms:.0f}ms"
        if ms > 0:
            return f"{ms * 1000:.0f}µs"
        return "0ms"

    # ── HTTP  ───────────────────────────────────────────────────

    def _render_http(self, d: dict) -> Text:
        method = self._pop(d, "method", "GET").upper()
        path = self._pop(d, "path", "/")
        status = self._pop(d, "status", 0)
        duration_ms = self._pop(d, "duration_ms", 0)
        trace_id = self._pop(d, "trace_id", "")
        user_id = self._pop(d, "user_id", "")
        error = self._pop(d, "error", "")

        t = Text()
        t.append("🌐  ", Style(color="cyan"))

        # Method
        method_colors = {"GET": "cyan", "POST": "green", "PUT": "yellow",
                         "PATCH": "yellow", "DELETE": "red"}
        t.append(f"{method:7s} ", Style(color=method_colors.get(method, "white"), bold=True))

        # Path
        t.append(f"{path}  ", Style(color="white"))

        # Status
        if status >= 500:
            st = Style(color="red", bold=True)
        elif status >= 400:
            st = Style(color="yellow", bold=True)
        elif status >= 300:
            st = Style(color="cyan")
        elif status >= 200:
            st = Style(color="green", bold=True)
        else:
            st = Style(dim=True)
        t.append(f"{status}  ", st)

        # Duration
        if duration_ms > 1000:
            ds = Style(color="red", bold=True)
        elif duration_ms > 500:
            ds = Style(color="yellow")
        else:
            ds = Style(dim=True)
        t.append(f"{self._fmt_ms(duration_ms)}  ", ds)

        if trace_id:
            t.append(f"[trace:{trace_id[:12]}]  ", Style(color="blue", dim=True))
        if user_id:
            t.append(f"user={user_id}  ", Style(color="green", dim=True))
        if error:
            t.append(f" ERR:{error[:120]}", Style(color="red", bold=True))

        return t

    # ── Agent  ──────────────────────────────────────────────────

    def _render_agent(self, d: dict) -> Text:
        event_type = self._pop(d, "event_type", "agent_event")
        agent_name = self._pop(d, "agent_name", "")
        run_id = self._pop(d, "run_id", "")
        duration_ms = self._pop(d, "duration_ms", 0)
        thought = self._pop(d, "thought", "")
        tool_name = self._pop(d, "tool_name", "")
        token_usage_raw = self._pop(d, "token_usage", None)
        message = self._pop(d, "message", "")
        llm_model = self._pop(d, "llm_model", "")
        error_type = self._pop(d, "error_type", "")
        error_message = self._pop(d, "error_message", "")
        level = self._pop(d, "level", "INFO").upper()

        # ── Determine indent level and icon ──
        if event_type in ("agent_start", "agent_finish"):
            indent = ""
        elif event_type == "thought":
            indent = "  "
        elif event_type == "tool_call":
            indent = "    "
        elif event_type == "tool_result":
            indent = "    "
        elif event_type == "error":
            indent = ""
        else:
            indent = ""

        # ── Icon ──
        icon_map = {
            "agent_start": "🚀",
            "agent_finish": "✅",
            "thought": "💭",
            "tool_call": "🔧",
            "tool_result": "📎",
            "error": "❌",
            "llm_call": "🤖",
        }
        icon = icon_map.get(event_type, "●")

        t = Text()
        t.append(f"{icon}  ", Style(color="magenta"))

        # Event type label
        t.append(f"{event_type:15s} ", Style(color="cyan", bold=True))

        # Agent name
        if agent_name:
            t.append(f"{agent_name:14s} ", Style(color="white"))

        # Run ID
        if run_id:
            t.append(f"run={run_id[:8]}  ", Style(color="blue", dim=True))

        # Duration
        if duration_ms:
            if duration_ms > 10000:
                dur_style = Style(color="red", bold=True)
            elif duration_ms > 2000:
                dur_style = Style(color="yellow")
            else:
                dur_style = Style(color="green")
            t.append(f"{self._fmt_ms(duration_ms)}  ", dur_style)

        # Model
        if llm_model:
            t.append(f"[{llm_model}]  ", Style(color="magenta", dim=True))

        # Token usage
        if token_usage_raw:
            if isinstance(token_usage_raw, dict):
                total = token_usage_raw.get("total_tokens", 0)
                prompt = token_usage_raw.get("prompt_tokens", 0)
                completion = token_usage_raw.get("completion_tokens", 0)
                cost = token_usage_raw.get("cost_usd", 0)
            else:
                total = token_usage_raw
                prompt = completion = cost = 0
            t.append(f"tokens={total} ", Style(color="cyan", dim=True))
            if prompt or completion:
                t.append(f"(↑{prompt}↓{completion}) ", Style(dim=True))
            if cost:
                t.append(f"${float(cost):.4f} ", Style(color="yellow", dim=True))

        # Thought preview (first 100 chars)
        if thought:
            preview = thought[:120].replace("\n", " ")
            t.append(f"\n{indent}  💭 {preview}", Style(color="grey50", italic=True))

        # Tool name
        if tool_name:
            tool_detail = self._pop(d, "tool_detail", "")
            if tool_detail:
                t.append(f" {tool_detail[:80]}", Style(color="yellow", dim=True))

        # Error
        if error_type:
            t.append(f" {error_type}: {error_message[:100]}", Style(color="red", bold=True))

        return t

    # ── LLM  ────────────────────────────────────────────────────

    def _render_llm(self, d: dict) -> Text:
        model = self._pop(d, "model", "unknown")
        prompt_tokens = self._pop(d, "prompt_tokens", 0)
        completion_tokens = self._pop(d, "completion_tokens", 0)
        total_tokens = self._pop(d, "total_tokens", 0)
        duration_ms = self._pop(d, "duration_ms", 0)
        cost = self._pop(d, "cost", 0)
        agent_type = self._pop(d, "agent_type", "")
        req_id = self._pop(d, "request_id", "")

        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens

        t = Text()
        t.append("🧠  ", Style(color="magenta"))

        # Model
        t.append(f"{model:18s} ", Style(color="magenta", bold=True))

        # Tokens
        t.append(f"{total_tokens}tokens ", Style(color="cyan"))
        if prompt_tokens or completion_tokens:
            t.append(f"(↑{prompt_tokens}↓{completion_tokens})  ", Style(dim=True))

        # Duration
        t.append(f"{self._fmt_ms(duration_ms)}  ",
                 Style(color="red", bold=True) if duration_ms > 10000 else Style(dim=True))

        # Cost
        if cost:
            t.append(f"${float(cost):.5f}  ", Style(color="yellow"))

        # Agent
        if agent_type:
            t.append(f"[{agent_type}]  ", Style(color="blue", dim=True))

        # Request ID
        if req_id and len(str(req_id)) > 6:
            t.append(f"req={req_id[:12]}", Style(dim=True))

        return t

    # ── DB  ─────────────────────────────────────────────────────

    def _render_db(self, d: dict) -> Text:
        statement = self._pop(d, "statement", "")
        duration_ms = self._pop(d, "duration_ms", 0)
        rows = self._pop(d, "rows", None)
        engine_name = self._pop(d, "engine", "default")

        t = Text()
        t.append("🗄️  ", Style(color="yellow"))

        # SQL summary — truncate to 80 chars, replace params with ?
        sql = statement.strip()
        if len(sql) > 80:
            sql = sql[:77] + "..."
        t.append(f"{sql}  ", Style(color="white"))

        # Duration
        if duration_ms > 500:
            ds = Style(color="red", bold=True)
        elif duration_ms > 100:
            ds = Style(color="yellow")
        else:
            ds = Style(color="green")
        t.append(f"{self._fmt_ms(duration_ms)}  ", ds)

        if rows is not None:
            t.append(f"{rows}行  ", Style(color="cyan", dim=True))

        t.append(f"[{engine_name}]", Style(dim=True))

        return t

    # ── System  ─────────────────────────────────────────────────

    def _render_system(self, d: dict) -> Text:
        event = self._pop(d, "event_name", "")
        message = self._pop(d, "message", "")
        status = self._pop(d, "status", "")
        detail = self._pop(d, "detail", "")

        t = Text()
        t.append("⚡  ", Style(color="green"))

        # Event label
        if event:
            t.append(f"{event:20s} ", Style(color="green", bold=True))

        # Message
        if message:
            t.append(f"{message}  ", Style(color="white"))

        # Status
        if status:
            status_colors = {"ok": "green", "passed": "green",
                             "failed": "red", "error": "red",
                             "running": "cyan", "started": "cyan",
                             "stopped": "yellow"}
            sc = status_colors.get(status.lower(), "white") if status else "white"
            t.append(f"[{status}]  ", Style(color=sc, bold=True))

        if detail:
            t.append(f"{detail}", Style(dim=True))

        return t
