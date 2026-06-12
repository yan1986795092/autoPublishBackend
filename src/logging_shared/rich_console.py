"""基于 Rich 的控制台渲染器，用于 structlog — 按类型输出彩色日志。

支持 5 种日志类别，每种类别有独特的图标、颜色和布局：

    🌐 HTTP    — 按状态码显示青色/绿色/黄色/红色
    🤖 Agent   — 缩进的步骤链（开始 → 思考 → 工具 → 完成）
    🧠 LLM     — 洋红色，显示 Token 详情、延迟、成本
    🗄️  DB     — 黄色，SQL 摘要、行数、耗时
    ⚡ System  — 绿色，启动/关闭/生命周期事件

用法（通过 structlog 配置）：

    processors.append(RichConsoleRenderer(colors=True))

每个日志事件的 event_dict 中必须携带一个 ``_type`` 键，
匹配上述 5 种类别之一。未分类的事件回退到 structlog.dev.ConsoleRenderer。
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
    """structlog 处理器：根据 ``_type`` 将 event_dict 渲染为 Rich 彩色输出。

    处理 5 种日志类别（http, agent, llm, db, system）。
    未分类的事件回退到 structlog.dev.ConsoleRenderer。

    该渲染器维护一个共享的 ``rich.Console``，连接到可复用的
    ``StringIO`` 缓冲区，避免每次调用时重新分配。
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

    # ── 公共 API（structlog 处理器契约）────────────────────────

    def __call__(self, logger: Any, name: str, event_dict: dict) -> str:
        event_type = event_dict.pop("_type", None)
        handler = getattr(self, f"_render_{event_type}", None) if event_type else None
        if handler is not None:
            # 添加带颜色的级别指示 — 根处理器不再添加 [LEVEL]
            level = event_dict.pop("level", "info").upper()
            text = self._render_level(level)
            text.append(handler(event_dict))
            return self._capture(text)
        return self._fallback(logger, name, event_dict)

    # ── 内部辅助方法 ─────────────────────────────────────

    @staticmethod
    def _render_level(level: str) -> Text:
        """渲染带颜色的级别前缀：``[INFO]  `` / ``[ERROR] `` 等"""
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
        """将 Rich ``Text`` 渲染为带 ANSI 颜色的字符串，通过共享缓冲区。"""
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

        # 请求方法
        method_colors = {"GET": "cyan", "POST": "green", "PUT": "yellow",
                         "PATCH": "yellow", "DELETE": "red"}
        t.append(f"{method:7s} ", Style(color=method_colors.get(method, "white"), bold=True))

        # 路径
        t.append(f"{path}  ", Style(color="white"))

        # 状态码
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

        # 耗时
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

        # ── 确定缩进级别和图标 ──
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

        # ── 图标 ──
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

        # 事件类型标签
        t.append(f"{event_type:15s} ", Style(color="cyan", bold=True))

        # 代理名称
        if agent_name:
            t.append(f"{agent_name:14s} ", Style(color="white"))

        # 运行 ID
        if run_id:
            t.append(f"run={run_id[:8]}  ", Style(color="blue", dim=True))

        # 耗时
        if duration_ms:
            if duration_ms > 10000:
                dur_style = Style(color="red", bold=True)
            elif duration_ms > 2000:
                dur_style = Style(color="yellow")
            else:
                dur_style = Style(color="green")
            t.append(f"{self._fmt_ms(duration_ms)}  ", dur_style)

        # 模型
        if llm_model:
            t.append(f"[{llm_model}]  ", Style(color="magenta", dim=True))

        # Token 用量
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

        # 思考过程预览（前 100 字符）
        if thought:
            preview = thought[:120].replace("\n", " ")
            t.append(f"\n{indent}  💭 {preview}", Style(color="grey50", italic=True))

        # 工具名称
        if tool_name:
            tool_detail = self._pop(d, "tool_detail", "")
            if tool_detail:
                t.append(f" {tool_detail[:80]}", Style(color="yellow", dim=True))

        # 错误
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

        # 模型
        t.append(f"{model:18s} ", Style(color="magenta", bold=True))

        # Token
        t.append(f"{total_tokens}tokens ", Style(color="cyan"))
        if prompt_tokens or completion_tokens:
            t.append(f"(↑{prompt_tokens}↓{completion_tokens})  ", Style(dim=True))

        # 耗时
        t.append(f"{self._fmt_ms(duration_ms)}  ",
                 Style(color="red", bold=True) if duration_ms > 10000 else Style(dim=True))

        # 成本
        if cost:
            t.append(f"${float(cost):.5f}  ", Style(color="yellow"))

        # Agent
        if agent_type:
            t.append(f"[{agent_type}]  ", Style(color="blue", dim=True))

        # 请求 ID
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

        # SQL 摘要 — 截断到 80 字符，将参数替换为 ?
        sql = statement.strip()
        if len(sql) > 80:
            sql = sql[:77] + "..."
        t.append(f"{sql}  ", Style(color="white"))

        # 耗时
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

        # 事件标签
        if event:
            t.append(f"{event:20s} ", Style(color="green", bold=True))

        # 消息
        if message:
            t.append(f"{message}  ", Style(color="white"))

        # 状态
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
