"""TraceContext — HTTP 请求级别的追踪上下文"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceContext:
    """请求级追踪上下文，记录生命周期的关键事件

    在 LoggingMiddleware 中创建，挂到 request.state.trace_ctx。
    用于输出 ⚡ REQUEST / ⚡ RESPONSE 摘要行。

    注意：Orchestrator 在 worker 中异步执行，不能直接传递此对象。
    trace_id 字符串已通过 skill_context["trace_id"] 传入 worker。
    """

    trace_id: str
    method: str = ""
    path: str = ""
    start_time: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    request_summary: str = ""
    response_summary: str = ""
    status_code: int = 0
    error: str = ""

    def add_event(self, name: str, **kwargs: Any) -> None:
        self.events.append({"name": name, "time": time.time(), **kwargs})

    @property
    def elapsed_ms(self) -> float:
        return round((time.time() - self.start_time) * 1000, 1)

    def to_request_line(self) -> str:
        return f"{self.method} {self.path}"

    def to_response_line(self) -> str:
        line = f"{self.status_code}"
        if self.response_summary:
            line += f"  {self.response_summary}"
        if self.error:
            line += f"  ERR: {self.error[:120]}"
        line += f"  {self.elapsed_ms}ms"
        return line
