"""SSE v3 事件类型定义

spec: docs/superpowers/specs/2026-07-02-chat-parts-model.md §3.3

9 种事件类型：
  message_start / plan / stage / tool_input_delta / tool_result /
  reasoning_delta / text_delta / artifact / message_end / error
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

EventType = Literal[
    "message_start",
    "plan",
    "stage",
    "tool_input_delta",
    "tool_result",
    "reasoning_delta",
    "text_delta",
    "artifact",
    "message_end",
    "error",
]


@dataclass
class StreamEventV3:
    """SSE v3 事件"""
    type: EventType
    data: dict = field(default_factory=dict)


# ── Payload helpers ──────────────────────────────────────────────────────────

def message_start(message_id: str, role: str) -> StreamEventV3:
    return StreamEventV3(
        type="message_start",
        data={"message_id": message_id, "role": role},
    )


def plan_event(part_id: str, goal: str, reasoning: str, tasks: list) -> StreamEventV3:
    return StreamEventV3(
        type="plan",
        data={"part_id": part_id, "goal": goal, "reasoning": reasoning, "tasks": tasks},
    )


def stage_event(
    part_id: str,
    agent: str,
    label: str,
    icon: str,
    status: str,
    duration_ms: int | None = None,
    round_: int | None = None,
    error: str | None = None,
) -> StreamEventV3:
    data: dict = {
        "part_id": part_id,
        "agent": agent,
        "label": label,
        "icon": icon,
        "status": status,
    }
    if duration_ms is not None:
        data["duration_ms"] = duration_ms
    if round_ is not None:
        data["round"] = round_
    if error is not None:
        data["error"] = error
    return StreamEventV3(type="stage", data=data)


def tool_input_delta_event(part_id: str, delta_json: str) -> StreamEventV3:
    return StreamEventV3(
        type="tool_input_delta",
        data={"part_id": part_id, "delta_json": delta_json},
    )


def tool_result_event(
    part_id: str,
    output: Optional[dict] = None,
    duration_ms: int | None = None,
    error: str | None = None,
) -> StreamEventV3:
    data: dict = {}
    if output is not None:
        data["output"] = output
    if duration_ms is not None:
        data["duration_ms"] = duration_ms
    if error is not None:
        data["error"] = error
    return StreamEventV3(type="tool_result", data={"part_id": part_id, **data})


def reasoning_delta_event(
    part_id: str,
    agent: str,
    delta: str,
    round_: int | None = None,
) -> StreamEventV3:
    data: dict = {"part_id": part_id, "agent": agent, "delta": delta}
    if round_ is not None:
        data["round"] = round_
    return StreamEventV3(type="reasoning_delta", data=data)


def text_delta_event(part_id: str, delta: str) -> StreamEventV3:
    return StreamEventV3(type="text_delta", data={"part_id": part_id, "delta": delta})


def artifact_event(
    part_id: str,
    artifact_type: str,
    body: str,
    title: str | None = None,
    platform: str | None = None,
    structured: dict | None = None,
) -> StreamEventV3:
    data: dict = {
        "part_id": part_id,
        "artifact_type": artifact_type,
        "body": body,
    }
    if title is not None:
        data["title"] = title
    if platform is not None:
        data["platform"] = platform
    if structured is not None:
        data["structured"] = structured
    return StreamEventV3(type="artifact", data=data)


def message_end_event(
    message_id: str,
    status: str,
    usage: dict | None = None,
) -> StreamEventV3:
    data: dict = {"message_id": message_id, "status": status}
    if usage is not None:
        data["usage"] = usage
    return StreamEventV3(type="message_end", data=data)


def error_event(
    code: str,
    message: str,
    details: str | None = None,
    part_id: str | None = None,
    retry_supported: bool = True,
) -> StreamEventV3:
    data: dict = {
        "code": code,
        "message": message,
        "retry_supported": retry_supported,
    }
    if details is not None:
        data["details"] = details
    if part_id is not None:
        data["part_id"] = part_id
    return StreamEventV3(type="error", data=data)
