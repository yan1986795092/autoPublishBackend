from __future__ import annotations

from typing import Any


def has_generation_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False

    stages = result.get("_stages")
    if isinstance(stages, list) and len(stages) > 0:
        return True

    if result.get("_draftContent"):
        return True

    llm_chunks = result.get("_llmChunks")
    if isinstance(llm_chunks, dict) and len(llm_chunks) > 0:
        return True

    if result.get("usage"):
        return True

    if result.get("agent_type") or result.get("task_id"):
        return True

    output = result.get("output")
    if not isinstance(output, dict):
        return False

    return bool(output.get("full_text") or output.get("content") or output.get("draft_content"))


def is_trigger_only_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    return bool(result.get("trigger")) and not has_generation_result(result)


def build_render_payload(content: str, result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None

    output = result.get("output")
    payload: dict[str, Any] = {}

    if isinstance(output, dict):
        body = output.get("full_text") or output.get("content")
        draft = output.get("draft_content")
        review = output.get("review_content")
        title = output.get("title")
        platform = output.get("platform")
        if body:
            payload["body"] = body
            payload["preview_text"] = body
        if draft:
            payload["draft"] = draft
        if review:
            payload["review"] = review
        if title:
            payload["title"] = title
        if platform:
            payload["platform"] = platform

    if result.get("_draftContent") and not payload.get("draft"):
        payload["draft"] = result["_draftContent"]

    if result.get("_reviewContent") and not payload.get("review"):
        payload["review"] = result["_reviewContent"]

    if content and not payload.get("body") and has_generation_result(result):
        payload["body"] = content
        payload["preview_text"] = content

    return payload or None


def infer_message_kind(role: str, status: str | None, content: str, result: Any, task_id: str | None, plan: Any) -> str:
    if role != "assistant":
        return role

    if is_trigger_only_result(result):
        return "conversation"

    if has_generation_result(result):
        return "generation"

    if status == "running":
        return "generation" if task_id else "conversation"

    if result is None and not task_id and not plan:
        return "conversation"

    return "conversation"


def build_message_ui_fields(
    *,
    role: str,
    status: str | None,
    content: str,
    result: Any,
    task_id: str | None,
    plan: Any,
) -> dict[str, Any]:
    kind = infer_message_kind(role, status, content, result, task_id, plan)

    if role == "user":
        render_mode = "bubble"
    else:
        render_mode = "generation_card" if kind == "generation" else "bubble"

    payload = build_render_payload(content, result)
    generation_meta = None
    if isinstance(result, dict):
        trigger = result.get("trigger")
        if isinstance(trigger, dict):
            generation_meta = {
                "action": trigger.get("action"),
                "platform": trigger.get("platform"),
                "revision_focus": trigger.get("revision_focus"),
                "topic": trigger.get("topic"),
            }

    return {
        "kind": kind,
        "render_mode": render_mode,
        "render_payload": payload,
        "is_generation": kind == "generation",
        "is_conversation": kind == "conversation",
        "generation_meta": generation_meta,
    }
