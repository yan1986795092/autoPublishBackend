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


def has_generation_parts(parts: Any) -> bool:
    """检查 parts 是否包含生成类内容（artifact / text / code / image 等）。

    注意：PartWriter 持久化的文本内容是 text 类型（非 artifact），
    但含有实际生成内容，应视为生成消息。
    """
    if not isinstance(parts, list):
        return False
    for part in parts:
        if not isinstance(part, dict):
            continue
        t = part.get("type", "")
        if t in ("artifact", "code", "image", "social_post", "tutorial", "blog"):
            return True
        # text 类型但含实际生成内容（非空）也视为生成消息
        if t == "text" and bool(part.get("content")):
            return True
    return False


def infer_message_kind(role: str, status: str | None, content: str, result: Any, task_id: str | None, plan: Any, parts: Any = None) -> str:
    if role != "assistant":
        return role

    # parts 含 artifact/text → generation（优先于 trigger-only 判断）
    # 修复：部分历史消息的 result_json 只有 trigger，但 parts 中有实际生成内容
    if has_generation_parts(parts):
        return "generation"

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
    parts: Any = None,
    is_generation_explicit: bool | None = None,  # [Task 5 / Phase 4] 显式标记
) -> dict[str, Any]:
    # [Task 5 / Phase 4] 显式标记优先于字段探测
    # 来源：DB 列 is_generation / SSE 完成时记录的 stream_status
    # 若调用方提供了 is_generation_explicit，直接使用，避免 kind 推导不确定性
    if is_generation_explicit is not None:
        kind = "generation" if is_generation_explicit else "conversation"
    else:
        kind = infer_message_kind(role, status, content, result, task_id, plan, parts)

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
