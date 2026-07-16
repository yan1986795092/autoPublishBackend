"""Parts 时代的前端 UI 数据构建器（spec 2026-07-02 §6.1）

职责：
  1. migrate_legacy_result_to_parts() — 旧 result_json 读取时转换为一个 fake artifact part
  2. build_message_parts() — 从 DB 读 parts，或走迁移 shim
  3. build_message_response_v2() — 返回带 parts[] 的 MessageResponse
  4. text→artifact 升级器（Task 2） — 通用升级函数，stream_core.py result 事件已用近似实现
"""

from __future__ import annotations
import json
import logging
from typing import Any, Optional

from .message_part_schema import ArtifactPart, ArtifactType, PartState, MessagePart

logger = logging.getLogger(__name__)


# 升级触发阈值（spec §2.3）：正文 >= 300 字才升级为 artifact
UPGRADE_MIN_CONTENT_LENGTH = 300

# 触发升级的 agent 类型（内容生成类，不是 reasoning 类）
UPGRADE_ELIGIBLE_AGENTS = frozenset({
    "WritingAgent", "FixAgent", "SEOAgent", "CodeAgent", "ReviewAgent",  # TEMP: DiagramAgent disabled
})


def should_upgrade_text_to_artifact(part: dict, agent_type: str = "") -> bool:
    """判断 text part 是否应该升级为 artifact。

    规则（来自 writing-engine spec §2.3）：
    - type == "text"
    - agent 属于内容生成类（WritingAgent / FixAgent / SEOAgent / CodeAgent / ReviewAgent）
    - content 长度 >= 300 字（短文案可能是临时回答）

    Args:
        part: 单个 part dict（包含 type/content/agent）
        agent_type: 当 part 自身缺 agent 字段时的兜底

    Returns:
        是否应该升级
    """
    if not isinstance(part, dict):
        return False
    if part.get("type") != "text":
        return False
    content = part.get("content", "")
    if not isinstance(content, str) or len(content) < UPGRADE_MIN_CONTENT_LENGTH:
        return False
    agent = part.get("agent") or agent_type or ""
    if agent not in UPGRADE_ELIGIBLE_AGENTS:
        return False
    return True


def upgrade_text_to_artifact(
    part: dict,
    *,
    title: str = "",
    platform: str = "",
    artifact_type: str = ArtifactType.BLOG.value,
) -> dict:
    """将 text part 升级为 artifact part（保持 id 不变）。

    注：实际生产环境升级点在 stream_core.py:283（result 事件触发）。
    本函数作为通用升级器，用于：
    - 迁移 shim（老数据没 artifact 时回填）
    - 单测
    """
    new_part = ArtifactPart(
        id=part.get("id", ""),
        state=PartState.DONE.value,
        artifact_type=artifact_type,
        body=part.get("content", ""),
        title=title,
        platform=platform,
        structured=part.get("structured"),
    )
    return new_part.to_dict()


def migrate_legacy_result_to_parts(result_json_str: str | None) -> list[dict]:
    """旧 result_json → parts 数组（spec §6.1，仅读取时转换，不改原数据）

    对应前端 migrateLegacyResult.ts 的后端等价实现。
    顺序：reasoning → tool_call → artifact → plan（按 spec §6.1）
    """
    parts: list[dict] = []
    if not result_json_str:
        return parts

    try:
        result = json.loads(result_json_str)
    except (json.JSONDecodeError, TypeError):
        return parts

    if not isinstance(result, dict):
        return parts

    # 1. reasoning（从 _reasoningChunks）
    reasoning_chunks = result.get("_reasoningChunks", {})
    if isinstance(reasoning_chunks, dict):
        for key, chunks in reasoning_chunks.items():
            if not isinstance(chunks, list):
                continue
            content = "".join(c.get("content", "") for c in chunks if isinstance(c, dict))
            if not content:
                continue
            # key 格式：agent|round
            agent = key.split("|")[0] if "|" in key else key
            round_str = key.split("|")[1] if "|" in key else "0"
            try:
                round_num = int(round_str)
            except ValueError:
                round_num = 0
            parts.append({
                "type": "reasoning",
                "state": "done",
                "content": content,
                "agent": agent,
                "round": round_num,
                "collapsed_by_default": True,
            })

    # 2. tool_call（从 _stages）
    stages = result.get("_stages", [])
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            parts.append({
                "type": "tool_call",
                "state": "done",
                "tool_name": stage.get("agent", ""),
                "display_name": stage.get("label", stage.get("agent", "")),
                "icon": stage.get("icon", "⚙️"),
                "input": {},
                "output": stage.get("output", {}),
                "duration_ms": int(float(stage.get("duration", 0)) * 1000),
                "round": stage.get("reviewRound", 0),
            })

    # 3. artifact（从 output）
    output = result.get("output", {})
    if isinstance(output, dict) and output:
        body = output.get("full_text") or output.get("content") or ""
        if body:
            parts.append({
                "type": "artifact",
                "state": "done",
                "artifact_type": _infer_artifact_type(output),
                "body": body,
                "title": output.get("title"),
                "platform": output.get("platform"),
                "structured": output.get("structured"),
            })

    # 4. [Task 3] trigger-only 兜底
    # 场景：result_json 只有 trigger（如用户触发创作但 pipeline 失败未生成 output），没有 output。
    # 老逻辑：parts = [] → 前端按 conversation 气泡渲染 → 用户看到结果丢失
    # 新设计：不再强制 shim 构造 artifact（避免凭空捏造内容），交给 Phase 4 的 is_generation 字段处理。
    # Phase 4 会在 message 上显式标记 is_generation=True，前端按 generation 卡片渲染 + 显示 trigger 元信息。
    # 此处仅留注释，不构造 part，保持数据真实性。
    if not any(p["type"] == "artifact" for p in parts):
        trigger = result.get("trigger")
        if isinstance(trigger, dict) and trigger:
            # 触发器存在但无 output → 用户触发了创作但结果丢失（极少发生）
            # 由 Phase 4 的 is_generation 显式标记 + 前端 generation 卡片兜底渲染
            logger.debug(
                "migrate_legacy_result_to_parts: trigger-only 场景，由 Phase 4 is_generation 处理",
                trigger_keys=list(trigger.keys()),
            )

    # 4. plan（从 plan_json — 作为独立 part）
    plan_json_str = result.get("plan_json") or result.get("_plan")
    if plan_json_str:
        try:
            plan_data = json.loads(plan_json_str) if isinstance(plan_json_str, str) else plan_json_str
            if isinstance(plan_data, dict) and plan_data.get("tasks"):
                parts.append({
                    "type": "plan",
                    "state": "done",
                    "goal": plan_data.get("goal", ""),
                    "reasoning": plan_data.get("reasoning", ""),
                    "tasks": plan_data.get("tasks", []),
                })
        except (json.JSONDecodeError, TypeError):
            pass

    return parts


def _infer_artifact_type(output: dict) -> str:
    """根据 output 内容推断 artifact_type"""
    platform = output.get("platform", "")
    if platform in ("xiaohongshu", "xiaohongshu_native"):
        return "social_post"
    content_type = output.get("content_type", "")
    if content_type == "code":
        return "code"
    if content_type in ("image", "img"):
        return "image"
    if content_type in ("tutorial", "howto"):
        return "tutorial"
    return "blog"


def build_message_parts_from_db(rows: list) -> list[dict]:
    """从 DB rows（ChatMessagePart）构建 parts 数组（有序）"""
    parts: list[dict] = []
    for row in sorted(rows, key=lambda r: r.seq):
        try:
            payload = json.loads(row.payload_json)
        except (json.JSONDecodeError, TypeError):
            payload = {}
        parts.append({
            "id": row.id,
            "server_id": row.server_id,
            "type": row.type,
            "state": row.state,
            **payload,
        })
    return parts


def build_message_parts(
    message_id: str,
    db_parts_rows: list | None,
    legacy_result_json: str | None,
) -> list[dict]:
    """获取消息的 parts：

    - DB 有 parts → 直接用
    - DB 无 parts 且有 legacy_result_json → 走迁移 shim
    - 否则 → 空数组
    """
    # DB parts 优先
    if db_parts_rows:
        return build_message_parts_from_db(db_parts_rows)

    # 旧数据迁移
    if legacy_result_json:
        parts = migrate_legacy_result_to_parts(legacy_result_json)
        if parts:
            logger.debug(
                "migrate_legacy_result_to_parts",
                message_id=message_id[:8],
                parts_count=len(parts),
            )
        return parts

    return []


def build_preview_from_parts(parts: list[dict]) -> str:
    """从 parts 推导 preview 文本（列表预览用）"""
    for part in reversed(parts):
        if part.get("type") == "artifact":
            body = part.get("body", "")
            if body:
                return body[:200]
        if part.get("type") == "text":
            content = part.get("content", "")
            if content:
                return content[:200]
    return ""


def build_content_from_parts(parts: list[dict]) -> str:
    """从 parts 推导 content（顶层文本字段）"""
    for part in parts:
        if part.get("type") == "artifact":
            body = part.get("body", "")
            if body:
                return body
        if part.get("type") == "text":
            content = part.get("content", "")
            if content:
                return content
    return ""
