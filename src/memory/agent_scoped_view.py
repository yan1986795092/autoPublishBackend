"""基于权限的多智能体上下文作用域视图。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .context_store import (
    AGENT_PERMISSIONS,
    SharedContextStore,
    CreativeAnchors,
    ContentBlock,
    ReviewBlock,
    SEOBlock,
    AgentPrivateState,
    CreativeDecision,
    WriteResult,
)
from src.logging_shared.config import get_logger

logger = get_logger("memory.agent_scoped_view")

# ──────────────────────────────────────────
# 任务字段映射
# ──────────────────────────────────────────

_TASK_FIELD_MAP: Dict[str, str] = {
    "writing_content": "writing_content",
    "review_feedback": "review_feedback",
    "seo_suggestions": "seo_suggestions",
    "creative_anchors": "creative_anchors",
}


# ──────────────────────────────────────────
# 按权限过滤任务字段
# ──────────────────────────────────────────


def filter_task_fields(agent_name: str, store: SharedContextStore) -> Dict[str, Any]:
    """返回该 Agent 有权读取的任务层字段。"""
    perm = AGENT_PERMISSIONS.get(agent_name)
    if not perm:
        logger.warning("未知 Agent 类型，返回空上下文", agent_name=agent_name)
        return {}

    result = {}
    task_values = {
        "writing_content": store.writing_content,
        "review_feedback": store.review_feedback,
        "seo_suggestions": store.seo_suggestions,
        "creative_anchors": store.creative_anchors,
    }

    for field_name, value in task_values.items():
        full_path = f"task.{field_name}"
        if full_path in perm.readable_fields or "task" in perm.readable_fields:
            result[field_name] = value

    return result


# ──────────────────────────────────────────
# 构建 Agent 上下文
# ──────────────────────────────────────────


def build_agent_context(agent_name: str, store: SharedContextStore) -> Dict[str, Any]:
    """
    返回给定 Agent 的 SharedContextStore 的权限过滤视图。

    - system: 系统层字段（只读契约）
    - task: Agent 有权读取的任务字段
    - self: Agent 自身的私有状态
    """
    perm = AGENT_PERMISSIONS.get(agent_name, AGENT_PERMISSIONS.get("WritingAgent"))

    # 系统层 —— 所有 Agent 始终包含
    system = {
        "user_request": store.user_request,
        "project_manifest": store.project_manifest,
        "global_memory": store.global_memory,
        "thinking_trace": store.thinking_trace[-20:],  # 限制 20 条以防止注入
        "creative_log": store.creative_log,  # 始终完整 —— 永不压缩
    }

    # 任务层 —— 按权限过滤
    task = filter_task_fields(agent_name, store)

    # 自身层 —— 仅属于当前 Agent
    self_state = store.agent_private.get(agent_name, AgentPrivateState())
    self_dict = {
        "internal_reasoning": self_state.internal_reasoning,
        "confidence_score": self_state.confidence_score,
        "local_draft_cache": self_state.local_draft_cache,
    }

    context = {
        "system": system,
        "task": task,
        "self": self_dict,
    }

    # 对于 WritingAgent，添加包含锚点和近期决策的 creative_view
    if agent_name == "WritingAgent":
        context["creative_view"] = _build_creative_view(store)

    return context


def _build_creative_view(store: SharedContextStore) -> Dict[str, Any]:
    """为 WritingAgent 构建 creative_view 部分。"""
    # 使用 last_* 属性实现向后兼容的访问
    return {
        "anchors": store.creative_anchors,
        "active_feedback": {
            "review": store.last_review_feedback,
            "seo": store.last_seo_suggestions,
        },
        "recent_content": store.writing_content[-3:] if store.writing_content else [],
        "recent_decisions": store.creative_log[-5:],
    }


def get_context_for_writing_agent(store: SharedContextStore) -> Dict[str, Any]:
    """便捷封装：为 WritingAgent 构建包含 creative_view 的完整上下文。"""
    return build_agent_context("WritingAgent", store)


# ──────────────────────────────────────────
# 写权限强制执行
# ──────────────────────────────────────────


def write_to_context(
    agent_name: str,
    field_path: str,
    value: Any,
    store: SharedContextStore,
) -> WriteResult:
    """
    尝试将值写入存储中的点分字段路径，
    强制执行权限。列表字段追加，标量字段替换。
    返回 WriteResult.ACCEPTED 或 WriteResult.REJECTED。
    """
    perm = AGENT_PERMISSIONS.get(agent_name)
    if not perm:
        logger.warning(f"write_to_context: 未知 agent {agent_name}")
        return WriteResult.REJECTED

    if field_path not in perm.writable_fields:
        logger.warning(
            f"权限拒绝: {agent_name} 尝试写入 {field_path}。"
            f"允许的字段: {perm.writable_fields}"
        )
        store.add_thinking_step(
            agent=agent_name,
            action="permission_denied",
            step_field=field_path,
            detail=f"writable_fields={perm.writable_fields}",
        )
        return WriteResult.REJECTED

    _set_nested(store, field_path, value, agent_name=agent_name)
    store.add_thinking_step(agent=agent_name, action="write", step_field=field_path)
    return WriteResult.ACCEPTED


def _set_nested(store: SharedContextStore, field_path: str, value: Any, agent_name: str = "") -> None:
    """设置存储上的点分字段路径（例如 'task.writing_content'）。

    对于任务层列表字段，追加值。对于标量字段，直接设置。
    """
    parts = field_path.split(".", 1)
    layer = parts[0]
    sub = parts[1] if len(parts) > 1 else ""

    if layer == "task":
        attr = _TASK_FIELD_MAP.get(sub, sub)
        if not hasattr(store, attr):
            return
        current = getattr(store, attr)
        if isinstance(current, list):
            current.append(value)
        else:
            setattr(store, attr, value)

    elif layer == "self":
        if sub in ("internal_reasoning", "confidence_score", "local_draft_cache"):
            priv = store.get_or_create_agent_private(agent_name=agent_name)
            setattr(priv, sub, value)

    elif layer == "system":
        if sub in store.__dict__:
            store.__dict__[sub] = value


# ──────────────────────────────────────────
# 创意决策日志
# ──────────────────────────────────────────


def log_creative_decision(
    decision_text: str,
    tags: List[str],
    agent: str,
    store: SharedContextStore,
) -> CreativeDecision:
    """
    将创意决策记录到存储的 creative_log 中。
    WritingAgent 在每次输出生成开始时调用此方法。
    即使是 '无重要决策' 也应作为延续记录。
    """
    turn_number = 0
    if store.project_manifest:
        turn_number = store.project_manifest.turn_count

    decision = CreativeDecision.create(
        decision_text=decision_text,
        tags=tags,
        agent=agent,
        turn_number=turn_number,
    )
    store.creative_log.append(decision)
    store.add_thinking_step(
        agent=agent,
        action="log_creative_decision",
        detail=f"tags={tags}, turn={turn_number}",
    )
    logger.debug(f"Creative decision 已由 {agent} 记录: {decision_text[:50]}")
    return decision


def log_review_anchor_drift(
    drift_text: str,
    severity: str,
    store: SharedContextStore,
) -> None:
    """当 ReviewAgent 检测到内容偏离锚点时调用。"""
    log_creative_decision(
        decision_text=f"[REVIEW ANCHOR DRIFT] {drift_text}",
        tags=["review_anchor_drift", f"severity:{severity}"],
        agent="ReviewAgent",
        store=store,
    )


# ──────────────────────────────────────────
# 从 WritingAgent 输出中提取锚点
# ──────────────────────────────────────────


def extract_anchors_from_content(content: str) -> dict:
    """
    从 WritingAgent 输出文本中提取阶段 1 锚点。
    查找结构化 Markdown 部分以填充创意锚点。

    返回包含以下键的字典: story_outline, character_profiles, plot_milestones,
    style_guide, world_setting。
    """
    import re

    anchors = {
        "story_outline": "",
        "character_profiles": {},
        "plot_milestones": [],
        "style_guide": "",
        "world_setting": "",
    }

    lines = content.split("\n")
    current_section = None

    for i, line in enumerate(lines):
        lower = line.lower().strip()

        # 检测章节标题
        if re.match(r"^#{1,3}\s*(outline|故事大纲|大纲)", lower):
            current_section = "outline"
            continue
        elif re.match(r"^#{1,3}\s*(character|人物|角色)", lower):
            current_section = "character"
            continue
        elif re.match(r"^#{1,3}\s*(milestone|plot|情节|节点)", lower):
            current_section = "milestone"
            continue
        elif re.match(r"^#{1,3}\s*(style|风格)", lower):
            current_section = "style"
            continue
        elif re.match(r"^#{1,3}\s*(world|setting|世界观|设定)", lower):
            current_section = "world"
            continue

        # 为当前活动章节收集内容（跳过标题）
        if current_section and line.strip() and not line.strip().startswith("#"):
            text = line.strip().lstrip("-*").strip()

            if current_section == "outline":
                anchors["story_outline"] += " " + text
            elif current_section == "character":
                # 格式: "Name: description" 或 "- Name: description"
                m = re.match(r"^\*?\-?\s*(\w[\w\s]*?):\s*(.+)", text)
                if m:
                    name, desc = m.group(1).strip(), m.group(2).strip()
                    anchors["character_profiles"][name] = desc
            elif current_section == "milestone":
                anchors["plot_milestones"].append(text)
            elif current_section == "style":
                anchors["style_guide"] += " " + text
            elif current_section == "world":
                anchors["world_setting"] += " " + text

    anchors["story_outline"] = anchors["story_outline"].strip()
    anchors["style_guide"] = anchors["style_guide"].strip()
    anchors["world_setting"] = anchors["world_setting"].strip()
    return anchors


def update_creative_anchors(
    writing_result: dict,
    store: SharedContextStore,
) -> bool:
    """
    从 WritingAgent 输出中提取锚点字段并更新存储。
    在 WritingAgent 完成后调用。如果任何锚点被更新则返回 True。
    """
    content = ""
    if isinstance(writing_result, dict):
        output = writing_result.get("output", {})
        if isinstance(output, dict):
            content = output.get("content", "")
        elif isinstance(output, str):
            content = output
    elif isinstance(writing_result, str):
        content = writing_result

    if not content:
        return False

    extracted = extract_anchors_from_content(content)
    updated = False

    current = store.creative_anchors

    if extracted["story_outline"] and not current.story_outline:
        current.story_outline = extracted["story_outline"]
        updated = True
    if extracted["character_profiles"] and not current.character_profiles:
        current.character_profiles = extracted["character_profiles"]
        updated = True
    if extracted["plot_milestones"] and not current.plot_milestones:
        current.plot_milestones = extracted["plot_milestones"]
        updated = True
    if extracted["style_guide"] and not current.style_guide:
        current.style_guide = extracted["style_guide"]
        updated = True
    if extracted["world_setting"] and not current.world_setting:
        current.world_setting = extracted["world_setting"]
        updated = True

    if updated:
        store.add_thinking_step(
            agent="WritingAgent",
            action="anchor_update",
            detail=f"从输出中更新锚点: {list(extracted.keys())}",
        )

    return updated


# ──────────────────────────────────────────
# 从 ReviewAgent 输出中检查锚点偏移
# ──────────────────────────────────────────


def check_anchor_drift_from_review(
    review_result: dict,
    store: SharedContextStore,
) -> int:
    """
    检查 ReviewAgent 输出中的锚点偏移指标并记录。
    返回检测到并记录的偏移数量。
    """
    output = review_result.get("output", {})
    if isinstance(output, dict):
        content = output.get("content", "")
        suggestions = output.get("suggestions", [])
    else:
        content = str(output)
        suggestions = []

    if not content:
        return 0

    # 阶段 1：基于关键词的偏移检测
    drift_keywords = [
        "与设定不符", "矛盾", "不一致", "偏离", "anchor drift",
        "contradict", "inconsistent", "deviation", "inconsistency",
    ]
    drifts_found = 0

    for kw in drift_keywords:
        if kw.lower() in content.lower():
            log_review_anchor_drift(
                drift_text=f"审阅标记了关键词 '{kw}'，提示锚点偏离",
                severity="medium",
                store=store,
            )
            drifts_found += 1

    # 同时检查建议中的偏移信号
    for suggestion in suggestions:
        for kw in drift_keywords:
            if kw.lower() in suggestion.lower():
                log_review_anchor_drift(
                    drift_text=f"建议包含偏移信号: '{suggestion[:100]}'",
                    severity="medium",
                    store=store,
                )
                drifts_found += 1
                break

    return drifts_found
