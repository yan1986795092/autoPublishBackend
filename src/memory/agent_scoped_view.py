"""Permission-scoped context views for multi-agent context management."""
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
# Task field map
# ──────────────────────────────────────────

_TASK_FIELD_MAP: Dict[str, str] = {
    "writing_content": "writing_content",
    "review_feedback": "review_feedback",
    "seo_suggestions": "seo_suggestions",
    "creative_anchors": "creative_anchors",
}


# ──────────────────────────────────────────
# Filter task fields by permission
# ──────────────────────────────────────────


def filter_task_fields(agent_name: str, store: SharedContextStore) -> Dict[str, Any]:
    """Return only the task-layer fields this agent is authorized to read."""
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
# Build context for agent
# ──────────────────────────────────────────


def build_agent_context(agent_name: str, store: SharedContextStore) -> Dict[str, Any]:
    """
    Returns a permission-filtered view of the SharedContextStore for the given agent.

    - system: SystemLayer fields (read-only contract)
    - task: task fields the agent is authorized to read
    - self: agent's own private state
    """
    perm = AGENT_PERMISSIONS.get(agent_name, AGENT_PERMISSIONS.get("WritingAgent"))

    # System layer — always included for all agents
    system = {
        "user_request": store.user_request,
        "project_manifest": store.project_manifest,
        "global_memory": store.global_memory,
        "thinking_trace": store.thinking_trace[-20:],  # cap at 20 for injection
        "creative_log": store.creative_log,  # always full — never compressed
    }

    # Task layer — filtered by permission
    task = filter_task_fields(agent_name, store)

    # Self layer — only the owning agent
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

    # For WritingAgent, add the creative_view with anchors and recent decisions
    if agent_name == "WritingAgent":
        context["creative_view"] = _build_creative_view(store)

    return context


def _build_creative_view(store: SharedContextStore) -> Dict[str, Any]:
    """Build the creative_view section for WritingAgent."""
    # Use last_* properties for backward-compatible access
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
    """Convenience wrapper: builds full context for WritingAgent with creative_view."""
    return build_agent_context("WritingAgent", store)


# ──────────────────────────────────────────
# Write permission enforcement
# ──────────────────────────────────────────


def write_to_context(
    agent_name: str,
    field_path: str,
    value: Any,
    store: SharedContextStore,
) -> WriteResult:
    """
    Attempt to write a value to a dotted field path in the store,
    enforcing permission. Appends to list fields, replaces scalar fields.
    Returns WriteResult.ACCEPTED or WriteResult.REJECTED.
    """
    perm = AGENT_PERMISSIONS.get(agent_name)
    if not perm:
        logger.warning(f"write_to_context: unknown agent {agent_name}")
        return WriteResult.REJECTED

    if field_path not in perm.writable_fields:
        logger.warning(
            f"Permission denied: {agent_name} attempted to write to {field_path}. "
            f"Allowed: {perm.writable_fields}"
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
    """Set a dotted field path on the store (e.g. 'task.writing_content').

    For task-layer list fields, appends the value. For scalar fields, sets directly.
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
# Creative decision logging
# ──────────────────────────────────────────


def log_creative_decision(
    decision_text: str,
    tags: List[str],
    agent: str,
    store: SharedContextStore,
) -> CreativeDecision:
    """
    Log a creative decision to the store's creative_log.
    WritingAgent calls this at the START of every output generation.
    Even 'no significant decision' should be logged as continuation.
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
    logger.debug(f"Creative decision logged by {agent}: {decision_text[:50]}")
    return decision


def log_review_anchor_drift(
    drift_text: str,
    severity: str,
    store: SharedContextStore,
) -> None:
    """Called when ReviewAgent detects content diverging from anchors."""
    log_creative_decision(
        decision_text=f"[REVIEW ANCHOR DRIFT] {drift_text}",
        tags=["review_anchor_drift", f"severity:{severity}"],
        agent="ReviewAgent",
        store=store,
    )


# ──────────────────────────────────────────
# Anchor extraction from WritingAgent output
# ──────────────────────────────────────────


def extract_anchors_from_content(content: str) -> dict:
    """
    Phase 1 anchor extraction from WritingAgent output text.
    Looks for structured markdown sections to populate creative anchors.

    Returns a dict with keys: story_outline, character_profiles, plot_milestones,
    style_guide, world_setting.
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

        # Detect section headers
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

        # Collect content for active section (skip headings)
        if current_section and line.strip() and not line.strip().startswith("#"):
            text = line.strip().lstrip("-*").strip()

            if current_section == "outline":
                anchors["story_outline"] += " " + text
            elif current_section == "character":
                # Pattern: "Name: description" or "- Name: description"
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
    Extract anchor fields from WritingAgent output and update the store.
    Called after WritingAgent completes. Returns True if any anchors were updated.
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
            detail=f"Updated anchors from output: {list(extracted.keys())}",
        )

    return updated


# ──────────────────────────────────────────
# Anchor drift check from ReviewAgent output
# ──────────────────────────────────────────


def check_anchor_drift_from_review(
    review_result: dict,
    store: SharedContextStore,
) -> int:
    """
    Check ReviewAgent output for anchor drift indicators and log them.
    Returns the number of drifts detected and logged.
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

    # Phase 1: keyword-based drift detection from review text
    drift_keywords = [
        "与设定不符", "矛盾", "不一致", "偏离", "anchor drift",
        "contradict", "inconsistent", "deviation", "inconsistency",
    ]
    drifts_found = 0

    for kw in drift_keywords:
        if kw.lower() in content.lower():
            log_review_anchor_drift(
                drift_text=f"Review flagged keyword '{kw}' suggesting anchor deviation",
                severity="medium",
                store=store,
            )
            drifts_found += 1

    # Also check suggestions for drift signals
    for suggestion in suggestions:
        for kw in drift_keywords:
            if kw.lower() in suggestion.lower():
                log_review_anchor_drift(
                    drift_text=f"Suggestion contains drift signal: '{suggestion[:100]}'",
                    severity="medium",
                    store=store,
                )
                drifts_found += 1
                break

    return drifts_found
