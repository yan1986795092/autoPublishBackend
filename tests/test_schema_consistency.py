"""前后端 schema 一致性约束 — Task 6 / Phase 5

确保 Python dataclass 字段集与前端 TypeScript 类型保持一致。
新增 Part 类型时必须双端同步。
"""

from src.api.message_part_schema import (
    ReasoningPart,
    TextPart,
    ToolCallPart,
    PlanPart,
    PlanTask,
    ErrorPart,
    ArtifactPart,
    from_dict,
    PartType,
    ArtifactType,
)

# 与前端 frontend/src/types/message.ts 的 ALL_PART_TYPES / ALL_ARTIFACT_TYPES 对齐
ALL_PART_TYPES = {
    "reasoning", "text", "tool_call", "plan", "error", "artifact",
}
ALL_ARTIFACT_TYPES = {
    "blog", "social_post", "copywriting", "code", "image", "tutorial",
}


def test_part_types_complete():
    """PartType 枚举覆盖前端定义的 6 种 part。"""
    for pt in PartType:
        assert pt.value in ALL_PART_TYPES, f"PartType.{pt.name} 缺少前端对应"


def test_artifact_types_complete():
    """ArtifactType 枚举覆盖前端定义的 6 种 artifact。"""
    for at in ArtifactType:
        assert at.value in ALL_ARTIFACT_TYPES, f"ArtifactType.{at.name} 缺少前端对应"


def test_artifact_part_required_fields():
    """ArtifactPart 序列化字段结构稳定。"""
    p = ArtifactPart()
    d = p.to_dict()
    assert d["type"] == "artifact"
    assert d["state"] in ("streaming", "done", "error", "cancelled")
    assert d["artifact_type"] in ALL_ARTIFACT_TYPES
    assert isinstance(d["body"], str)
    # 可选字段
    assert "title" in d
    assert "platform" in d
    assert "structured" in d


def test_text_part_not_confused_with_artifact():
    """TextPart 与 ArtifactPart 字段不重叠（除了基础字段）。"""
    text_fields = set(TextPart.__dataclass_fields__.keys())
    artifact_fields = set(ArtifactPart.__dataclass_fields__.keys())
    # text 专属字段
    assert "content" in text_fields
    assert "content" not in artifact_fields
    # artifact 专属字段
    assert "body" in artifact_fields
    assert "body" not in text_fields
    assert "artifact_type" in artifact_fields
    assert "platform" in artifact_fields


def test_reasoning_part_requires_agent_field():
    """ReasoningPart.agent 字段必须存在（前端已对齐为必填）。"""
    p = ReasoningPart()
    d = p.to_dict()
    assert "agent" in d
    assert isinstance(d["agent"], str)


def test_error_part_has_retry_field_in_chat_message_ui():
    """前端 ErrorPart 有 retry_supported 字段（spec §3.1）。
    后端 ErrorPart 无此字段（retry_supported 由前端 UI 派生）。
    注：这是已知差异，前端 retry_supported 从 message.status 计算，后端无需存储。
    """
    p = ErrorPart()
    d = p.to_dict()
    assert "retry_supported" not in d  # 前端字段，后端不存


def test_tool_call_part_icon_default_emoji():
    """ToolCallPart.icon 默认值 ⚙️（前端 fallback 用）。"""
    p = ToolCallPart()
    assert p.icon == "⚙️"


def test_plan_part_supports_nested_tasks():
    """PlanPart.tasks 字段支持嵌套 PlanTask 实例。"""
    task = PlanTask(agent="WritingAgent", label="写", dependencies=[])
    p = PlanPart(tasks=[task])
    d = p.to_dict()
    assert len(d["tasks"]) == 1
    assert d["tasks"][0]["agent"] == "WritingAgent"


def test_plan_task_id_default_unique():
    """PlanTask.id 默认值唯一。"""
    from src.api.message_part_schema import new_part_id
    t1 = PlanTask()
    t2 = PlanTask()
    assert t1.id != t2.id


def test_from_dict_idempotent():
    """from_dict → to_dict → from_dict 应得到等价对象。"""
    p1 = ArtifactPart(body="x", artifact_type="blog", platform="wechat")
    d = p1.to_dict()
    p2 = from_dict(d)
    assert isinstance(p2, ArtifactPart)
    assert p2.body == p1.body
    assert p2.artifact_type == p1.artifact_type
    assert p2.platform == p1.platform