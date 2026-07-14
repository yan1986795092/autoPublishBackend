"""MessagePart Schema 单测 — Task 1 Step 2

参考 docs/superpowers/plans/2026-07-13-message-parts-schema-driven-rendering.md §Task 1。
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
    new_part_id,
    PartState,
    PartType,
    ArtifactType,
)


def test_artifact_part_round_trip():
    """ArtifactPart 序列化 / 反序列化保持一致。"""
    p = ArtifactPart(
        state=PartState.DONE.value,
        artifact_type=ArtifactType.BLOG.value,
        body="hello world",
        title="My Post",
        platform="wechat",
    )
    d = p.to_dict()
    assert d["type"] == "artifact"
    assert d["body"] == "hello world"
    assert d["platform"] == "wechat"

    p2 = from_dict(d)
    assert isinstance(p2, ArtifactPart)
    assert p2.body == "hello world"
    assert p2.title == "My Post"


def test_from_dict_discriminates_type():
    """from_dict 根据 type 字段分派到正确的 dataclass。"""
    assert isinstance(from_dict({"type": "reasoning"}), ReasoningPart)
    assert isinstance(from_dict({"type": "text"}), TextPart)
    assert isinstance(from_dict({"type": "tool_call"}), ToolCallPart)
    assert isinstance(from_dict({"type": "plan"}), PlanPart)
    assert isinstance(from_dict({"type": "error"}), ErrorPart)
    assert isinstance(from_dict({"type": "artifact"}), ArtifactPart)


def test_from_dict_strips_unknown_fields():
    """from_dict 丢弃未知字段（避免前端加新字段破坏后端）。"""
    p = from_dict({"type": "text", "content": "x", "unknown_field": "y"})
    assert p.content == "x"
    assert not hasattr(p, "unknown_field")


def test_reasoning_part_default_state_is_streaming():
    """ReasoningPart 默认 streaming，方便流式增量写入。"""
    p = ReasoningPart()
    assert p.state == "streaming"
    assert p.type == "reasoning"
    assert p.agent == ""


def test_all_part_types_have_correct_type_field():
    """每个 Part dataclass 的 type 字面量必须与其 PartType 一致。"""
    assert ReasoningPart().type == PartType.REASONING.value
    assert TextPart().type == PartType.TEXT.value
    assert ToolCallPart().type == PartType.TOOL_CALL.value
    assert PlanPart().type == PartType.PLAN.value
    assert ErrorPart().type == PartType.ERROR.value
    assert ArtifactPart().type == PartType.ARTIFACT.value


def test_artifact_type_includes_copywriting():
    """计划新增的 copywriting 类型必须存在（spec §3.1）。"""
    assert ArtifactType.COPYWRITING.value == "copywriting"


def test_plan_part_handles_nested_tasks():
    """PlanPart 反序列化时 tasks 嵌套字段正确转换。"""
    data = {
        "type": "plan",
        "state": "done",
        "plan_data": {"tasks": []},
        "goal": "写一篇博客",
        "reasoning": "因为用户要求",
        "tasks": [
            {"agent": "WritingAgent", "label": "生成正文", "dependencies": []},
            {"agent": "ReviewAgent", "label": "审校", "dependencies": ["t1"]},
        ],
    }
    p = from_dict(data)
    assert isinstance(p, PlanPart)
    assert isinstance(p.tasks, list)
    assert len(p.tasks) == 2
    assert isinstance(p.tasks[0], PlanTask)
    assert p.tasks[0].agent == "WritingAgent"


def test_from_dict_rejects_unknown_type():
    """from_dict 对未知 type 抛 ValueError，不静默吞错。"""
    import pytest

    with pytest.raises(ValueError, match="Unknown part type"):
        from_dict({"type": "unknown_future_type"})


def test_new_part_id_is_unique():
    """new_part_id 每次生成不同 ID（避免流式 chunks 互相覆盖）。"""
    ids = {new_part_id() for _ in range(100)}
    assert len(ids) == 100


def test_error_part_default_state_is_error():
    """ErrorPart 默认 state=error（错误场景专用）。"""
    p = ErrorPart()
    assert p.state == "error"
    assert p.type == "error"


def test_artifact_part_structured_optional():
    """ArtifactPart.structured 可以为 None（没有 platform 数据时）。"""
    p = ArtifactPart(body="x", structured=None)
    assert p.to_dict()["structured"] is None


def test_tool_call_part_icon_default():
    """ToolCallPart.icon 默认 ⚙️（前端有 fallback 渲染）。"""
    p = ToolCallPart()
    assert p.icon == "⚙️"
