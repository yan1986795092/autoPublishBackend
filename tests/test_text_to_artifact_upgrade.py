"""text→artifact 升级函数单测 — Task 2 Step 4

参考 docs/superpowers/plans/2026-07-13-message-parts-schema-driven-rendering.md §Task 2.

注意：实际生产环境升级发生在 stream_core.py:283（result 事件触发），
本单测覆盖通用升级函数 should_upgrade_text_to_artifact / upgrade_text_to_artifact。
"""

from src.api.chat_message_ui_v2 import (
    should_upgrade_text_to_artifact,
    upgrade_text_to_artifact,
    UPGRADE_MIN_CONTENT_LENGTH,
)


def test_short_text_not_upgraded():
    """短正文（<300 字）不升级，可能是临时回答。"""
    p = {"type": "text", "content": "短回答", "agent": "WritingAgent"}
    assert should_upgrade_text_to_artifact(p, "WritingAgent") is False


def test_writing_agent_long_text_upgraded():
    """WritingAgent 长正文（≥300 字）触发升级。"""
    p = {"type": "text", "content": "x" * UPGRADE_MIN_CONTENT_LENGTH, "agent": "WritingAgent"}
    assert should_upgrade_text_to_artifact(p, "WritingAgent") is True


def test_reasoning_part_not_upgraded():
    """reasoning part 即使内容长也不升级。"""
    p = {"type": "reasoning", "content": "x" * 500, "agent": "ManagerAgent"}
    assert should_upgrade_text_to_artifact(p, "ManagerAgent") is False


def test_artifact_part_not_upgraded():
    """已经是 artifact 的不重复升级。"""
    p = {"type": "artifact", "body": "x" * 500, "artifact_type": "blog"}
    assert should_upgrade_text_to_artifact(p, "WritingAgent") is False


def test_non_eligible_agent_not_upgraded():
    """非内容生成类 agent 不升级（如 ManagerAgent 编排）。"""
    p = {"type": "text", "content": "x" * 500, "agent": "ManagerAgent"}
    assert should_upgrade_text_to_artifact(p, "ManagerAgent") is False


def test_part_without_agent_falls_back_to_agent_type_arg():
    """part 自身缺 agent 字段时，用 agent_type 参数兜底。"""
    p = {"type": "text", "content": "x" * 500}
    assert should_upgrade_text_to_artifact(p, "FixAgent") is True
    assert should_upgrade_text_to_artifact(p, "ManagerAgent") is False


def test_non_dict_input_safe():
    """防御：非 dict 输入返回 False，不抛异常。"""
    assert should_upgrade_text_to_artifact("not a dict", "WritingAgent") is False  # type: ignore[arg-type]
    assert should_upgrade_text_to_artifact(None, "WritingAgent") is False  # type: ignore[arg-type]
    assert should_upgrade_text_to_artifact(123, "WritingAgent") is False  # type: ignore[arg-type]


def test_upgrade_preserves_id_and_converts_content_to_body():
    """升级保留 part.id，content → body。"""
    p = {"id": "part_abc", "type": "text", "content": "hello", "state": "done"}
    a = upgrade_text_to_artifact(p, title="T", platform="wechat")
    assert a["id"] == "part_abc"
    assert a["type"] == "artifact"
    assert a["body"] == "hello"
    assert a["title"] == "T"
    assert a["platform"] == "wechat"
    assert "content" not in a
    assert a["state"] == "done"


def test_upgrade_uses_default_artifact_type_when_unspecified():
    """默认 artifact_type = blog。"""
    p = {"type": "text", "content": "hello"}
    a = upgrade_text_to_artifact(p)
    assert a["artifact_type"] == "blog"


def test_upgrade_preserves_structured_metadata():
    """structured 字段透传（封面/标签/metrics 等 platform-specific 数据）。"""
    p = {"type": "text", "content": "x" * 500, "structured": {"tags": ["AI", "tech"]}}
    a = upgrade_text_to_artifact(p)
    assert a["structured"] == {"tags": ["AI", "tech"]}


def test_upgrade_infers_social_post_from_platform():
    """platform=xiaohongshu → artifact_type=social_post（chat_message_ui_v2._infer_artifact_type 已支持）。"""
    from src.api.chat_message_ui_v2 import _infer_artifact_type
    assert _infer_artifact_type({"platform": "xiaohongshu"}) == "social_post"
    assert _infer_artifact_type({"platform": "wechat"}) == "blog"
    assert _infer_artifact_type({"content_type": "code"}) == "code"
    assert _infer_artifact_type({}) == "blog"  # 兜底