"""migrate_legacy_result_to_parts 单测 — Task 3

参考 docs/superpowers/plans/2026-07-13-message-parts-schema-driven-rendering.md §Task 3.

核心：trigger-only 场景不构造 artifact，由 Phase 4 的 is_generation 字段处理。
"""

import json
from src.api.chat_message_ui_v2 import migrate_legacy_result_to_parts


def test_empty_input_returns_empty_parts():
    """空输入 → 空 parts。"""
    assert migrate_legacy_result_to_parts(None) == []
    assert migrate_legacy_result_to_parts("") == []


def test_invalid_json_returns_empty_parts():
    """非法 JSON 不抛异常，返回空 parts。"""
    assert migrate_legacy_result_to_parts("not json") == []
    assert migrate_legacy_result_to_parts("{invalid") == []


def test_non_dict_json_returns_empty_parts():
    """JSON 是数组/字符串/数字 → 返回空 parts。"""
    assert migrate_legacy_result_to_parts(json.dumps([1, 2, 3])) == []
    assert migrate_legacy_result_to_parts(json.dumps("string")) == []
    assert migrate_legacy_result_to_parts(json.dumps(123)) == []


def test_output_only_produces_artifact():
    """正常 output → 1 个 artifact part。"""
    result = json.dumps({
        "output": {
            "full_text": "hello world",
            "title": "T",
            "platform": "wechat",
        }
    })
    parts = migrate_legacy_result_to_parts(result)
    assert len(parts) == 1
    assert parts[0]["type"] == "artifact"
    assert parts[0]["body"] == "hello world"
    assert parts[0]["title"] == "T"


def test_trigger_only_no_artifact_part_added():
    """trigger-only 场景（无 output）→ 不构造 artifact（Phase 4 is_generation 处理）。"""
    result = json.dumps({
        "trigger": {
            "action": "create_blog",
            "platform": "wechat",
            "topic": "AI",
        }
    })
    parts = migrate_legacy_result_to_parts(result)
    # 不应有任何 artifact part
    assert all(p["type"] != "artifact" for p in parts)
    # 但也不应抛异常 — parts 应为空列表或仅包含其他类型的 part
    assert isinstance(parts, list)


def test_full_legacy_result_includes_all_part_types():
    """完整 legacy result（reasoning + stages + output + plan）→ 4 类 parts。"""
    result = json.dumps({
        "_reasoningChunks": {"ManagerAgent|0": [{"content": "thinking..."}]},
        "_stages": [{"agent": "WritingAgent", "label": "写作", "duration": 1.5}],
        "output": {"full_text": "body text", "platform": "wechat"},
        "plan_json": json.dumps({"tasks": [{"agent_type": "WritingAgent"}]}),
    })
    parts = migrate_legacy_result_to_parts(result)
    types = {p["type"] for p in parts}
    assert "reasoning" in types
    assert "tool_call" in types
    assert "artifact" in types
    assert "plan" in types


def test_output_with_only_metadata_no_artifact():
    """output 存在但 body 为空 → 不构造空 artifact。"""
    result = json.dumps({"output": {"title": "T", "platform": "wechat"}})
    parts = migrate_legacy_result_to_parts(result)
    # body 为空，artifact 不应被构造
    assert all(p["type"] != "artifact" for p in parts)


def test_xiaohongshu_inferred_as_social_post():
    """platform=xiaohongshu → artifact_type=social_post。"""
    result = json.dumps({
        "output": {"full_text": "xhs post", "platform": "xiaohongshu"}
    })
    parts = migrate_legacy_result_to_parts(result)
    assert parts[0]["artifact_type"] == "social_post"