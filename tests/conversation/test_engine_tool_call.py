"""对话引擎 tool_call 路径单元测试

v3 改造：对话 LLM 用 trigger_generation 函数调用触发创作阶段。
v3 已彻底移除 [GENERATE]{json}[/GENERATE] 文本标记协议——不再有"双路径"。
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.conversation.engine import TRIGGER_TOOL_SCHEMA
from src.conversation.schemas import ConversationContext, GenerateTrigger


# ─────────────── TRIGGER_TOOL_SCHEMA ───────────────


class TestTriggerToolSchema:
    """tool schema 基本形态验证"""

    def test_schema_is_list_with_one_function(self):
        """schema 是 list 类型，含 1 个 function 定义"""
        assert isinstance(TRIGGER_TOOL_SCHEMA, list)
        assert len(TRIGGER_TOOL_SCHEMA) == 1
        assert TRIGGER_TOOL_SCHEMA[0]["type"] == "function"
        assert TRIGGER_TOOL_SCHEMA[0]["function"]["name"] == "trigger_generation"

    def test_required_fields(self):
        """必填字段：action + topic"""
        params = TRIGGER_TOOL_SCHEMA[0]["function"]["parameters"]
        assert "action" in params["required"]
        assert "topic" in params["required"]

    def test_action_enum(self):
        """action 只能是 create/revise/expand"""
        action = TRIGGER_TOOL_SCHEMA[0]["function"]["parameters"]["properties"]["action"]
        assert set(action["enum"]) == {"create", "revise", "expand"}

    def test_length_enum_allows_empty(self):
        """length 字段允许空字符串（前端兜底场景）"""
        length = TRIGGER_TOOL_SCHEMA[0]["function"]["parameters"]["properties"]["length"]
        assert "" in length["enum"]

    def test_description_does_not_recommend_text_marker(self):
        """v3：tool description 不再让 LLM 用 [GENERATE] 文本标记"""
        desc = TRIGGER_TOOL_SCHEMA[0]["function"]["description"]
        # 不应包含"用 [GENERATE] 标记"的措辞（虽然可能有"不要"的措辞）
        # 这里只校验 description 提到了"必须用此函数调用"
        assert "必须用此函数调用" in desc or "唯一方式" in desc


# ─────────────── chunk 工厂 ───────────────


def _make_tool_chunk(name, args, chunk_id="call_1"):
    """构造带 tool_call 的 chunk"""
    tc = MagicMock()
    tc.name = name
    tc.args = args
    tc.id = chunk_id
    chunk = MagicMock()
    chunk.content = ""
    chunk.tool_calls = [tc]
    return chunk


def _make_text_chunk(text):
    chunk = MagicMock()
    chunk.content = text
    chunk.tool_calls = None
    return chunk


# ─────────────── 工具调用路径 ───────────────


class TestConversationEngineToolCallPath:
    """ConversationEngine.chat() 流式迭代时识别 tool_call 的能力"""

    @pytest.fixture
    def ctx(self):
        return ConversationContext(
            message="[目标平台：微博]\n直接生成",
            conversation_id="conv-1",
            user_id="user-1",
            history=[],
            platform="weibo",
            content_type="",
            preferences={"tone": "friendly", "length": "medium", "audience": "intermediate"},
        )

    @pytest.fixture
    def engine(self):
        from src.conversation.engine import ConversationEngine
        return ConversationEngine()

    @pytest.mark.asyncio
    async def test_tool_call_payload_extracted_from_chunk(self, engine, ctx):
        """tool_call 返回完整 dict args → 引擎识别 yield trigger。"""
        text_chunk = _make_text_chunk("好，信息够了，开始创作！")
        tool_chunk = _make_tool_chunk(
            "trigger_generation",
            {"action": "create", "topic": "测试主题", "platform": ""},
        )

        mock_llm = MagicMock()
        async def fake_astream(messages):
            yield text_chunk
            yield tool_chunk
        bound_llm = MagicMock()
        bound_llm.astream = fake_astream
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        with patch.object(engine, "_get_llm", AsyncMock(return_value=mock_llm)):
            events = []
            async for ev in engine.chat(ctx):
                events.append(ev)

        text_events = [e for e in events if e["type"] == "text"]
        trigger_events = [e for e in events if e["type"] == "trigger"]
        assert len(text_events) >= 1
        assert text_events[0]["content"] == "好，信息够了，开始创作！"

        assert len(trigger_events) == 1
        trigger = trigger_events[0]["data"]
        assert trigger["action"] == "create"
        assert trigger["topic"] == "测试主题"
        # ctx.platform="weibo" 应覆盖 trigger.platform
        assert trigger["platform"] == "weibo"
        # preferences 应平铺到 parameters 顶层
        assert "parameters" in trigger
        assert trigger["parameters"]["tone"] == "friendly"
        # tool_call 路径不应被标记为 fallback
        assert trigger.get("_fallback") is not True

    @pytest.mark.asyncio
    async def test_tool_call_with_args_as_str(self, engine, ctx):
        """tool_call args 是 JSON 字符串时也能解析。"""
        text_chunk = _make_text_chunk("好！")
        tool_chunk = _make_tool_chunk(
            "trigger_generation",
            json.dumps({"action": "create", "topic": "字符串参数", "platform": ""}, ensure_ascii=False),
            chunk_id="call_2",
        )

        mock_llm = MagicMock()
        async def fake_astream(messages):
            yield text_chunk
            yield tool_chunk
        bound_llm = MagicMock()
        bound_llm.astream = fake_astream
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        with patch.object(engine, "_get_llm", AsyncMock(return_value=mock_llm)):
            events = []
            async for ev in engine.chat(ctx):
                events.append(ev)

        trigger_events = [e for e in events if e["type"] == "trigger"]
        assert len(trigger_events) == 1
        assert trigger_events[0]["data"]["topic"] == "字符串参数"

    @pytest.mark.asyncio
    async def test_tool_call_with_empty_topic_falls_back(self, engine, ctx):
        """tool_call 缺 topic（required）→ engine 忽略 tool_call
        → 走到兜底分支（用户消息含"生成"关键词）→ 产出 _fallback=True 的 trigger。"""
        text_chunk = _make_text_chunk("好！")
        tool_chunk = _make_tool_chunk(
            "trigger_generation",
            {"action": "create"},  # 缺 topic
            chunk_id="call_3",
        )

        mock_llm = MagicMock()
        async def fake_astream(messages):
            yield text_chunk
            yield tool_chunk
        bound_llm = MagicMock()
        bound_llm.astream = fake_astream
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        with patch.object(engine, "_get_llm", AsyncMock(return_value=mock_llm)):
            events = []
            async for ev in engine.chat(ctx):
                events.append(ev)

        # tool_call 路径不应产出非 fallback 触发器
        tool_path_triggers = [
            e for e in events
            if e["type"] == "trigger" and not e["data"].get("_fallback")
        ]
        assert tool_path_triggers == []

        # 兜底路径可能产出 _fallback=True 的 trigger（用户消息含"生成"关键词）
        fallback_triggers = [
            e for e in events
            if e["type"] == "trigger" and e["data"].get("_fallback")
        ]
        assert len(fallback_triggers) == 1

    @pytest.mark.asyncio
    async def test_other_tool_call_does_not_set_trigger_via_tool_path(self, engine, ctx):
        """非 trigger_generation 函数调用被忽略 → tool_call 路径不产 trigger。"""
        text_chunk = _make_text_chunk("好！")
        tool_chunk = _make_tool_chunk(
            "unrelated_function",
            {"foo": "bar"},
            chunk_id="call_4",
        )

        mock_llm = MagicMock()
        async def fake_astream(messages):
            yield text_chunk
            yield tool_chunk
        bound_llm = MagicMock()
        bound_llm.astream = fake_astream
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        with patch.object(engine, "_get_llm", AsyncMock(return_value=mock_llm)):
            events = []
            async for ev in engine.chat(ctx):
                events.append(ev)

        non_fallback_triggers = [
            e for e in events
            if e["type"] == "trigger" and not e["data"].get("_fallback")
        ]
        assert non_fallback_triggers == []

    @pytest.mark.asyncio
    async def test_text_marker_is_no_longer_parsed(self, engine, ctx):
        """v3 改造：tool_call 解析失败 + 文本含 [GENERATE] 标记 → 标记路径不再触发。
        标记协议已彻底废弃，文本里的 [GENERATE]{json}[/GENERATE] 会被当作普通文本流过。
        兜底分支可能产出 _fallback=True 的 trigger（用户消息含生成关键词）。"""
        tc = MagicMock()
        tc.name = "trigger_generation"
        tc.args = 12345  # 非法类型，解析失败
        tc.id = "call_5"
        tool_chunk = MagicMock()
        tool_chunk.content = ""
        tool_chunk.tool_calls = [tc]

        # 即使文本里包含 [GENERATE] 标记，v3 也不再解析它
        marker_text = (
            "好的，开始创作！\n"
            "[GENERATE]\n"
            '{"action": "create", "topic": "文本标记主题", "platform": ""}\n'
            "[/GENERATE]"
        )
        marker_chunk = _make_text_chunk(marker_text)

        mock_llm = MagicMock()
        async def fake_astream(messages):
            yield tool_chunk
            yield marker_chunk
        bound_llm = MagicMock()
        bound_llm.astream = fake_astream
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        with patch.object(engine, "_get_llm", AsyncMock(return_value=mock_llm)):
            events = []
            async for ev in engine.chat(ctx):
                events.append(ev)

        # 关键断言：标记路径不应产出 topic="文本标记主题" 的 trigger（已废弃）
        marker_path_triggers = [
            e for e in events
            if e["type"] == "trigger"
            and e["data"].get("topic") == "文本标记主题"
        ]
        assert marker_path_triggers == []

        # 兜底分支可能产出 _fallback=True trigger（用户消息含生成关键词）
        # 这个 _fallback trigger 的 topic 不应是 "文本标记主题"，而是兜底提取的结果
        fallback_triggers = [
            e for e in events
            if e["type"] == "trigger" and e["data"].get("_fallback")
        ]
        # 兜底会清理掉文本里的 [GENERATE] 标记当作普通文本
        assert all(
            t["data"].get("topic") != "文本标记主题"
            for t in fallback_triggers
        )

    @pytest.mark.asyncio
    async def test_pure_text_marker_in_text_only_no_longer_triggers(self, engine, ctx):
        """v3 改造：纯文本里包含 [GENERATE]{json}[/GENERATE] 但 LLM 没调用工具
        → 不应触发 trigger_generation 路径。"""
        marker_text = (
            "好的，开始创作！\n"
            "[GENERATE]\n"
            '{"action": "create", "topic": "纯文本标记主题", "platform": ""}\n'
            "[/GENERATE]"
        )
        text_chunk = _make_text_chunk(marker_text)

        mock_llm = MagicMock()
        async def fake_astream(messages):
            yield text_chunk
        bound_llm = MagicMock()
        bound_llm.astream = fake_astream
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        with patch.object(engine, "_get_llm", AsyncMock(return_value=mock_llm)):
            events = []
            async for ev in engine.chat(ctx):
                events.append(ev)

        # tool_call 路径不应产出 topic="纯文本标记主题" 的 trigger
        marker_topic_triggers = [
            e for e in events
            if e["type"] == "trigger"
            and e["data"].get("topic") == "纯文本标记主题"
        ]
        assert marker_topic_triggers == []


# ─────────────── GenerateTrigger.from_dict 健壮性 ───────────────


class TestGenerateTriggerFromDict:
    """schema 解析的基本健壮性"""

    def test_minimal_payload(self):
        t = GenerateTrigger.from_dict({"action": "create", "topic": "测试"})
        assert t.action == "create"
        assert t.topic == "测试"
        assert t.platform == ""
        assert t.style == ""
        assert t.keywords == []

    def test_full_payload(self):
        t = GenerateTrigger.from_dict({
            "action": "revise",
            "topic": "改写",
            "platform": "weibo",
            "style": "幽默",
            "length": "short",
            "tone": "humorous",
            "audience": "beginner",
            "keywords": ["AI", "测试"],
            "revision_focus": "把开头改得更吸引人",
        })
        assert t.action == "revise"
        assert t.revision_focus == "把开头改得更吸引人"
        assert t.keywords == ["AI", "测试"]

    def test_default_action(self):
        t = GenerateTrigger.from_dict({"topic": "测试"})
        assert t.action == "create"