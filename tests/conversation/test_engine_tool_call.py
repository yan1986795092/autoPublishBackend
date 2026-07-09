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
        """v3：tool description 必须明确告诉 LLM 必须用工具调用，且 [GENERATE] 已废弃"""
        desc = TRIGGER_TOOL_SCHEMA[0]["function"]["description"]
        # 关键断言：description 强调必须调用工具
        assert "必须调用" in desc or "实际执行工具调用" in desc, (
            "tool description 必须明确告诉 LLM 触发创作必须用工具调用"
        )
        # 关键断言：description 明确 [GENERATE] 已废弃
        assert "已废弃" in desc, (
            "tool description 必须明确说明 [GENERATE] 文本标记已废弃"
        )


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
        → 可能走兜底分支（如果有有效 topic）→ 否则转追问。

        v3 关键修复：兜底 topic 提取必须避免纯动作词，
        因此当 ctx.message 是纯动作词且没有对话历史时，**应当转追问**而不是产出无效 trigger。
        """
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

        # 关键：tool_call 路径不应产出非 fallback 触发器（缺 topic 不合法）
        tool_path_triggers = [
            e for e in events
            if e["type"] == "trigger" and not e["data"].get("_fallback")
        ]
        assert tool_path_triggers == []

        # 当前 ctx.message="[目标平台：微博]\n直接生成"，纯动作词 + 无历史 →
        #   兜底 topic 提取失败，转追问（不产出 fallback trigger）
        fallback_triggers = [
            e for e in events
            if e["type"] == "trigger" and e["data"].get("_fallback")
        ]
        # 不一定有 fallback trigger（取决于 topic 是否能提取出来）
        # 但即使有，也不应是纯动作词
        for t in fallback_triggers:
            assert t["data"].get("topic") not in ("", "直接", "直接生成"), \
                f"兜底触发器 topic 不应是纯动作词: {t['data'].get('topic')!r}"

        # 应当有 text_cleanup 事件（让前端展示追问话）
        cleanup_events = [e for e in events if e["type"] == "text_cleanup"]
        # 至少有一个 text_cleanup（追问话） OR trigger（兜底成功）
        assert cleanup_events or fallback_triggers

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


# ─────────────── 兜底分支 topic 提取（v3 关键修复）────────────────


class TestFallbackTopicExtraction:
    """当 LLM 没调工具但用户要求生成时，兜底 topic 提取的正确性。

    场景：LLM 反复说"开始创作"但拒绝调工具 → 后端走兜底分支，
    必须能从 LLM 输出 / 对话历史里提取出正确的 topic（而不是"直接开始"）。
    """

    @pytest.fixture
    def engine(self):
        from src.conversation.engine import ConversationEngine
        return ConversationEngine()

    @pytest.fixture
    def ctx_with_history(self):
        """模拟用户多轮对话：先问主题，再要求生成"""
        return ConversationContext(
            message="[目标平台：公众号]\n直接开始写",
            conversation_id="conv-test",
            user_id="user-test",
            history=[
                {"role": "user", "content": "[目标平台：公众号]\n聊聊 AI 工具如何改变了我的日常工作流"},
                {"role": "assistant", "content": "好的，这个话题很有意思..."},
                {"role": "user", "content": "[目标平台：公众号]\n开始创作"},
                {"role": "assistant", "content": "好的，信息够了，开始创作！再确认一下..."},
            ],
            platform="wechat",
            content_type="",
            preferences={"tone": "friendly", "length": "medium", "audience": "intermediate"},
        )

    @pytest.mark.asyncio
    async def test_llm_explicit_topic_signal_in_output(self, engine, ctx_with_history):
        """LLM 输出里包含'就以X为主题'信号时，topic 应当取 X。
        复现真实场景：LLM 输出 '就以AI工具如何改变日常工作流为主题，采用第一人称个人经验分享的风格来写'"""
        # LLM 自然语言说"开始创作"+ 显式说主题
        text_chunk = _make_text_chunk(
            '好的，那我来推断一下最合理的主题——就以"AI工具如何改变日常工作流"为主题，'
            '采用第一人称个人经验分享的风格来写。开始创作！'
        )

        mock_llm = MagicMock()
        async def fake_astream(messages):
            yield text_chunk
        bound_llm = MagicMock()
        bound_llm.astream = fake_astream
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        with patch.object(engine, "_get_llm", AsyncMock(return_value=mock_llm)):
            events = []
            async for ev in engine.chat(ctx_with_history):
                events.append(ev)

        # 兜底分支应触发，topic 应来自 LLM 输出的显式信号（去掉引号）
        fallback_triggers = [
            e for e in events
            if e["type"] == "trigger" and e["data"].get("_fallback")
        ]
        assert len(fallback_triggers) == 1
        topic = fallback_triggers[0]["data"]["topic"]
        assert "AI工具" in topic or "日常工作流" in topic, \
            f"topic 应该是 LLM 显式说的'AI工具如何改变日常工作流'，实际是 {topic!r}"
        # 不应该是动作词
        assert topic not in ("直接", "开始写", "开始", "直接开始")

    @pytest.mark.asyncio
    async def test_topic_from_history_when_llm_no_signal(self, engine, ctx_with_history):
        """LLM 输出里没有'就以X为主题'信号 → 从对话历史找最近实质主题
        排除 '直接生成''开始创作' 这种动作词消息。"""
        # LLM 只说开始创作但没显式主题
        text_chunk = _make_text_chunk("好的，信息够了，开始创作！")

        mock_llm = MagicMock()
        async def fake_astream(messages):
            yield text_chunk
        bound_llm = MagicMock()
        bound_llm.astream = fake_astream
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        with patch.object(engine, "_get_llm", AsyncMock(return_value=mock_llm)):
            events = []
            async for ev in engine.chat(ctx_with_history):
                events.append(ev)

        fallback_triggers = [
            e for e in events
            if e["type"] == "trigger" and e["data"].get("_fallback")
        ]
        assert len(fallback_triggers) == 1
        topic = fallback_triggers[0]["data"]["topic"]
        # 应该取历史里"聊聊 AI 工具..."那条
        assert "AI工具" in topic or "日常工作流" in topic, \
            f"topic 应该来自对话历史的实质内容，实际是 {topic!r}"

    @pytest.mark.asyncio
    async def test_topic_not_pure_action_words(self, engine, ctx_with_history):
        """兜底触发不应产出纯动作词 topic（如'直接''开始''直接开始'）。"""
        # 即使 LLM 输出简短、history 全是动作词，也应避免错误的纯动作 topic
        ctx = ConversationContext(
            message="[目标平台：公众号]\n直接",
            conversation_id="conv-test",
            user_id="user-test",
            history=[
                {"role": "user", "content": "[目标平台：公众号]\n直接"},
                {"role": "assistant", "content": "好的，收到！"},
                {"role": "user", "content": "[目标平台：公众号]\n直接生成"},
                {"role": "assistant", "content": "好的，信息够了，开始创作！"},
            ],
            platform="wechat",
            content_type="",
            preferences={},
        )

        text_chunk = _make_text_chunk("好的，信息够了，开始创作！")
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

        # 要么产出有效 topic 的 trigger，要么不触发（转追问）
        # 不应产出 topic="直接" 这种无效的纯动作词 trigger
        invalid_topic_triggers = [
            e for e in events
            if e["type"] == "trigger"
            and e["data"].get("topic") in ("直接", "直接开始", "开始", "开始写", "生成", "")
        ]
        assert invalid_topic_triggers == [], \
            f"不应产出纯动作词 topic 的 trigger，实际: {invalid_topic_triggers}"


# ─────────────── 安全网：LLM 就绪信号检测（_llm_ready_signal）────────────────


class TestSafetyNetLlmReadySignal:
    """对话引擎安全网：LLM 输出'够了/开始创作'但没调工具 → 自动触发 pipeline

    场景：用户回答了 LLM 的追问（如专辑类型、风格），LLM 说"信息够了，开始创作"
    但没调 trigger_generation 工具（可能因 Gateway 未透传 tool_calls）。
    引擎应进入兜底分支，从上下文提取 topic 后触发 pipeline。
    """

    @pytest.fixture
    def engine(self):
        from src.conversation.engine import ConversationEngine
        return ConversationEngine()

    @pytest.fixture
    def ctx_ready_to_generate(self):
        """模拟用户已回答 LLM 的追问，LLM 确认已就绪的场景"""
        return ConversationContext(
            message="[目标平台：微博]\n轻快的流行       走心治愈        独处时光",
            conversation_id="conv-safety",
            user_id="user-safety",
            history=[
                {"role": "user", "content": "[目标平台：微博]\n今日份小确幸：通勤路上听完了整张专辑"},
                {"role": "assistant", "content": "听起来是个很惬意的早晨呀！..."},
                {"role": "user", "content": "[目标平台：微博]\n微博    写一篇更丰富的内容"},
                {"role": "assistant", "content": "好嘞，明白了！...是什么类型的专辑呀？"},
            ],
            platform="weibo",
            content_type="",
            preferences={"tone": "friendly", "length": "medium", "audience": "intermediate"},
        )

    @pytest.mark.asyncio
    async def test_llm_ready_signal_triggers_fallback(self, engine, ctx_ready_to_generate):
        """LLM 说'信息够了，开始创作'但没调工具 → 安全网触发生成触发器

        模拟日志中出现的真实场景：LLM 输出
        '明白了！轻快流行 + 走心治愈 + 独处时光，这组合太棒了～信息够了，开始创作！'
        """
        text_chunk = _make_text_chunk(
            "明白了！轻快流行 + 走心治愈 + 独处时光，"
            "这组合太棒了～信息够了，开始创作！"
        )

        mock_llm = MagicMock()
        async def fake_astream(messages):
            yield text_chunk
        bound_llm = MagicMock()
        bound_llm.astream = fake_astream
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        with patch.object(engine, "_get_llm", AsyncMock(return_value=mock_llm)):
            events = []
            async for ev in engine.chat(ctx_ready_to_generate):
                events.append(ev)

        # 应触发兜底分支 → 产出 _fallback=True 的 trigger
        fallback_triggers = [
            e for e in events
            if e["type"] == "trigger" and e["data"].get("_fallback")
        ]
        assert len(fallback_triggers) == 1, (
            f"安全网应产出 1 个 fallback trigger，实际: {len(fallback_triggers)}"
        )
        trigger = fallback_triggers[0]["data"]
        assert trigger["action"] == "create"
        # topic 应从上下文提取（消息中的有效内容）
        assert trigger["topic"], "topic 不应为空"
        # topic 应从上下文提取（历史中的实质内容或当前消息）
        assert len(trigger["topic"]) >= 4, \
            f"topic 应有实质内容，实际: {trigger['topic']!r}"
        # platform 应从前端预选值覆盖
        assert trigger["platform"] == "weibo"
        # parameters 应包含 preferences 信息
        assert "parameters" in trigger
        assert trigger["parameters"]["tone"] == "friendly"

    @pytest.mark.asyncio
    async def test_llm_ready_signal_no_followup(self, engine, ctx_ready_to_generate):
        """LLM 输出包含就绪信号但不含追问特征 → 安全网触发

        验证：_llm_ready_signal 的正确性取决于
        looks_like_followup_question 返回 False + looks_like_ready_to_generate 返回 True。
        """
        # "信息够了"（就绪信号）但没有问号（非追问）
        text_chunk = _make_text_chunk("好的，信息够了，开始创作吧！")

        mock_llm = MagicMock()
        async def fake_astream(messages):
            yield text_chunk
        bound_llm = MagicMock()
        bound_llm.astream = fake_astream
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        with patch.object(engine, "_get_llm", AsyncMock(return_value=mock_llm)):
            events = []
            async for ev in engine.chat(ctx_ready_to_generate):
                events.append(ev)

        fallback_triggers = [
            e for e in events
            if e["type"] == "trigger" and e["data"].get("_fallback")
        ]
        assert len(fallback_triggers) >= 1, "安全网应产出 fallback trigger"

    @pytest.mark.asyncio
    async def test_normal_greeting_not_triggered(self, engine, ctx_ready_to_generate):
        """普通问候不含就绪信号 → 安全网不应触发"""
        text_chunk = _make_text_chunk("您好！很高兴见到你！有什么可以帮你的吗？")

        mock_llm = MagicMock()
        async def fake_astream(messages):
            yield text_chunk
        bound_llm = MagicMock()
        bound_llm.astream = fake_astream
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        with patch.object(engine, "_get_llm", AsyncMock(return_value=mock_llm)):
            events = []
            async for ev in engine.chat(ctx_ready_to_generate):
                events.append(ev)

        # 不应产出 trigger
        triggers = [e for e in events if e["type"] == "trigger"]
        assert triggers == [], f"问候文本不应触发 pipeline: {triggers}"

    @pytest.mark.asyncio
    async def test_user_says_generate_still_works(self, engine):
        """用户说'直接生成'且 LLM 没调工具 → 仍通过 _wants_generation 进入兜底

        验证：_wants_generation 路径不受安全网影响（互不阻塞）
        """
        ctx = ConversationContext(
            message="直接生成",
            conversation_id="conv-normal",
            user_id="user-normal",
            history=[
                {"role": "user", "content": "聊聊 AI 工具如何改变了工作流"},
                {"role": "assistant", "content": "很好的话题！"},
            ],
            platform="weibo",
            content_type="",
            preferences={},
        )
        text_chunk = _make_text_chunk("好的，开始创作！")

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

        fallback_triggers = [
            e for e in events
            if e["type"] == "trigger" and e["data"].get("_fallback")
        ]
        assert len(fallback_triggers) >= 1
        topic = fallback_triggers[0]["data"]["topic"]
        assert "AI" in topic or "工作流" in topic, \
            f"topic 应从历史提取: {topic!r}"