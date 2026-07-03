"""对话引擎核心

每次用户消息：
1. 加载对话历史和用户偏好
2. 调用对话 LLM 流式生成回复（v3: bind_tools 让 LLM 用 trigger_generation 函数触发创作）
3. 提取 tool_call → 返回 trigger（不再支持文本 [GENERATE] 标记）
4. 返回对话文本 + 可选的触发指令
"""

from __future__ import annotations
import json
import logging
import re
from typing import AsyncGenerator, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from .schemas import ConversationContext, GenerateTrigger
from .intent_detector import IntentDetector
from src.core.model_router import ModelRouter
from src.config import get_settings
from src.memory.manager import memory_manager

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Trigger tool schema（OpenAI function calling 格式）
# ──────────────────────────────────────────────────────────────────────
# 对话 LLM 触发创作阶段的唯一方式。
# v3 已彻底移除旧的 [GENERATE]{json}[/GENERATE] 文本标记协议——LLM 必须
# 通过调用 trigger_generation 函数来触发，引擎不再尝试解析文本中的标记。
#
# 字段说明：
# - action: create (新创作) / revise (修改已有内容) / expand (扩写)
# - topic: 创作主题（必填，revise 时可填历史主题）
# - platform: 平台 ID（如 xiaohongshu / weibo），可以留空让系统用前端预选值兜底
# - style: 写作风格 / 体裁 ID（copywriting / tutorial 等）
# - length / tone / audience: 篇幅/语调/读者
# - keywords: 关键词列表
# - revision_focus: 仅 revise 时必填，用户的具体修改要求
TRIGGER_TOOL_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "trigger_generation",
            "description": (
                "当信息收集完毕、可以开始创作时调用此函数。"
                "这是触发创作 pipeline 的唯一方式——必须用此函数调用，"
                "**不要**在回复文本里嵌入 `[GENERATE]{json}[/GENERATE]` 标记"
                "（这种格式已废弃，引擎不再解析）。"
                "调用前你应该先在文本中简短确认（如'好，信息够了，开始创作！'），"
                "然后立即调用此函数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "revise", "expand"],
                        "description": "create = 创作新内容；revise = 修改已有内容；expand = 扩写",
                    },
                    "topic": {
                        "type": "string",
                        "description": "用户要创作的主题（必填）",
                    },
                    "platform": {
                        "type": "string",
                        "description": (
                            "目标平台 ID（xiaohongshu / wechat / weibo / douyin / "
                            "bilibili / zhihu 等）。如果系统已告知用户已选平台，"
                            "可以传空字符串让系统用前端预选值兜底"
                        ),
                    },
                    "style": {
                        "type": "string",
                        "description": "写作风格/体裁描述（free text），如'幽默风趣'",
                    },
                    "length": {
                        "type": "string",
                        "enum": ["", "short", "medium", "long"],
                        "description": "篇幅：short=短篇 / medium=中篇 / long=长篇",
                    },
                    "tone": {
                        "type": "string",
                        "description": "语调（friendly / professional / humorous / warm）",
                    },
                    "audience": {
                        "type": "string",
                        "description": "目标读者（beginner / intermediate / expert）",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "关键词列表（可选）",
                    },
                    "revision_focus": {
                        "type": "string",
                        "description": "仅 revise 时填：用户的具体修改要求（如'把开头改得更吸引人'）",
                    },
                },
                "required": ["action", "topic"],
            },
        },
    },
]


# 对话 LLM System Prompt
SYSTEM_PROMPT = """你是一个友好的内容创作助手，通过自然对话帮助用户创作内容。

## 核心原则
- 像真人一样自然交流，不要一次性问所有问题
- 从问候开始，根据用户回复逐步深入了解需求
- 保持友好、热情的语气

## 信息收集
通过自然对话了解以下信息（非全部必填，由你自行判断需要哪些）：
- topic: 创作主题
- platform: 目标平台（xiaohongshu/wechat/douyin/weibo/bilibili/zhihu 等）
- style: 写作风格
- length: 篇幅（short/medium/long）
- tone: 语调（friendly/professional/humorous/warm）
- audience: 目标读者（beginner/intermediate/expert）

## 决策规则 — 必须严格遵守

你是「需求收集者」，**不是**「写作者」。你**绝对不可以**自己写正文、标题、章节。

### 规则 A：判断用户当前是否在要求"生成新内容"
判断关键词（不限于）：写、生成、写一份、重新写、再来一份、重新生成、重写、扩写、改写。

- 用户**没有**明确要生成 → 继续对话、追问
- 用户**明确**要生成但信息不足（缺主题/平台/风格） → 追问 1-2 个问题
- 用户**明确**要生成且信息充分 → 调用 `trigger_generation` 工具函数

### 规则 B：禁止越权（核心约束）
**绝对禁止**在回复中包含任何形式的：
- 以 `## 思考：` / `## 正文：` / `## 标题：` / `## 引言：` / `## 结论：` / `## 章节：` 开头的结构化元描述
- 任何超过 3 行的"标题 + 引言 + 多章节 + 结论"框架
- Markdown 标题（`#` / `##`）超过 1 个

### 规则 C：触发方式（结构化 Function Calling）
确认要生成时，**调用 `trigger_generation` 工具函数**（你已自动获得此工具）。
这是触发创作 pipeline 的最可靠方式——比在文本里嵌入 `[GENERATE]{json}[/GENERATE]` 标记更稳定。

正确流程示例：
- 用户：「直接生成」
- 你：先在文本中简短确认（一句话即可），例如"好，信息够了，开始创作！"
- 然后**立即**调用 `trigger_generation` 工具函数，传入参数：`{action, topic, platform, tone, length, audience, ...}`

注意：
1. 工具调用**与文本回复可以同时出现**——文本是确认话，工具调用是触发动作
2. 工具调用的 topic 字段必须**填实际创作主题**（用户没明确时填你推断的最合理主题）
3. 平台/篇幅/语调/读者等用户已选的可以直接从系统提示里读取，**不要凭空编造**
4. revise 模式（用户对已有内容做修改）必须填 `revision_focus` 字段

### 规则 1：简单修改（改词、调语气、改 1-2 句）
→ 直接改好回复，不调用工具。例：
用户：把"很好"改成"非常好"
你：好的，改好了：原来的"...很好..." → "...非常好..."

### 规则 3：润色/修改已有内容
用户对**已经生成好的内容**提修改意见（如"润色一下""开头改得更吸引人""第二段太啰嗦了"），
→ 调用 `trigger_generation`，参数：`action="revise"`、`revision_focus=用户的具体要求`。

### 规则 4：大篇幅修改（扩写、改风格、重写）
→ 调用 `trigger_generation`，参数：`action="revise"`、`revision_focus=用户要求`。

### 规则 5：信息不足
→ 继续追问，不调用工具。一次只问 1-2 个问题，不要全部列出来。

### 关键：当信息收集完毕时
用户回答了最后一个问题后，按以下顺序：
1. 输出一句确认话（如"好，信息够了，开始创作！"）
2. **立即**调用 `trigger_generation` 工具
不准在触发前写任何内容正文。

## 强制约束：不要在文本里输出 `[GENERATE]` 标记

- **不要**用 `[GENERATE]{json}[/GENERATE]` 文本格式触发——这会失败，引擎已经不再解析这种格式。
- **唯一**的触发方式是调用 `trigger_generation` 工具。
- 如果你不知道当前是否支持工具调用：**先假设支持**，直接调用工具；
  如果调用报错，会进入兜底流程由系统代你触发。
- 不要试图在文本里模仿工具参数输出（如 `{action: create, ...}`）——同样会被忽略。
"""
# 注意：System Prompt 是纯静态指令，不夹带动态数据。
# 用户偏好、历史记录等上下文通过单独的 USER 消息注入。


def _log_llm_messages(messages: list) -> None:
    """打印 LLM 收到的完整 messages（绕过 structlog 渲染）"""
    _log = logging.getLogger("conversation.llm")
    parts = [f"\n{'='*60}", "  LLM 输入 [ConversationAgent]", f"{'='*60}"]
    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "").upper()
        content = msg.content or ""
        parts.append(f"\n──[{role}]──\n{content}")
    parts.append(f"\n{'='*60}")
    text = "".join(parts)
    try:
        for i in range(0, len(text), 2000):
            _log.info("📋 %s", text[i:i+2000])
    except Exception:
        pass


def _build_trigger_data(trigger: "GenerateTrigger", preferences: dict | None = None) -> dict:
    """从 GenerateTrigger 构造触发器 dict（避免重复代码）

    preferences 中的 creativity / length / tone / audience 会作为 fallback：
    - 若 LLM 没在 trigger 里填某个字段（trigger.tone == ""），用 preference 兜底
    - 同时原样写入顶层 `parameters`，前端 runStreaming 用来转发给 /agent/stream
    """
    _t = {
        "action": trigger.action,
        "topic": trigger.topic,
        "platform": trigger.platform,
        "style": trigger.style,
        "length": trigger.length,
        "tone": trigger.tone,
        "audience": trigger.audience,
        "keywords": trigger.keywords,
        "revision_focus": trigger.revision_focus,
        "_prompt": trigger.to_prompt(),
        "pipeline_request": {
            "action": trigger.action,
            "topic": trigger.topic,
            "platform": trigger.platform,
            "style": trigger.style,
            "length": trigger.length,
            "tone": trigger.tone,
            "audience": trigger.audience,
            "keywords": trigger.keywords,
            "revision_focus": trigger.revision_focus,
        },
    }
    if preferences is not None:
        # 把 preferences 平铺到 trigger 顶层（triggerData.parameters
        # 会被前端 runStreaming 当成第三个参数发到 /agent/stream）
        # 注意：用 `is not None` 判断而不是 truthiness，否则空 dict 会被吞掉，
        # 导致前端拿不到 parameters 字段，runStreaming 走 fallback。
        _t["parameters"] = dict(preferences)
    return _t


class ConversationEngine:
    """对话引擎 — 单次消息处理"""

    # 网关可用性缓存 TTL（秒）
    _GATEWAY_CACHE_TTL = 30

    def __init__(self):
        self._router = ModelRouter()
        self._llm_cache: dict = {}
        self._gateway_up: bool | None = None
        self._gateway_checked_at: float = 0.0

    async def _get_llm(self) -> object:
        """获取对话 LLM 实例（异步，带网关缓存）

        优先 Gateway（model=auto 走 AutoRouter），
        Gateway 不可用时降级为直连（用第一个可用的 API key）。
        """
        # 先检测 Gateway 是否可达（带缓存）
        gateway_up = await self._check_gateway()
        if gateway_up:
            logger.info("对话引擎 → 使用 Gateway (auto-route)")
            return self._router.get_chat_model("ConversationAgent")

        # Gateway 不可用 → 直连降级
        settings = get_settings()
        for provider, key_attr in [
            ("deepseek", "deepseek_api_key"),
            ("openai", "openai_api_key"),
            ("zhipu", "zhipu_api_key"),
            ("qwen", "qwen_api_key"),
        ]:
            api_key = getattr(settings, key_attr, None)
            if api_key:
                base_urls = {
                    "deepseek": "https://api.deepseek.com/v1",
                    "openai": "https://api.openai.com/v1",
                    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
                    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                }
                model_names = {
                    "deepseek": "deepseek-chat",
                    "openai": "gpt-4o-mini",
                    "zhipu": "glm-4-flash",
                    "qwen": "qwen-plus",
                }
                logger.info(
                    "对话引擎 → Gateway 不可用，降级直连 %s",
                    provider,
                )
                return ChatOpenAI(
                    model=model_names.get(provider, "deepseek-chat"),
                    api_key=api_key,
                    base_url=base_urls.get(provider),
                    temperature=0.7,
                    max_tokens=2048,
                    streaming=True,
                    timeout=30.0,
                )

        # 全部不可用 → 抛出明确的错误
        raise ConnectionError(
            "LLM Gateway 未运行，且未配置任何 API key。"
            "请启动 Gateway (llm-gateway) 或在 .env 中配置 deepseek_api_key。"
        )

    async def _check_gateway(self) -> bool:
        """检查 LLM Gateway 是否可用（异步连接测试 + 30 秒缓存）"""
        import asyncio
        now = asyncio.get_event_loop().time()
        # 缓存未过期且上次检查有效 → 直接返回缓存值
        if self._gateway_up is not None and (now - self._gateway_checked_at) < self._GATEWAY_CACHE_TTL:
            return self._gateway_up
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("localhost", 9010), timeout=1.0,
            )
            writer.close()
            await writer.wait_closed()
            self._gateway_up = True
        except (OSError, asyncio.TimeoutError, Exception):
            self._gateway_up = False
        self._gateway_checked_at = now
        return self._gateway_up is True

    async def chat(
        self,
        ctx: ConversationContext,
    ) -> AsyncGenerator[dict, None]:
        """处理一条用户消息，流式生成回复

        Yields:
            {"type": "text", "content": str}         — 文本片段
            {"type": "text_cleanup", "content": str} — 移除标记后的完整文本
            {"type": "trigger", "data": dict}        — 生成触发指令
        """
        llm = await self._get_llm()

        # 1. System Prompt（纯静态，不夹带动态数据）
        system = SystemMessage(content=SYSTEM_PROMPT)
        messages = [system]

        # 1.5 注入用户已选的平台/体裁（前端 PlatformSelector 传入）
        # 让 LLM 知道这些信息已经确认，不要再重复询问
        _known_parts = []
        if ctx.platform:
            _name = {"xiaohongshu": "小红书", "wechat": "公众号", "weibo": "微博",
                     "douyin": "抖音", "bilibili": "B站", "zhihu": "知乎"}.get(ctx.platform, ctx.platform)
            _known_parts.append(f"发布平台：{_name}")
        if ctx.content_type:
            _type_labels = {"copywriting": "产品文案", "brand-story": "品牌故事",
                           "ad-copy": "广告文案", "fiction": "小说故事",
                           "script": "剧本脚本", "tech-doc": "技术文档",
                           "tutorial": "教程指南", "email": "邮件营销"}
            _label = _type_labels.get(ctx.content_type, ctx.content_type)
            _known_parts.append(f"写作体裁：{_label}")
        # 1.6 用户生成参数（创造力/篇幅/语调/读者）也作为已知偏好
        _pref_labels = {
            "creativity": ("创造力", {0: "保守", 0.5: "中等", 1: "发散"}),
            "length": ("篇幅", {"short": "短篇", "medium": "中篇", "long": "长篇"}),
            "tone": ("语调", {"friendly": "友好", "professional": "专业",
                             "humorous": "幽默", "warm": "温暖"}),
            "audience": ("目标读者", {"beginner": "初学者",
                                 "intermediate": "进阶", "expert": "专家"}),
        }
        if isinstance(ctx.preferences, dict):
            for key, (label, mapping) in _pref_labels.items():
                v = ctx.preferences.get(key)
                if v is None:
                    continue
                # 数字类用 mapping 取最接近，否则直接显示
                if isinstance(v, (int, float)) and key == "creativity":
                    # 取最接近档位的标签
                    nearest = min(mapping.keys(), key=lambda x: abs(x - float(v)))
                    _known_parts.append(f"{label}：{mapping[nearest]}({float(v):.1f})")
                elif v in mapping:
                    _known_parts.append(f"{label}：{mapping[v]}")
                else:
                    _known_parts.append(f"{label}：{v}")
        if _known_parts:
            messages.append(SystemMessage(content=
                "用户已选择的配置（请基于此信息进行创作，不要再询问这些已确认的内容）：\n"
                + "\n".join(_known_parts)))

        # 2. 对话历史（按 conversation_id 严格隔离）
        # 每个窗口有独立的 conversation_id → ctx.history 天然是窗口级
        for msg in ctx.history[-6:]:  # 最近 3 轮
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # 4. 当前用户消息
        messages.append(HumanMessage(content=ctx.message))

        # 3.5 记录 LLM 收到的完整 messages
        try:
            _log_llm_messages(messages)
        except Exception:
            pass

        # 4. 流式调用 LLM — 绑定 trigger_generation 工具
        # 工具调用优先级高于文本 [GENERATE] 标记（fallback）。
        # 旧版 LLM 不支持 tool_calls 时自动退回纯文本模式。
        try:
            _llm_with_tools = llm.bind_tools(TRIGGER_TOOL_SCHEMA, tool_choice="auto")
        except Exception:
            _llm_with_tools = llm

        full_text = ""
        tool_call_payload: Optional[dict] = None
        try:
            async for chunk in _llm_with_tools.astream(messages):
                # 提取文本片段（yield 给前端 streaming 渲染）
                content = chunk.content if hasattr(chunk, "content") else None
                if isinstance(content, str) and content:
                    full_text += content
                    yield {"type": "text", "content": content}

                # 提取 tool_calls（OpenAI Chat Completions 格式）
                # chunk.tool_calls 是 List[ToolCall]（每次流式追加）
                chunk_tool_calls = getattr(chunk, "tool_calls", None)
                if chunk_tool_calls:
                    for tc in chunk_tool_calls:
                        # 每个 ToolCall 有 name / args（dict 或 str）/ id
                        # 部分 chunk 里 args 是空 dict（增量），等下一个 chunk 补全
                        tc_name = (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None))
                        if tc_name != "trigger_generation":
                            continue
                        tc_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
                        if isinstance(tc_args, str):
                            try:
                                tc_args = json.loads(tc_args) if tc_args.strip() else None
                            except json.JSONDecodeError:
                                tc_args = None
                        if isinstance(tc_args, dict) and tc_args.get("topic"):
                            tool_call_payload = tc_args
        except Exception as e:
            logger.error("对话 LLM 调用失败: %s", e)
            yield {"type": "text", "content": f"抱歉，我暂时遇到了问题: {e}"}
            return

        # 5. 提取 trigger：只从 tool_call 提取（v3 协议已完全移除 [GENERATE] 文本标记）。
        trigger: Optional[GenerateTrigger] = None
        if tool_call_payload:
            try:
                trigger = GenerateTrigger.from_dict(tool_call_payload)
                logger.info(
                    "对话 LLM 通过 function calling 触发 trigger_generation "
                    "(action=%s, topic=%r, platform=%r)",
                    trigger.action, trigger.topic[:60], trigger.platform,
                )
            except Exception as e:
                logger.warning("tool_call 解析失败: %s", e)

        # 5.0 用前端预选值覆盖 LLM 输出的标签（LLM 看到的是中文标签，不是 ID）
        if trigger:
            if ctx.platform:
                trigger.platform = ctx.platform
            if ctx.content_type:
                trigger.style = ctx.content_type  # GenerateTrigger.style 存储体裁 ID
            # 用 preference 兜底 LLM 没填的 tone/length/audience
            # 这样 /agent/stream 拿到的 trigger 字段总是有值
            if isinstance(ctx.preferences, dict):
                if not trigger.tone and ctx.preferences.get("tone"):
                    trigger.tone = str(ctx.preferences["tone"])
                if not trigger.length and ctx.preferences.get("length"):
                    trigger.length = str(ctx.preferences["length"])
                if not trigger.audience and ctx.preferences.get("audience"):
                    trigger.audience = str(ctx.preferences["audience"])

        # 5.1 有 trigger → 检查 LLM 是否在确认话里越权夹带正文（## 标题：xxx 等元描述）。
        #     注意：v3 已不解析 [GENERATE] 文本标记，所以不再有 strip 步骤。
        #     文本是合法的"确认话 + tool_call"，但若 LLM 越权自写正文，仍要剥离。
        if trigger and IntentDetector.looks_like_unauthorized_write(full_text):
            logger.warning(
                "对话 LLM 通过 tool_call 触发，但文本含越权元描述，剥离正文部分 "
                "(full_text_len=%s)",
                len(full_text),
            )
            # 用一个简单的"剥离 ## 元描述行"的清理（不依赖 [GENERATE] 标记）
            cleaned = re.sub(
                r'(?m)^#{1,3}\s*(思考|正文|标题|引言|结论|章节)\s*[:：]?.*$',
                '',
                full_text,
            ).strip()
            if cleaned and cleaned != full_text:
                yield {"type": "text_cleanup", "content": cleaned}
            yield {"type": "trigger", "data": _build_trigger_data(trigger, ctx.preferences)}
            return

        if trigger:
            yield {"type": "trigger", "data": _build_trigger_data(trigger, ctx.preferences)}
            return

        # 5.2 越权自写但没 trigger → 拦截，强制重发（让前端展示错误 + 不触发 pipeline）
        if IntentDetector.looks_like_unauthorized_write(full_text):
            logger.error(
                "对话 LLM 越权自写且未调用 trigger_generation，强制拦截，避免污染 pipeline "
                "(full_text_len=%s, preview=%r)",
                len(full_text), full_text[:80],
            )
            yield {
                "type": "text_cleanup",
                "content": "抱歉，我刚刚的回复格式有误，请重试一次。"
                          "请用一句话告诉我：你想写什么主题、发布在哪个平台。",
            }
            return

        # ── 兜底：LLM 没有调用 trigger_generation 但用户明确要求生成内容 ──
        _user_msg = (ctx.message or "").strip()
        _user_msg_lower = _user_msg.lower()
        _gen_keywords = ["写", "生成", "重新写", "重新生成", "重写", "再来",
                         "新的一篇", "重新", "生成一份", "写一份", "来一份"]
        # "重做"类关键词：用户想用相同主题再生成一次，必须保留上次 topic
        _redo_keywords = ["重新生成", "重新写", "重新", "重写", "再来", "新的一篇",
                          "再来一份", "重新来"]
        # 简化：用户消息几乎完全是重做关键词（如 "重新生成"、"再写一份吧"）
        _is_pure_redo = (
            len(_user_msg) <= 8
            and any(_user_msg.strip() == kw or _user_msg.strip().startswith(kw)
                    for kw in _redo_keywords)
        )
        _wants_generation = any(kw in _user_msg_lower for kw in _gen_keywords)

        # ── 关键防御：LLM 已主动追问 = 规则 A 正常执行，不要被兜底覆盖 ──
        # 当用户消息里有生成关键词，但 LLM 正确表达了"信息不足、需要追问"
        # （如"请告诉我主题和平台"），应尊重 LLM 的判断，不再兜底触发 pipeline。
        # 兜底分支只用于：LLM 没追问、也没调用工具的异常情况（如模型卡住 / 返回空 / 旧模型不支持 tools）。
        if (
            _wants_generation
            and not IntentDetector.looks_like_unauthorized_write(full_text)
            and IntentDetector.looks_like_followup_question(full_text)
        ):
            logger.info(
                "对话 LLM 已主动追问（信息不足），尊重其判断，不触发 pipeline "
                "(msg_len=%s, user_msg=%r, llm_preview=%r)",
                len(full_text), _user_msg[:50], full_text[:80],
            )
            return

        if _wants_generation:
            logger.info("LLM 未调用 trigger_generation 但用户要求生成，自动触发 pipeline (msg_len=%s, user_msg=%s)",
                        len(full_text), _user_msg[:50])

            # ── 重做场景：保留上次的 topic，不要从 history 随便取 ──
            if _is_pure_redo:
                def _get_last_topic_from_memory(ctx_obj):
                    return memory_manager.get_last_conversation_topic(
                        user_id=ctx_obj.user_id or "",
                        conversation_id=ctx_obj.conversation_id or "",
                    )
                _topic = _get_last_topic_from_memory(ctx)
                if not _topic:
                    logger.info("重做场景未找到上次 topic，转追问")
                    yield {
                        "type": "text_cleanup",
                        "content": "这是我们的第一次对话，还没有可重新生成的内容。"
                                  "请告诉我你想写什么主题、发布在哪个平台。",
                    }
                    return
                logger.info("重做场景，复用上次 topic=%r", _topic[:50])
            else:
                # topic 提取：剥离用户输入里的"动作词"，剩余作为 topic 候选
                _topic = _user_msg
                for _pre in sorted(_gen_keywords, key=len, reverse=True):
                    if _pre in _topic:
                        _topic = _topic.replace(_pre, "").strip()
                        break

            # topic 太短 → 从对话历史里**安全**地提取
            # 安全 = 排除 LLM 越权自写的元描述（## 思考、## 正文 等）
            if not _topic or len(_topic) < 3:
                _meta_re = re.compile(r'(?m)^#{1,3}\s*(思考|正文|标题|引言|结论)\s*[:：]?\s*')
                # 按钮占位/系统消息（前端 useAgentStore 注水）：[内容生成]、[重写] 等
                _placeholder_re = re.compile(r'^\[[^\]]+\]$')
                for msg in reversed(ctx.history):
                    if msg.get("role") == "user" and msg.get("content", "").strip():
                        candidate = msg["content"].strip()
                        # 跳过方括号占位（如 "[内容生成]"）—— 这是前端按钮注水消息
                        if _placeholder_re.match(candidate):
                            continue
                        # 用户消息理论上不会带"## 思考："，但保险起见过滤
                        candidate = _meta_re.sub('', candidate)
                        candidate = re.sub(r'\s+', ' ', candidate).strip()
                        if 3 <= len(candidate) <= 200:
                            _topic = candidate
                            break

            # 还提不出来 → 不要瞎猜，直接走追问而不是兜底
            if not _topic or len(_topic) < 3:
                logger.info("兜底分支 topic 提取失败，转追问（不触发 pipeline）")
                yield {
                    "type": "text_cleanup",
                    "content": "请告诉我你想写什么主题、发布在哪个平台（例如：小红书、公众号、知乎）。"
                              "回复示例：'写一篇关于 Python 入门的笔记，发布在小红书'。",
                }
                return

            # 修复兜底 BUG：优先用前端已选的 ctx.platform / ctx.content_type，
            #     不再依赖 user_profile 字符串匹配（之前 _user_profile="" 会导致 platform=""）。
            _platform = ctx.platform or ""
            _style = ctx.content_type or ""
            if not _platform and _user_prefs:
                _platform = "xiaohongshu" if ("xiaohongshu" in _user_prefs or "小红书" in _user_prefs) else ""
                _style = "真实体验分享" if "真实" in _user_prefs else ""

            # 修复兜底 BUG：剥离用户输入里的 [目标平台：xxx] 前缀（前端 Workspace.tsx 注入），
            #     避免 topic 里夹带平台标签导致后续 pipeline 主题错乱
            _PLATFORM_HINT_RE = re.compile(r"^\[目标平台：[^\]]+\]\s*\n?")
            _topic = _PLATFORM_HINT_RE.sub("", _topic).strip()
            if not _topic:
                _topic = ctx.message

            _fallback_trigger = GenerateTrigger(
                action="create",
                topic=_topic,
                platform=_platform,
                style=_style,
            )
            # 把 preferences 平铺到 fallback trigger，让前端能透传到 /agent/stream
            _fb_data = _build_trigger_data(_fallback_trigger, ctx.preferences)
            _fb_data["_fallback"] = True
            yield {"type": "trigger", "data": _fb_data}
