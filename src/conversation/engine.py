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
import os
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

# 统一 LLM 入口：所有对话流必须走 Gateway，禁止直连
_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL") or "http://localhost:9010/v1"


# ──────────────────────────────────────────────────────────────────────
# Trigger tool schema（OpenAI function calling 格式）
# ──────────────────────────────────────────────────────────────────────
# 对话 LLM 触发创作阶段的唯一方式。
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
                "【必须调用】当且仅当你要开始写作时调用此函数。"
                "这是触发创作 pipeline 的**唯一**方式——你必须实际执行工具调用，"
                "**不要**只在文本里说'开始创作'而**不调**此函数（那样系统无法知道你已就绪）。"
                "**不要**在回复文本里嵌入 `[GENERATE]{json}[/GENERATE]` 标记"
                "（这种格式已废弃，引擎不再解析）。"
                "调用前你应该先在文本中简短确认（如'好，信息够了，开始创作！'），"
                "然后立即在同一轮调用此函数。"
                "topic 字段必须填实际创作主题（例如'AI 工具如何改变日常工作流'），"
                "不要填'开始写'或'直接开始'这种动作词。"
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

### 规则 C：触发方式（结构化 Function Calling）— 关键

**当且仅当你想开始写作时，必须调用 `trigger_generation` 工具函数**（你已自动获得此工具）。

🚨 **绝对禁止**在文本里描述"开始创作"而**不调用**工具。如果你说"好的，开始创作"但没有调 `trigger_generation`，系统无法知道你已就绪——用户会卡在等待状态。

正确流程示例：
- 用户：「直接生成」
- 你：先在文本中简短确认（一句话即可），例如"好，信息够了，开始创作！"
- **然后立即在同一轮调用 `trigger_generation` 工具函数**，传入参数：`{action, topic, platform, tone, length, audience, ...}`

注意：
1. 工具调用**与文本回复可以同时出现**——文本是确认话，工具调用是触发动作，**两者必须同时存在**
2. 工具调用的 topic 字段必须**填实际创作主题**（用户没明确时填你推断的最合理主题，**不要**写"直接开始"这种占位词）
3. 平台/篇幅/语调/读者等用户已选的可以直接从系统提示里读取，**不要凭空编造**
4. revise 模式（用户对已有内容做修改）必须填 `revision_focus` 字段

**反例（绝不允许）**：
- ❌ 只输出"好的，信息够了，开始创作！"但不调工具 → 用户卡住
- ❌ 在文本里输出 `{action: create, topic: "..."}` → 不会被识别
- ❌ topic 写"开始写"或"开始创作"等动作词 → topic 应该是创作内容的主题，不是动作

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

1. 输出一句确认话（如"好，信息够了，开始创作！"）—— **只是**告诉用户你要开始了
2. **立刻在同一轮对话里调用 `trigger_generation` 工具**—— 这是真正触发 pipeline 的动作

🚨 **必须两件事一起做**：文本确认 + 工具调用。漏掉任何一件都不行。
- 只输出文本 → 系统无法知道你已就绪，用户卡在等待
- 只调工具 → 用户看不到状态，会以为你没反应
- 拆成两轮 → 用户看到第一轮"开始创作"但 pipeline 没启动，第二轮才真的开始，体验割裂

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

    async def _get_llm(
        self,
        provider: str | None = None,
        hint: str | None = None,
    ) -> ChatOpenAI:
        """获取对话 LLM 实例（统一走 Gateway，禁止直连）

        参数：
          provider: 强制使用某 provider（deepseek/qwen/openai/zhipu）。
                   None 时由 Gateway AutoRouter 按成本自动选择。
          hint:     路由提示（simple/balanced/creative/analysis），
                   配合 provider=None 时生效。

        注意：Gateway 不可用时直接抛异常，不走直连 fallback。
              直连 fallback 会绕过 Gateway 的路由策略、Circuit Breaker、
              观测日志和统一鉴权，且无法保证 function calling 能力。
        """
        gateway_up = await self._check_gateway()
        if not gateway_up:
            raise ConnectionError(
                f"LLM Gateway ({_GATEWAY_URL}) 不可用。"
                "请启动 llm-gateway 服务或检查网络连通性。"
            )

        logger.info(
            "对话引擎 → 使用 Gateway (provider=%r, hint=%r)",
            provider or "auto", hint or "balanced",
        )

        llm = ChatOpenAI(
            model=provider or "auto",
            api_key="gateway",
            base_url=_GATEWAY_URL,
            temperature=0.7,
            max_tokens=2048,
            streaming=True,
            timeout=120.0,
            default_headers={
                "X-Route-Hint": hint or "balanced",
                "X-Agent-Type": "ConversationAgent",
            },
        )
        return llm

    async def _check_gateway(self) -> bool:
        """检查 LLM Gateway 是否可用（异步，带缓存）

        不只是测 TCP 连通，而是真正测 /v1/chat/completions 是否返回 200。
        避免：TCP 能连上但 API 实际不可用（认证失败、限流、模型过期等）。
        """
        import asyncio
        now = asyncio.get_event_loop().time()
        # 缓存未过期且上次检查有效 → 直接返回缓存值
        if self._gateway_up is not None and (now - self._gateway_checked_at) < self._GATEWAY_CACHE_TTL:
            return self._gateway_up
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{_GATEWAY_URL}/v1/chat/completions",
                    json={
                        "model": "auto",
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                    },
                    headers={"X-Route-Hint": "simple"},
                )
                self._gateway_up = resp.status_code < 500
                if not self._gateway_up:
                    logger.warning(
                        "Gateway /v1/chat/completions 返回 %d: %s",
                        resp.status_code, resp.text[:200],
                    )
        except Exception as e:
            self._gateway_up = False
            logger.warning("Gateway 健康检查失败: %s", e)
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

        # 5. 提取 trigger：只从 tool_call 提取。
        trigger: Optional[GenerateTrigger] = None
        _user_msg = (ctx.message or "").strip()
        if tool_call_payload:
            try:
                trigger = GenerateTrigger.from_dict(tool_call_payload)
                # ── 关键防御：tool_call 的 topic 与当前用户消息严重不匹配时，用当前消息覆盖 ──
                # 典型场景：LLM 的 tool_call 携带了上一轮的历史 topic，但文本回复说的是新主题
                # 例：tool_call topic="降噪耳机"，但 LLM 文本说"聊聊 AI 工具改变工作流"
                _user_topic_candidate = _user_msg
                if _user_topic_candidate and trigger.topic:
                    _user_topic_stripped = re.sub(
                        r'^\[[^\]]+\]\s*', '', _user_topic_candidate
                    ).strip()
                    if _user_topic_stripped:
                        # 检查 tool_call topic 是否与当前用户消息相关
                        # 真正的相关：包含关系 或 共同关键词（不含纯动作词）
                        _common_kw = [
                            "AI", "工具", "工作流", "耳机", "降噪", "推荐",
                            "分享", "生活", "日常", "科技", "产品", "测评",
                        ]
                        _has_common_kw = any(
                            kw in _user_topic_stripped and kw in trigger.topic
                            for kw in _common_kw
                        )
                        _is_related = (
                            _user_topic_stripped in trigger.topic
                            or trigger.topic in _user_topic_stripped
                            or _has_common_kw
                        )
                        if not _is_related and len(_user_topic_stripped) > 5:
                            logger.warning(
                                "tool_call topic 与当前用户消息不匹配，覆盖 trigger topic "
                                "(tool_call_topic=%r → user_msg_topic=%r)",
                                trigger.topic[:50], _user_topic_stripped[:50],
                            )
                            trigger.topic = _user_topic_stripped
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
        # ── 关键修复：剥离 [目标平台：xxx] 前缀再做"重做"判断 ──
        # 用户消息 "[目标平台：微博]\n重新生成" 长度 13，超过 8 字符阈值，
        # 导致 _is_pure_redo=False → 走兜底 topic 提取 → 误把"重新生成"当主题。
        # 先清洗再判断：用 _user_msg_clean 作为统一的清洗后消息。
        _hint_strip_re = re.compile(r'^\[目标平台：[^\]]+\]\s*\n?')
        _user_msg_clean = _hint_strip_re.sub('', _user_msg).strip()
        # 简化：用户消息（剥前缀后）几乎完全是重做关键词（如 "重新生成"、"再写一份吧"）
        _is_pure_redo = (
            0 < len(_user_msg_clean) <= 8
            and any(_user_msg_clean == kw or _user_msg_clean.startswith(kw)
                    for kw in _redo_keywords)
        )
        _wants_generation = any(kw in _user_msg_lower for kw in _gen_keywords)

        # ── 安全网：LLM 输出"够了/开始创作"但没调工具 ──
        # 可能原因：Gateway 未正确透传 tool_calls、LLM 不支持工具调用。
        # 当 LLM 明确表示已足够但没调工具时，进入兜底 pipeline。
        _llm_ready_signal = (
            not _wants_generation
            and not trigger
            and not IntentDetector.looks_like_followup_question(full_text)
            and IntentDetector.looks_like_ready_to_generate(full_text)
        )

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

        if _wants_generation or _llm_ready_signal:
            if _llm_ready_signal:
                logger.info(
                    "安全网：LLM 输出含'足够/开始创作'信号但未调工具，自动触发 pipeline "
                    "(full_text_preview=%r, topic_from=%s)",
                    full_text[:80], "context_extraction",
                )
            else:
                logger.info("LLM 未调用 trigger_generation 但用户要求生成，自动触发 pipeline (msg_len=%s, user_msg=%s)",
                            len(full_text), _user_msg[:50])

            # ── 重做场景：保留上次的 topic，不要从 history 随便取 ──
            _topic: Optional[str] = None
            if _is_pure_redo and not _llm_ready_signal:
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
                # 兜底 topic 提取优先级（关键修复）：
                #   1. 从 LLM 输出里识别"就以X为主题""以X为题"等显式信号
                #      → 即使 LLM 没调工具，它可能已经说出主题
                #   2. 从对话历史里找最近一条有实质内容的 user 消息
                #      （排除"直接生成""开始创作"这种无意义短消息）
                #   3. 才考虑剥离当前用户消息里的动作词（fallback）

                # 优先级 1: LLM 输出里识别显式主题
                _llm_topic_re = re.compile(
                    r'(?:就以|以|围绕|关于)(.{2,40}?)(?:为主题|为题|写|创作)',
                    re.DOTALL,
                )
                _m = _llm_topic_re.search(full_text)
                if _m:
                    _topic = _m.group(1).strip().strip('"\'""''「」『』')
                    logger.info("兜底：从 LLM 输出提取 topic=%r", _topic[:50])

                # 优先级 2: 对话历史里找最近实质主题
                if not _topic or len(_topic) < 3:
                    _placeholder_re = re.compile(r'^\[[^\]]+\]$')
                    # "纯动作词"检测：开始创作/直接生成/再来一份/重写/帮我写个...
                    # 注意：中文里常带各种标点（吧/吧。/！/"等）+ 单引号 + 全角空格
                    # 这里用 (主题内容) 与 (尾标点) 分开剥离，避免 误判
                    #
                    # 关键修复：增加"重新生成""重新写""新的一篇"这种 **两段动词组合**
                    # 不被长尾巴（如"生成"作为第二段）漏判并误当 topic。
                    _gen_action_only_re = re.compile(
                        r'^[直接的马快立刻帮我给我]*(?:'
                        r'生成|写|创作|开始|再来|重写|重新)'
                        r'(?:一份|一篇|一下|个|吧|创作|写|生成)?'
                        r'[的吧！!。.，,"\'\s]*$'
                    )
                    # 单独的"重做"动作短消息（如"重新生成""重新写""重写""再来"）
                    _redo_only_re = re.compile(
                        r'^[重新]*(?:'
                        r'生成|写|创作|开始|再来|重写|重新)'
                        r'(?:一份|一篇|一下|个|吧|创作|写|生成)?'
                        r'[的吧！!。.，,"\'\s]*$'
                    )
                    _hint_re = re.compile(r'^\[目标平台：[^\]]+\]\s*\n?')
                    for msg in reversed(ctx.history):
                        if msg.get("role") != "user":
                            continue
                        candidate = msg.get("content", "").strip()
                        if not candidate:
                            continue
                        # 剥 [目标平台：xxx] 前缀
                        candidate = _hint_re.sub('', candidate).strip()
                        # 跳过按钮占位
                        if _placeholder_re.match(candidate):
                            continue
                        # 跳过"直接生成""开始创作""开始写""重新生成""再来一份"等纯动作词
                        # 注意两次用不同 regex，覆盖"直接+动作"和"重新+动作"两种模式
                        if _gen_action_only_re.match(candidate):
                            continue
                        if _redo_only_re.match(candidate):
                            logger.info("兜底：跳过历史中的纯重做动作词 msg=%r", candidate[:30])
                            continue
                        # 跳过太短/太长
                        candidate = re.sub(r'\s+', ' ', candidate).strip()
                        if 3 <= len(candidate) <= 200:
                            _topic = candidate
                            logger.info("兜底：从对话历史提取 topic=%r", _topic[:50])
                            break

                # 优先级 3: 剥离当前用户消息里的动作词（最后兜底）
                if not _topic or len(_topic) < 3:
                    _topic = _user_msg_clean  # 复用上面已清洗过的消息
                    for _pre in sorted(_gen_keywords, key=len, reverse=True):
                        if _pre in _topic:
                            _topic = _topic.replace(_pre, "").strip()
                    # 还要剥"重新""重新生成""重新写"等重做前缀
                    for _pre in sorted(_redo_keywords, key=len, reverse=True):
                        if _topic.startswith(_pre):
                            _topic = _topic[len(_pre):].strip()
                    # 还要剥"直接""马上""开始"等纯动作词
                    for _pre in ["直接", "马上", "立刻", "快点", "快"]:
                        if _topic.startswith(_pre):
                            _topic = _topic[len(_pre):].strip()
                    logger.info("兜底：从当前用户消息提取 topic=%r", _topic[:50])

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
