"""对话引擎核心

每次用户消息：
1. 加载对话历史和用户偏好
2. 系统判断当前是否需要触发创作（状态机）
3. 调用对话 LLM 流式生成回复
   - 如果需要触发创作：tool_choice="required"，强制 LLM 调用工具
   - 如果正常对话：tool_choice="auto"，LLM 自行决定
4. 提取 tool_call → 返回 trigger
5. 返回对话文本 + 可选的触发指令

核心设计：
- 系统维护状态机，自己知道什么时候"需要触发创作"
- 不依赖 LLM/正则判断"是否应该调用工具"
- 不需要 fallback 重试来检测"是否调用了工具"
"""

from __future__ import annotations
import json
import logging
import os
import re
from enum import Enum
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
# 对话状态机
# ──────────────────────────────────────────────────────────────────────
class ConversationState(Enum):
    """对话状态枚举

    系统根据状态决定是否需要触发创作：
    - COLLECTING: 收集参数中，正常对话
    - WAITING_GENERATION: 用户已确认，系统需要触发创作
    """

    COLLECTING = "collecting"  # 收集参数中，正常对话
    WAITING_GENERATION = "waiting_generation"  # 等待触发创作


# 生成意图关键词：用户明确要求生成内容
_GENERATION_KEYWORDS = [
    "开始创作", "开始生成", "开始写", "生成", "创作", "写一篇",
    "写一个", "写一份", "生成一份", "重新生成", "重新写", "重写",
    "再来", "重新", "来一份", "写吧", "开始吧", "生成吧",
]


def _user_wants_generation(message: str) -> bool:
    """判断用户是否明确要求生成内容（确定性判断）

    只用于系统状态判断，不用于 LLM prompt。
    """
    if not message:
        return False
    msg_lower = message.strip().lower()
    return any(kw in msg_lower for kw in _GENERATION_KEYWORDS)


def _is_pure_redo(message: str) -> bool:
    """判断用户消息是否是纯重做指令（如"重新生成"、"再来一份"）

    用于检测用户想用相同主题再生成一次。
    """
    if not message:
        return False
    msg = message.strip()
    # 纯动作词，长度 <= 8
    if len(msg) > 8:
        return False
    redo_keywords = ["重新生成", "重新写", "重写", "再来", "重新", "开始", "生成", "写"]
    return any(msg == kw or msg.startswith(kw) for kw in redo_keywords)


# ──────────────────────────────────────────────────────────────────────
# 系统状态判断
# ──────────────────────────────────────────────────────────────────────
def _determine_conversation_state(
    user_message: str,
    ctx_platform: Optional[str],
    ctx_content_type: Optional[str],
    has_topic_in_context: bool = False,
) -> ConversationState:
    """系统根据用户输入和上下文，判断当前对话状态

    这是确定性的布尔判断，不会有失误。
    只有当用户明确要求生成内容时，才返回 WAITING_GENERATION。

    Args:
        user_message: 当前用户消息
        ctx_platform: 前端传入的平台预选
        ctx_content_type: 前端传入的体裁预选
        has_topic_in_context: 对话历史/上下文中是否已有主题

    Returns:
        WAITING_GENERATION: 用户明确要求生成，且系统等待触发创作
        COLLECTING: 正常对话/收集参数中
    """
    msg = (user_message or "").strip()

    # 用户说了生成关键词
    if _user_wants_generation(msg):
        return ConversationState.WAITING_GENERATION

    # 纯重做指令（如"重新生成"）也视为需要触发
    if _is_pure_redo(msg):
        return ConversationState.WAITING_GENERATION

    # 正常对话/收集参数
    return ConversationState.COLLECTING


def _build_state_hint(state: ConversationState) -> str:
    """根据状态构建系统提示片段，注入到 LLM 的 System Prompt 中

    让 LLM 知道当前对话状态，以便做出正确的响应。
    """
    if state == ConversationState.WAITING_GENERATION:
        return (
            "\n\n[系统状态] 当前对话状态: WAITING_GENERATION\n"
            "用户已确认需要开始创作。\n"
            "你必须调用 trigger_generation 工具来启动创作流程。\n"
            "仅输出文本（如'好的，马上创作'）而不调用工具是无效的。"
        )
    return ""  # COLLECTING 状态不需要额外提示


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
            "description": "【必须调用】用户明确要求生成内容时，必须调用此工具启动创作 pipeline。仅输出文本而未调用此工具，前端将无法执行创作。调用前无需征得用户同意。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "revise", "expand"], "description": "create=创作新内容, revise=修改已有, expand=扩写"},
                    "topic": {"type": "string", "description": "创作主题，填写实际的主题或标题"},
                    "platform": {"type": "string", "description": "目标平台: xiaohongshu/wechat/douyin/weibo/bilibili/zhihu"},
                    "style": {"type": "string", "description": "写作风格"},
                    "length": {"type": "string", "enum": ["short", "medium", "long"], "description": "篇幅"},
                    "tone": {"type": "string", "enum": ["friendly", "professional", "humorous", "warm"], "description": "语调"},
                    "audience": {"type": "string", "enum": ["beginner", "intermediate", "expert"], "description": "目标读者"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "核心关键词"},
                    "revision_focus": {"type": "string", "description": "修改重点（仅 revise/expand）"},
                },
                "required": ["action", "topic"],
            },
        },
    },
]


# 对话 LLM System Prompt
SYSTEM_PROMPT = """你是专业的 AI 内容创作助手，通过自然对话引导用户完成内容创作。

## 职责

1. 了解用户想创作什么（主题、平台、风格、篇幅、语调、受众等）
2. 信息不足时自然追问，一次 1-2 个问题
3. 信息充足后调用 trigger_generation 启动创作

## 约束

- 不要自行撰写正文、标题、章节——那是创作 pipeline 的工作
- 不要使用 [GENERATE] 标记——此格式已废弃
- 如果用户只是修改措辞（改词、调一句），可以直接改好回复，无需触发创作
- **强制规则：当你认为应该开始创作时，必须同时调用 trigger_generation 工具。**
  **仅输出文本（如"好，马上创作"）而没有工具调用是无效的，前端无法继续执行。**

## 触发时机

当用户明确要求生成内容（如说"开始生成"、"开始创作"、"生成文章"、"写一篇"等），
且至少主题已确认时，必须：
1. 输出一句简短确认（如"好，开始创作"）
2. **同时调用 trigger_generation 工具**，传入已收集的参数（action、topic 必填）

## 平台信息

用户已选择的配置会以系统消息提供，直接使用即可，无需重复询问。
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

        # ──────────────────────────────────────────────────────────────────────
        # 4. 系统状态判断：系统自己知道当前是否需要触发创作
        # ──────────────────────────────────────────────────────────────────────
        # 判断用户是否要求生成内容（确定性判断，不依赖 LLM/正则）
        conv_state = _determine_conversation_state(
            user_message=ctx.message or "",
            ctx_platform=ctx.platform,
            ctx_content_type=ctx.content_type,
        )
        logger.info("系统状态: %s, 用户消息: %r", conv_state.value, (ctx.message or "")[:50])

        # 根据状态决定 tool_choice
        # - WAITING_GENERATION: 必须调用工具，不调用就是 API 问题（不需要 fallback）
        # - COLLECTING: 正常对话，auto 让 LLM 自行决定
        needs_tool = conv_state == ConversationState.WAITING_GENERATION
        tool_choice = "required" if needs_tool else "auto"

        # 注入状态提示到 System Prompt，让 LLM 知道当前状态
        if needs_tool:
            state_hint = _build_state_hint(conv_state)
            messages.insert(-1, SystemMessage(content=state_hint))

        # ──────────────────────────────────────────────────────────────────────
        # 5. 流式调用 LLM（使用 astream_events 同时获取流式文本和完整 tool_call）
        # ──────────────────────────────────────────────────────────────────────
        # 问题：astream 流式传输中，tool_call 的 args 是增量传输的
        # 解决：使用 astream_events，文本通过事件流输出，最终从完整响应提取 tool_call
        try:
            _llm_with_tools = llm.bind_tools(TRIGGER_TOOL_SCHEMA, tool_choice=tool_choice)
        except Exception:
            _llm_with_tools = llm

        full_text = ""
        tool_call_payload: Optional[dict] = None

        try:
            # 使用 astream_events：文本流式输出 + 最终获取完整 tool_call
            async for event in _llm_with_tools.astream_events(messages, version="v2"):
                # 确保 event 是字典
                if not isinstance(event, dict):
                    continue
                event_type = event.get("event", "")
                name = event.get("name", "")

                # 文本片段：yield 给前端
                if event_type == "on_chat_model_stream":
                    event_data = event.get("data")
                    if event_data is None:
                        continue
                    # data 可能是字典或对象
                    if isinstance(event_data, dict):
                        chunk = event_data.get("chunk", {})
                    else:
                        chunk = getattr(event_data, "chunk", None) or getattr(event_data, "content", "")
                    if chunk:
                        content = getattr(chunk, "content", "") if hasattr(chunk, "content") else str(chunk)
                    else:
                        content = ""
                    if content:
                        full_text += content
                        yield {"type": "text", "content": content}

                # LLM 完成：提取完整的 tool_call
                elif event_type == "on_chat_model_end":
                    # on_chat_model_end 的 data.output 包含完整响应
                    event_data = event.get("data")
                    if event_data is None:
                        continue
                    # 获取 output（AIMessage 对象）
                    if isinstance(event_data, dict):
                        response = event_data.get("output", {})
                    else:
                        response = getattr(event_data, "output", None) or event_data
                    # response 可能是 AIMessage 对象
                    response_tool_calls = getattr(response, "tool_calls", None) or []
                    for tc in response_tool_calls:
                        # 安全获取 tc_name
                        if isinstance(tc, dict):
                            tc_name = tc.get("name")
                            tc_args = tc.get("args")
                        else:
                            tc_name = getattr(tc, "name", None)
                            tc_args = getattr(tc, "args", None)
                        if tc_name != "trigger_generation":
                            continue
                        if isinstance(tc_args, str):
                            try:
                                tc_args = json.loads(tc_args) if tc_args.strip() else {}
                            except json.JSONDecodeError:
                                tc_args = {}
                        if isinstance(tc_args, dict):
                            if tc_args.get("topic"):
                                tool_call_payload = tc_args
                            elif needs_tool and ctx.message:
                                logger.info("tool_call args 为空，用用户消息作为 topic fallback")
                                tool_call_payload = {
                                    "action": tc_args.get("action", "create"),
                                    "topic": ctx.message.strip(),
                                }

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

        # ── 核心设计：系统维护状态机，不需要复杂的正则 fallback ──
        #
        # 判断逻辑：
        # 1. 系统判断状态（确定性）
        # 2. 根据状态决定 tool_choice
        #    - WAITING_GENERATION: "required" → LLM 必须调用工具
        #    - COLLECTING: "auto" → 正常对话，LLM 自行决定
        # 3. 有了 tool_choice="required"，不再需要 fallback 重试来检测"是否调用了工具"
        #
        # 保留的检测：
        # - 越权检测：防止 LLM 越权写正文
        # - 参数校验：LLM 调用了工具但参数有问题（如缺 topic）
        # ──────────────────────────────────────────────────────────────────────

        if trigger:
            yield {"type": "trigger", "data": _build_trigger_data(trigger, ctx.preferences)}
            return

        # ── 越权自写但没 trigger → 拦截 ──
        # 即使在 required 模式下也要检测，防止 LLM 越权
        if IntentDetector.looks_like_unauthorized_write(full_text):
            logger.error(
                "对话 LLM 越权自写且未调用 trigger_generation，强制拦截 "
                "(full_text_len=%s, preview=%r)",
                len(full_text), full_text[:80],
            )
            yield {
                "type": "text_cleanup",
                "content": "抱歉，我刚刚的回复格式有误，请重试一次。"
                          "请用一句话告诉我：你想写什么主题、发布在哪个平台。",
            }
            return

        # ── needs_tool=True 但没有 tool_call_payload 的兜底处理 ──
        # 理论上在 "required" 模式下不会发生，但作为安全网保留
        if needs_tool and not tool_call_payload:
            logger.warning(
                "安全网：tool_choice=required 但没有 tool_call_payload，尝试从上下文提取 topic "
                "(full_text_preview=%r)",
                full_text[:80],
            )
            # 从对话历史或用户消息提取 topic
            _topic = _extract_topic_from_context(ctx, full_text)
            if _topic:
                _fallback_trigger = GenerateTrigger(
                    action="create",
                    topic=_topic,
                    platform=ctx.platform or "",
                    style=ctx.content_type or "",
                )
                _fb_data = _build_trigger_data(_fallback_trigger, ctx.preferences)
                _fb_data["_fallback"] = True
                yield {"type": "trigger", "data": _fb_data}
                return
            # 提取失败，转追问
            yield {
                "type": "text_cleanup",
                "content": "请告诉我你想写什么主题、发布在哪个平台。",
            }
            return

        # needs_tool=False（正常对话）：LLM 没有调用工具是正确的行为，直接返回文本
        # 不需要任何 fallback
        logger.debug(
            "正常对话（needs_tool=False），LLM 未调用工具，文本回复: %r",
            full_text[:80],
        )


def _extract_topic_from_context(ctx, full_text: str) -> Optional[str]:
    """从对话上下文提取 topic（兜底用）

    仅在 tool_choice=required 但 LLM 未调用工具的异常情况下使用。
    """
    import re as _re

    # 1. 从 LLM 输出里识别"就以X为主题"等显式信号
    _llm_topic_re = _re.compile(
        r'(?:就以|以|围绕|关于)(.{2,40}?)(?:为主题|为题|写|创作)',
        _re.DOTALL,
    )
    _m = _llm_topic_re.search(full_text)
    if _m:
        return _m.group(1).strip().strip('"\'""''「」『』')

    # 2. 从对话历史里找最近实质主题
    _hint_re = _re.compile(r'^\[目标平台：[^\]]+\]\s*\n?')
    _gen_action_re = _re.compile(
        r'^[直接的马快立刻帮我给我]*(?:'
        r'生成|写|创作|开始|再来|重写|重新)'
        r'(?:一份|一篇|一下|个|吧|创作|写|生成)?'
        r'[的吧！!。.，,"\'\s]*$'
    )
    _placeholder_re = _re.compile(r'^\[[^\]]+\]$')

    for msg in reversed(ctx.history):
        if msg.get("role") != "user":
            continue
        candidate = msg.get("content", "").strip()
        if not candidate:
            continue
        candidate = _hint_re.sub('', candidate).strip()
        if _placeholder_re.match(candidate):
            continue
        if _gen_action_re.match(candidate):
            continue
        candidate = _re.sub(r'\s+', ' ', candidate).strip()
        if 3 <= len(candidate) <= 200:
            return candidate

    # 3. 从当前用户消息剥离动作词
    _user_msg = (ctx.message or "").strip()
    _user_msg = _hint_re.sub('', _user_msg).strip()
    _gen_keywords = ["写", "生成", "创作", "开始", "重新", "再来"]
    for _pre in sorted(_gen_keywords, key=len, reverse=True):
        if _pre in _user_msg:
            _user_msg = _user_msg.replace(_pre, "").strip()
    if len(_user_msg) >= 3:
        return _user_msg

    return None
