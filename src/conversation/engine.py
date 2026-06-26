"""ConversationEngine — 对话引擎核心

每次用户消息：
1. 加载对话历史和用户偏好
2. 调用对话 LLM 流式生成回复
3. 检查回复中是否包含 [GENERATE] 标记
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
from .intent_detector import TriggerExtractor
from src.core.model_router import ModelRouter
from src.config import get_settings
from src.memory.manager import memory_manager

logger = logging.getLogger(__name__)

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
- 用户**明确**要生成且信息充分 → 触发 [GENERATE]

### 规则 B：禁止越权（核心约束）
**绝对禁止**在回复中包含任何形式的：
- 以 `## 思考：` / `## 正文：` / `## 标题：` / `## 引言：` / `## 结论：` / `## 章节：` 开头的结构化元描述
- 任何超过 3 行的"标题 + 引言 + 多章节 + 结论"框架
- Markdown 标题（`#` / `##`）超过 1 个

如果违反上面任何一条，本次回复**作废**，用户将看到错误，必须输出 [GENERATE] 标记由专门的 Pipeline 处理。

### 规则 C：trigger 格式
仅在确认要生成时，在回复**末尾**附加：

[GENERATE]
{{
  "action": "create|revise|expand",
  "topic": "...",
  "platform": "...",
  "style": "...",
  "length": "short|medium|long",
  "tone": "...",
  "audience": "...",
  "keywords": ["..."]
}}
[/GENERATE]

trigger 之外**不得**有任何正文、标题、章节内容。最多一句话确认（如"好，信息够了，开始创作！"）。

### 规则 1：简单修改（改词、调语气、改 1-2 句）
→ 直接改好回复，不加标记。例：
用户：把"很好"改成"非常好"
你：好的，改好了：原来的"...很好..." → "...非常好..."

### 规则 3：润色/修改已有内容
用户对**已经生成好的内容**提修改意见（如"润色一下""开头改得更吸引人""第二段太啰嗦了"），
→ 对话 LLM **不得**自己改，必须加 [GENERATE]action=revise[/GENERATE]，
让 FixAgent 处理。
注意：这种情况下 `revision_focus` 字段必须填用户的具体修改要求。

### 规则 4：大篇幅修改（扩写、改风格、重写）
→ 加 [GENERATE]action=revise[/GENERATE]

### 规则 5：信息不足
→ 继续追问，不加标记。一次只问 1-2 个问题，不要全部列出来。

### 关键：当信息收集完毕时
用户回答了最后一个问题后，你只能说一句确认话（如"好，信息够了，开始创作！"），**然后立即输出 [GENERATE]**。
不准在触发前写任何内容正文。

## [GENERATE] 格式

当你确认要触发生成时，在回复末尾添加：

[GENERATE]
{{
  "action": "create|revise|expand",
  "topic": "主题",
  "platform": "平台代码",
  "style": "风格描述",
  "length": "short|medium|long",
  "tone": "语调",
  "audience": "目标读者",
  "keywords": ["关键词1", "关键词2"],
  "revision_focus": "修改重点（仅 revise 时需要）"
}}
[/GENERATE]

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


def _build_trigger_data(trigger: "GenerateTrigger") -> dict:
    """从 GenerateTrigger 构造触发器 dict（避免重复代码）"""
    return {
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


class ConversationEngine:
    """对话引擎 — 单次消息处理"""

    def __init__(self):
        self._router = ModelRouter()
        self._llm_cache: dict = {}

    def _get_llm(self) -> object:
        """获取对话 LLM 实例

        优先 Gateway（model=auto 走 AutoRouter），
        Gateway 不可用时降级为直连（用第一个可用的 API key）。
        """
        # 先检测 Gateway 是否可达
        gateway_up = self._check_gateway()
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

    def _check_gateway(self) -> bool:
        """检查 LLM Gateway 是否可用（快速连接测试）"""
        import socket
        try:
            s = socket.create_connection(("localhost", 9010), timeout=1.0)
            s.close()
            return True
        except (OSError, Exception):
            return False

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
        llm = self._get_llm()

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

        # 4. 流式调用 LLM
        full_text = ""
        try:
            async for chunk in llm.astream(messages):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    full_text += content
                    yield {"type": "text", "content": content}
        except Exception as e:
            logger.error("对话 LLM 调用失败: %s", e)
            yield {"type": "text", "content": f"抱歉，我暂时遇到了问题: {e}"}
            return

        # 5. 检查触发标记
        trigger = TriggerExtractor.extract(full_text)
        has_marker = TriggerExtractor.has_marker(full_text)

        # 5.0 用前端预选值覆盖 LLM 输出的标签（LLM 看到的是中文标签，不是 ID）
        if trigger:
            if ctx.platform:
                trigger.platform = ctx.platform
            if ctx.content_type:
                trigger.style = ctx.content_type  # GenerateTrigger.style 存储体裁 ID

        # 5.1 越权自写检测：有 trigger 但有大量正文/元描述 → 警告并把正文部分剥离
        if trigger and TriggerExtractor.looks_like_unauthorized_write(full_text, has_marker=True):
            logger.warning(
                "对话 LLM 越权自写：触发器已识别但伴随大量正文/元描述，剥离正文部分 "
                "(full_text_len=%s)",
                len(full_text),
            )
            clean_text = TriggerExtractor.strip(full_text)
            if clean_text:
                yield {"type": "text_cleanup", "content": clean_text}
            yield {"type": "trigger", "data": _build_trigger_data(trigger)}
            return

        if trigger:
            clean_text = TriggerExtractor.strip(full_text)
            if clean_text:
                yield {"type": "text_cleanup", "content": clean_text}
            yield {"type": "trigger", "data": _build_trigger_data(trigger)}
            return

        # 5.2 越权自写但没 trigger → 拦截，强制重发（让前端展示错误 + 不触发 pipeline）
        if TriggerExtractor.looks_like_unauthorized_write(full_text, has_marker=False):
            logger.error(
                "对话 LLM 越权自写且未输出 [GENERATE]，强制拦截，避免污染 pipeline "
                "(full_text_len=%s, preview=%r)",
                len(full_text), full_text[:80],
            )
            yield {
                "type": "text_cleanup",
                "content": "抱歉，我刚刚的回复格式有误，请重试一次。"
                          "请用一句话告诉我：你想写什么主题、发布在哪个平台。",
            }
            return

        # ── 兜底：LLM 没有输出 [GENERATE] 但用户明确要求生成内容 ──
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
        # 兜底分支只用于：LLM 没追问、也没触发标记的异常情况（如模型卡住 / 返回空）。
        if (
            _wants_generation
            and not TriggerExtractor.has_marker(full_text)
            and not TriggerExtractor.looks_like_unauthorized_write(full_text, has_marker=False)
            and TriggerExtractor.looks_like_followup_question(full_text)
        ):
            logger.info(
                "对话 LLM 已主动追问（信息不足），尊重其判断，不触发 pipeline "
                "(msg_len=%s, user_msg=%r, llm_preview=%r)",
                len(full_text), _user_msg[:50], full_text[:80],
            )
            return

        if _wants_generation and not TriggerExtractor.has_marker(full_text):
            logger.info("LLM 未输出 [GENERATE] 但用户要求生成，自动触发 pipeline (msg_len=%s, user_msg=%s)",
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

            _user_prefs = ctx.user_profile or ""
            _platform = "xiaohongshu" if ("xiaohongshu" in _user_prefs or "小红书" in _user_prefs) else ""
            _style = "真实体验分享" if "真实" in _user_prefs else ""

            _fallback_trigger = GenerateTrigger(
                action="create",
                topic=_topic,
                platform=_platform,
                style=_style,
            )
            yield {"type": "trigger", "data": {
                "action": _fallback_trigger.action,
                "topic": _fallback_trigger.topic,
                "platform": _fallback_trigger.platform,
                "style": _fallback_trigger.style,
                "_prompt": _fallback_trigger.to_prompt(),
                "_fallback": True,
            }}
