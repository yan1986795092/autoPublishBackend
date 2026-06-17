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
from typing import AsyncGenerator, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from .schemas import ConversationContext, GenerateTrigger
from .intent_detector import TriggerExtractor
from src.core.model_router import ModelRouter
from src.config import get_settings

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

## 决策规则

根据用户需求选择处理方式，在回复末尾添加 [GENERATE] 标记来触发生成：

1. **简单修改**（改几个词、调语气、改 1-2 句）
   → 直接修改后回复，不加标记

2. **全新创作**（写文章/笔记/文案/脚本 等）
   → 确认需求足够明确后，在回复末尾加 [GENERATE]action=create[/GENERATE]

3. **大篇幅修改**（扩写、全文改风格、重写结构）
   → 在回复末尾加 [GENERATE]action=revise[/GENERATE]

4. **信息不足** → 继续自然追问，不加标记

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

## 用户偏好
以下是从记忆中读取的用户历史偏好，供参考：
{user_profile}
"""


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

        # 1. 构建消息列表
        system = SystemMessage(
            content=SYSTEM_PROMPT.format(
                user_profile=ctx.user_profile or "暂无用户数据"
            )
        )

        messages = [system]

        # 2. 添加对话历史（最多 20 条）
        for msg in ctx.history[-20:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # 3. 添加当前消息
        messages.append(HumanMessage(content=ctx.message))

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
        if trigger:
            # 返回移除标记后的纯文本
            clean_text = TriggerExtractor.strip(full_text)
            if clean_text:
                yield {"type": "text_cleanup", "content": clean_text}

            # 注入记忆标签到 trigger（用于后续记忆沉淀）
            trigger_data = {
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
            }
            yield {"type": "trigger", "data": trigger_data}
