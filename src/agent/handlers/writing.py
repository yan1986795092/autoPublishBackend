"""WritingAgent — 多风格内容写作

使用 PromptEngine 构建 prompt，支持记忆上下文注入和工具调用（memory_search 等）。
"""

import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..agent_utils import extract_output
from src.utils.tokenizer import estimate_tokens
from src.prompts import PromptBuilder, CONTENT_TYPE_PROMPTS, STYLE_GUIDES
from src.core.exceptions import HandlerError
from .tool_utils import call_llm

logger = logging.getLogger(__name__)


async def handle_writing(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    try:
        topic = task_input.get("topic", task_input.get("query", ""))
        content_type = task_input.get("content_type", "blog")
        style = task_input.get("style", "")
        platform = task_input.get("platform", "")

        # ── 构建上下文 ──
        dep_context = task_input.get("_context", {})
        context_str_parts = []
        for cdata in dep_context.values():
            if cdata.get("content"):
                context_str_parts.append(cdata["content"][:1000])

        # 记忆上下文（由 MemoryManager 注入）
        memory_context = task_input.get("_memory_context", "")

        # ── 用 PromptBuilder 构建 ──
        builder = PromptBuilder()

        # system prompt
        builder.system("system.base")

        # 任务指令
        type_guide = CONTENT_TYPE_PROMPTS.get(content_type, CONTENT_TYPE_PROMPTS["blog"])
        style_guide = STYLE_GUIDES.get(style, "自然流畅，有自己的观点和态度")
        platform_line = f"**目标平台：** {platform}" if platform else ""

        builder.instructions("writing.task", {
            "topic": topic,
            "content_type": content_type,
            "style_guide": style_guide,
            "platform_line": platform_line,
            "type_guide": type_guide,
        })

        # 注入工具描述
        tool_descriptions = task_input.get("_tool_descriptions", "")
        if tool_descriptions:
            builder.context(f"\n\n可用工具：\n{tool_descriptions}")

        # 引用上下文
        if context_str_parts:
            builder.context(f"\n[已有内容参考]\n{chr(10).join(context_str_parts)}")

        # 记忆上下文
        if memory_context:
            builder.memory(f"\n\n{memory_context}")

        # 创作上下文（CreativeAnchors、反馈、最近内容等）
        extra_context = task_input.get("_extra_context", "")
        if extra_context:
            builder.memory(f"\n\n{extra_context}")

        # ── 构建消息 ──
        composed = builder.build()
        messages = composed.messages

        # 转为 LangChain 消息
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))

        # ── 工具感知的 LLM 调用 ──
        raw = await call_llm(
            llm, lc_messages,
            tool_registry=task_input.get("_tool_registry"),
            agent_type="WritingAgent",
            stream=True,
            on_chunk=task_input.get("_on_llm_chunk"),
        )

        # 提取正文（extract_output 统一处理多种分隔符）
        content = extract_output(raw)
        if not content:
            content = raw.strip()
            if content:
                logger.warning("WritingAgent: extract_output returned empty, using raw response")

        tokens = estimate_tokens(raw)

        return {
            "status": "completed", "agent_type": "WritingAgent",
            "output": {"content": content, "type": "text", "content_type": content_type},
            "tokens_used": tokens,
            "prompt_versions": composed.used_templates,
            "thinking_steps": [{
                "agent": "WritingAgent", "action": "写作完成",
                "reasoning": f"撰写{content_type}文章：{topic[:30]}...",
                "output_summary": f"生成了约 {len(content)} 字内容",
                "tokens_used": tokens, "status": "done",
            }],
        }
    except Exception as e:
        logger.error("WritingAgent failed: %s", e, exc_info=True)
        raise HandlerError(str(e), handler="WritingAgent") from e
