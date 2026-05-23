"""WritingAgent — 多风格内容写作

使用 PromptEngine 构建 prompt，支持记忆上下文注入。
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..agent_utils import extract_output
from src.prompts import PromptBuilder, CONTENT_TYPE_PROMPTS, STYLE_GUIDES


async def handle_writing(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
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

    # 引用上下文
    if context_str_parts:
        builder.context(f"\n[已有内容参考]\n{chr(10).join(context_str_parts)}")

    # 记忆上下文
    if memory_context:
        builder.memory(f"\n\n{memory_context}")

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

    # ── 流式 LLM 调用 ──
    on_llm_chunk = task_input.get("_on_llm_chunk")
    chunks: list[str] = []
    async for chunk in llm.astream(lc_messages):
        if chunk.content:
            chunks.append(chunk.content)
            if on_llm_chunk:
                await on_llm_chunk(chunk.content)

    raw = "".join(chunks)

    # 提取正文
    if "正文：" in raw:
        content = raw.split("正文：", 1)[1].strip()
    else:
        content = extract_output(raw)

    tokens = len(raw) // 4

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
