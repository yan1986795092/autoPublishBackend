"""WritingAgent — 多风格内容写作"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..agent_utils import CONTENT_TYPE_PROMPTS, STYLE_GUIDES, SYSTEM_PROMPT_TEMPLATE, extract_output


async def handle_writing(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    topic = task_input.get("topic", task_input.get("query", ""))
    content_type = task_input.get("content_type", "blog")
    style = task_input.get("style", "")
    platform = task_input.get("platform", "")

    type_guide = CONTENT_TYPE_PROMPTS.get(content_type, CONTENT_TYPE_PROMPTS["blog"])
    style_guide = STYLE_GUIDES.get(style, "")

    dep_context = task_input.get("_context", {})
    context_str = ""
    for cdata in dep_context.values():
        if cdata.get("content"):
            context_str += f"\n[已有内容参考]\n{cdata['content'][:1000]}"

    prompt = f"""## 写作任务

**主题：** {topic}
**内容形式：** {content_type}
{('**目标平台：** ' + platform) if platform else ''}

**风格偏好：** {style_guide if style_guide else '自然流畅，有自己的观点和态度'}

**正文要求：**
{type_guide}
{context_str}

直接开始写吧，标题和正文都要。"""

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT_TEMPLATE),
        HumanMessage(content=prompt),
    ])

    raw = response.content

    # 提取正文：优先找「正文：」标记，否则找第一个标题后开始的内容
    if "正文：" in raw:
        content = raw.split("正文：", 1)[1].strip()
    else:
        content = extract_output(raw)

    tokens = len(response.content) // 4

    return {
        "status": "completed", "agent_type": "WritingAgent",
        "output": {"content": content, "type": "text", "content_type": content_type},
        "tokens_used": tokens,
        "thinking_steps": [{
            "agent": "WritingAgent", "action": "写作完成",
            "reasoning": f"撰写{content_type}文章：{topic[:30]}...",
            "output_summary": f"生成了约 {len(content)} 字内容",
            "tokens_used": tokens, "status": "done",
        }],
    }
