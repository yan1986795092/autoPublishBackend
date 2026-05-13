"""WritingAgent — 多风格内容写作"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..agent_utils import CONTENT_TYPE_PROMPTS, STYLE_GUIDES, SYSTEM_PROMPT_TEMPLATE, extract_output


async def handle_writing(task_input: dict, llm: ChatOpenAI) -> dict:
    topic = task_input.get("topic", task_input.get("query", ""))
    content_type = task_input.get("content_type", "blog")
    style = task_input.get("style", "")
    platform = task_input.get("platform", "")

    type_guide = CONTENT_TYPE_PROMPTS.get(content_type, CONTENT_TYPE_PROMPTS["blog"])
    style_guide = STYLE_GUIDES.get(style, "专业、清晰的写作风格")

    dep_context = task_input.get("_context", {})
    context_str = ""
    for cdata in dep_context.values():
        if cdata.get("content"):
            context_str += f"\n[参考信息]\n{cdata['content'][:1000]}"

    prompt = f"""## 写作任务

主题：{topic}
内容类型：{content_type}
{('目标平台：' + platform) if platform else ''}

## 风格要求
{style_guide}

### {type_guide}
{context_str}

请先写 REASONING，然后输出 OUTPUT。"""

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(role="专业内容创作者", description=f"撰写{content_type}内容")),
        HumanMessage(content=prompt),
    ])

    content = extract_output(response.content)
    tokens = len(response.content) // 4

    return {
        "status": "completed", "agent_type": "WritingAgent",
        "output": {"content": content, "type": "text", "content_type": content_type},
        "tokens_used": tokens,
        "thinking_steps": [{
            "agent": "WritingAgent", "action": "generate",
            "reasoning": f"Writing {content_type} about: {topic}",
            "output_summary": f"Generated {len(content)} chars",
            "tokens_used": tokens, "status": "done",
        }],
    }
