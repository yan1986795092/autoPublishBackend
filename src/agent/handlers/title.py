"""TitleAgent — 根据文章正文生成标题

输入（由 Schema 约束）:
  article_body — 文章正文（consume 字段）

产出:
  article_title — 文章标题
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from .tool_utils import call_llm
from ..result import agent_response
from src.logging_shared.config import get_logger

logger = get_logger("agent.title")

SYSTEM_PROMPT = """你是一个专业的标题创作专家。

你的任务是根据文章正文，生成一个吸引人的标题。

要求：
1. 标题要有信息量，让读者知道文章在讲什么
2. 要有情绪价值，引发点击欲望
3. 长度控制在 15-30 字之间
4. 直接输出标题文字，不要任何前缀或引号"""


async def handle_title(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    try:
        # ── 输入 ──
        article_body: str = task_input["article_body"]
        topic: str = task_input.get("topic", "")

        # 截断正文到 2000 字
        body_preview = article_body[:2000]

        lc_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"## 文章正文\n\n{body_preview}\n\n## 请生成标题"),
        ]

        llm_result = await call_llm(
            llm, lc_messages,
            agent_type="TitleAgent",
            stream=True,
            on_chunk=task_input.get("_on_llm_chunk"),
        )

        title = llm_result.content.strip().strip('"').strip("'").strip()
        if not title:
            title = topic[:60]

        return agent_response(
            "TitleAgent",
            content=title,
            output_type="text",
            tokens_used=llm_result.total_tokens,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            output_fields={
                "article_title": title,
            },
            thinking_steps=[{
                "agent": "TitleAgent", "action": "标题生成完成",
                "reasoning": f"根据{len(article_body)}字正文生成标题",
                "output_summary": title[:60],
                "tokens_used": llm_result.total_tokens,
                "status": "done",
            }],
        )
    except Exception as e:
        logger.error("TitleAgent 执行失败: %s", e, exc_info=True)
        raise __import__("src.core.exceptions", fromlist=["HandlerError"]).HandlerError(str(e), handler="TitleAgent") from e
