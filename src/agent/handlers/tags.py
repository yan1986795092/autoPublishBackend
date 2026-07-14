"""TagAgent — 根据文章正文生成标签

输入（由 Schema 约束）:
  article_body — 文章正文（consume 字段）

产出:
  tags — 标签列表
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from .tool_utils import call_llm
from ..result import agent_response
from src.logging_shared.config import get_logger
import json

logger = get_logger("agent.tags")

SYSTEM_PROMPT = """你是一个关键词标签生成专家。

你的任务是根据文章正文，提取 2-5 个关键词标签。

要求：
1. 标签要准确反映文章核心内容
2. 每个标签 2-6 个字
3. 用中文
4. 直接输出 JSON 数组，如 ["标签1", "标签2", "标签3"]，不要其他内容"""


async def handle_tags(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    try:
        # ── 输入 ──
        article_body: str = task_input["article_body"]

        # 截断正文到 2000 字
        body_preview = article_body[:2000]

        lc_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"## 文章正文\n\n{body_preview}\n\n## 请生成标签（JSON 数组）"),
        ]

        llm_result = await call_llm(
            llm, lc_messages,
            agent_type="TagAgent",
            stream=True,
            on_chunk=task_input.get("_on_llm_chunk"),
        )

        # 解析 JSON 数组
        tags: list[str] = []
        raw = llm_result.content.strip()
        if raw.startswith("["):
            try:
                tags = json.loads(raw)
            except json.JSONDecodeError:
                pass
        if not tags:
            # fallback: 按行拆分
            tags = [t.strip().strip('"-') for t in raw.split("\n") if t.strip()]
        if not tags:
            tags = ["AI", "科技"]

        return agent_response(
            "TagAgent",
            content=json.dumps(tags, ensure_ascii=False),
            output_type="text",
            tokens_used=llm_result.total_tokens,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            output_fields={
                "tags": tags,
            },
            thinking_steps=[{
                "agent": "TagAgent", "action": "标签生成完成",
                "reasoning": f"根据{len(article_body)}字正文生成 {len(tags)} 个标签",
                "output_summary": str(tags),
                "tokens_used": llm_result.total_tokens,
                "status": "done",
            }],
        )
    except Exception as e:
        logger.error("TagAgent 执行失败: %s", e, exc_info=True)
        raise __import__("src.core.exceptions", fromlist=["HandlerError"]).HandlerError(str(e), handler="TagAgent") from e
