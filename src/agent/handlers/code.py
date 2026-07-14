"""CodeAgent — 代码示例生成、技术文档

输入（由 Schema 约束）:
  topic    — 编程任务描述（必填）
  language — 编程语言（可选，默认 python）

产出:
  code_block — 生成的代码
"""

import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..agent_utils import extract_output
from src.prompts import PromptBuilder
from src.tools.code_runner import CodeRunner
from src.core.exceptions import HandlerError
from src.utils.tokenizer import estimate_tokens
from .tool_utils import call_llm

logger = logging.getLogger(__name__)


async def handle_code(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    try:
        # ── 输入：只读 Schema 声明的字段 ──
        topic: str = task_input["topic"]
        language: str = task_input.get("language", "python")
        run_code: bool = task_input.get("run_code", False)

        # ── 业务逻辑 ──
        builder = PromptBuilder()
        builder.system("system.role", {"role": "技术写作者", "description": f"生成{language}代码示例"})
        builder.instructions("code.task", {"topic": topic, "language": language})

        tool_descriptions = task_input.get("_tool_descriptions", "")
        if tool_descriptions:
            builder.context(f"\n可用工具：\n{tool_descriptions}")

        composed = builder.build()
        lc_messages = builder.build_langchain()

        response_text = await call_llm(
            llm, lc_messages,
            agent_type="CodeAgent",
            stream=False,
        )

        content = extract_output(response_text)
        tokens = estimate_tokens(response_text)

        # ── 产出 ──
        from ..result import agent_response

        return agent_response(
            "CodeAgent",
            content=content,
            output_type="code",
            tokens_used=tokens,
            output_extra={"language": language, "stats": {"code_blocks": content.count("```") // 2}},
            output_fields={
                "code_block": content,
            },
            prompt_versions=composed.used_templates,
            thinking_steps=[{
                "agent": "CodeAgent",
                "action": "代码生成完成",
                "reasoning": f"生成 {language} 代码：{topic[:30]}...",
                "output_summary": f"生成代码块 {content.count('```') // 2} 个",
                "tokens_used": tokens,
                "status": "done",
            }],
        )
    except Exception as e:
        logger.error("CodeAgent 执行失败: %s", e, exc_info=True)
        raise HandlerError(str(e), handler="CodeAgent") from e
