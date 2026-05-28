"""CodeAgent — 代码示例生成、技术文档、沙箱执行

使用 PromptEngine 构建 prompt，支持工具调用。
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
        topic = task_input.get("topic", task_input.get("query", ""))
        language = task_input.get("language", "python")
        run_code = task_input.get("run_code", False)

        # ── 用 PromptBuilder 构建 ──
        builder = PromptBuilder()
        builder.system("system.role", {"role": "技术写作者", "description": f"生成{language}代码示例"})
        builder.instructions("code.task", {"topic": topic, "language": language})

        # 注入工具描述
        tool_descriptions = task_input.get("_tool_descriptions", "")
        if tool_descriptions:
            builder.context(f"\n可用工具：\n{tool_descriptions}")

        composed = builder.build()
        lc_messages = builder.build_langchain()

        # ── 工具感知的 LLM 调用 ──
        response_text = await call_llm(
            llm, lc_messages,
            tool_registry=task_input.get("_tool_registry"),
            agent_type="CodeAgent",
            stream=False,
        )

        content = extract_output(response_text)
        tokens = estimate_tokens(response_text)

        output: dict = {
            "content": content,
            "language": language,
            "type": "text",
            "stats": {"code_blocks": content.count("```") // 2},
        }

        # 沙箱执行
        if run_code and content:
            code = _extract_code_block(content, language)
            if code:
                runner = CodeRunner()
                if language in ("python", "py"):
                    run_result = await runner.run_python(code)
                elif language in ("javascript", "js", "node"):
                    run_result = await runner.run_javascript(code)
                    run_result.language = "javascript"
                else:
                    run_result = await runner.run_python(code)

                output["execution"] = {
                    "success": run_result.success,
                    "stdout": run_result.stdout,
                    "stderr": run_result.stderr,
                    "error": run_result.error,
                    "execution_time_ms": run_result.execution_time_ms,
                }

        return {
            "status": "completed",
            "agent_type": "CodeAgent",
            "output": output,
            "tokens_used": tokens,
            "prompt_versions": composed.used_templates,
            "thinking_steps": [{
                "agent": "CodeAgent",
                "action": "代码执行完成" if run_code else "代码生成完成",
                "reasoning": f"生成 {language} 代码：{topic[:30]}...",
                "output_summary": f"生成代码块 {content.count('```') // 2} 个" +
                    (f"，执行结果：{'成功 ✓' if output.get('execution', {}).get('success') else '失败 ✗'}" if run_code else ""),
                "tokens_used": tokens,
                "status": "done",
            }],
        }
    except Exception as e:
        logger.error("CodeAgent failed: %s", e, exc_info=True)
        raise HandlerError(str(e), handler="CodeAgent") from e


def _extract_code_block(text: str, language: str) -> str:
    """从 markdown 响应中提取代码块"""
    lang_markers = {
        "python": ["```python", "```py", "```"],
        "py": ["```python", "```py", "```"],
        "javascript": ["```javascript", "```js", "```node", "```"],
        "js": ["```javascript", "```js", "```node", "```"],
        "node": ["```javascript", "```js", "```node", "```"],
    }
    markers = lang_markers.get(language, ["```"])

    for marker in markers:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) > 1:
                after = parts[1]
                if "```" in after:
                    return after.split("```", 1)[0].strip()

    return text.strip()
