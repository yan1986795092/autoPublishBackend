"""DiagramAgent — 知识架构图、流程图、脑图生成

使用 PromptEngine 构建 prompt，支持工具调用。
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from src.prompts import PromptBuilder
from src.utils.tokenizer import estimate_tokens
from .tool_utils import call_llm
import logging

logger = logging.getLogger(__name__)


async def handle_diagram(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    topic = task_input.get("topic", "")
    diagram_type = task_input.get("diagram_type", "mindmap")

    content_lower = (topic + " ").lower()
    if diagram_type == "auto":
        if any(kw in content_lower for kw in ["流程", "步骤", "工作流", "部署", "架构", "模块"]):
            diagram_type = "flowchart"
        elif any(kw in content_lower for kw in ["交互", "api", "请求"]):
            diagram_type = "sequenceDiagram"
        else:
            diagram_type = "mindmap"

    if llm:
        try:
            builder = PromptBuilder()
            builder.system("system.diagram")

            # ── Skill system prompt injection ──
            skill_system_prompt = task_input.get("_skill_system_prompt", "")
            if skill_system_prompt:
                builder.context(f"\n\n=== 创作指导 (Skill: {task_input.get('_skill_name', 'Custom')}) ===\n{skill_system_prompt}\n===")

            builder.instructions("diagram.task", {
                "topic": topic,
                "diagram_type": diagram_type,
            })
            composed = builder.build()
            lc_messages = builder.build_langchain()
            response_text = await call_llm(
                llm, lc_messages,
                tool_registry=task_input.get("_tool_registry"),
                agent_type="DiagramAgent",
                session_id=str(task_input.get("_session_id") or task_input.get("session_id") or "default"),
                run_id=str(task_input.get("_task_id") or ""),
                stream=False,
            )
            code = response_text.strip()
            if "```mermaid" in code:
                code = code.split("```mermaid")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].strip()
            return {
                "status": "completed", "agent_type": "DiagramAgent",
                "output": {"content": code, "diagram_type": diagram_type, "code": code, "rendered": True},
                "tokens_used": estimate_tokens(code),
                "prompt_versions": composed.used_templates,
                "thinking_steps": [{"agent": "DiagramAgent", "action": "图表生成完成", "reasoning": f"生成{diagram_type}图：{topic}", "status": "done"}],
            }
        except Exception as e:
            logger.warning("DiagramAgent LLM 调用失败，回退到模板: %s", e)

    code = f"mindmap\n  root(({topic}))\n    核心概念\n      定义\n    关键技术\n      技术A\n    应用场景"
    return {
        "status": "completed", "agent_type": "DiagramAgent",
        "output": {"content": code, "diagram_type": "mindmap", "code": code, "rendered": True},
        "tokens_used": 0, "thinking_steps": [],
    }
