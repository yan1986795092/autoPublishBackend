"""DiagramAgent — 知识架构图、流程图、脑图生成

输入（由 Schema 约束）:
  topic        — 图表主题（必填）
  diagram_type — 图表类型: flowchart/mindmap/sequence（可选，默认 mindmap）

产出:
  diagram — Mermaid 格式图表定义
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from src.prompts import PromptBuilder
from src.utils.tokenizer import estimate_tokens
from .tool_utils import call_llm
import logging

logger = logging.getLogger(__name__)


async def handle_diagram(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    # ── 输入：只读 Schema 声明的字段 ──
    topic: str = task_input["topic"]
    diagram_type: str = task_input.get("diagram_type", "mindmap")

    # ── 业务逻辑 ──
    content_lower = (topic + " ").lower()
    if diagram_type == "auto":
        if any(kw in content_lower for kw in ["流程", "步骤", "工作流", "部署", "架构", "模块"]):
            diagram_type = "flowchart"
        elif any(kw in content_lower for kw in ["交互", "api", "请求"]):
            diagram_type = "sequenceDiagram"
        else:
            diagram_type = "mindmap"

    code = ""
    if llm:
        try:
            builder = PromptBuilder()
            builder.system("system.diagram")
            builder.instructions("diagram.task", {
                "topic": topic,
                "diagram_type": diagram_type,
            })
            composed = builder.build()
            lc_messages = builder.build_langchain()
            response_text = await call_llm(
                llm, lc_messages,
                agent_type="DiagramAgent",
                stream=False,
            )
            code = response_text.strip()
            if "```mermaid" in code:
                code = code.split("```mermaid")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].strip()
        except Exception as e:
            logger.warning("DiagramAgent LLM 调用失败，回退到模板: %s", e)

    if not code:
        code = f"mindmap\n  root(({topic}))\n    核心概念\n      定义\n    关键技术\n      技术A\n    应用场景"

    from ..result import agent_response

    # ── 产出：只填 Schema 声明的 output_fields ──
    return agent_response(
        "DiagramAgent",
        content=code,
        output_type="diagram",
        tokens_used=estimate_tokens(code) if code else 0,
        output_extra={"diagram_type": diagram_type, "code": code, "rendered": True},
        output_fields={
            "diagram": code,
        },
        thinking_steps=[{
            "agent": "DiagramAgent", "action": "图表生成完成",
            "reasoning": f"生成{diagram_type}图：{topic}",
            "output_summary": code[:80],
            "status": "done",
        }] if code else [],
    )
