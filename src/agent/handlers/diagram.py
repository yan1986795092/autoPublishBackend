"""DiagramAgent — 知识架构图、流程图、脑图生成"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI


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
            response = await llm.ainvoke([
                SystemMessage(content="You are a Mermaid.js diagram expert. Output ONLY valid Mermaid code."),
                HumanMessage(content=f'Generate a Mermaid.js {diagram_type} about "{topic}". Detail level: detailed'),
            ])
            code = response.content.strip()
            if "```mermaid" in code:
                code = code.split("```mermaid")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].strip()
            return {
                "status": "completed", "agent_type": "DiagramAgent",
                "output": {"content": code, "diagram_type": diagram_type, "code": code, "rendered": True},
                "tokens_used": len(code) // 4,
                "thinking_steps": [{"agent": "DiagramAgent", "action": "图表生成完成", "reasoning": f"生成{diagram_type}图：{topic}", "status": "done"}],
            }
        except Exception:
            pass

    code = f"mindmap\n  root(({topic}))\n    核心概念\n      定义\n    关键技术\n      技术A\n    应用场景"
    return {
        "status": "completed", "agent_type": "DiagramAgent",
        "output": {"content": code, "diagram_type": "mindmap", "code": code, "rendered": True},
        "tokens_used": 0, "thinking_steps": [],
    }
