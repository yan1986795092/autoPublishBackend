"""MermaidGen — LangChain BaseTool 版本"""
from __future__ import annotations
import json
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class MermaidGenInput(BaseModel):
    topic: str = Field(description="图表主题")
    diagram_type: str = Field(
        default="mindmap",
        description="图表类型: flowchart, mindmap, sequenceDiagram, classDiagram, graph",
    )
    detail_level: str = Field(default="detailed", description="详细程度: simple, detailed")


class MermaidGenTool(BaseTool):
    name: str = "MermaidGen"
    description: str = "根据描述生成 Mermaid.js 图表（流程图、时序图、思维导图等）"
    args_schema: Type[BaseModel] = MermaidGenInput
    chat_model: Optional[ChatOpenAI] = None

    async def _arun(self, topic: str, diagram_type: str = "mindmap", detail_level: str = "detailed", **kwargs) -> str:
        if self.chat_model:
            return await self._llm_generate(topic, diagram_type, detail_level)
        return json.dumps(self._template_generate(topic, diagram_type, detail_level), ensure_ascii=False)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async _arun")

    async def _llm_generate(self, topic: str, diagram_type: str, detail_level: str) -> str:
        prompt = f'Generate a Mermaid.js {diagram_type} about "{topic}".\nOnly output the Mermaid code, no explanations. Detail level: {detail_level}'
        try:
            response = await self.chat_model.ainvoke([
                SystemMessage(content="You are a Mermaid.js diagram expert. Output ONLY valid Mermaid code."),
                HumanMessage(content=prompt),
            ])
            code = response.content.strip()
            if "```mermaid" in code:
                code = code.split("```mermaid")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].strip()
            return json.dumps({
                "success": True,
                "data": {"code": code, "type": diagram_type, "topic": topic},
                "summary": f"生成 {diagram_type} 图表：{topic}",
            }, ensure_ascii=False)
        except Exception:
            return json.dumps(self._template_generate(topic, diagram_type, detail_level), ensure_ascii=False)

    def _template_generate(self, topic: str, diagram_type: str, detail_level: str) -> dict:
        templates = {
            "mindmap": f"""mindmap\n  root(({topic}))\n    核心概念\n      定义与原理\n      发展历程\n    关键技术\n      技术A\n      技术B\n    应用场景\n      场景1\n      场景2\n    最佳实践\n      实践1\n      实践2""",
            "flowchart": f"""flowchart TD\n    A[{topic}] --> B[需求分析]\n    A --> C[技术选型]\n    B --> D[方案设计]\n    C --> D\n    D --> E[开发实现]\n    E --> F[测试验证]\n    F --> G[部署上线]""",
            "sequenceDiagram": f"""sequenceDiagram\n    participant User as 用户\n    participant System as 系统\n    participant Service as 服务\n    User->>System: 发起{topic}请求\n    System->>Service: 处理请求\n    Service-->>System: 返回结果\n    System-->>User: 展示结果""",
        }
        code = templates.get(diagram_type, templates["mindmap"])
        return {
            "success": True,
            "data": {"code": code, "type": diagram_type, "topic": topic},
            "summary": f"模板生成 {diagram_type} 图表：{topic}",
        }
