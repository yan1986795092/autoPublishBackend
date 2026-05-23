"""DiagramAgent 模板 — 图表生成"""

from ..engine import PromptTemplateDef
from ..variables import VariableDef

templates = [
    PromptTemplateDef(
        name="diagram.task",
        template='Generate a Mermaid.js {diagram_type} about "{topic}". Detail level: detailed',
        description="图表生成任务指令",
        version="1.0",
        role="DiagramAgent",
        group="diagram",
        variables=[
            VariableDef("topic", required=True, label="图表主题"),
            VariableDef("diagram_type", required=False, default="mindmap",
                        options=["mindmap", "flowchart", "sequenceDiagram", "classDiagram"],
                        label="图表类型"),
        ],
    ),
]
