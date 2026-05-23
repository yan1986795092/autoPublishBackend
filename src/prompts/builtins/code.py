"""CodeAgent 模板 — 代码生成和技术文档"""

from ..engine import PromptTemplateDef
from ..variables import VariableDef

templates = [
    PromptTemplateDef(
        name="code.task",
        template="""## 技术内容创作

主题：{topic}
语言：{language}

要求：
- 代码可运行
- 添加中文注释
- 考虑边界情况""",
        description="代码生成任务指令",
        version="1.0",
        role="CodeAgent",
        group="code",
        variables=[
            VariableDef("topic", required=True, label="主题"),
            VariableDef("language", required=False, default="python",
                        options=["python", "javascript", "go", "rust", "java", "typescript"],
                        label="编程语言"),
        ],
    ),
]
