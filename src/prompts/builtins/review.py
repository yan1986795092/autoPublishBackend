"""ReviewAgent 模板 — 内容审核和质量检查"""

from ..engine import PromptTemplateDef
from ..variables import VariableDef

templates = [
    PromptTemplateDef(
        name="review.task",
        template="""## 审核任务{memory_section}

请对以下{content_type}内容进行审核，重点关注：
- 内容是否完整，有没有虎头蛇尾
- 逻辑是否通顺，有没有前后矛盾
- 语言是否自然，读起来顺不顺口
- 有没有违禁词、敏感词或者容易引发争议的表达
- 段落节奏怎么样，长短搭配是否舒适

内容如下：
{content}

请用中文回复，直接给出审核意见，不要用英文。格式参考：
**审核结论：** [通过 / 需要修改]
**主要问题：** [列出发现的问题，如果没有就说"整体质量良好"]
**改进建议：** [具体的改进方向，如果有的话]""",
        description="审核任务指令",
        version="2.0",
        role="ReviewAgent",
        group="review",
        variables=[
            VariableDef("content", required=True, label="待审核内容"),
            VariableDef("content_type", required=False, default="general", label="内容类型"),
            VariableDef("memory_section", required=False, default="", label="记忆上下文"),
        ],
    ),
]
