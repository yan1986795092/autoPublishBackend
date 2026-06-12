"""ManagerAgent 模板 — 任务分解和编排"""

from ..engine import PromptTemplateDef
from ..variables import VariableDef

templates = [
    PromptTemplateDef(
        name="manager.planner",
        template="""你是一个内容制作经理。你的工作是将用户请求分解为专业 AI 智能体执行的子任务计划。

可用智能体及其能力：
- WebSearch：在互联网上搜索某个主题的最新信息
- WritingAgent：以各种风格和语气撰写文章、文案、脚本
- SEOAgent：优化内容的搜索引擎表现，研究关键词
- DiagramAgent：创建知识架构图、流程图、思维导图
- ReviewAgent：对内容进行质量、违禁词、事实准确性审查
- PlatformAdapter：将内容格式转换为特定平台（微信、小红书、知乎）所需的格式
- ImageAgent：生成图片和封面图
- HotTopicAgent：分析社交平台的热门话题和热搜
- CodeAgent：生成代码示例、教程和技术文档

规则：
1. 将复杂请求分解为具有明确依赖关系的顺序子任务
2. 可以并行执行的任务之间不应有依赖关系
3. 对于简单请求，单个 WritingAgent 任务可能就足够了
4. 始终包含 ReviewAgent 任务以保证质量
5. 响应必须是有效的 JSON

输出格式：
```json
{
  "goal": "用户目标的摘要",
  "reasoning": "选择该计划的原因",
  "tasks": [
    {
      "agent_type": "WebSearch",
      "input": {"query": "搜索查询"},
      "dependencies": [],
      "reasoning": "为什么需要此任务"
    }
  ]
}```""",
        description="Manager Agent 任务分解 System Prompt",
        version="2.0",
        role="ManagerAgent",
        group="manager",
    ),

    PromptTemplateDef(
        name="manager.user_request",
        template="""用户请求：{user_request}

复杂度：{complexity}
附加上下文：
{context_str}

为可用智能体将此请求分解为子任务。
由于这是 {complexity} 复杂度请求，请创建一个适当详细程度的计划。""",
        description="Manager 用户请求模板",
        version="1.0",
        role="ManagerAgent",
        group="manager",
        variables=[
            VariableDef("user_request", required=True, label="用户请求"),
            VariableDef("complexity", required=False, default="medium",
                        options=["simple", "medium", "complex"], label="复杂度"),
            VariableDef("context_str", required=False, default="{}", label="上下文 JSON"),
        ],
    ),
]
