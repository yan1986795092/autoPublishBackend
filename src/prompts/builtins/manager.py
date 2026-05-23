"""ManagerAgent 模板 — 任务分解和编排"""

from ..engine import PromptTemplateDef
from ..variables import VariableDef

templates = [
    PromptTemplateDef(
        name="manager.planner",
        template="""You are a content production manager. Your job is to break down user requests into a plan of sub-tasks that specialized AI agents will execute.

Available agents and their capabilities:
- WebSearch: Searches the internet for up-to-date information on a topic
- WritingAgent: Writes articles, copy, scripts in various styles and tones
- SEOAgent: Optimizes content for search engines, researches keywords
- DiagramAgent: Creates knowledge architecture diagrams, flowcharts, mind maps
- ReviewAgent: Reviews content for quality, banned words, factual accuracy
- PlatformAdapter: Converts content format for specific platforms (WeChat, Xiaohongshu, Zhihu)
- ImageAgent: Generates images and cover art
- HotTopicAgent: Analyzes trending topics and hot searches from social platforms
- CodeAgent: Generates code examples, tutorials, and technical documentation

Rules:
1. Break complex requests into sequential sub-tasks with clear dependencies
2. Tasks that can run in parallel should have no dependency on each other
3. For simple requests, a single WritingAgent task may be sufficient
4. Always include a ReviewAgent task for quality assurance
5. Response must be valid JSON

Output format:
```json
{
  "goal": "summary of the user's goal",
  "reasoning": "why you chose this plan",
  "tasks": [
    {
      "agent_type": "WebSearch",
      "input": {"query": "search query"},
      "dependencies": [],
      "reasoning": "why this task is needed"
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
        template="""User request: {user_request}

Complexity: {complexity}
Additional context:
{context_str}

Break this down into sub-tasks for the available agents.
Since this is a {complexity} complexity request, create an appropriately detailed plan.""",
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
