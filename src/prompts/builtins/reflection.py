"""ReflectionAgent 反思 Prompt 模板"""
from ..engine import PromptTemplateDef


templates = [
    # ── 自省 ──
    PromptTemplateDef(
        name="reflection.self",
        template="""你是一位质量评审专家，请评审你自己的产出。

你扮演的角色：{agent_type}
原始任务：{task_summary}

你的产出内容：
{output_content}

请从以下维度评分（1-5分）：
1. 完整性：是否完整回应了任务要求？
2. 准确性：是否存在事实错误？
3. 清晰度：内容是否清晰结构化？
4. 相关性：是否切题，没有偏题？

请指出具体问题和改进建议。

以 JSON 格式输出（不要包含其他内容）：
```json
{{
  "scores": {{"completeness": 1-5, "accuracy": 1-5, "clarity": 1-5, "relevance": 1-5}},
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"],
  "overall_score": 0.0-1.0
}}```""",
        description="Agent 自省：对自身产出进行质量评分",
        version="1.0",
        role="ReflectionAgent",
        group="reflection",
    ),

    # ── 同行评审 ──
    PromptTemplateDef(
        name="reflection.peer",
        template="""你是一位同行评审专家（{reviewer_agent}），请评审 {target_agent} 的产出。

任务背景：{task_summary}

待评审内容：
{output_content}

请从以下方面评审：
1. 内容质量：是否专业、准确、有深度？
2. 结构逻辑：是否层次清晰、逻辑连贯？
3. 风格匹配：是否符合任务要求的风格？
4. 改进空间：哪些地方可以更好？

以 JSON 格式输出：
```json
{{
  "overall_score": 0.0-1.0,
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"]
}}```""",
        description="同行评审：一个 Agent 评审另一个 Agent 的产出",
        version="1.0",
        role="ReflectionAgent",
        group="reflection",
    ),
]
