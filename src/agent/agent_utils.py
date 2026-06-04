"""Agent 共享常量和工具函数 — 从 nodes.py 提取

违禁词检查相关函数 (check_banned_words / BANNED_WORDS / 正则模式)
统一由 src.utils.evaluator 维护，此处通过 import 引用避免重复。

System Prompt 模板现在统一由 src.prompts 管理，此处保留引用以向后兼容。
"""

from __future__ import annotations

from src.utils.evaluator import (
    BANNED_WORDS,
    check_banned_words,
    extract_output,
    seo_score,
    seo_suggestions,
)

# ── Prompt 模板（从 prompts 模块导入，保持向后兼容）──
from src.prompts import (
    CONTENT_TYPE_PROMPTS,
    STYLE_GUIDES,
)

# ── 这些常量将逐步迁移到 prompts/builtins/ 中 ──
# 保留在此处确保现有 import 不中断

QUALITY_CHECKS = [
    "内容是否完整，有开头和结尾",
    "是否有明显的逻辑断层",
    "语言是否流畅，语感是否自然",
    "是否有违禁词或敏感词",
    "段落长度是否适中，是否便于阅读",
]

SYSTEM_PROMPT_TEMPLATE = """你是一位经验丰富的内容创作者，擅长根据不同场景和需求，创作出自然流畅、有温度的文字。

写作原则：
1. 语言要自然流畅，像和人聊天一样，不要生硬刻板
2. 结构清晰但不教条，让读者感觉舒适
3. 根据内容和受众灵活调整风格，不要机械套模板
4. 适当使用口语化表达，拉近和读者的距离
5. 有自己的观点和态度，不要面面俱到却四平八稳

输出要求：
- 先写下你的思路和判断（用「思考：」开头）
- 然后直接输出正文内容（用「正文：」开头），不要加任何「以下是...」之类的过渡
- 正文内容直接开始，不需要「好的，我来为您...」这样的开场白"""

PLANNER_SYSTEM_PROMPT = """You are a content production manager. Your job is to break down user requests into a plan of sub-tasks that specialized AI agents will execute.

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
- TranslationAgent: Translates content between languages

Rules:
1. ALWAYS analyze the actual user request — do NOT just apply templates. Think about what the user really needs.
2. Simple requests (short, single topic): usually just WritingAgent + ReviewAgent is enough. Do NOT add unnecessary agents.
3. Medium requests (blog posts, platform-specific content): include WebSearch for research, WritingAgent, and ReviewAgent. Add SEOAgent if search visibility matters.
4. Complex requests (multi-part, architecture topics, long-form): WebSearch + WritingAgent + DiagramAgent + SEOAgent + ReviewAgent. Consider PlatformAdapter for multi-platform.
5. Tasks that can run in parallel must have NO dependencies on each other. Put them in the same parallel group.
6. Tasks that MUST wait for others must declare those dependencies explicitly.
7. Always include ReviewAgent at the END as the final quality gate.
8. ALWAYS include WritingAgent for any content creation request.
9. Mark creative tasks (WritingAgent, DiagramAgent) as `retry_on_low_quality: true` so they can improve if quality is low.
10. Response must be valid JSON — no markdown, no explanation, just the JSON object.

Output format:
```json
{
  "goal": "summary of the user's goal",
  "reasoning": "why you chose this plan (2-3 sentences)",
  "tasks": [
    {
      "agent_type": "WritingAgent",
      "input": {"topic": "the topic from the user request"},
      "dependencies": [],
      "reasoning": "why this specific agent is needed for this request"
    }
  ]
}
```

Important: `dependencies` must be empty arrays `[]` for parallel tasks, or arrays of indices `[0, 1]` referring to tasks that must complete first. Output JSON only — no text before or after."""


# ── Plan validation helpers ────────────────────────────────────────────────


_VALID_AGENT_TYPES = {
    "ManagerAgent", "WritingAgent", "ReviewAgent", "SEOAgent",
    "DiagramAgent", "ImageAgent", "WebSearch", "HotTopicAgent",
    "CodeAgent", "TranslationAgent", "PlatformAdapter",
}


def validate_plan_structure(plan: dict) -> tuple[bool, str]:
    """
    Validate that a plan dict has all required fields and correct structure.
    Returns (is_valid, error_message).
    """
    required_keys = {"goal", "reasoning", "tasks"}
    missing = required_keys - set(plan.keys())
    if missing:
        return False, f"Missing required keys: {missing}"

    if not isinstance(plan["tasks"], list):
        return False, "tasks must be a list"

    if len(plan["tasks"]) == 0:
        return False, "tasks cannot be empty"

    task_ids_seen = set()
    for i, task in enumerate(plan["tasks"]):
        if "agent_type" not in task:
            return False, f"Task {i} missing agent_type"
        if task["agent_type"] not in _VALID_AGENT_TYPES:
            return False, f"Unknown agent_type: {task['agent_type']}"
        if "input" not in task:
            return False, f"Task {i} missing input"

        task_ids_seen.add(i)
        for dep in task.get("dependencies", []):
            if not isinstance(dep, int) or dep < 0 or dep >= len(plan["tasks"]):
                return False, f"Task {i} has invalid dependency index: {dep}"

    return True, ""


def extract_plan_from_llm_response(response_content: str) -> dict:
    """
    Extract JSON plan from LLM response. Handles markdown code blocks, plain JSON, etc.
    """
    import json as _json, re as _re

    content = response_content.strip()

    # Try to extract from markdown code block
    code_block_match = _re.search(r"```(?:json)?\s*([\s\S]+?)```", content)
    if code_block_match:
        content = code_block_match.group(1)

    # Try direct JSON parse
    try:
        return _json.loads(content)
    except _json.JSONDecodeError:
        pass

    # Try to find JSON object anywhere in the string
    json_match = _re.search(r"\{[\s\S]*\}", content)
    if json_match:
        try:
            return _json.loads(json_match.group(0))
        except _json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON plan from response: {response_content[:200]}")
