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
- SEOAgent: Optimizes and rewrites content for search engines/platforms; when rewrite=True, outputs the revised content directly
- DiagramAgent: Creates knowledge architecture diagrams, flowcharts, mind maps
- ReviewAgent: Reviews content for quality, banned words, factual accuracy
- PlatformAdapter: Converts content format for specific platforms (WeChat, Xiaohongshu, Zhihu)
- ImageAgent: Generates images and cover art
- HotTopicAgent: Analyzes trending topics and hot searches from social platforms
- CodeAgent: Generates code examples, tutorials, and technical documentation
- TranslationAgent: Translates content between languages

Platform-specific guidance (use the platform code matching the frontend PlatformSelector):
- Xiaohongshu (xiaohongshu): content_type="xiaohongshu", conversational tone, emojis, short paragraphs, authentic voice, CTA at end
- WeChat Official Account (wechat): content_type="wechat", deeper analysis, warmer tone, structured with subheadings, CTA at end
- Weibo (weibo): content_type="weibo", punchy opinions, 140-500 chars, hashtags, shareable
- Douyin (douyin): content_type="douyin", video script format, 3s hook, spoken language
- Bilibili (bilibili): content_type="bilibili", medium-long video script, UP主 style, danmaku interaction
- Copywriting (copywriting): content_type="copywriting", punchy headlines, clear value prop, strong CTA
- Brand Story (brand-story): content_type="brand-story", narrative voice, emotional resonance, brand identity
- Ad Copy (ad-copy): content_type="ad-copy", conversion-driven, urgent, targeted
- Fiction (fiction): content_type="fiction", narrative voice, vivid characters, engaging plot
- Script (script): content_type="script", natural dialogue, stage directions
- Technical Doc (tech-doc): content_type="tech-doc", clear structure, code examples, actionable
- Tutorial (tutorial): content_type="tutorial", step-by-step, beginner-friendly, common pitfalls noted
- SEO Article (seo): content_type="seo", keyword-optimized, search intent, featured snippets
- Email Marketing (email): content_type="email", personalized, valuable, clear CTA, mobile-friendly

SEO mode by platform:
- Xiaohongshu/copywriting/ad-copy/seo/tutorial/tech-doc: set SEOAgent rewrite=True (rewrite content)
- WeChat/Weibo/brand-story: set SEOAgent rewrite=False (analyze only)
- Fiction/script/bilibili/douyin: skip SEOAgent entirely
- Email: SEOAgent analyze title (rewrite=False)

Rules:
1. Analyze the actual user request and decide which agents to use. There is no fixed template — the right plan depends entirely on what the user asked for.
2. Use WebSearch ONLY if the topic requires current information, facts, data, or trending topics. Don't use it for purely creative requests (fiction, personal essays, etc.).
3. Tasks that can run in parallel must have NO dependencies on each other. Put them in the same parallel group.
4. Tasks that MUST wait for others must declare those dependencies explicitly.
5. Always include ReviewAgent at the END as the final quality gate.
6. ALWAYS include WritingAgent for any content creation request.
7. Mark creative tasks (WritingAgent, SEOAgent with rewrite) as `retry_on_low_quality: true`.
8. Response must be valid JSON — no markdown, no explanation, just the JSON object.

Output format:
```json
{
  "goal": "summary of the user's goal",
  "reasoning": "why you chose this plan (2-3 sentences)",
  "tasks": [
    {
      "agent_type": "WritingAgent",
      "input": {"topic": "the topic", "content_type": "xhs_post or relevant type"},
      "dependencies": [],
      "reasoning": "why this specific agent is needed for this request"
    }
  ]
}
```

Important: `dependencies` must be empty arrays `[]` for parallel tasks, or arrays of indices `[0, 1]` referring to tasks that must complete first. Output JSON only — no text before or after."""


# ── 计划验证辅助函数 ────────────────────────────────────────────────


_VALID_AGENT_TYPES = {
    "ManagerAgent", "WritingAgent", "ReviewAgent", "SEOAgent",
    "DiagramAgent", "ImageAgent", "WebSearch", "HotTopicAgent",
    "CodeAgent", "TranslationAgent", "PlatformAdapter",
}


def validate_plan_structure(plan: dict) -> tuple[bool, str]:
    """
    验证计划字典是否包含所有必需字段且结构正确。
    返回 (is_valid, error_message)。
    """
    required_keys = {"goal", "reasoning", "tasks"}
    missing = required_keys - set(plan.keys())
    if missing:
        return False, f"缺少必需字段: {missing}"

    if not isinstance(plan["tasks"], list):
        return False, "tasks 必须是列表"

    if len(plan["tasks"]) == 0:
        return False, "tasks 不能为空"

    task_ids_seen = set()
    for i, task in enumerate(plan["tasks"]):
        if "agent_type" not in task:
            return False, f"任务 {i} 缺少 agent_type"
        if task["agent_type"] not in _VALID_AGENT_TYPES:
            return False, f"未知的 agent_type: {task['agent_type']}"
        if "input" not in task:
            return False, f"任务 {i} 缺少 input"

        task_ids_seen.add(i)
        for dep in task.get("dependencies", []):
            if not isinstance(dep, int) or dep < 0 or dep >= len(plan["tasks"]):
                return False, f"任务 {i} 包含无效的依赖索引: {dep}"

    return True, ""


def extract_plan_from_llm_response(response_content: str) -> dict:
    """
    从 LLM 响应中提取 JSON 计划。处理 markdown 代码块、纯 JSON 等。
    """
    import json as _json, re as _re

    content = response_content.strip()

    # 尝试从 markdown 代码块中提取
    code_block_match = _re.search(r"```(?:json)?\s*([\s\S]+?)```", content)
    if code_block_match:
        content = code_block_match.group(1)

    # 尝试直接解析 JSON
    try:
        return _json.loads(content)
    except _json.JSONDecodeError:
        pass

    # 尝试在字符串中查找 JSON 对象
    json_match = _re.search(r"\{[\s\S]*\}", content)
    if json_match:
        try:
            return _json.loads(json_match.group(0))
        except _json.JSONDecodeError:
            pass

    raise ValueError(f"无法从响应中提取有效的 JSON 计划: {response_content[:200]}")
