"""SEOAgent — 关键词研究和搜索优化"""

from langchain_openai import ChatOpenAI
from ..agent_utils import seo_score, seo_suggestions


async def handle_seo(task_input: dict, llm: ChatOpenAI) -> dict:
    topic = task_input.get("topic", task_input.get("query", ""))
    content = task_input.get("content", "")
    if not content:
        dep_context = task_input.get("_context", {})
        for cdata in dep_context.values():
            if cdata.get("content"):
                content = cdata["content"]
                break

    keywords = [
        {"keyword": topic, "source": "topic", "density_target": 0.02},
        {"keyword": f"{topic} 指南", "source": "suggestion", "density_target": 0.02},
    ]

    score = seo_score(content, keywords, topic) if content else None
    suggestions = seo_suggestions(content, keywords, score) if score else ["请提供内容以获取 SEO 优化建议"]

    return {
        "status": "completed", "agent_type": "SEOAgent",
        "output": {
            "keywords": keywords,
            "seo_score": score,
            "suggestions": suggestions,
            "meta": {
                "title": topic[:60],
                "description": (content or f"关于{topic}的深度内容")[:160],
                "keywords": ", ".join(k["keyword"] for k in keywords[:5]),
            },
        },
        "tokens_used": 0,
        "thinking_steps": [{
            "agent": "SEOAgent", "action": "seo_score",
            "reasoning": f"SEO analysis for: {topic}",
            "output_summary": f"Score: {score['overall'] if score else 'N/A'}/100",
            "status": "done",
        }],
    }
