"""SEOAgent — 关键词研究和搜索优化"""

from langchain_openai import ChatOpenAI
from ..agent_utils import seo_score, seo_suggestions
from .keyword_tool import extract_keywords, suggest_related_keywords, keyword_density_analysis


async def handle_seo(task_input: dict, llm: ChatOpenAI) -> dict:
    topic = task_input.get("topic", task_input.get("query", ""))
    content = task_input.get("content", "")
    if not content:
        dep_context = task_input.get("_context", {})
        for cdata in dep_context.values():
            if cdata.get("content"):
                content = cdata["content"]
                break

    # 使用 KeywordTool 从主题生成扩展关键词
    related = suggest_related_keywords(topic, content)
    extracted = extract_keywords(content, 5) if content else []

    keywords = [
        {"keyword": topic, "source": "topic", "density_target": 0.02},
        {"keyword": f"{topic} 指南", "source": "suggestion", "density_target": 0.02},
    ]
    for kw in related[1:4]:
        keywords.append({"keyword": kw, "source": "keyword_tool", "density_target": 0.015})
    for kw in extracted[:3]:
        kw_str = kw["keyword"]
        if kw_str != topic and kw_str not in [k["keyword"] for k in keywords]:
            keywords.append({"keyword": kw_str, "source": "extraction", "density_target": 0.01})

    score = seo_score(content, keywords, topic) if content else None
    suggestions = seo_suggestions(content, keywords, score) if score else ["请提供内容以获取 SEO 优化建议"]

    # 添加关键词密度分析
    density = None
    if content and keywords:
        density = keyword_density_analysis(content, [k["keyword"] for k in keywords[:5]])

    return {
        "status": "completed", "agent_type": "SEOAgent",
        "output": {
            "keywords": keywords,
            "seo_score": score,
            "density_analysis": density,
            "related_keywords": related[:8],
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
            "output_summary": f"Score: {score['overall'] if score else 'N/A'}/100, {len(related)} related keywords",
            "status": "done",
        }],
    }
