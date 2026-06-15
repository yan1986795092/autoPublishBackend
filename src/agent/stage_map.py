"""Agent 类型 → 前端 Stage 映射表"""

STAGE_MAP: dict[str, dict] = {
    "ManagerAgent":  {"label": "思考分析中", "icon": "🧠", "color": "purple"},
    "WebSearch":     {"label": "搜索资料中", "icon": "🔍", "color": "blue"},
    "URLCrawler":    {"label": "搜索资料中", "icon": "🔍", "color": "blue"},
    "WritingAgent":  {"label": "创作初稿中", "icon": "✍️", "color": "amber"},
    "ReviewAgent":   {"label": "评审优化中", "icon": "📋", "color": "violet"},
    "FixAgent":      {"label": "质量检查中", "icon": "✅", "color": "green"},
    "CodeAgent":     {"label": "代码生成中", "icon": "💻", "color": "emerald"},
    "SEOAgent":      {"label": "SEO 优化中", "icon": "🚀", "color": "orange"},
    "DiagramAgent":  {"label": "图表生成中", "icon": "📊", "color": "pink"},
}

_DEFAULT_STAGE = {"label": "处理中", "icon": "⚙️", "color": "gray"}


def get_stage_info(agent_type: str) -> dict:
    """获取 Agent 类型对应的 Stage 信息，未知类型返回 fallback"""
    return STAGE_MAP.get(agent_type, _DEFAULT_STAGE)


def get_stage_label(agent_type: str) -> str:
    return get_stage_info(agent_type)["label"]


def get_stage_icon(agent_type: str) -> str:
    return get_stage_info(agent_type)["icon"]
