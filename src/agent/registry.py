"""Agent Registry — 基于 AgentCard 模式的统一注册中心

每个 Agent 的 input_schema / output_schema 是唯一的数据契约。
orchestrator 按 schema 装配输入，handler 只消费 schema 声明的字段。

source 规则:
  "param:xxx"   — 从 task input params 原样取
  "consume"     — 沿依赖链 BFS 找最近产出该字段名的任务
  "meta"        — 从 task 元数据取（如 _review_round）
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Any

# ── 类型 ──

AgentCategory = Literal["search", "writing", "review", "fix", "code", "diagram", "analysis", "title_tag"]


@dataclass
class AgentCard:
    """Agent 能力自描述卡片"""

    # 身份
    name: str
    description: str

    # 分类（用于依赖链推断）
    category: AgentCategory

    # 模型分配
    model: str

    # 前端展示
    stage_label: str
    stage_icon: str
    stage_color: str

    # 输入契约（JSON Schema 格式 + 扩展 source/consume）
    input_schema: dict[str, Any] = field(default_factory=dict)
    # 输出契约（JSON Schema 格式，字段名即产出物名）
    output_schema: dict[str, Any] = field(default_factory=dict)


# ── 注册表 ──

_REGISTRY: dict[str, AgentCard] = {}

CATEGORY_DEPENDENCY_RULES: dict[AgentCategory, list[AgentCategory]] = {
    "title_tag": ["writing"],
    "review":    ["writing", "title_tag"],
    "fix":       ["writing", "review"],
    "analysis":  ["search"],
}


def register(card: AgentCard) -> None:
    _REGISTRY[card.name] = card


def get(name: str) -> AgentCard | None:
    return _REGISTRY.get(name)


def all() -> list[AgentCard]:
    return list(_REGISTRY.values())


def by_category(category: AgentCategory) -> list[AgentCard]:
    return [c for c in _REGISTRY.values() if c.category == category]


def categories() -> dict[AgentCategory, list[AgentCard]]:
    result: dict[AgentCategory, list[AgentCard]] = {}
    for c in _REGISTRY.values():
        result.setdefault(c.category, []).append(c)
    return result


def format_agent_descriptions() -> str:
    lines = []
    for card in _REGISTRY.values():
        parts = [f"- {card.name}：{card.description}"]
        if card.input_schema:
            props = card.input_schema.get("properties", {})
            required = card.input_schema.get("required", [])
            if props:
                param_str = ", ".join(
                    f"{k}({'必填' if k in required else '可选'})"
                    for k in props
                )
                parts.append(f"  参数: {param_str}")
        lines.append("".join(parts))
    return "\n".join(lines)


def format_agent_schemas_for_prompt() -> str:
    blocks = []
    for card in _REGISTRY.values():
        if card.input_schema:
            import json
            schema_str = json.dumps(card.input_schema, ensure_ascii=False, indent=2)
            blocks.append(f"## {card.name}\n```json\n{schema_str}\n```")
    return "\n\n".join(blocks)


# ── 注册全部 Agent ──

register(AgentCard(
    name="WebSearch",
    description="【实时联网搜索】通过搜索引擎获取最新信息。",
    category="search",
    model="deepseek-chat",
    stage_label="搜索资料中",
    stage_icon="🔍",
    stage_color="blue",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "source": "param:query", "description": "搜索关键词"},
            "count": {"type": "integer", "source": "param:count", "description": "返回结果数量", "optional": True},
        },
        "required": ["query"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "search_results": {"type": "array", "description": "搜索结果列表"},
        },
        "required": ["search_results"],
    },
))

# TEMP: URLCrawler disabled
# register(AgentCard(
#     name="URLCrawler",
#     description="抓取并提取指定 URL 的内容。",
#     category="search",
#     model="deepseek-chat",
#     stage_label="搜索资料中",
#     stage_icon="🔍",
#     stage_color="blue",
#     input_schema={
#         "type": "object",
#         "properties": {
#             "url": {"type": "string", "source": "param:url", "description": "要抓取的网页地址"},
#         },
#         "required": ["url"],
#     },
#     output_schema={
#         "type": "object",
#         "properties": {
#             "crawled_content": {"type": "string", "description": "抓取到的页面内容"},
#         },
#         "required": ["crawled_content"],
#     },
# ))

register(AgentCard(
    name="WritingAgent",
    description="内容写作（文章、文案、脚本）。只输出正文，不包含标题和标签。标题和标签由 TitleAgent 和 TagAgent 生成。",
    category="writing",
    model="deepseek-chat",
    stage_label="创作初稿中",
    stage_icon="✍️",
    stage_color="amber",
    input_schema={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "source": "param:topic", "description": "写作主题"},
            "content_type": {"type": "string", "source": "param:content_type", "description": "内容类型: blog/article/wechat/...", "optional": True},
            "platform": {"type": "string", "source": "param:platform", "description": "目标平台", "optional": True},
            "tone": {"type": "string", "source": "param:tone", "description": "语气: professional/friendly/humorous/...", "optional": True},
            "audience": {"type": "string", "source": "param:audience", "description": "目标受众: beginner/intermediate/expert", "optional": True},
            "search_analysis": {"type": "string", "source": "consume", "description": "搜索结果分析（来自 SearchAnalyzer）", "optional": True},
            # 平台写作 DNA（由 nodes.py 从 platform_profile 注入，纯透传）
            "_platform_system_prompt": {"type": "string", "source": "meta", "description": "平台角色定位 system prompt", "optional": True},
            "_platform_writing_rules": {"type": "object", "source": "meta", "description": "平台结构化写作规则", "optional": True},
            "_content_genre": {"type": "string", "source": "meta", "description": "内容体裁: text(图文)/video(视频脚本)/audio(音频)", "optional": True},
        },
        "required": ["topic"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "article_body": {"type": "string", "description": "文章正文"},
        },
        "required": ["article_body"],
    },
))

# ── TitleAgent：根据文章正文生成标题 ──
register(AgentCard(
    name="TitleAgent",
    description="【标题生成】根据文章正文生成吸引人的标题。可独立使用（内容管理调用）。",
    category="title_tag",
    model="deepseek-chat",
    stage_label="生成标题中",
    stage_icon="🏷️",
    stage_color="amber",
    input_schema={
        "type": "object",
        "properties": {
            "article_body": {"type": "string", "source": "consume", "description": "文章正文"},
        },
        "required": ["article_body"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "article_title": {"type": "string", "description": "文章标题"},
        },
        "required": ["article_title"],
    },
))

# ── TagAgent：根据文章正文生成标签 ──
register(AgentCard(
    name="TagAgent",
    description="【标签生成】根据文章正文提取关键词标签。可独立使用（内容管理调用）。",
    category="title_tag",
    model="deepseek-chat",
    stage_label="生成标签中",
    stage_icon="🔖",
    stage_color="amber",
    input_schema={
        "type": "object",
        "properties": {
            "article_body": {"type": "string", "source": "consume", "description": "文章正文"},
        },
        "required": ["article_body"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}, "description": "标签列表"},
        },
        "required": ["tags"],
    },
))

register(AgentCard(
    name="ReviewAgent",
    description="审核内容质量、准确性、违禁词。Review 之后必须紧跟 FixAgent。",
    category="review",
    model="deepseek-chat",
    stage_label="评审优化中",
    stage_icon="📋",
    stage_color="violet",
    input_schema={
        "type": "object",
        "properties": {
            "article_body": {"type": "string", "source": "consume", "description": "待审核的文章正文"},
            "platform": {"type": "string", "source": "param:platform", "description": "目标平台", "optional": True},
        },
        "required": ["article_body"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "review_text": {"type": "string", "description": "审核结论 + 主要问题 + 改进建议"},
            "passed": {"type": "boolean", "description": "审核是否通过"},
            "suggestions": {"type": "array", "items": {"type": "string"}, "description": "改进建议列表"},
        },
        "required": ["review_text", "passed"],
    },
))

register(AgentCard(
    name="FixAgent",
    description="根据审核意见修复内容。",
    category="fix",
    model="deepseek-chat",
    stage_label="润色优化中",
    stage_icon="✨",
    stage_color="green",
    input_schema={
        "type": "object",
        "properties": {
            "article_body": {"type": "string", "source": "consume", "description": "原文（WritingAgent 初稿 或 上一轮 FixAgent 润色版）"},
            "review_text": {"type": "string", "source": "consume", "description": "审核意见（ReviewAgent 输出）"},
            "platform": {"type": "string", "source": "param:platform", "description": "目标平台", "optional": True},
        },
        "required": ["article_body", "review_text"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "article_body": {"type": "string", "description": "润色后的文章正文"},
        },
        "required": ["article_body"],
    },
))

# TEMP: DiagramAgent disabled
# register(AgentCard(
#     name="DiagramAgent",
#     description="创建知识图谱、流程图、思维导图。",
#     category="diagram",
#     model="deepseek-chat",
#     stage_label="图表生成中",
#     stage_icon="📊",
#     stage_color="pink",
#     input_schema={
#         "type": "object",
#         "properties": {
#             "topic": {"type": "string", "source": "param:topic", "description": "图表主题"},
#             "diagram_type": {"type": "string", "source": "param:diagram_type", "enum": ["flowchart", "mindmap", "knowledge_graph", "sequence"], "description": "图表类型", "optional": True},
#         },
#         "required": ["topic"],
#     },
#     output_schema={
#         "type": "object",
#         "properties": {
#             "diagram": {"type": "string", "description": "图表定义（Mermaid/JSON）"},
#         },
#         "required": ["diagram"],
#     },
# ))

register(AgentCard(
    name="CodeAgent",
    description="生成代码、教程、技术文档。",
    category="code",
    model="deepseek-chat",
    stage_label="代码生成中",
    stage_icon="💻",
    stage_color="emerald",
    input_schema={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "source": "param:topic", "description": "编程任务描述"},
            "language": {"type": "string", "source": "param:language", "description": "编程语言", "optional": True},
        },
        "required": ["topic"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "code_block": {"type": "string", "description": "生成的代码"},
        },
        "required": ["code_block"],
    },
))

register(AgentCard(
    name="PlannerAgent",
    description="【任务规划】根据用户意图自动拆解为子任务序列，决定后续 Agent 调用链。",
    category="analysis",
    model="deepseek-chat",
    stage_label="规划任务中",
    stage_icon="📋",
    stage_color="slate",
    input_schema={
        "type": "object",
        "properties": {
            "user_request": {"type": "string", "source": "param:user_request", "description": "用户原始请求"},
            "platform": {"type": "string", "source": "param:platform", "description": "目标平台", "optional": True},
        },
        "required": ["user_request"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "plan": {"type": "object", "description": "任务规划"},
        },
        "required": ["plan"],
    },
))

register(AgentCard(
    name="SearchAnalyzer",
    description="【搜索结果分析】基于 WebSearch 获取的搜索结果，提取关键主题、结构和洞察。",
    category="analysis",
    model="deepseek-chat",
    stage_label="整合分析中",
    stage_icon="🔬",
    stage_color="orange",
    input_schema={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "source": "param:topic", "description": "分析主题"},
            "search_results": {"type": "array", "source": "consume", "description": "搜索结果（来自 WebSearch）", "optional": True},
        },
        "required": ["topic"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "search_analysis": {"type": "string", "description": "搜索结果分析"},
        },
        "required": ["search_analysis"],
    },
))
