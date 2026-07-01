"""Agent Registry — 基于 AgentCard 模式的统一注册中心

将分散在各处的 Agent 元数据（描述、模型分配、前端展示、类别）集中管理。
新 Agent 只需在此注册，其余模块自动感知。

灵感来自 Google A2A 协议的 AgentCard，但适配同进程调用场景（无 HTTP 开销）。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Any

# ── 类型 ──

AgentCategory = Literal["search", "writing", "review", "fix", "code", "diagram", "analysis"]


@dataclass
class AgentCard:
    """Agent 能力自描述卡片（同进程版 AgentCard）

    对应 A2A AgentCard 的核心字段，移除 url/streaming 等 HTTP 相关字段。
    """

    # 身份
    name: str                                                # "WritingAgent"
    description: str                                         # 给人看的描述

    # 分类（用于依赖链推断）
    category: AgentCategory                                  # "writing" | "review" | ...

    # 模型分配
    model: str                                               # "deepseek-chat" | "glm-4-flash"

    # 前端展示
    stage_label: str                                         # "创作初稿中"
    stage_icon: str                                          # "✍️"
    stage_color: str                                         # "amber"

    # 技能定义（JSON Schema，供 LLM 了解精确参数格式，替代"猜字段"）
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)


# ── 注册表 ──

_REGISTRY: dict[str, AgentCard] = {}

# 类别 → 依赖规则：下游类自动依赖上游类的最新任务
# 搜索 → 写作/代码 → 审核 → 修复
CATEGORY_DEPENDENCY_RULES: dict[AgentCategory, list[AgentCategory]] = {
    "writing":  ["search", "analysis"],  # writing 依赖搜索和热点分析
    "code":     ["search"],
    "review":   ["writing"],
    "fix":      ["writing", "review"],
    "analysis": ["search"],              # 热点分析依赖搜索
}


def register(card: AgentCard) -> None:
    """注册一个 AgentCard"""
    _REGISTRY[card.name] = card


def get(name: str) -> AgentCard | None:
    """按 Agent 类型名获取卡片"""
    return _REGISTRY.get(name)


def all() -> list[AgentCard]:
    """获取所有注册的 Agent 卡片"""
    return list(_REGISTRY.values())


def by_category(category: AgentCategory) -> list[AgentCard]:
    """按类别获取 Agent 卡片列表"""
    return [c for c in _REGISTRY.values() if c.category == category]


def categories() -> dict[AgentCategory, list[AgentCard]]:
    """获取按类别分组的 Agent 列表"""
    result: dict[AgentCategory, list[AgentCard]] = {}
    for c in _REGISTRY.values():
        result.setdefault(c.category, []).append(c)
    return result


def format_agent_descriptions() -> str:
    """生成给 LLM Planner 看的 Agent 描述文本（替代硬编码字符串）"""
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
    """为每个 Agent 生成 JSON Schema 描述，供 LLM 精确理解输入格式"""
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
    description="【实时联网搜索】通过搜索引擎获取最新信息。当你需要当前日期、实时新闻、最新数据、流行趋势、或任何 LLM 训练数据中不存在的信息时，必须用此 Agent。",
    category="search",
    model="qwen3.7-max",
    stage_label="搜索资料中",
    stage_icon="🔍",
    stage_color="blue",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "count": {"type": "integer", "description": "返回结果数量"},
        },
        "required": ["query"],
    },
))

register(AgentCard(
    name="URLCrawler",
    description="抓取并提取指定 URL 的内容。",
    category="search",
    model="qwen3.7-max",
    stage_label="搜索资料中",
    stage_icon="🔍",
    stage_color="blue",
    input_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的网页地址"},
        },
        "required": ["url"],
    },
))

register(AgentCard(
    name="WritingAgent",
    description="内容写作（文章、文案、脚本）。内容创作的核心 Agent。",
    category="writing",
    model="qwen3.7-max",
    stage_label="创作初稿中",
    stage_icon="✍️",
    stage_color="amber",
    input_schema={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "写作主题，核心必填参数"},
            "content_type": {"type": "string", "enum": ["blog", "article", "copywriting", "wechat", "script", "tutorial"], "description": "内容类型"},
            "tone": {"type": "string", "description": "语气风格，如 professional/friendly/humorous"},
            "audience": {"type": "string", "description": "目标受众描述"},
            "language": {"type": "string", "description": "输出语言"},
        },
        "required": ["topic"],
    },
))

register(AgentCard(
    name="ReviewAgent",
    description="审核内容质量、准确性、违禁词。Review 之后必须紧跟 FixAgent。",
    category="review",
    model="qwen3.7-max",
    stage_label="评审优化中",
    stage_icon="📋",
    stage_color="violet",
    input_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "待审核的内容"},
            "rules": {"type": "array", "items": {"type": "string"}, "description": "审核规则列表"},
        },
    },
))

register(AgentCard(
    name="FixAgent",
    description="根据审核意见修复内容。",
    category="fix",
    model="qwen3.7-max",
    stage_label="润色优化中",
    stage_icon="✨",
    stage_color="green",
    input_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "原始内容"},
            "feedback": {"type": "string", "description": "审核反馈意见"},
        },
    },
))

register(AgentCard(
    name="DiagramAgent",
    description="创建知识图谱、流程图、思维导图。",
    category="diagram",
    model="qwen3.7-max",
    stage_label="图表生成中",
    stage_icon="📊",
    stage_color="pink",
    input_schema={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "图表主题"},
            "diagram_type": {"type": "string", "enum": ["flowchart", "mindmap", "knowledge_graph", "sequence"], "description": "图表类型"},
        },
        "required": ["topic"],
    },
))

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
            "task": {"type": "string", "description": "编程任务详细描述"},
            "language": {"type": "string", "description": "编程语言"},
        },
        "required": ["task"],
    },
))

register(AgentCard(
    name="HotTopicAgent",
    description="【热点分析】基于 WebSearch 获取的搜索结果，分析社交平台的热点话题趋势和洞察。必须依赖 WebSearch 的输出，不可独立搜索。",
    category="analysis",
    model="qwen3.7-max",
    stage_label="整合分析中",
    stage_icon="🔬",
    stage_color="orange",
    input_schema={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "分析主题"},
            "platform": {"type": "string", "description": "关注的社交平台"},
        },
    },
))
