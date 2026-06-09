"""Agent 角色身份定义

统一收敛当前散落在 nodes.py / memory/manager.py / factory.py 中的 Agent 元数据。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AgentProfile:
    """Agent 角色身份定义

    每个 Agent 实例拥有唯一的 profile，描述其角色、人格、专长。
    """

    # ── 身份标识 ──
    name: str                         # Agent 类型名称, e.g. "WritingAgent"
    role: str                         # 可读角色名, e.g. "内容创作者"
    description: str = ""             # 一句话描述

    # ── 对话人格（供角色化 prompt 使用）──
    persona: str = ""                 # 角色设定, e.g. "你是一位有 10 年经验的技术写作者…"
    expertise: list[str] = field(default_factory=list)  # 专长领域
    communication_style: str = ""     # "专业严谨" / "轻松幽默" / "简洁直接"

    # ── 执行配置 ──
    token_budget: int = 2000
    memory_namespace: str = "general"
    tools_enabled: bool = False
    available_tools: list[str] = field(default_factory=list)  # 允许使用的工具名称列表，空=全部
    cacheable: bool = False           # 是否可缓存（对应 cacheable_types）
    memory_enabled: bool = False      # 是否注入记忆上下文
    remember_enabled: bool = False    # 产出是否值得沉淀

    # ── 路由 ──
    handler_key: str = ""             # HANDLER_MAP 的 key，空则按 name 查找

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AgentProfile:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ──────────────────────────────────────────────
# 预定义 Agent 角色（单一定义源）
# 统一了 nodes.py AGENT_TOKEN_BUDGETS、
# memory/manager.py AGENT_NAMESPACE_MAP / AGENT_MEMORY_ENABLED / AGENT_REMEMBER_ENABLED、
# nodes.py cacheable_types
# ──────────────────────────────────────────────

PREDEFINED_PROFILES: dict[str, AgentProfile] = {
    "ManagerAgent": AgentProfile(
        name="ManagerAgent",
        role="内容生产经理",
        description="负责任务拆解、调度编排、质量把控",
        persona="你是一位资深内容生产经理，擅长将复杂需求拆解为可执行的子任务，并协调各角色协作。",
        expertise=["任务规划", "质量把控", "资源调度"],
        communication_style="专业严谨",
        token_budget=4000,
        tools_enabled=True,
    ),
    "WritingAgent": AgentProfile(
        name="WritingAgent",
        role="内容创作者",
        description="多风格内容写作",
        persona="你是一位经验丰富的写作者，擅长各种内容形式的创作，从技术博客到社交媒体文案都得心应手。",
        expertise=["博客", "社交媒体", "营销文案", "视频脚本", "Newsletter", "技术写作"],
        communication_style="自然流畅",
        token_budget=4000,
        memory_namespace="content",
        memory_enabled=True,
        remember_enabled=True,
        tools_enabled=True,
    ),
    "ReviewAgent": AgentProfile(
        name="ReviewAgent",
        role="质量审核官",
        description="内容质量审核、违禁词检查、事实核查",
        persona="你是一位严谨的内容审核专家，注重细节，擅长发现逻辑漏洞和事实错误。",
        expertise=["质量审核", "事实核查", "风格校验"],
        communication_style="严谨细致",
        token_budget=3000,
        memory_namespace="feedback",
        memory_enabled=True,
        cacheable=True,
        tools_enabled=True,
    ),
    "SEOAgent": AgentProfile(
        name="SEOAgent",
        role="SEO 优化师",
        description="搜索引擎优化、关键词研究",
        persona="你是一位 SEO 专家，深谙搜索引擎排名算法，擅长关键词策略和内容优化。",
        expertise=["关键词研究", "SEO 优化", "数据分析"],
        communication_style="数据驱动",
        token_budget=2500,
        memory_namespace="domain",
        memory_enabled=True,
        remember_enabled=True,
        cacheable=True,
        tools_enabled=True,
    ),
}
