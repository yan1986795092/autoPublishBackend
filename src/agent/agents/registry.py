"""AgentRegistry — Agent 注册中心

管理所有 Agent 实例，提供按类型、能力查找的能力。
Factory 中构造，通过 state._agent_registry 注入 Graph。
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from .profile import AgentProfile, PREDEFINED_PROFILES
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Agent 注册中心

    维护 agent_type → BaseAgent 的映射。
    提供按类型、能力检索 Agent 的能力。
    """

    def __init__(
        self,
        profiles: Optional[dict[str, AgentProfile]] = None,
        llm_models: Optional[dict[str, ChatOpenAI]] = None,
    ):
        if profiles is None:
            profiles = PREDEFINED_PROFILES
        llm_models = llm_models or {}

        self._agents: dict[str, BaseAgent] = {}
        for name, profile in profiles.items():
            llm = llm_models.get(name)
            self._agents[name] = BaseAgent(profile=profile, llm=llm)

    # ── 查询 ──

    def get(self, agent_type: str) -> Optional[BaseAgent]:
        """按类型获取 Agent"""
        return self._agents.get(agent_type)

    def get_profile(self, agent_type: str) -> Optional[AgentProfile]:
        """按类型获取 Profile"""
        agent = self._agents.get(agent_type)
        return agent.profile if agent else None

    def list_types(self) -> list[str]:
        """列出所有已注册的 Agent 类型"""
        return list(self._agents.keys())

    def list_profiles(self) -> list[AgentProfile]:
        """列出所有 Profile"""
        return [a.profile for a in self._agents.values()]

    # ── 按能力查找 ──

    def find_by_expertise(self, keyword: str) -> list[AgentProfile]:
        """按专长关键词查找 Agent"""
        return [
            a.profile
            for a in self._agents.values()
            if any(keyword in e for e in a.profile.expertise)
        ]

    def find_by_memory_enabled(self) -> list[str]:
        """查找启用了记忆的 Agent 列表"""
        return [name for name, a in self._agents.items() if a.profile.memory_enabled]

    def find_by_remember_enabled(self) -> list[str]:
        """查找启用了记忆沉淀的 Agent 列表"""
        return [name for name, a in self._agents.items() if a.profile.remember_enabled]

    def find_cacheable(self) -> list[str]:
        """查找可缓存的 Agent 列表"""
        return [name for name, a in self._agents.items() if a.profile.cacheable]

    def find_tool_enabled(self) -> list[str]:
        """查找可使用 MCP 工具的 Agent 列表"""
        return [name for name, a in self._agents.items() if a.profile.tools_enabled]

    # ── 运行时注册 ──

    def register(self, agent_type: str, agent: BaseAgent) -> None:
        """注册一个 Agent 实例"""
        self._agents[agent_type] = agent

    def __repr__(self) -> str:
        return f"AgentRegistry({len(self._agents)} agents: {', '.join(self._agents.keys())})"
