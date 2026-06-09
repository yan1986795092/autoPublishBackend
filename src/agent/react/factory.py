"""SubAgentFactory — 为 AgentRegistry 中的每个 BaseAgent 创建带 ReAct 能力的 SubAgent"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

from .sub_agent import SubAgent
from ..agents.registry import AgentRegistry
from ..schemas import SubAgentConfig

logger = logging.getLogger(__name__)


class SubAgentFactory:
    """SubAgent 工厂

    从 AgentRegistry 获取所有已注册的 BaseAgent，
    为其包装 SubAgent（增加 ReAct 循环 + 对等协作能力）。

    注入依赖：
    - ToolRegistry → 工具调用能力
    - MessageBus → 对等协作能力
    - SubAgentConfig → ReAct 配置参数
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        tool_registry: Optional[Any] = None,
        message_bus: Optional[Any] = None,
        config: Optional[SubAgentConfig] = None,
    ):
        self._registry = agent_registry
        self._tool_registry = tool_registry
        self._message_bus = message_bus
        self._config = config or SubAgentConfig()
        self._sub_agents: dict[str, SubAgent] = {}

    def create_all(self) -> dict[str, SubAgent]:
        """为 AgentRegistry 中所有 Agent 创建 SubAgent 实例"""
        for agent_type in self._registry.list_types():
            sub = self.create(agent_type)
            if sub is not None:
                self._sub_agents[agent_type] = sub
                logger.info("SubAgentFactory: 创建 SubAgent[%s]", agent_type)
        return self._sub_agents

    def create(self, agent_type: str) -> Optional[SubAgent]:
        """为指定 Agent 类型创建 SubAgent"""
        base_agent = self._registry.get(agent_type)
        if base_agent is None:
            logger.warning("SubAgentFactory: AgentRegistry 中未找到 %s", agent_type)
            return None

        profile = base_agent.profile
        llm = base_agent._llm

        sub = SubAgent(
            profile=profile,
            llm=llm,
            tool_registry=self._tool_registry,
            message_bus=self._message_bus,
            config=self._config,
        )
        return sub

    def get(self, agent_type: str) -> Optional[SubAgent]:
        """获取已创建的 SubAgent"""
        if agent_type not in self._sub_agents:
            return self.create(agent_type)
        return self._sub_agents.get(agent_type)

    def list_types(self) -> list[str]:
        """列出所有已创建的 SubAgent 类型"""
        return list(self._sub_agents.keys())

    def has_agent(self, agent_type: str) -> bool:
        """检查 SubAgent 是否已创建"""
        return agent_type in self._sub_agents
