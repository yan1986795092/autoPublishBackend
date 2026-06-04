"""共享工厂：创建带 ModelRouter + LangGraph 的 Orchestrator 实例。"""
from __future__ import annotations
import logging
from typing import Any, Optional

from src.core.model_router import ModelRouter
from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def create_orchestrator(
    router: Optional[ModelRouter] = None,
    mcp_tools: Optional[list] = None,
    enable_agent_registry: Optional[bool] = None,
    enable_message_bus: Optional[bool] = None,
    enable_reflection: Optional[bool] = None,
    enable_hitl: Optional[bool] = None,
    enable_tools: Optional[bool] = None,
    **kwargs,
) -> Orchestrator:
    """创建 Orchestrator 实例，用 ModelRouter 为各 Agent 分配独立 LLM 实例。

    支持 Phase 1-5 所有组件 + Tool System（条件启用，默认从 Settings 读取）。

    Args:
        router: ModelRouter 实例，不传则自动创建
        mcp_tools: 可选的 MCP 工具列表
        enable_agent_registry: 是否启用 AgentRegistry，默认读取 settings.enable_agent_registry
        enable_message_bus: 是否启用 P2P 通信，默认读取 settings.enable_message_bus
        enable_reflection: 是否启用反思回路，默认读取 settings.enable_reflection
        enable_hitl: 是否启用 HITL，默认读取 settings.enable_hitl
        enable_tools: 是否启用 ToolRegistry，默认读取 settings.enable_tools
        **kwargs: 传递给 Orchestrator 的额外参数

    Returns:
        已就绪的 Orchestrator 实例
    """
    from src.config import get_settings
    _settings = get_settings()
    enable_agent_registry = enable_agent_registry if enable_agent_registry is not None else _settings.enable_agent_registry
    enable_message_bus = enable_message_bus if enable_message_bus is not None else _settings.enable_message_bus
    enable_reflection = enable_reflection if enable_reflection is not None else _settings.enable_reflection
    enable_hitl = enable_hitl if enable_hitl is not None else _settings.enable_hitl
    enable_tools = enable_tools if enable_tools is not None else _settings.enable_tools

    router = router or ModelRouter()

    # 按 Agent 类型获取 LLM（工具绑定统一由 ToolRegistry 处理）
    chat_models = {}
    for agent_type in [
        "ManagerAgent", "WritingAgent", "SEOAgent", "ReviewAgent",
        "DiagramAgent", "CodeAgent", "WebSearch", "HotTopicAgent",
        "PlatformAdapter", "TranslationAgent",
    ]:
        try:
            chat_models[agent_type] = router.get_chat_model(agent_type)
        except Exception:
            pass

    # ── Phase 1: AgentRegistry ──
    agent_registry = _build_agent_registry(chat_models) if enable_agent_registry else None

    # ── Phase 2: MessageBus ──
    message_bus = _build_message_bus(agent_registry) if enable_message_bus and agent_registry else None

    # ── Phase 3: ReflectionEngine ──
    reflection_engine = _build_reflection_engine(agent_registry) if enable_reflection else None

    # ── Phase 4: HITLManager ──
    hitl_manager = _build_hitl_manager(kwargs.get("task_queue")) if enable_hitl else None

    # ── Tool System: ToolRegistry ──
    tool_registry = _build_tool_registry(
        agent_registry=agent_registry,
        mcp_tools=mcp_tools,
        message_bus=message_bus,
        reflection_engine=reflection_engine,
        hitl_manager=hitl_manager,
    ) if enable_tools else None

    # 如果 ToolRegistry 就绪且 AgentRegistry 就绪，为各 Agent LLM 绑定工具
    if tool_registry and agent_registry:
        for agent_type in agent_registry.list_types():
            agent = agent_registry.get(agent_type)
            if agent and agent.profile.tools_enabled:
                agent.bind_tools(tool_registry)

    orchestrator = Orchestrator(
        llm_gateway=router.get_gateway("ManagerAgent"),
        chat_models=chat_models,
        mcp_tools=mcp_tools,
        agent_registry=agent_registry,
        message_bus=message_bus,
        reflection_engine=reflection_engine,
        hitl_manager=hitl_manager,
        tool_registry=tool_registry,
        **kwargs,
    )
    return orchestrator


def _build_agent_registry(chat_models: dict) -> Any:
    """构建 AgentRegistry"""
    from .agents import AgentRegistry
    return AgentRegistry(llm_models=chat_models)


def _build_message_bus(registry: Any) -> Any:
    """构建 MessageBus"""
    from .communication import MessageBus
    return MessageBus(registry=registry)


def _build_reflection_engine(registry: Any) -> Any:
    """构建 ReflectionEngine"""
    from .reflection import ReflectionEngine, ReflectionConfig
    return ReflectionEngine(config=ReflectionConfig(), registry=registry)


def _build_hitl_manager(task_queue: Any) -> Any:
    """构建 HITLManager"""
    from .hitl import HITLManager
    return HITLManager(task_queue=task_queue)


def _build_tool_registry(
    agent_registry: Any = None,
    mcp_tools: Optional[list] = None,
    message_bus: Any = None,
    reflection_engine: Any = None,
    hitl_manager: Any = None,
) -> Any:
    """构建 ToolRegistry 并注册所有工具

    注册顺序：
    1. 内置工具（memory / communication / reflection / hitl）
    2. MCP 工具（如果提供了 mcp_tools）
    """
    from .tools import ToolRegistry, create_memory_tools

    registry = ToolRegistry()

    # 1. 记忆工具（始终可用，依赖 memory_manager 单例）
    for tool in create_memory_tools():
        registry.register(tool)

    # 2. 通信工具（依赖 MessageBus 实例）
    if message_bus is not None:
        from .tools import create_communication_tools
        for tool in create_communication_tools(message_bus):
            registry.register(tool)

    # 3. 反思工具（依赖 ReflectionEngine 实例）
    if reflection_engine is not None:
        from .tools import create_reflection_tools
        for tool in create_reflection_tools(reflection_engine):
            registry.register(tool)

    # 4. HITL 工具（依赖 HITLManager 实例）
    if hitl_manager is not None:
        from .tools import create_hitl_tools
        for tool in create_hitl_tools(hitl_manager):
            registry.register(tool)

    # 5. MCP 工具（仅限搜索类 Agent 使用，避免污染写作/审核 Agent 的 LLM 调用）
    if mcp_tools:
        for lc_tool in mcp_tools:
            from .tools import AgentToolMeta
            registry.register_from_langchain(
                lc_tool,
                meta=AgentToolMeta(
                    category="search",
                    agent_types=["WebSearch", "HotTopicAgent"],
                    builtin=False,
                ),
            )

    logger.info(
        "ToolRegistry: 已注册 %d 个工具（记忆=%s, 通信=%s, 反思=%s, HITL=%s, MCP=%d）",
        registry.count,
        "✓", "✓" if message_bus else "✗",
        "✓" if reflection_engine else "✗",
        "✓" if hitl_manager else "✗",
        len(mcp_tools) if mcp_tools else 0,
    )
    return registry
