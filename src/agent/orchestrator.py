"""Orchestrator — LangGraph StateGraph 包装器

保持原有的 Orchestrator 类名和方法签名（run()），
内部使用 LangGraph 代替原有的 DAG 调度器。
"""

from __future__ import annotations
import asyncio
import json
import time
import uuid
from typing import Any, Optional

import structlog
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from langchain_openai import ChatOpenAI

from .graph import build_agent_graph
from .state import AgentGraphState
from .schemas import AgentResult, AgentTask, ExecutionPlan, TaskStatus, UsageInfo
from .agent_utils import PLANNER_SYSTEM_PROMPT, extract_plan_from_llm_response, validate_plan_structure
from src.observability import get_tracer
from src.observability.metrics import record_agent_run
from src.logging_shared import AgentEvent, EventType, TokenUsage, get_agent_event_emitter

logger = structlog.get_logger(__name__)

_tracer = None


class _AsyncioTimeout:
    """兼容 Python 3.10 的 asyncio.timeout 替代"""
    def __init__(self, timeout: float):
        self._timeout = timeout
        self._task: asyncio.Task | None = None
        self._cancelled = False

    async def __aenter__(self) -> "_AsyncioTimeout":
        self._task = asyncio.current_task()
        self._cancelled = False
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is asyncio.CancelledError:
            self._cancelled = True
            if self._task and not self._task.done():
                self._task.cancel()
            raise TimeoutError(f"异步操作在 {self._timeout}s 后超时")
        return False


def _get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = get_tracer("agent.orchestrator")
    return _tracer


class Orchestrator:
    """Manager Agent：负责任务拆解、Agent 调度、结果合并

    内部使用 LangGraph StateGraph 进行编排，
    对外保持原有的 run() 接口不变。
    """

    def __init__(
        self,
        chat_models: dict[str, ChatOpenAI] | None = None,
        agent_registry: dict[str, Any] | None = None,
        model: str = "",
        temperature: float = 0.3,
        task_queue: Any = None,
        async_task_id: str = "",
        max_concurrent: int = 3,
        mcp_tools: list | None = None,
        hitl_manager: Any | None = None,
        message_bus: Any | None = None,
        reflection_engine: Any | None = None,
        tool_registry: Any | None = None,
        subagent_factory: Any | None = None,
    ):
        self.chat_models = chat_models or {}
        self._planner_llm = self.chat_models.get("ManagerAgent")
        self.model = model or (
            getattr(self._planner_llm, "model_name", "") if self._planner_llm else ""
        )
        self.temperature = temperature
        self.task_queue = task_queue
        self.async_task_id = async_task_id
        self.max_concurrent = max_concurrent
        self.mcp_tools = mcp_tools or []
        self.agent_registry = agent_registry
        self.hitl_manager = hitl_manager
        self.message_bus = message_bus
        self.reflection_engine = reflection_engine
        self.tool_registry = tool_registry
        self.subagent_factory = subagent_factory
        self._graph = build_agent_graph()
        self._context_manager = None
        self._context_store = None

    def register_agent(self, agent_type: str, agent: Any) -> None:
        """无操作 — 代理现通过 chat_models 字典注册。"""
        pass

    async def request_user_decision(
        self,
        question: str,
        options: list[dict],
        context: dict | None = None,
    ) -> str | None:
        """暂停执行，等待用户决策。返回选择值或 None（超时/非异步模式）。

        当启用了 HITLManager 时委托给 HITLManager。否则使用原有逻辑。
        """
        if self.hitl_manager:
            from .hitl.gates import DecisionGate
            gate = DecisionGate(
                gate_id=str(uuid.uuid4()),
                task_id=self.async_task_id,
                question=question,
                options=options,
                context=context or {},
            )
            return await self.hitl_manager.request_decision(gate)

        # 原有逻辑（向后兼容）
        if not self.async_task_id or not self.task_queue:
            return options[0]["value"] if options else None

        from .workflow.task_queue import PendingDecision
        decision_id = str(uuid.uuid4())
        decision = PendingDecision(
            decision_id=decision_id,
            task_id=self.async_task_id,
            question=question,
            options=options,
            context=context or {},
        )
        self.task_queue.set_decision_required(self.async_task_id, decision)
        choice = await self.task_queue.wait_for_decision(
            self.async_task_id, timeout=300.0
        )
        return choice

    async def run(
        self, user_request: str, context: dict | None = None, timeout: float = 180.0
    ) -> AgentResult:
        """执行一次完整的 Agent 编排流程（使用 LangGraph StateGraph）"""
        tracer = _get_tracer()
        emitter = get_agent_event_emitter()
        start_time = time.time()
        task_id = str(uuid.uuid4())
        session_id = ((context or {}).get("session_id") or (context or {}).get("conversation_id") or "default")
        emitter.reset_counter(task_id)

        # ── [插桩] agent.run 根 Span ─────────────────────────────────────────
        run_span_name = f"agent.run"
        skill_name = (context or {}).get("skill_name", "unknown")

        with tracer.start_as_current_span(
            run_span_name,
            kind=SpanKind.INTERNAL,
            attributes={
                # OTel 语义约定
                "agent.system": "langgraph",
                "agent.graph.version": "2.0.0",
                "agent.orchestrator.class": "Orchestrator",
                "agent.run.id": self.async_task_id or task_id,
                "agent.skill.name": skill_name,
                "agent.user_request.preview": user_request[:200],
                "agent.max_concurrent": self.max_concurrent,
                "agent.timeout_seconds": timeout,
                "agent.complexity.hint": context.get("complexity_hint", "balanced") if context else "balanced",
                "agent.phases.enabled": json.dumps({
                    "phase_1_agent_registry": bool(self.agent_registry),
                    "phase_2_message_bus": bool(self.message_bus),
                    "phase_3_reflection": bool(self.reflection_engine),
                    "phase_4_hitl": bool(self.hitl_manager),
                    "phase_5_tools": bool(self.tool_registry),
                }),
                "custom.run_id": task_id,
                "custom.task_id": self.async_task_id or "",
            },
        ) as root_span:
            await emitter.emit(
                AgentEvent(
                    session_id=session_id,
                    run_id=task_id,
                    agent_name="ManagerAgent",
                    event_type=EventType.AGENT_START.value,
                    level="INFO",
                    message="ManagerAgent 运行已开始",
                    tool_output=user_request[:500],
                    llm_model=self.model,
                )
            )
            try:
                # Python 3.10 兼容
                async with asyncio.timeout(timeout) if hasattr(asyncio, 'timeout') else _AsyncioTimeout(timeout):
                    precomputed = (context or {}).get("precomputed_plan")
                    skill_system_prompt = (context or {}).get("skill_system_prompt", "")
                    initial_state: AgentGraphState = {
                        "user_request": user_request,
                        "context": context or {},
                        "skill_system_prompt": skill_system_prompt,
                        "skill_name": skill_name,
                        "plan": precomputed,
                        "plan_reasoning": (precomputed or {}).get("reasoning", ""),
                        "task_queue": [],
                        "completed_task_ids": [],
                        "failed_task_ids": [],
                        "task_results": {},
                        "ready_tasks": [],
                        "concurrency_limit": self.max_concurrent,
                        "thinking_steps": [],
                        "trace_task_id": task_id,
                        "start_time": start_time,
                        "pending_decision": None,
                        "decision_result": None,
                        "final_output": None,
                        "error": None,
                        "status": "running",
                        "_async_task_id": self.async_task_id or task_id,
                        "_task_queue_ref": self.task_queue,
                        "_llm_models": self.chat_models,
                        "_mcp_tools": self.mcp_tools,
                        "_streaming_mode": True,
                        "_agent_registry": self.agent_registry,
                        "use_message_bus": self.message_bus is not None,
                        "_message_bus": self.message_bus,
                        "_messages": [],
                        "enable_reflection": self.reflection_engine is not None,
                        "reflection_config": None,
                        "_reflection_engine": self.reflection_engine,
                        "reflection_results": [],
                        "self_reflection_rounds": {},
                        "enable_hitl": self.hitl_manager is not None,
                        "_hitl_manager": self.hitl_manager,
                        "pending_gates": [],
                        "_tool_registry": self.tool_registry,
                        "enable_context_management": True,
                        "_context_store": None,
                        "_context_manager": None,
                        "_subagent_factory": self.subagent_factory,
                        "enable_subagent_react": self.subagent_factory is not None,
                    }

                    from src.memory import ContextManager, SharedContextStore
                    self._context_manager = ContextManager()
                    conversation_id = (context or {}).get("conversation_id", "default")
                    self._context_store = self._context_manager.handle_message(
                        user_message=user_request,
                        conversation_id=conversation_id,
                        project_id=(context or {}).get("project_id"),
                        project_directive=(context or {}).get("project_directive"),
                    )
                    initial_state["_context_manager"] = self._context_manager
                    initial_state["_context_store"] = self._context_store

                    config = {"recursion_limit": 200}
                    final_state = await self._graph.ainvoke(initial_state, config=config)

                    elapsed = round(time.time() - start_time, 2)
                    status = final_state.get("status", "completed")
                    final_output = final_state.get("final_output")
                    error = final_state.get("error")

                    thinking_steps = final_state.get("thinking_steps", [])
                    agent_events = {
                        "events": [
                            {
                                "session_id": session_id,
                                "run_id": task_id,
                                "step_index": index,
                                "agent_name": step.get("agent", "ManagerAgent"),
                                "event_type": EventType.THOUGHT.value,
                                "level": "DEBUG",
                                "message": step.get("action", "thinking_step"),
                                "thought": step.get("reasoning", ""),
                                "tool_output": step.get("output_summary", ""),
                                "duration_ms": None,
                                "token_usage": {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": step.get("tokens_used", 0),
                                    "cost_usd": 0.0,
                                },
                            }
                            for index, step in enumerate(thinking_steps)
                        ]
                    }

                    result_status = TaskStatus.COMPLETED
                    if status == "failed":
                        result_status = TaskStatus.FAILED
                    elif status == "partial":
                        result_status = TaskStatus.PARTIAL

                    total_tokens = sum(
                        r.get("tokens_used", 0)
                        for r in final_state.get("task_results", {}).values()
                    )

                    root_span.set_status(Status(StatusCode.OK))
                    root_span.set_attribute("agent.status", status)
                    root_span.set_attribute("agent.completed_tasks", len(final_state.get("completed_task_ids", [])))
                    root_span.set_attribute("agent.failed_tasks", len(final_state.get("failed_task_ids", [])))
                    root_span.set_attribute("agent.total_tokens", total_tokens)
                    root_span.set_attribute("agent.duration_ms", elapsed * 1000)

                    await emitter.emit(
                        AgentEvent(
                            session_id=session_id,
                            run_id=task_id,
                            agent_name="ManagerAgent",
                            event_type=EventType.AGENT_FINISH.value,
                            level="INFO",
                            message="ManagerAgent 运行已完成",
                            tool_output=json.dumps(final_output or {}, ensure_ascii=False)[:500],
                            duration_ms=elapsed * 1000,
                            token_usage=TokenUsage(total_tokens=total_tokens),
                            llm_model=self.model,
                        )
                    )

                    record_agent_run(status=status, skill_name=skill_name, duration_s=elapsed)

                    return AgentResult(
                        task_id=task_id,
                        status=result_status,
                        agent_type="ManagerAgent",
                        output=final_output or {},
                        error=error,
                        usage=UsageInfo(
                            tokens_used=total_tokens,
                            time_elapsed=elapsed,
                            model=self.model,
                        ),
                        agent_events=agent_events,
                    )

            except asyncio.TimeoutError:
                elapsed = round(time.time() - start_time, 2)
                root_span.set_status(Status(StatusCode.ERROR, "图执行超时"))
                root_span.record_exception(TimeoutError(f"图在 {timeout}s 后超时"))
                root_span.set_attribute("agent.status", "timeout")
                record_agent_run(status="timeout", skill_name=skill_name, duration_s=elapsed)
                return AgentResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    agent_type="ManagerAgent",
                    error="图执行超时",
                    usage=UsageInfo(time_elapsed=elapsed),
                    agent_events={
                        "events": [
                            {
                                "session_id": session_id,
                                "run_id": task_id,
                                "step_index": 0,
                                "agent_name": "ManagerAgent",
                                "event_type": EventType.ERROR.value,
                                "level": "ERROR",
                                "message": "图执行超时",
                                "error_type": "TimeoutError",
                                "error_message": "图执行超时",
                            }
                        ]
                    },
                )
            except Exception as e:
                elapsed = round(time.time() - start_time, 2)
                root_span.set_status(Status(StatusCode.ERROR, str(e)))
                root_span.record_exception(e)
                root_span.set_attribute("agent.status", "failed")
                root_span.set_attribute("agent.error_type", type(e).__name__)
                logger.error("图执行失败", error=str(e), exc_info=True)
                record_agent_run(status="failed", skill_name=skill_name, duration_s=elapsed)
                return AgentResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    agent_type="ManagerAgent",
                    error=str(e),
                    usage=UsageInfo(time_elapsed=elapsed),
                    agent_events={
                        "events": [
                            {
                                "session_id": session_id,
                                "run_id": task_id,
                                "step_index": 0,
                                "agent_name": "ManagerAgent",
                                "event_type": EventType.ERROR.value,
                                "level": "ERROR",
                                "message": str(e),
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                            }
                        ]
                    },
                )

    # ══════════════════════════════════════════════
    # 以下方法保留供 /agent/plan 和 /agent/repurpose 使用
    # ══════════════════════════════════════════════

    def _classify_complexity(self, user_request: str) -> str:
        """对用户需求进行复杂度分类"""
        text = user_request.lower()
        complex_keywords = [
            "图表", "架构图", "流程图", "思维导图", "配图", "封面",
            "多平台", "全平台", "报告", "分析报告", "行业分析",
            "视频", "分镜", "多格式",
            "newsletter", "博客", "深度", "万字",
        ]
        medium_keywords = [
            "搜索", "热点", "热搜", "趋势", "优化", "seo", "微博", "平台",
        ]
        simple_writing_keywords = [
            "推广", "营销", "文案", "广告", "slogan",
        ]
        complex_count = sum(1 for kw in complex_keywords if kw in text)
        medium_count = sum(1 for kw in medium_keywords if kw in text)
        simple_writing_count = sum(1 for kw in simple_writing_keywords if kw in text)

        if complex_count >= 2 or (complex_count >= 1 and len(text) > 50):
            return "complex"
        elif medium_count >= 1 or len(text) > 80:
            return "medium"
        elif simple_writing_count >= 1:
            return "simple"
        return "simple"

    def _detect_platform(self, user_request: str) -> str | None:
        """检测用户需求对应的平台，返回 code 或 None"""
        from .platform_registry import detect_platform
        return detect_platform(user_request)

    def _build_fallback_plan(self, user_request: str) -> "ExecutionPlan":
        """极简兜底：只用 WritingAgent + ReviewAgent。

        Pipeline 完全由 LLM 自主决定，这里只在 LLM 不可用时保证系统能跑起来。
        """
        from .schemas import AgentTask, ExecutionPlan
        from .platform_registry import PLATFORMS, detect_platform

        goal = user_request[:200]
        detected = detect_platform(user_request)
        p = PLATFORMS.get(detected) if detected else None
        content_type = p.content_type if p else "blog"

        task0 = AgentTask(
            id=str(uuid.uuid4()), agent_type="WritingAgent",
            input={"topic": user_request, "content_type": content_type},
            dependencies=[], can_parallel_with=[], retry_on_low_quality=True, priority="high",
        )
        task1 = AgentTask(
            id=str(uuid.uuid4()), agent_type="ReviewAgent",
            input={"topic": user_request, "content_type": content_type},
            dependencies=[0], can_parallel_with=[], retry_on_low_quality=False, priority="medium",
        )
        platform_hint = f"（{p.name}）" if p else ""
        return ExecutionPlan(
            goal=goal, tasks=[task0, task1],
            reasoning=f"兜底计划{platform_hint}：写作 + 审核",
            estimated_steps=2, plan_method="template",
        )

    # 保留向后兼容（api.py 使用）
    _build_default_plan = _build_fallback_plan
    _build_platform_plan = _build_fallback_plan

    async def _decompose(self, user_request: str, context: dict) -> ExecutionPlan:
        """将用户请求分解为执行计划。

        LLM 是主路径，完全自主决定 Agent 组合和流程。
        只有在 LLM 完全不可用时才用极简兜底（只保证系统能跑起来）。
        """
        detected = self._detect_platform(user_request)
        detected_name = ""
        if detected:
            from .platform_registry import PLATFORMS
            detected_name = PLATFORMS.get(detected, type("", (), {"name": ""})()).name

        if self._planner_llm:
            try:
                complexity = self._classify_complexity(user_request)
                context_str = json.dumps(context, ensure_ascii=False, indent=2) if context else "{}"
                prompt = f"""用户请求: {user_request}

复杂度: {complexity}
检测到的平台: {detected_name or "无"}
额外上下文:
{context_str}

请将其分解为可分配给可用代理的子任务。"""

                llm_result = await self._planner_llm.ainvoke(
                    [
                        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    config={"temperature": self.temperature},
                )
                plan = self._parse_plan(llm_result.content, user_request)
                if len(plan.tasks) == 0:
                    logger.warning("LLM 计划无任务，使用兜底方案")
                    return self._build_fallback_plan(user_request)
                logger.info("LLM 生成了计划: %d 个任务", len(plan.tasks))
                return plan

            except Exception as e:
                logger.warning("编排器中 LLM 分解失败，使用兜底方案", error=str(e))

        # LLM 不可用时的极简兜底
        return self._build_fallback_plan(user_request)

    def _parse_plan(self, response: str, goal: str) -> ExecutionPlan:
        """解析 LLM 返回的计划 JSON"""
        try:
            plan_dict = extract_plan_from_llm_response(response)
            is_valid, error_msg = validate_plan_structure(plan_dict)
            if not is_valid:
                raise ValueError(f"无效的计划结构: {error_msg}")

            tasks = []
            for i, t in enumerate(plan_dict.get("tasks", [])):
                tasks.append(AgentTask(
                    id=str(uuid.uuid4()),
                    agent_type=t.get("agent_type", "WritingAgent"),
                    input=t.get("input", {}),
                    dependencies=t.get("dependencies", []),
                    can_parallel_with=t.get("can_parallel_with", []),
                    retry_on_low_quality=t.get("retry_on_low_quality", False),
                    priority=t.get("priority", "medium"),
                    config={"reasoning": t.get("reasoning", ""), "priority": i},
                ))
            return ExecutionPlan(
                goal=plan_dict.get("goal", goal),
                tasks=tasks,
                reasoning=plan_dict.get("reasoning", ""),
                estimated_steps=len(tasks),
                plan_method="llm",
            )
        except Exception as e:
            logger.warning("编排器中解析计划失败: %s", e)
            return self._build_fallback_plan(goal)

    async def _fallback_llm_call(
        self, task: AgentTask, context: dict, previous_results: dict[str, AgentResult]
    ) -> AgentResult:
        """当子 Agent 未注册时，直接用 LLM 生成（用于 /agent/repurpose）"""
        session_id = context.get("session_id", "default") if context else "default"

        previous_text = ""
        for dep_id in list(previous_results.keys()):
            res = previous_results[dep_id]
            previous_text += f"\n## Previous: {dep_id}\n{res.output.get('content', '')}"

        prompt = f"任务类型: {task.agent_type}\n输入: {json.dumps(task.input, ensure_ascii=False, indent=2)}"
        if previous_text:
            prompt += f"\n\n前置任务的上下文:{previous_text}"

        try:
            llm_result = await self._planner_llm.ainvoke(
                [
                    {"role": "system", "content": f"你正在扮演一个 {task.agent_type}。请完成分配的任务。"},
                    {"role": "user", "content": prompt},
                ],
                config={"temperature": 0.7},
            )
            response = llm_result.content

            return AgentResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                agent_type=task.agent_type,
                output={"content": response, "type": "text"},
                agent_events={
                    "events": [
                        {
                            "session_id": session_id,
                            "run_id": task.id,
                            "step_index": 0,
                            "agent_name": str(task.agent_type),
                            "event_type": EventType.AGENT_FINISH.value,
                            "level": "INFO",
                            "message": "兜底 LLM 任务已完成",
                            "tool_output": response[:500],
                        }
                    ]
                },
            )
        except Exception as e:
            return AgentResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                agent_type=task.agent_type,
                error=str(e),
                agent_events={
                    "events": [
                        {
                            "session_id": session_id,
                            "run_id": task.id,
                            "step_index": 0,
                            "agent_name": str(task.agent_type),
                            "event_type": EventType.ERROR.value,
                            "level": "ERROR",
                            "message": str(e),
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        }
                    ]
                },
            )
