"""Orchestrator — LangGraph StateGraph 包装器

保持原有的 Orchestrator 类名和方法签名（run()），
内部使用 LangGraph 代替原有的 DAG 调度器。

Tracing: 使用轻量级 AgentTracer 写入 UnifiedEsWriter (traces-*),
不再依赖 OTel SDK exporter 管线。
"""

from __future__ import annotations
import asyncio
import json
import time
import uuid
from typing import Any, Optional

import structlog

from langchain_openai import ChatOpenAI

from .graph import build_agent_graph
from .state import AgentGraphState
from .schemas import AgentResult, AgentTask, ExecutionPlan, TaskStatus, UsageInfo
from src.observability.metrics import record_agent_run
from src.observability.agent_tracer import get_agent_tracer, AgentTracer

logger = structlog.get_logger(__name__)

_tracer: AgentTracer | None = None


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


def _get_tracer() -> AgentTracer:
    global _tracer
    if _tracer is None:
        _tracer = get_agent_tracer()
    return _tracer


class Orchestrator:
    """Manager Agent：负责任务拆解、Agent 调度、结果合并

    内部使用 LangGraph StateGraph 进行编排，
    对外保持原有的 run() 接口不变。
    """

    def __init__(
        self,
        chat_models: dict[str, ChatOpenAI] | None = None,
        model: str = "",
        temperature: float = 0.3,
        task_queue: Any = None,
        async_task_id: str = "",
        max_concurrent: int = 3,
        mcp_tools: list | None = None,
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
        self._graph = build_agent_graph()

    async def run(
        self, user_request: str, context: dict | None = None, timeout: float = 180.0
    ) -> AgentResult:
        """执行一次完整的 Agent 编排流程（使用 LangGraph StateGraph）"""
        tracer = _get_tracer()
        start_time = time.time()
        task_id = str(uuid.uuid4())
        session_id = ((context or {}).get("session_id") or (context or {}).get("conversation_id") or "default")

        # 从 context 获取 HTTP trace_id 做端到端关联
        http_trace_id = (context or {}).get("trace_id", "")
        trace_task_id = http_trace_id or task_id

        run_span_name = f"agent.run"
        skill_name = (context or {}).get("skill_name", "unknown")

        with tracer.start_as_current_span(
            run_span_name,
            kind="INTERNAL",
            attributes={
                "agent.system": "langgraph",
                "agent.version": "3.0.0",
                "agent.orchestrator.class": "Orchestrator",
                "agent.run.id": self.async_task_id or task_id,
                "agent.skill.name": skill_name,
                "agent.user_request.preview": user_request[:200],
                "agent.max_concurrent": self.max_concurrent,
                "agent.timeout_seconds": timeout,
                "custom.run_id": task_id,
                "custom.task_id": self.async_task_id or "",
                "http.trace_id": http_trace_id or "(none)",
            },
        ) as root_span:
            try:
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
                        "trace_task_id": trace_task_id,
                        "start_time": start_time,
                        "final_output": None,
                        "error": None,
                        "status": "running",
                        "_async_task_id": self.async_task_id or task_id,
                        "_task_queue_ref": self.task_queue,
                        "_llm_models": self.chat_models,
                        "_mcp_tools": self.mcp_tools,
                        "_streaming_mode": True,
                        "_review_fix_cycles": 0,
                        "_loop_created_in_cycle": False,
                        "_last_review_passed": False,
                        "_shared_results": {},
                        "_review_feedback_for_fix": "",
                    }

                    config = {"recursion_limit": 200}
                    final_state = await self._graph.ainvoke(initial_state, config=config)

                    elapsed = round(time.time() - start_time, 2)
                    status = final_state.get("status", "completed")
                    final_output = final_state.get("final_output")
                    error = final_state.get("error")

                    result_status = TaskStatus.COMPLETED
                    if status == "failed":
                        result_status = TaskStatus.FAILED
                    elif status == "partial":
                        result_status = TaskStatus.PARTIAL

                    total_tokens = sum(
                        r.get("tokens_used", 0)
                        for r in final_state.get("task_results", {}).values()
                    )

                    root_span.set_status("OK")
                    root_span.set_attribute("agent.status", status)
                    root_span.set_attribute("agent.completed_tasks", len(final_state.get("completed_task_ids", [])))
                    root_span.set_attribute("agent.failed_tasks", len(final_state.get("failed_task_ids", [])))
                    root_span.set_attribute("agent.total_tokens", total_tokens)
                    root_span.set_attribute("agent.duration_ms", elapsed * 1000)

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
                    )

            except asyncio.TimeoutError:
                elapsed = round(time.time() - start_time, 2)
                root_span.set_status("ERROR")
                root_span.record_exception(TimeoutError(f"图在 {timeout}s 后超时"))
                root_span.set_attribute("agent.status", "timeout")
                record_agent_run(status="timeout", skill_name=skill_name, duration_s=elapsed)
                return AgentResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    agent_type="ManagerAgent",
                    error="图执行超时",
                    usage=UsageInfo(time_elapsed=elapsed),
                )
            except Exception as e:
                elapsed = round(time.time() - start_time, 2)
                root_span.set_status("ERROR")
                root_span.record_exception(e)
                root_span.set_attribute("agent.status", "failed")
                root_span.set_attribute("agent.error_type", type(e).__name__)
                logger.error("图执行失败", exception=e, exc_info=True)
                record_agent_run(status="failed", skill_name=skill_name, duration_s=elapsed)
                return AgentResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    agent_type="ManagerAgent",
                    error=str(e),
                    usage=UsageInfo(time_elapsed=elapsed),
                )

    # ══════════════════════════════════════════════
    # 以下方法保留供 /agent/plan 和 /agent/repurpose 使用
    # ══════════════════════════════════════════════

    def _build_fallback_plan(self, user_request: str, *args, **kwargs) -> "ExecutionPlan":
        from .nodes import _build_fallback_plan as _bfp
        plan_dict = _bfp(user_request)
        tasks = []
        for i, t in enumerate(plan_dict.get("tasks", [])):
            tasks.append(AgentTask(
                id=t["id"],
                agent_type=t["agent_type"],
                input=t["input"],
                dependencies=t.get("dependencies", []),
                can_parallel_with=t.get("can_parallel_with", []),
                retry_on_low_quality=t.get("retry_on_low_quality", False),
                priority=t.get("priority", "medium"),
            ))
        return ExecutionPlan(
            goal=plan_dict.get("goal", user_request[:200]),
            tasks=tasks,
            reasoning=plan_dict.get("reasoning", ""),
            estimated_steps=len(tasks),
            plan_method="template",
        )

    _build_default_plan = _build_fallback_plan
    _build_platform_plan = _build_fallback_plan
    _build_template_plan = _build_fallback_plan

    async def _decompose(self, user_request: str, context: dict) -> ExecutionPlan:
        """将用户请求分解为执行计划（用于 /agent/plan 端点）"""
        from .nodes import (
            UNDERSTANDING_SYSTEM_PROMPT, PLANNING_SYSTEM_PROMPT,
            _build_fallback_understanding, _parse_plan_from_response, _build_fallback_plan,
        )

        # Step 1: 理解（纯 LLM）
        understanding = None
        platform_tag = context.get("platform", "")
        platform_context = ""
        if platform_tag:
            from .platform_registry import PLATFORMS as _P
            meta = _P.get(platform_tag)
            if meta:
                platform_context = f"\nPlatform tag from UI: {meta.name} ({platform_tag})"

        if self._planner_llm:
            try:
                response = await self._planner_llm.ainvoke(
                    [
                        {"role": "system", "content": UNDERSTANDING_SYSTEM_PROMPT},
                        {"role": "user", "content": f"User request: {user_request}{platform_context}"},
                    ],
                    config={"temperature": 0.3},
                )
                understanding = json.loads(response.content)
                if not isinstance(understanding, dict) or "intent" not in understanding:
                    understanding = None
            except Exception as e:
                logger.warning("编排器 LLM 理解失败", exception=str(e))

        if not understanding:
            understanding = _build_fallback_understanding(user_request, context)

        # Step 2: 规划（纯 LLM，仅基于 understanding）
        context_data = {
            "intent": understanding.get("intent", ""),
            "cleaned_prompt": understanding.get("cleaned_prompt", user_request)[:500],
            "content_type": understanding.get("content_type", "blog"),
            "platform": understanding.get("platform"),
            "platform_name": understanding.get("platform_name", ""),
        }

        plan_dict = None
        if self._planner_llm:
            try:
                response = await self._planner_llm.ainvoke(
                    [
                        {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                        {"role": "user", "content": f"""## User Request\n{context_data['cleaned_prompt']}\n\n## Intent Understanding\n{json.dumps(context_data, ensure_ascii=False, indent=2)}\n\nBased on the above understanding, select the most appropriate agents to complete the task."""},
                    ],
                    config={"temperature": self.temperature},
                )
                plan_dict = _parse_plan_from_response(response.content, user_request)
            except Exception as e:
                logger.warning("编排器 LLM 规划失败", exception=str(e))

        if not plan_dict:
            plan_dict = _build_fallback_plan(user_request)

        tasks = []
        for i, t in enumerate(plan_dict.get("tasks", [])):
            tasks.append(AgentTask(
                id=t["id"],
                agent_type=t.get("agent_type", "WritingAgent"),
                input=t.get("input", {}),
                dependencies=t.get("dependencies", []),
                can_parallel_with=t.get("can_parallel_with", []),
                retry_on_low_quality=t.get("retry_on_low_quality", False),
                priority=t.get("priority", "medium"),
                config={"reasoning": t.get("config", {}).get("reasoning", ""), "priority": i},
            ))
        return ExecutionPlan(
            goal=plan_dict.get("goal", user_request[:200]),
            tasks=tasks,
            reasoning=plan_dict.get("reasoning", ""),
            estimated_steps=len(tasks),
            plan_method=plan_dict.get("plan_method", "template"),
        )

    def _parse_plan(self, response: str, goal: str) -> ExecutionPlan:
        try:
            from .nodes import _parse_plan_from_response as _ppr
            plan_dict = _ppr(response, goal)
            if not plan_dict:
                return self._build_fallback_plan(goal)
            tasks = []
            for i, t in enumerate(plan_dict.get("tasks", [])):
                tasks.append(AgentTask(
                    id=t["id"],
                    agent_type=t.get("agent_type", "WritingAgent"),
                    input=t.get("input", {}),
                    dependencies=t.get("dependencies", []),
                    can_parallel_with=t.get("can_parallel_with", []),
                    retry_on_low_quality=t.get("retry_on_low_quality", False),
                    priority=t.get("priority", "medium"),
                    config={"reasoning": t.get("config", {}).get("reasoning", ""), "priority": i},
                ))
            return ExecutionPlan(
                goal=plan_dict.get("goal", goal),
                tasks=tasks,
                reasoning=plan_dict.get("reasoning", ""),
                estimated_steps=len(tasks),
                plan_method=plan_dict.get("plan_method", "llm"),
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
            )
        except Exception as e:
            return AgentResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                agent_type=task.agent_type,
                error=str(e),
            )
