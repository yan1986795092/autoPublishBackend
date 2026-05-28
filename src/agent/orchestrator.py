"""Orchestrator — LangGraph StateGraph 包装器

保持原有的 Orchestrator 类名和方法签名（run()），
内部使用 LangGraph 代替原有的 DAG 调度器。
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from typing import Any, Optional


# Python 3.10 兼容的 asyncio.timeout 包装
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
            # 与 asyncio.timeout (3.11+) 行为对齐：抛 TimeoutError 而非传播 CancelledError
            raise TimeoutError(f"async operation timed out after {self._timeout}s")
        return False

from langchain_openai import ChatOpenAI

from .graph import build_agent_graph
from .state import AgentGraphState
from .schemas import AgentResult, AgentTask, ExecutionPlan, TaskStatus, UsageInfo
from .thinking_trace import ThinkingTrace
from src.core.llm_gateway import LLMGateway
from .agent_utils import PLANNER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Orchestrator:
    """Manager Agent：负责任务拆解、Agent 调度、结果合并

    内部使用 LangGraph StateGraph 进行编排，
    对外保持原有的 run() 接口不变。
    """

    def __init__(
        self,
        llm_gateway: LLMGateway | None = None,
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
    ):
        self.llm_gateway = llm_gateway
        self.chat_models = chat_models or {}
        self.model = model or (llm_gateway.model if llm_gateway else "")
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

        self._graph = build_agent_graph()

    def register_agent(self, agent_type: str, agent: Any) -> None:
        """No-op — agents are now registered via chat_models dict."""
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
        trace = ThinkingTrace(task_id=str(uuid.uuid4()))
        trace.begin()
        start_time = time.time()
        task_id = trace.task_id

        try:
            # Python 3.10 兼容：asyncio.timeout 在 3.11+ 才支持上下文管理器
            async with asyncio.timeout(timeout) if hasattr(asyncio, 'timeout') else _AsyncioTimeout(timeout):
                # ── 构建初始 state（支持预计算计划，避免 SSE plan ≠ 执行计划）──
                precomputed = (context or {}).get("precomputed_plan")
                initial_state: AgentGraphState = {
                    "user_request": user_request,
                    "context": context or {},
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
                    "_streaming_mode": True,  # 开启实时流式推送
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
                }

                # ── 运行 Graph ──
                config = {"recursion_limit": 200}
                final_state = await self._graph.ainvoke(initial_state, config=config)

                elapsed = round(time.time() - start_time, 2)
                status = final_state.get("status", "completed")
                final_output = final_state.get("final_output")
                error = final_state.get("error")

                # ── 构建 AgentResult（与原有格式一致） ──
                thinking_steps = final_state.get("thinking_steps", [])
                if thinking_steps:
                    trace.steps = []
                    for s in thinking_steps:
                        step = trace.add_step(
                            agent=s.get("agent", ""),
                            action=s.get("action", ""),
                            reasoning=s.get("reasoning", ""),
                            input_summary=s.get("input_summary", ""),
                        )
                        trace.complete_step(
                            step_index=step,
                            output_summary=s.get("output_summary", ""),
                            tokens_used=s.get("tokens_used", 0),
                        )

                result_status = TaskStatus.COMPLETED
                if status == "failed":
                    result_status = TaskStatus.FAILED
                elif status == "partial":
                    result_status = TaskStatus.PARTIAL

                total_tokens = sum(
                    r.get("tokens_used", 0)
                    for r in final_state.get("task_results", {}).values()
                )

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
                    thinking_trace=trace.to_dict(),
                )

        except asyncio.TimeoutError:
            elapsed = round(time.time() - start_time, 2)
            return AgentResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                agent_type="ManagerAgent",
                error="Graph execution timed out",
                usage=UsageInfo(time_elapsed=elapsed),
                thinking_trace=trace.to_dict(),
            )
        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            return AgentResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                agent_type="ManagerAgent",
                error=str(e),
                usage=UsageInfo(time_elapsed=elapsed),
                thinking_trace=trace.to_dict(),
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
            "视频", "脚本", "分镜", "多格式", "全部",
            "newsletter", "博客", "深度", "万字",
        ]
        medium_keywords = [
            "搜索", "热点", "热搜", "趋势", "优化", "seo",
            "小红书", "公众号", "知乎", "微博", "平台",
            "推广", "营销", "文案",
        ]
        complex_count = sum(1 for kw in complex_keywords if kw in text)
        medium_count = sum(1 for kw in medium_keywords if kw in text)

        if complex_count >= 2 or (complex_count >= 1 and len(text) > 50):
            return "complex"
        elif medium_count >= 1 or len(text) > 80:
            return "medium"
        return "simple"

    def _build_template_plan(self, user_request: str, complexity: str) -> ExecutionPlan:
        """根据复杂度返回预设模板计划"""
        goal = user_request[:200]
        plans = {
            "simple": ExecutionPlan(
                goal=goal,
                reasoning="简单任务：直接写作 + 质量审核",
                estimated_steps=2,
                tasks=[
                    AgentTask(id=str(uuid.uuid4()), agent_type="WritingAgent", input={"topic": user_request, "content_type": "blog"}, dependencies=[]),
                    AgentTask(id=str(uuid.uuid4()), agent_type="ReviewAgent", input={"topic": user_request}, dependencies=[0]),
                ],
            ),
            "medium": ExecutionPlan(
                goal=goal,
                reasoning="中等任务：搜索资料 → 写作 → SEO 优化 → 审核",
                estimated_steps=4,
                tasks=[
                    AgentTask(id=str(uuid.uuid4()), agent_type="WebSearch", input={"query": user_request, "count": 5}, dependencies=[]),
                    AgentTask(id=str(uuid.uuid4()), agent_type="WritingAgent", input={"topic": user_request, "content_type": "blog"}, dependencies=[0]),
                    AgentTask(id=str(uuid.uuid4()), agent_type="SEOAgent", input={"topic": user_request}, dependencies=[1]),
                    AgentTask(id=str(uuid.uuid4()), agent_type="ReviewAgent", input={"topic": user_request}, dependencies=[1, 2]),
                ],
            ),
            "complex": ExecutionPlan(
                goal=goal,
                reasoning="复杂任务：搜索 → 写作 → 图表 → SEO → 审核",
                estimated_steps=5,
                tasks=[
                    AgentTask(id=str(uuid.uuid4()), agent_type="WebSearch", input={"query": user_request, "count": 8}, dependencies=[]),
                    AgentTask(id=str(uuid.uuid4()), agent_type="WritingAgent", input={"topic": user_request, "content_type": "blog"}, dependencies=[0]),
                    AgentTask(id=str(uuid.uuid4()), agent_type="DiagramAgent", input={"topic": user_request, "diagram_type": "auto"}, dependencies=[1]),
                    AgentTask(id=str(uuid.uuid4()), agent_type="SEOAgent", input={"topic": user_request}, dependencies=[1]),
                    AgentTask(id=str(uuid.uuid4()), agent_type="ReviewAgent", input={"topic": user_request}, dependencies=[1, 3]),
                ],
            ),
        }
        return plans.get(complexity, plans["simple"])

    async def _decompose(self, user_request: str, context: dict) -> ExecutionPlan:
        """将用户请求拆解为子任务列表（用于 /agent/plan 预览）"""
        force_template = context.get("force_template", "")
        if force_template in ("simple", "medium", "complex"):
            return self._build_template_plan(user_request, force_template)

        complexity = self._classify_complexity(user_request)
        if complexity == "simple":
            return self._build_template_plan(user_request, complexity)

        context_str = json.dumps(context, ensure_ascii=False, indent=2) if context else "{}"
        prompt = f"""User request: {user_request}

Complexity: {complexity}
Additional context:
{context_str}

Break this down into sub-tasks for the available agents.
Since this is a {complexity} complexity request, create an appropriately detailed plan."""

        try:
            response = await self.llm_gateway.generate_sync(
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=3000,
            )
            plan = self._parse_plan(response, user_request)
            if complexity == "medium" and len(plan.tasks) > 8:
                return self._build_template_plan(user_request, complexity)
            if complexity == "complex" and len(plan.tasks) < 2:
                return self._build_template_plan(user_request, complexity)
            return plan
        except Exception as e:
            logger.warning(f"LLM decompose failed ({e}), falling back to template")
            return self._build_template_plan(user_request, complexity)

    def _parse_plan(self, response: str, goal: str) -> ExecutionPlan:
        """解析 LLM 返回的计划 JSON"""
        try:
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].strip()
            data = json.loads(json_str)
            tasks = []
            for i, t in enumerate(data.get("tasks", [])):
                tasks.append(AgentTask(
                    id=str(uuid.uuid4()),
                    agent_type=t.get("agent_type", "WritingAgent"),
                    input=t.get("input", {}),
                    dependencies=t.get("dependencies", []),
                    config={"reasoning": t.get("reasoning", ""), "priority": t.get("priority", 0)},
                ))
            return ExecutionPlan(
                goal=data.get("goal", goal),
                tasks=tasks,
                reasoning=data.get("reasoning", ""),
                estimated_steps=len(tasks),
            )
        except (json.JSONDecodeError, KeyError) as e:
            return ExecutionPlan(
                goal=goal,
                tasks=[
                    AgentTask(id=str(uuid.uuid4()), agent_type="WritingAgent", input={"topic": goal}, dependencies=[]),
                    AgentTask(id=str(uuid.uuid4()), agent_type="ReviewAgent", input={"topic": goal}, dependencies=[0]),
                ],
                reasoning=f"Auto-generated fallback plan (parse error: {e})",
                estimated_steps=2,
            )

    async def _fallback_llm_call(
        self, task: AgentTask, context: dict, previous_results: dict[str, AgentResult]
    ) -> AgentResult:
        """当子 Agent 未注册时，直接用 LLM 生成（用于 /agent/repurpose）"""
        trace = ThinkingTrace(task_id=task.id)
        trace.begin()

        previous_text = ""
        for dep_id in list(previous_results.keys()):
            res = previous_results[dep_id]
            previous_text += f"\n## Previous: {dep_id}\n{res.output.get('content', '')}"

        prompt = f"Task type: {task.agent_type}\nInput: {json.dumps(task.input, ensure_ascii=False, indent=2)}"
        if previous_text:
            prompt += f"\n\nContext from previous tasks:{previous_text}"

        try:
            response = await self.llm_gateway.generate_sync(
                messages=[
                    {"role": "system", "content": f"You are acting as a {task.agent_type}. Complete the assigned task."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            return AgentResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                agent_type=task.agent_type,
                output={"content": response, "type": "text"},
                thinking_trace=trace.to_dict(),
            )
        except Exception as e:
            return AgentResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                agent_type=task.agent_type,
                error=str(e),
                thinking_trace=trace.to_dict(),
            )
