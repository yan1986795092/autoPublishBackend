"""LangGraph 图节点函数 — Agent 编排的全部逻辑"""

from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .state import AgentGraphState
from .agent_utils import PLANNER_SYSTEM_PROMPT
from .handlers import HANDLER_MAP, handle_fallback

logger = logging.getLogger(__name__)

# 常量/模板已迁移至 agent_utils.py 和各 Agent 模块

# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _classify_complexity(user_request: str) -> str:
    """对用户需求进行复杂度分类"""
    text = user_request.lower()
    complex_keywords = ["图表", "架构图", "流程图", "思维导图", "配图", "封面", "多平台", "报告", "行业分析", "视频", "脚本", "newsletter", "博客", "深度"]
    medium_keywords = ["搜索", "热点", "热搜", "趋势", "优化", "seo", "小红书", "公众号", "知乎", "微博", "推广", "营销"]

    complex_count = sum(1 for kw in complex_keywords if kw in text)
    medium_count = sum(1 for kw in medium_keywords if kw in text)

    if complex_count >= 2 or (complex_count >= 1 and len(text) > 50):
        return "complex"
    elif medium_count >= 1 or len(text) > 80:
        return "medium"
    return "simple"


def _build_template_plan(user_request: str, complexity: str) -> dict:
    """根据复杂度返回预设模板计划"""
    plans = {
        "simple": {
            "goal": user_request[:200],
            "reasoning": "简单任务：直接写作 + 质量审核",
            "estimated_steps": 2,
            "tasks": [
                {"id": str(uuid.uuid4()), "agent_type": "WritingAgent", "input": {"topic": user_request, "content_type": "blog"}, "dependencies": []},
                {"id": str(uuid.uuid4()), "agent_type": "ReviewAgent", "input": {"topic": user_request}, "dependencies": []},
            ],
        },
        "medium": {
            "goal": user_request[:200],
            "reasoning": "中等任务：搜索 → 写作 → SEO → 审核",
            "estimated_steps": 4,
            "tasks": [
                {"id": str(uuid.uuid4()), "agent_type": "WebSearch", "input": {"query": user_request, "count": 5}, "dependencies": []},
                {"id": str(uuid.uuid4()), "agent_type": "WritingAgent", "input": {"topic": user_request, "content_type": "blog"}, "dependencies": []},
                {"id": str(uuid.uuid4()), "agent_type": "SEOAgent", "input": {"topic": user_request}, "dependencies": []},
                {"id": str(uuid.uuid4()), "agent_type": "ReviewAgent", "input": {"topic": user_request}, "dependencies": []},
            ],
        },
        "complex": {
            "goal": user_request[:200],
            "reasoning": "复杂任务：搜索 → 写作 → 图表 → SEO → 审核",
            "estimated_steps": 6,
            "tasks": [
                {"id": str(uuid.uuid4()), "agent_type": "WebSearch", "input": {"query": user_request, "count": 8}, "dependencies": []},
                {"id": str(uuid.uuid4()), "agent_type": "WritingAgent", "input": {"topic": user_request, "content_type": "blog"}, "dependencies": []},
                {"id": str(uuid.uuid4()), "agent_type": "DiagramAgent", "input": {"topic": user_request, "diagram_type": "auto"}, "dependencies": []},
                {"id": str(uuid.uuid4()), "agent_type": "SEOAgent", "input": {"topic": user_request}, "dependencies": []},
                {"id": str(uuid.uuid4()), "agent_type": "ReviewAgent", "input": {"topic": user_request}, "dependencies": []},
            ],
        },
    }
    return plans.get(complexity, plans["simple"])


def _get_llm(state: AgentGraphState, agent_type: str) -> ChatOpenAI:
    """从 state 中获取对应 agent 类型的 LLM 实例"""
    models = state.get("_llm_models", {})
    return models.get(agent_type, models.get("default"))


def _update_progress(state: AgentGraphState, progress: float, message: str = "", step: str = "") -> None:
    """更新 TaskQueue 中的进度（用于 WebSocket 推送）"""
    task_queue = state.get("_task_queue_ref")
    task_id = state.get("_async_task_id", "")
    if task_queue and task_id:
        task_queue.update_task_progress(task_id, progress, message=message, step=step)


def _add_thinking_step(state: AgentGraphState, agent: str, action: str, reasoning: str = "",
                       input_summary: str = "", output_summary: str = "",
                       status: str = "done", tokens: int = 0, elapsed: float = 0) -> dict:
    """添加一个思考步骤到 state"""
    step = {
        "step_number": len(state.get("thinking_steps", [])) + 1,
        "agent": agent,
        "action": action,
        "reasoning": reasoning,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "tokens_used": tokens,
        "time_elapsed": elapsed,
        "status": status,
    }
    return {"thinking_steps": [step]}  # list reducer will append

# ──────────────────────────────────────────────
# 1. classify_and_decompose 节点
# ──────────────────────────────────────────────

async def classify_and_decompose(state: AgentGraphState) -> dict:
    """复杂度分类 + 任务分解

    复用原有系统的分类器逻辑：
    - 简单任务 → 模板计划（跳过 LLM）
    - 中等/复杂任务 → LLM 拆解 + 模板兜底
    """
    user_request = state.get("user_request", "")
    context = state.get("context", {})
    trace_id = str(uuid.uuid4())

    updates = {
        "trace_task_id": trace_id,
        "start_time": time.time(),
        "thinking_steps": [],
        "status": "running",
    }

    # 复杂度分类
    updates.update(_add_thinking_step(
        state, "ManagerAgent", "decompose",
        reasoning=f"Analyzing user request: {user_request[:100]}...",
        input_summary=user_request[:200],
    ))

    complexity = _classify_complexity(user_request)
    logger.info(f"Classified request as '{complexity}': {user_request[:80]}")

    if complexity == "simple":
        plan = _build_template_plan(user_request, complexity)
        updates["plan"] = plan
        updates["plan_reasoning"] = plan["reasoning"]
        updates["task_queue"] = plan["tasks"]
        # 更新 thinking step
        step = {
            "step_number": 1,
            "agent": "ManagerAgent",
            "action": "decompose",
            "reasoning": f"Simple task → template plan ({len(plan['tasks'])} steps)",
            "input_summary": user_request[:200],
            "output_summary": f"Plan: {plan['reasoning'][:80]}...",
            "tokens_used": 0,
            "time_elapsed": 0,
            "status": "done",
        }
        updates["thinking_steps"] = [step]
        return updates

    # 中等/复杂 → LLM 拆解
    llm = _get_llm(state, "ManagerAgent")
    if not llm:
        # 回退到模板
        plan = _build_template_plan(user_request, complexity)
        updates["plan"] = plan
        updates["plan_reasoning"] = plan["reasoning"]
        updates["task_queue"] = plan["tasks"]
        return updates

    try:
        context_str = json.dumps(context, ensure_ascii=False, indent=2) if context else "{}"
        response = await llm.ainvoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User request: {user_request}\n\nComplexity: {complexity}\nContext: {context_str}\n\nBreak this down into sub-tasks."),
        ])
        plan = _parse_plan(response.content, user_request)

        # 验证计划
        if complexity == "medium" and len(plan.get("tasks", [])) > 8:
            plan = _build_template_plan(user_request, complexity)
        elif complexity == "complex" and len(plan.get("tasks", [])) < 2:
            plan = _build_template_plan(user_request, complexity)

        updates["plan"] = plan
        updates["plan_reasoning"] = plan.get("reasoning", "")
        updates["task_queue"] = plan.get("tasks", [])
    except Exception as e:
        logger.warning(f"LLM decompose failed ({e}), using template")
        plan = _build_template_plan(user_request, complexity)
        updates["plan"] = plan
        updates["plan_reasoning"] = plan["reasoning"]
        updates["task_queue"] = plan["tasks"]

    return updates


def _parse_plan(response: str, goal: str) -> dict:
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
            tasks.append({
                "id": str(uuid.uuid4()),
                "agent_type": t.get("agent_type", "WritingAgent"),
                "input": t.get("input", {}),
                "dependencies": t.get("dependencies", []),
                "config": {"reasoning": t.get("reasoning", ""), "priority": i},
            })
        return {"goal": data.get("goal", goal), "tasks": tasks, "reasoning": data.get("reasoning", ""), "estimated_steps": len(tasks)}
    except (json.JSONDecodeError, KeyError):
        plan = _build_template_plan(goal, "simple")
        plan["reasoning"] = f"Auto-generated fallback plan (parse error)"
        return plan

# ──────────────────────────────────────────────
# 2. find_ready_tasks 节点
# ──────────────────────────────────────────────

async def find_ready_tasks(state: AgentGraphState) -> dict:
    """检查 DAG 依赖，找出下一批可执行的任务"""
    task_queue = state.get("task_queue", [])
    completed_ids = set(state.get("completed_task_ids", []))
    failed_ids = set(state.get("failed_task_ids", []))

    def deps_met(task: dict) -> bool:
        for dep in task.get("dependencies", []):
            dep_id = dep
            if isinstance(dep, int) and 0 <= dep < len(task_queue):
                # 通过索引引用
                dep_id = task_queue[dep].get("id", "")
            if dep_id not in completed_ids and dep_id not in failed_ids:
                # 检查是否是索引引用
                if isinstance(dep, int) and 0 <= dep < len(task_queue):
                    ref_id = task_queue[dep].get("id", "")
                    if ref_id in completed_ids or ref_id in failed_ids:
                        continue
                return False
        return True

    ready = [t for t in task_queue if t.get("id") not in completed_ids and t.get("id") not in failed_ids and deps_met(t)]

    if not ready and len(completed_ids) + len(failed_ids) < len(task_queue):
        # 死锁检测
        stuck = [t for t in task_queue if t.get("id") not in completed_ids and t.get("id") not in failed_ids]
        logger.warning(f"Deadlock detected: {len(stuck)} tasks stuck")
        return {"ready_tasks": [], "failed_task_ids": [t["id"] for t in stuck]}

    return {"ready_tasks": ready}

# ──────────────────────────────────────────────
# 3. execute_task_batch 节点
# ──────────────────────────────────────────────

async def execute_task_batch(state: AgentGraphState) -> dict:
    """并行执行当前批次的所有任务"""
    ready_tasks = state.get("ready_tasks", [])
    if not ready_tasks:
        return {}

    concurrency = state.get("concurrency_limit", 3)
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(task: dict) -> tuple[str, dict]:
        async with semaphore:
            result = await _run_single_agent_task(task, state)
            return task["id"], result

    results_list = await asyncio.gather(*[run_one(t) for t in ready_tasks])

    task_results = dict(state.get("task_results", {}))
    completed = []
    failed = []
    steps = []

    for task_id, result in results_list:
        task_results[task_id] = result
        if result.get("status") in ("completed", "partial"):
            completed.append(task_id)
        else:
            failed.append(task_id)
        if result.get("thinking_steps"):
            steps.extend(result["thinking_steps"])

    return {
        "task_results": task_results,
        "completed_task_ids": completed,
        "failed_task_ids": failed,
        "thinking_steps": steps,
    }

# ──────────────────────────────────────────────
# 4. 单个 Agent 任务执行
# ──────────────────────────────────────────────

async def _run_single_agent_task(task: dict, state: AgentGraphState) -> dict:
    """根据 agent_type 路由到对应的处理函数"""
    agent_type = task.get("agent_type", "")
    llm = _get_llm(state, agent_type)
    task_input = task.get("input", {})
    start_time = time.time()

    handler = HANDLER_MAP.get(agent_type)
    if handler:
        try:
            t_input = {**task_input}
            dep_context = _build_dependency_context(task, state)
            if dep_context:
                t_input["_context"] = dep_context

            result = await handler(t_input, llm)
            elapsed = round(time.time() - start_time, 2)
            result["time_elapsed"] = elapsed
            if "thinking_steps" not in result:
                result["thinking_steps"] = []
            return result
        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            logger.error(f"Agent {agent_type} failed: {e}")
            return {
                "status": "failed",
                "agent_type": agent_type,
                "error": str(e),
                "time_elapsed": elapsed,
                "thinking_steps": [{"agent": agent_type, "action": "error", "reasoning": str(e), "status": "failed"}],
            }

    return await handle_fallback(task, llm, state)


def _build_dependency_context(task: dict, state: AgentGraphState) -> dict:
    """构建依赖上下文（只传递 public content，防止 Context Bleeding）"""
    task_queue = state.get("task_queue", [])
    task_results = state.get("task_results", {})
    context = {}
    for dep in task.get("dependencies", []):
        dep_id = dep
        if isinstance(dep, int) and 0 <= dep < len(task_queue):
            dep_id = task_queue[dep].get("id", "")
        if dep_id in task_results:
            prev = task_results[dep_id]
            context[dep_id] = {
                "content": prev.get("output", {}).get("content", ""),
                "agent": prev.get("agent_type", ""),
            }
    return context if context else {}


# ──────────────────────────────────────────────
# 5. check_task_results 节点
# ──────────────────────────────────────────────

async def check_task_results(state: AgentGraphState) -> dict:
    """检查批次执行结果，标记失败任务"""
    return {}  # 目前不做额外处理，失败已在 execute_batch 中标记

# ──────────────────────────────────────────────
# 6. merge_and_finish 节点
# ──────────────────────────────────────────────

async def merge_and_finish(state: AgentGraphState) -> dict:
    """合并所有子任务结果为最终输出 + 持久化"""
    plan = state.get("plan") or {}
    task_results = state.get("task_results", {})
    task_queue = state.get("task_queue", [])
    completed_ids = set(state.get("completed_task_ids", []))
    failed_ids = set(state.get("failed_task_ids", []))

    sections = []
    full_text_parts = []

    for task in task_queue:
        tid = task.get("id", "")
        result = task_results.get(tid, {})
        if tid in completed_ids or tid in failed_ids:
            sections.append({
                "agent": result.get("agent_type", task.get("agent_type", "")),
                "task_input": task.get("input", {}),
                "output": result.get("output", {}),
                "status": result.get("status", "unknown"),
                "error": result.get("error"),
            })
            content = result.get("output", {}).get("content", "")
            if content:
                full_text_parts.append(f"\n\n--- {result.get('agent_type', task.get('agent_type', ''))} ---\n{content}")

    completed = len(completed_ids)
    failed = len(failed_ids)

    status = "completed"
    if failed > 0 and completed == 0:
        status = "failed"
    elif failed > 0:
        status = "partial"

    # 持久化
    _persist_results(state, sections, full_text_parts, status)

    return {
        "final_output": {
            "goal": plan.get("goal", ""),
            "plan_reasoning": plan.get("reasoning", ""),
            "sections": sections,
            "full_text": "".join(full_text_parts),
        },
        "status": status,
    }


def _persist_results(state: AgentGraphState, sections: list, full_text_parts: list, status: str) -> None:
    """持久化执行结果到数据库"""
    try:
        from .persistence import save_agent_task, save_subtask

        task_id = state.get("_async_task_id", "")
        if not task_id:
            return

        # 任务级持久化
        save_agent_task(
            task_id=task_id or state.get("trace_task_id", ""),
            user_id=state.get("context", {}).get("user_id", ""),
            prompt=state.get("user_request", ""),
            status=status,
            result={"sections": sections, "full_text": "".join(full_text_parts)},
            total_steps=len(state.get("task_queue", [])),
            completed_steps=len(state.get("completed_task_ids", [])),
            total_tokens=sum(s.get("tokens_used", 0) for s in state.get("task_results", {}).values()),
            thinking_trace={"steps": state.get("thinking_steps", [])},
        )

        # 子任务级持久化
        for task in state.get("task_queue", []):
            tid = task.get("id", "")
            result = state.get("task_results", {}).get(tid, {})
            if result:
                save_subtask(
                    task_id=task_id,
                    agent_type=result.get("agent_type", task.get("agent_type", "")),
                    status=result.get("status", "unknown"),
                    input_data=task.get("input", {}),
                    output_data=result.get("output", {}),
                    tokens_used=result.get("tokens_used", 0),
                    time_elapsed=result.get("time_elapsed", 0.0),
                    error=result.get("error"),
                )
    except Exception as e:
        logger.warning(f"Persistence failed: {e}")


# ──────────────────────────────────────────────
# 路由函数（用于条件边）
# ──────────────────────────────────────────────

def should_continue(state: AgentGraphState) -> str:
    """检查是否还有未完成的任务"""
    task_queue = state.get("task_queue", [])
    completed = len(state.get("completed_task_ids", []))
    failed = len(state.get("failed_task_ids", []))

    if completed + failed >= len(task_queue):
        return "done"
    return "continue"
