"""Agent 任务持久化服务：将内存中的任务记录写入 SQLAlchemy 模型"""
from __future__ import annotations
import json
from typing import Optional
from datetime import datetime

from src.database import SessionLocal
from src.models import AgentTask as AgentTaskModel, SubTask as SubTaskModel, ThinkingTraceRecord, ToolCall as ToolCallModel
import logging
logger = logging.getLogger(__name__)


def save_agent_task(
    task_id: str,
    user_id: str,
    prompt: str,
    status: str,
    result: Optional[dict] = None,
    error: Optional[str] = None,
    progress: float = 0.0,
    total_steps: int = 1,
    completed_steps: int = 0,
    total_tokens: int = 0,
    agent_events: Optional[dict] = None,
) -> None:
    """保存或更新 Agent 任务到数据库（幂等：重复 task_id 时覆盖而非报错）"""
    with SessionLocal() as db:
        try:
            task = db.query(AgentTaskModel).filter_by(id=task_id).first()
            events_json = json.dumps(agent_events, ensure_ascii=False) if agent_events else None

            if task:
                task.status = status
                task.progress = progress
                task.completed_steps = completed_steps
                task.total_tokens = total_tokens
                task.error_message = error
                if result:
                    task.result_json = json.dumps(result, ensure_ascii=False)
                if events_json:
                    task.agent_events_json = events_json
                if status in ("completed", "failed"):
                    task.completed_at = datetime.utcnow()
            else:
                task = AgentTaskModel(
                    id=task_id,
                    user_id=user_id,
                    status=status,
                    prompt=prompt,
                    input_summary=prompt[:200],
                    total_steps=total_steps,
                    completed_steps=completed_steps,
                    progress=progress,
                    total_tokens=total_tokens,
                    result_json=json.dumps(result, ensure_ascii=False) if result else None,
                    error_message=error,
                    agent_events_json=events_json,
                    completed_at=datetime.utcnow() if status in ("completed", "failed") else None,
                )
                db.add(task)

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"保存 Agent 任务失败: {e}")


def save_agent_events(task_id: str, agent_events: dict) -> None:
    """兼容旧接口：结构化事件先落在任务聚合 JSON 中。"""
    with SessionLocal() as db:
        try:
            task = db.query(AgentTaskModel).filter_by(id=task_id).first()
            if task:
                task.agent_events_json = json.dumps(agent_events, ensure_ascii=False)
                db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"保存 Agent 事件失败: {e}")


def save_thinking_trace(task_id: str, trace: dict) -> None:
    """兼容旧调用：将旧 thinking trace 归档到聚合 JSON，并尽量同步写明细表。"""
    save_agent_events(task_id, {"events": trace.get("steps", [])})
    with SessionLocal() as db:
        try:
            for step in trace.get("steps", []):
                step_num = step.get("step_number", 0)
                existing = db.query(ThinkingTraceRecord).filter_by(
                    task_id=task_id, step_number=step_num
                ).first()

                if existing:
                    existing.agent = step.get("agent", "")
                    existing.action = step.get("action", "")
                    existing.reasoning = step.get("reasoning", "")
                    existing.input_summary = step.get("input_summary", "")
                    existing.output_summary = step.get("output_summary", "")
                    existing.tokens_used = step.get("tokens_used", 0)
                    existing.time_elapsed = step.get("time_elapsed", 0.0)
                    existing.status = step.get("status", "done")
                else:
                    record = ThinkingTraceRecord(
                        task_id=task_id,
                        step_number=step_num,
                        agent=step.get("agent", ""),
                        action=step.get("action", ""),
                        reasoning=step.get("reasoning", ""),
                        input_summary=step.get("input_summary", ""),
                        output_summary=step.get("output_summary", ""),
                        tokens_used=step.get("tokens_used", 0),
                        time_elapsed=step.get("time_elapsed", 0.0),
                        status=step.get("status", "done"),
                    )
                    db.add(record)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"保存思维轨迹失败: {e}")


def save_subtask(
    task_id: str,
    agent_type: str,
    status: str,
    input_data: Optional[dict] = None,
    output_data: Optional[dict] = None,
    tokens_used: int = 0,
    time_elapsed: float = 0.0,
    error: Optional[str] = None,
) -> None:
    """保存单个子 Agent 的执行结果（幂等：相同 (task_id, agent_type) 且 input_json 相同时跳过）"""
    with SessionLocal() as db:
        try:
            input_json_str = json.dumps(input_data, ensure_ascii=False) if input_data else None
            output_json_str = json.dumps(output_data, ensure_ascii=False) if output_data else None
            existing = db.query(SubTaskModel).filter_by(
                task_id=task_id,
                agent_type=agent_type,
                input_json=input_json_str,
            ).first()

            if existing:
                existing.status = status
                existing.output_json = output_json_str
                existing.tokens_used = tokens_used
                existing.time_elapsed = time_elapsed
                existing.error_message = error
            else:
                record = SubTaskModel(
                    task_id=task_id,
                    agent_type=agent_type,
                    status=status,
                    input_json=input_json_str,
                    output_json=output_json_str,
                    tokens_used=tokens_used,
                    time_elapsed=time_elapsed,
                    error_message=error,
                )
                db.add(record)

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"保存子任务失败: {e}")


def get_subtask_history(task_id: str) -> list[dict]:
    """查询指定任务的所有子 Agent 执行记录"""
    with SessionLocal() as db:
        try:
            records = db.query(SubTaskModel).filter_by(task_id=task_id).order_by(SubTaskModel.created_at).all()
            return [
                {
                    "id": r.id,
                    "agent_type": r.agent_type,
                    "status": r.status,
                    "input": json.loads(r.input_json) if r.input_json else None,
                    "output": json.loads(r.output_json) if r.output_json else None,
                    "tokens_used": r.tokens_used,
                    "time_elapsed": r.time_elapsed,
                    "error": r.error_message,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in records
            ]
        except Exception as e:
            logger.error(f"获取子任务历史失败: {e}")
            return []


def save_tool_call(
    task_id: str,
    tool_name: str,
    parameters: Optional[dict] = None,
    result: Optional[dict] = None,
    status: str = "completed",
    error: Optional[str] = None,
    tokens_used: int = 0,
    time_elapsed: float = 0.0,
    subtask_id: Optional[str] = None,
) -> None:
    """保存工具调用记录（幂等：相同 (task_id, tool_name, parameters_json) 则跳过）"""
    with SessionLocal() as db:
        try:
            params_json = json.dumps(parameters, ensure_ascii=False) if parameters else None
            existing = db.query(ToolCallModel).filter_by(
                task_id=task_id,
                tool_name=tool_name,
                parameters_json=params_json,
            ).first()
            if existing:
                existing.status = status
                existing.result_json = json.dumps(result, ensure_ascii=False) if result else None
                existing.error_message = error
                existing.tokens_used = tokens_used
                existing.time_elapsed = time_elapsed
            else:
                record = ToolCallModel(
                    task_id=task_id,
                    subtask_id=subtask_id,
                    tool_name=tool_name,
                    parameters_json=params_json,
                    result_json=json.dumps(result, ensure_ascii=False) if result else None,
                    status=status,
                    error_message=error,
                    tokens_used=tokens_used,
                    time_elapsed=time_elapsed,
                )
                db.add(record)

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"保存工具调用失败: {e}")


def cleanup_old_records(retention_days: Optional[dict] = None) -> dict[str, int]:
    """清理过期的 Agent 数据（AGENT_ARCHITECTURE.md §11.3 数据保留策略）"""
    retention = retention_days or {
        "thinking_traces": 7,
        "tool_calls": 30,
        "agent_tasks": 90,
    }
    now = datetime.utcnow()
    counts: dict[str, int] = {}

    with SessionLocal() as db:
        try:
            from datetime import timedelta

            trace_cutoff = now - timedelta(days=retention["thinking_traces"])
            old_traces = db.query(ThinkingTraceRecord).filter(
                ThinkingTraceRecord.created_at < trace_cutoff
            ).delete(synchronize_session=False)
            counts["thinking_traces"] = old_traces

            tool_cutoff = now - timedelta(days=retention["tool_calls"])
            old_tools = db.query(ToolCallModel).filter(
                ToolCallModel.created_at < tool_cutoff
            ).delete(synchronize_session=False)
            counts["tool_calls"] = old_tools

            from src.models import AgentTask as AgentTaskModel
            task_cutoff = now - timedelta(days=retention["agent_tasks"])
            old_tasks = db.query(AgentTaskModel).filter(
                AgentTaskModel.status.in_(["completed", "failed", "cancelled"]),
                AgentTaskModel.completed_at < task_cutoff,
            ).delete(synchronize_session=False)
            counts["agent_tasks"] = old_tasks

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"清理旧记录失败: {e}")

    return counts
