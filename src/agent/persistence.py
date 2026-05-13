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
    thinking_trace: Optional[dict] = None,
) -> None:
    """保存或更新 Agent 任务到数据库"""
    with SessionLocal() as db:
        try:
            task = db.query(AgentTaskModel).filter_by(id=task_id).first()

            if task:
                task.status = status
                task.progress = progress
                task.completed_steps = completed_steps
                task.total_tokens = total_tokens
                task.error_message = error
                if result:
                    task.result_json = json.dumps(result, ensure_ascii=False)
                if thinking_trace:
                    task.thinking_trace_json = json.dumps(thinking_trace, ensure_ascii=False)
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
                    thinking_trace_json=json.dumps(thinking_trace, ensure_ascii=False) if thinking_trace else None,
                    completed_at=datetime.utcnow() if status in ("completed", "failed") else None,
                )
                db.add(task)

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save agent task: {e}")


def save_thinking_trace(task_id: str, trace: dict) -> None:
    """保存 Thinking Trace 到数据库"""
    with SessionLocal() as db:
        try:
            for step in trace.get("steps", []):
                record = ThinkingTraceRecord(
                    task_id=task_id,
                    step_number=step.get("step_number", 0),
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
            logger.error(f"Failed to save thinking trace: {e}")


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
    """保存单个子 Agent 的执行结果"""
    with SessionLocal() as db:
        try:
            record = SubTaskModel(
                task_id=task_id,
                agent_type=agent_type,
                status=status,
                input_json=json.dumps(input_data, ensure_ascii=False) if input_data else None,
                output_json=json.dumps(output_data, ensure_ascii=False) if output_data else None,
                tokens_used=tokens_used,
                time_elapsed=time_elapsed,
                error_message=error,
            )
            db.add(record)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save subtask: {e}")


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
            logger.error(f"Failed to get subtask history: {e}")
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
    """保存工具调用记录（AGENT_ARCHITECTURE.md §11.1 ToolCall 表）"""
    with SessionLocal() as db:
        try:
            record = ToolCallModel(
                task_id=task_id,
                subtask_id=subtask_id,
                tool_name=tool_name,
                parameters_json=json.dumps(parameters, ensure_ascii=False) if parameters else None,
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
            logger.error(f"Failed to save tool call: {e}")


def cleanup_old_records(retention_days: Optional[dict] = None) -> dict[str, int]:
    """清理过期的 Agent 数据（AGENT_ARCHITECTURE.md §11.3 数据保留策略）

    Args:
        retention_days: 各类型的保留天数，默认：
            - thinking_traces: 7 天
            - tool_calls: 30 天
            - agent_tasks: 90 天

    Returns:
        各类型清理的记录数
    """
    retention = retention_days or {
        "thinking_traces": 7,
        "tool_calls": 30,
        "agent_tasks": 90,
    }
    now = datetime.utcnow()
    counts: dict[str, int] = {}

    with SessionLocal() as db:
        try:
            # 清理 ThinkingTrace（保留 7 天）
            trace_cutoff = now.timestamp() - retention["thinking_traces"] * 86400
            old_traces = db.query(ThinkingTraceRecord).filter(
                ThinkingTraceRecord.created_at < datetime.fromtimestamp(trace_cutoff)
            ).delete(synchronize_session=False)
            counts["thinking_traces"] = old_traces

            # 清理 ToolCall（保留 30 天）
            tool_cutoff = now.timestamp() - retention["tool_calls"] * 86400
            old_tools = db.query(ToolCallModel).filter(
                ToolCallModel.created_at < datetime.fromtimestamp(tool_cutoff)
            ).delete(synchronize_session=False)
            counts["tool_calls"] = old_tools

            # 清理已完成的 AgentTask（保留 90 天）
            task_cutoff = now.timestamp() - retention["agent_tasks"] * 86400
            from src.models import AgentTask as AgentTaskModel
            old_tasks = db.query(AgentTaskModel).filter(
                AgentTaskModel.status.in_(["completed", "failed", "cancelled"]),
                AgentTaskModel.completed_at < datetime.fromtimestamp(task_cutoff),
            ).delete(synchronize_session=False)
            counts["agent_tasks"] = old_tasks

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to cleanup old records: {e}")

    return counts
