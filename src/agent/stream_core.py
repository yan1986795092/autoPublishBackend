"""Agent SSE 核心生成器

被 /agent/stream/v3 和对话引擎 trigger 路径共用。
封装 Agent Pipeline 事件循环，统一输出 SSE 事件。
"""

from __future__ import annotations
import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Optional

import structlog

from src.models import User, ChatMessageRecord
from src.database import SessionLocal
from src.agent.part_writer import PartWriter
from src.agent.workflow.task_queue import task_queue, event_bus, StreamEvent
from src.api.chat_message_ui_v2 import _infer_artifact_type

logger = structlog.get_logger(__name__)


def _make_part_id(prefix: str, suffix: str = "") -> str:
    """生成 PartWriter part_id"""
    ts = int(time.time() * 1000)
    return f"{prefix}_{suffix}_{ts}" if suffix else f"{prefix}_{ts}"


async def run_agent_sse_generator(
    *,
    user_prompt: str,
    skill_context: dict,
    current_user: User,
    http_trace_id: str = "",
    message_id: str | None = None,
    conversation_id: str | None = None,
) -> AsyncGenerator[dict, None]:
    """运行 Agent Pipeline，生成 SSE 事件流

    与 /agent/stream/v3 端点行为完全一致（part_writer 持久化 + 事件翻译）。

    当 conversation_id 提供时，自动创建 chat_message_records 记录，
    避免 PartWriter flush 时外键约束失败。

    Yields:
        message_start, plan, task_created, text_delta, stage,
        reasoning_delta, artifact, message_end, error
    """
    db = SessionLocal()
    part_writer: PartWriter | None = None
    task_id_str = ""

    try:
        _message_id = message_id or str(uuid.uuid4())

        # 创建 message 记录，满足 chat_message_parts 外键约束
        if conversation_id:
            try:
                msg_record = ChatMessageRecord(
                    id=_message_id,
                    conversation_id=conversation_id,
                    role="assistant",
                    content="",
                    status="running",
                )
                db.add(msg_record)
                db.commit()
            except Exception as msg_err:
                logger.warning("创建 message_record 失败: %s", msg_err)
                db.rollback()

        part_writer = PartWriter(_message_id, db)

        # ── message_start ──
        yield {
            "event": "message_start",
            "data": json.dumps(
                {"message_id": _message_id, "role": "assistant"},
                ensure_ascii=False,
            ),
        }

        # ── plan 占位 ──
        yield {
            "event": "plan",
            "data": json.dumps({
                "part_id": _make_part_id("plan"),
                "goal": user_prompt[:200],
                "reasoning": "ManagerAgent 正在动态规划...",
                "tasks": [],
            }, ensure_ascii=False),
        }

        # ── EventBus 订阅 ──
        evt_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        async def on_event(event: StreamEvent) -> None:
            await evt_queue.put(event)

        task_id_str = task_queue.reserve_task_id()
        event_bus.subscribe(task_id_str, on_event)

        await task_queue.enqueue(
            prompt=user_prompt,
            user_id=current_user.id,
            plan=None,
            skill_context=skill_context if skill_context else None,
            task_id=task_id_str,
        )

        yield {
            "event": "task_created",
            "data": json.dumps(
                {"task_id": task_id_str, "message_id": _message_id},
                ensure_ascii=False,
            ),
        }

        # ── 事件循环 ──
        _got_terminal = False
        timeout_seconds = 600
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_seconds:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "code": "TIMEOUT",
                        "message": "任务执行超时",
                        "task_id": task_id_str,
                        "trace_id": http_trace_id,
                    }, ensure_ascii=False),
                }
                break

            try:
                event = await asyncio.wait_for(evt_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                task = task_queue.get_task(task_id_str)
                if not task or task.state.value in ("completed", "failed", "cancelled"):
                    break
                continue

            et = event.event_type
            data = event.data or {}

            # ── plan ──
            if et == "plan":
                part_id = _make_part_id("plan")
                yield {
                    "event": "message_start",
                    "data": json.dumps(
                        {"message_id": _message_id, "role": "assistant"},
                        ensure_ascii=False,
                    ),
                }
                yield {
                    "event": "plan",
                    "data": json.dumps({
                        "part_id": part_id,
                        "goal": data.get("goal", ""),
                        "reasoning": data.get("reasoning", ""),
                        "tasks": data.get("tasks", []),
                    }, ensure_ascii=False),
                }
                if part_writer:
                    await part_writer.append("plan", part_id, {
                        "goal": data.get("goal", ""),
                        "reasoning": data.get("reasoning", ""),
                        "tasks": data.get("tasks", []),
                    })

            # ── stage ──
            elif et == "stage":
                agent = data.get("agent", "")
                label = data.get("label", "")
                icon = data.get("icon", "⚙️")
                status = data.get("status", "running")
                round_ = data.get("reviewRound", 0)
                duration_ms = data.get("duration_ms")
                error_msg = data.get("error")
                part_id = data.get("part_id") or f"tool_{agent}_{round_}_{int(time.time() * 1000)}"

                if status == "running" and part_writer:
                    await part_writer.append_tool_call(part_id, agent, label, icon, round_)

                yield {
                    "event": "stage",
                    "data": json.dumps({
                        "part_id": part_id,
                        "agent": agent,
                        "label": label,
                        "icon": icon,
                        "status": status,
                        "round": round_,
                        **({"duration_ms": duration_ms} if duration_ms else {}),
                        **({"error": error_msg} if error_msg else {}),
                    }, ensure_ascii=False),
                }

                if status in ("done", "completed", "error") and part_writer:
                    state = "error" if error_msg else "done"
                    await part_writer.update(part_id, {
                        "state": state,
                        "output": data.get("output"),
                        "duration_ms": duration_ms,
                        "error_message": error_msg,
                    })
                    yield {
                        "event": "tool_result",
                        "data": json.dumps({
                            "part_id": part_id,
                            "output": data.get("output"),
                            "duration_ms": duration_ms,
                            "error": error_msg,
                        }, ensure_ascii=False),
                    }

            # ── llm_chunk → text_delta ──
            elif et == "llm_chunk":
                delta = data.get("chunk") or data.get("delta") or data.get("content") or ""
                if delta and part_writer:
                    await part_writer.append_text("text_main", delta)
                # 把 agent / channel / round 透传给前端，让 useConversationChat
                # 能把流式文本路由到对应 agent 的 llmChunks[agent]，StageTimeline
                # 才能实时显示中间产物。
                agent = data.get("agent", "")
                channel = data.get("channel", "body")
                review_round = data.get("_review_round", 0)
                yield {
                    "event": "text_delta",
                    "data": json.dumps(
                        {
                            "part_id": "text_main",
                            "delta": delta,
                            **({"agent": agent} if agent else {}),
                            **({"channel": channel} if channel else {}),
                            **({"round": review_round} if review_round else {}),
                        },
                        ensure_ascii=False,
                    ),
                }

            # ── reasoning → reasoning_delta ──
            elif et == "reasoning":
                agent = data.get("agent", "")
                round_ = data.get("round")
                content = data.get("content") or data.get("delta") or ""
                sid_suffix = f"{agent}" + (f"_r{round_}" if round_ else "")
                part_id = f"reasoning_{sid_suffix}_{int(time.time() * 1000)}"
                if content and part_writer:
                    await part_writer.append_reasoning(
                        part_id,
                        content,
                        {"agent": agent, "round": round_} if round_ else {"agent": agent},
                    )
                yield {
                    "event": "reasoning_delta",
                    "data": json.dumps({
                        "part_id": part_id,
                        "agent": agent,
                        "delta": content,
                        **({"round": round_} if round_ else {}),
                    }, ensure_ascii=False),
                }

            # ── result → artifact + message_end ──
            elif et == "result":
                _got_terminal = True
                output = data.get("output", {})
                full_text = output.get("full_text") or output.get("content", "")
                if part_writer:
                    await part_writer.append("artifact", "artifact_main", {
                        "artifact_type": _infer_artifact_type(output),
                        "body": full_text,
                        "title": output.get("title"),
                        "platform": output.get("platform"),
                        "structured": output.get("structured"),
                    })
                yield {
                    "event": "artifact",
                    "data": json.dumps({
                        "part_id": "artifact_main",
                        "artifact_type": _infer_artifact_type(output),
                        "body": full_text,
                        "title": output.get("title"),
                        "platform": output.get("platform"),
                        "structured": output.get("structured"),
                    }, ensure_ascii=False),
                }
                if part_writer:
                    await part_writer.finalize()
                task = task_queue.get_task(task_id_str)
                usage: dict = {}
                if task and task.result:
                    res = task.result or {}
                    usage = {
                        "tokens_used": res.get("usage", {}).get("tokens_used"),
                        "time_elapsed": res.get("usage", {}).get("time_elapsed"),
                        "model": res.get("usage", {}).get("model"),
                    }
                yield {
                    "event": "message_end",
                    "data": json.dumps({
                        "message_id": _message_id,
                        "status": "completed",
                        "usage": usage,
                    }, ensure_ascii=False),
                }
                break

            # ── error ──
            elif et == "error":
                _got_terminal = True
                if part_writer:
                    await part_writer.finalize()
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "code": "AGENT_ERROR",
                        "message": data.get("message", str(data)),
                    }, ensure_ascii=False),
                }
                yield {
                    "event": "message_end",
                    "data": json.dumps(
                        {"message_id": _message_id, "status": "failed"},
                        ensure_ascii=False,
                    ),
                }
        # ── 兜底：事件循环退出时未收到 result/error，从 task 状态读取 ──
        # 场景：EventBus 未推送事件，但任务已完成。直接从 task_queue 拉取结果。
        task = task_queue.get_task(task_id_str) if task_id_str else None
        if not _got_terminal and task and task.state.value == "completed" and task.result:
            output = task.result.get("output", {})
            if output:
                full_text = output.get("full_text") or output.get("content", "")
                if part_writer:
                    await part_writer.append("artifact", "artifact_main", {
                        "artifact_type": _infer_artifact_type(output),
                        "body": full_text,
                        "title": output.get("title"),
                        "platform": output.get("platform"),
                        "structured": output.get("structured"),
                    })
                yield {
                    "event": "artifact",
                    "data": json.dumps({
                        "part_id": "artifact_main",
                        "artifact_type": _infer_artifact_type(output),
                        "body": full_text,
                        "title": output.get("title"),
                        "platform": output.get("platform"),
                        "structured": output.get("structured"),
                    }, ensure_ascii=False),
                }
                if part_writer:
                    await part_writer.finalize()
                res = task.result or {}
                usage = {
                    "tokens_used": res.get("usage", {}).get("tokens_used"),
                    "time_elapsed": res.get("usage", {}).get("time_elapsed"),
                    "model": res.get("usage", {}).get("model"),
                }
                yield {
                    "event": "message_end",
                    "data": json.dumps({
                        "message_id": _message_id,
                        "status": "completed",
                        "usage": usage,
                    }, ensure_ascii=False),
                }
        elif not _got_terminal and task and task.state.value == "failed":
            if part_writer:
                await part_writer.finalize()
            yield {
                "event": "error",
                "data": json.dumps({
                    "code": "TASK_FAILED",
                    "message": task.error or "任务执行失败",
                }, ensure_ascii=False),
            }
            yield {
                "event": "message_end",
                "data": json.dumps(
                    {"message_id": _message_id, "status": "failed"},
                    ensure_ascii=False,
                ),
            }

    except Exception as e:
        logger.exception(
            "Agent SSE 流错误",
            _type="system",
            event_name="agent_sse_error",
            message=f"Agent SSE 错误: {e}",
            status="error",
        )
        yield {
            "event": "error",
            "data": json.dumps(
                {"code": "INTERNAL_ERROR", "message": str(e)},
                ensure_ascii=False,
            ),
        }

    finally:
        if task_id_str:
            event_bus.unsubscribe(task_id_str, on_event)
        if part_writer:
            try:
                await part_writer.finalize()
            except Exception:
                pass
        db.close()
