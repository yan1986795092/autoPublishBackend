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


def _build_artifact_structured(output: dict) -> dict | None:
    """构建 artifact 的 structured 数据。
    
    1. 优先使用 output.structured
    2. WebSearch 等工具的 results 数据自动合并到 structured.search_results
    """
    structured = output.get("structured")
    
    # WebSearch 工具：把 results 自动放入 structured.search_results
    results = output.get("results")
    if results and isinstance(results, list):
        if structured is None:
            structured = {}
        if "search_results" not in structured:
            structured["search_results"] = {
                "type": "search_results",
                "query": output.get("query", ""),
                "original_query": output.get("original_query", ""),
                "source": output.get("source", "bocha"),
                "results": results,
                "count": len(results),
            }
    
    return structured if isinstance(structured, dict) and structured else None


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
        # 如果 message_id 已存在（由 conversation.py 预先创建），跳过
        if conversation_id:
            existing = None
            try:
                existing = db.query(ChatMessageRecord).filter_by(id=_message_id).first()
            except Exception:
                pass
            if not existing:
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

        # ── 累积 stages / reasoning / llm_chunks，消息结束时写入 result_json ──
        # 让历史记录加载时能完整还原思考过程，不依赖 PartWriter 持久化
        _accumulated_stages: list[dict] = []
        _accumulated_reasoning: dict[str, list[dict]] = {}
        _accumulated_llm_chunks: dict[str, list[dict]] = {}
        _accumulated_thinking_steps: list[dict] = []
        # 追踪是否已接收过 stages（用于 retry 时重置累积器）
        _has_received_any_stage = False

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

                # ── 修复：检测到重试（已有 stages 后又来新 plan）→ 重置累积器 ──
                # 避免重试 pipeline 的 stages 追加在旧 stages 后面，导致 result_json
                # 中 stage 顺序错乱（如 WebSearch 出现在 FixAgent 之后）。
                if _has_received_any_stage:
                    logger.info("检测到重试 pipeline，重置 stage/reasoning/llmChunk 累积器")
                    _accumulated_stages.clear()
                    _accumulated_reasoning.clear()
                    _accumulated_llm_chunks.clear()
                    _accumulated_thinking_steps.clear()
                    _has_received_any_stage = False

            # ── stage ──
            elif et == "stage":
                agent = data.get("agent", "")
                label = data.get("label", "")
                icon = data.get("icon", "⚙️")
                status = data.get("status", "running")
                round_ = data.get("reviewRound", 0)
                duration_ms = data.get("duration_ms")
                error_msg = data.get("error")
                part_id = data.get("part_id") or f"tool_{agent}_{round_}"

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
                        "reviewRound": round_,  # 前端期望 reviewRound，同时保留 round 兼容
                        **({"duration_ms": duration_ms} if duration_ms else {}),
                        **({"error": error_msg} if error_msg else {}),
                    }, ensure_ascii=False),
                }

                # P0: 累积 stage，写入 result_json
                _stage_entry: dict = {
                    "agent": agent,
                    "label": label,
                    "icon": icon,
                    "status": "completed" if status in ("done", "completed") else ("failed" if status == "error" else "running"),
                    "reviewRound": round_,
                }
                if duration_ms:
                    _stage_entry["duration"] = duration_ms
                if error_msg:
                    _stage_entry["error"] = error_msg
                # 覆盖已存在的同名 stage（running → completed）
                for i, s in enumerate(_accumulated_stages):
                    if s["agent"] == agent and s.get("reviewRound", 0) == round_:
                        _accumulated_stages[i] = _stage_entry
                        break
                else:
                    _accumulated_stages.append(_stage_entry)

                _has_received_any_stage = True

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
                agent = data.get("agent", "unknown")
                channel = data.get("channel", "body")
                review_round = data.get("_review_round", 0)
                # 按 agent + round 分拆 text part，多轮审核/润色的内容各自独立
                text_part_id = f"text_{agent}_{review_round}" if review_round else f"text_{agent}"
                if delta and part_writer:
                    await part_writer.append_text(text_part_id, delta)
                yield {
                    "event": "text_delta",
                    "data": json.dumps(
                        {
                            "part_id": text_part_id,
                            "delta": delta,
                            **({"agent": agent} if agent else {}),
                            **({"channel": channel} if channel else {}),
                            **({"round": review_round} if review_round else {}),
                        },
                        ensure_ascii=False,
                    ),
                }
                # P0: 累积 llm_chunk，写入 result_json
                # 按 (agent, review_round) 分拆 key，多轮审核/润色各自独立
                if delta and agent != "unknown":
                    _llm_key = f"{agent}|{review_round}" if review_round else agent
                    if _llm_key not in _accumulated_llm_chunks:
                        _accumulated_llm_chunks[_llm_key] = []
                    _accumulated_llm_chunks[_llm_key].append({
                        "agent": agent,
                        "content": delta,
                        "timestamp": int(time.time() * 1000),
                    })

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
                # P0: 累积 reasoning，写入 result_json
                if content:
                    rkey = f"{agent}|{round_}" if round_ is not None else agent
                    if rkey not in _accumulated_reasoning:
                        _accumulated_reasoning[rkey] = []
                    _accumulated_reasoning[rkey].append({
                        "agent": agent,
                        "content": content,
                        "timestamp": int(time.time() * 1000),
                    })

            # ── agent_start → stage (running) ──
            # [修复] SearchAnalyzer 等 agent 第一条日志是 agent_start，不是 stage。
            # 透传为 stage 事件让前端能感知 agent 开始。
            elif et == "agent_start":
                agent = data.get("agent_name", data.get("agent", "unknown"))
                label = data.get("label", agent)
                # 使用和 stage 事件一致的 server_id 格式（不含时间戳），
                # 避免和后续 stage(running) 重复创建两个 tool_call part
                part_id = f"tool_{agent}_0"
                if part_writer:
                    await part_writer.append_tool_call(part_id, agent, label, "⚙️", 0)
                yield {
                    "event": "stage",
                    "data": json.dumps({
                        "part_id": part_id,
                        "agent": agent,
                        "label": label,
                        "icon": "⚙️",
                        "status": "running",
                        "round": 0,
                    }, ensure_ascii=False),
                }
                # P0: 累积 agent_start 为 running stage
                _accumulated_stages.append({
                    "agent": agent,
                    "label": label,
                    "icon": "⚙️",
                    "status": "running",
                    "reviewRound": 0,
                })
                _has_received_any_stage = True

            # ── agent_event → 透传到 SSE ──
            # [修复] WritingAgent 中间产物日志（"📎 工具结果"、"✅ 完成"、"● agent_event"）
            # 不再仅打日志，而是透传到前端让用户看到每个 agent 的完成状态。
            elif et == "agent_event":
                content = data.get("content", "") or data.get("message", "")
                if not content:
                    content = str(data)
                agent = data.get("agent", "unknown")
                part_id = _make_part_id(f"event_{agent}")
                yield {
                    "event": "agent_event",
                    "data": json.dumps({
                        "part_id": part_id,
                        "agent": agent,
                        "content": content,
                    }, ensure_ascii=False),
                }

            # ── thinking_step → 累积到 result_json ──
            # 每个 agent 完成时记录的步骤，包含 input_summary / output_summary / reasoning
            elif et == "thinking_step":
                step_data = data.get("thinking_step", data)
                if isinstance(step_data, dict) and step_data.get("agent"):
                    _accumulated_thinking_steps.append({
                        "agent": step_data.get("agent", ""),
                        "action": step_data.get("action", ""),
                        "reasoning": step_data.get("reasoning", ""),
                        "input_summary": step_data.get("input_summary", ""),
                        "output_summary": step_data.get("output_summary", ""),
                        "tokens_used": step_data.get("tokens_used", 0),
                        "time_elapsed": step_data.get("time_elapsed", 0),
                        "status": step_data.get("status", "done"),
                    })

            # ── task_partial → artifact（部分结果）──
            # [修复] 第一轮 pipeline 完成后，虽然 task_partial 被 worker 标记为失败重试，
            # 但内容已生成；直接 yield artifact 让用户能看到中间产物。
            elif et == "task_partial":
                output = data.get("output", {})
                full_text = output.get("full_text") or output.get("content", "")
                if full_text and part_writer:
                    # 追加而非覆盖（多轮生成时追加）
                    await part_writer.append("artifact", "artifact_partial", {
                        "artifact_type": _infer_artifact_type(output),
                        "body": full_text,
                        "title": output.get("title"),
                        "tags": output.get("tags"),
                        "platform": output.get("platform"),
                        "structured": _build_artifact_structured(output),
                    })
                yield {
                    "event": "artifact",
                    "data": json.dumps({
                        "part_id": "artifact_partial",
                        "artifact_type": _infer_artifact_type(output),
                        "body": full_text,
                        "title": output.get("title"),
                        "tags": output.get("tags"),
                        "platform": output.get("platform"),
                        "structured": _build_artifact_structured(output),
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
                        "tags": output.get("tags"),
                        "platform": output.get("platform"),
                        "structured": _build_artifact_structured(output),
                    })
                yield {
                    "event": "artifact",
                    "data": json.dumps({
                        "part_id": "artifact_main",
                        "artifact_type": _infer_artifact_type(output),
                        "body": full_text,
                        "title": output.get("title"),
                        "tags": output.get("tags"),
                        "platform": output.get("platform"),
                        "structured": _build_artifact_structured(output),
                    }, ensure_ascii=False),
                }
                # 更新消息正文 + result_json（含 accumulated stages/reasoning/llmChunks）
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
                if conversation_id and full_text:
                    try:
                        result_payload = {
                            "_stages": _accumulated_stages,
                            "_reasoningChunks": _accumulated_reasoning,
                            "_llmChunks": _accumulated_llm_chunks,
                            "_thinkingSteps": _accumulated_thinking_steps,
                            "output": output,
                            "usage": usage,
                        }
                        db.query(ChatMessageRecord).filter_by(id=_message_id).update({
                            "content": full_text,
                            "status": "completed",
                            "result_json": json.dumps(result_payload, ensure_ascii=False),
                        })
                        db.commit()
                    except Exception:
                        db.rollback()

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
                        "tags": output.get("tags"),
                        "platform": output.get("platform"),
                        "structured": _build_artifact_structured(output),
                    })
                yield {
                    "event": "artifact",
                    "data": json.dumps({
                        "part_id": "artifact_main",
                        "artifact_type": _infer_artifact_type(output),
                        "body": full_text,
                        "title": output.get("title"),
                        "tags": output.get("tags"),
                        "platform": output.get("platform"),
                        "structured": _build_artifact_structured(output),
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
                # P0: 更新 result_json（含 accumulated stages/reasoning/llmChunks）
                if conversation_id and full_text:
                    try:
                        result_payload = {
                            "_stages": _accumulated_stages,
                            "_reasoningChunks": _accumulated_reasoning,
                            "_llmChunks": _accumulated_llm_chunks,
                            "_thinkingSteps": _accumulated_thinking_steps,
                            "output": output,
                            "usage": usage,
                        }
                        db.query(ChatMessageRecord).filter_by(id=_message_id).update({
                            "content": full_text,
                            "status": "completed",
                            "result_json": json.dumps(result_payload, ensure_ascii=False),
                        })
                        db.commit()
                    except Exception:
                        db.rollback()
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
