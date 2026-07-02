"""Agent Streaming v2 / observability / result schema 收敛的单元测试

对应 spec: docs/superpowers/specs/2026-07-01-streaming-frontend-contract.md

测试矩阵：
1. _transform_result_for_frontend — 收敛 schema 字段映射
2. _result_for_persistence — 运行时缓冲字段拼接
3. _should_skip_in_business_track — 业务轨过滤（thinking_step / reasoning / observability-only）
4. StreamEvent audience 字段默认值
5. publish_thinking_step / publish_reasoning 的 audience 默认值
"""

from __future__ import annotations

import sys
import asyncio
import inspect
from pathlib import Path

import pytest

# ── 路径 setup ──
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from src.agent.api import (  # noqa: E402
    _transform_result_for_frontend,
    _result_for_persistence,
    _should_skip_in_business_track,
)
from src.agent.workflow.task_queue import (  # noqa: E402
    StreamEvent,
    task_queue,
)


# ─────────────────────────────────────────────────────────
# 1. _transform_result_for_frontend
# ─────────────────────────────────────────────────────────


class TestTransformResultForFrontend:
    """result 事件收敛 schema（spec §3.3）"""

    def test_basic_fields_present(self):
        """必填字段都在"""
        result = {
            "task_id": "abc-123",
            "status": "completed",
            "agent_type": "WritingAgent",
            "output": {"full_text": "hello world"},
            "usage": {"tokens_used": 100, "time_elapsed": 2.5, "model": "gpt-4o"},
        }
        transformed = _transform_result_for_frontend(result)

        assert transformed["task_id"] == "abc-123"
        assert transformed["status"] == "completed"
        assert transformed["agent_type"] == "WritingAgent"
        assert transformed["output"]["full_text"] == "hello world"
        assert transformed["usage"]["tokens_used"] == 100
        assert transformed["usage"]["time_elapsed"] == 2.5
        assert transformed["usage"]["model"] == "gpt-4o"

    def test_runtime_buffer_fields_excluded(self):
        """运行时缓冲字段绝不出现在收敛 schema 里"""
        result = {
            "task_id": "abc",
            "output": {"full_text": "x"},
            "_stages": [{"agent": "WritingAgent"}],
            "_thinkingSteps": [{"step_number": 1}],
            "_llmChunks": {"WritingAgent": [{"content": "x"}]},
            "_reasoningChunks": {"WritingAgent|0": [{"content": "x"}]},
            "_draftContent": "draft",
            "agent_events": {"events": [{"timestamp": "..."}]},
        }
        transformed = _transform_result_for_frontend(result)

        for forbidden in ("_stages", "_thinkingSteps", "_llmChunks",
                          "_reasoningChunks", "_draftContent", "agent_events"):
            assert forbidden not in transformed, (
                f"前端契约不应包含 {forbidden}（运行时缓冲 / observability-only）"
            )

    def test_handles_none_result(self):
        """result 是 None 时安全返回（不抛异常）"""
        transformed = _transform_result_for_frontend(None)
        assert isinstance(transformed, dict)
        # output / usage 应保持 schema 形状（key 都在，值是 None），不是空 dict
        assert set(transformed["output"].keys()) == {
            "full_text", "title", "platform", "content_type",
            "draft_content", "review_content",
        }
        assert all(v is None for v in transformed["output"].values())
        assert set(transformed["usage"].keys()) == {
            "tokens_used", "time_elapsed", "model",
        }
        # 标识字段应为 None
        assert transformed["task_id"] is None
        assert transformed["status"] is None
        assert transformed["agent_type"] is None

    def test_handles_non_dict_output(self):
        """output 不是 dict 时安全返回（不抛异常）"""
        result = {"output": "broken", "usage": []}
        transformed = _transform_result_for_frontend(result)
        # output/usage 应该 safe return
        assert transformed["output"]["full_text"] is None
        assert transformed["usage"]["tokens_used"] is None

    def test_full_text_falls_back_to_content(self):
        """output.full_text 缺失时回退到 output.content"""
        result = {"output": {"content": "from-content"}}
        transformed = _transform_result_for_frontend(result)
        assert transformed["output"]["full_text"] == "from-content"

    def test_usage_only_keeps_whitelisted_fields(self):
        """usage 收敛到 3 字段（tokens_used / time_elapsed / model），estimated_cost 不下发"""
        result = {
            "usage": {
                "tokens_used": 100,
                "time_elapsed": 2.0,
                "model": "gpt-4o",
                "estimated_cost": 0.001,
                "internal_field": "should-not-leak",
            }
        }
        transformed = _transform_result_for_frontend(result)
        assert "estimated_cost" not in transformed["usage"]
        assert "internal_field" not in transformed["usage"]
        # 白名单字段都在
        assert transformed["usage"]["tokens_used"] == 100
        assert transformed["usage"]["time_elapsed"] == 2.0
        assert transformed["usage"]["model"] == "gpt-4o"

    def test_output_keeps_all_business_fields(self):
        """output 子字段白名单：full_text / title / platform / content_type / draft_content / review_content"""
        result = {
            "output": {
                "full_text": "正文",
                "title": "标题",
                "platform": "xiaohongshu",
                "content_type": "blog",
                "draft_content": "草稿",
                "review_content": "评审",
                "internal_garbage": "no",
            }
        }
        transformed = _transform_result_for_frontend(result)
        out = transformed["output"]
        for expected in ("full_text", "title", "platform", "content_type",
                         "draft_content", "review_content"):
            assert expected in out
        assert "internal_garbage" not in out


# ─────────────────────────────────────────────────────────
# 2. _result_for_persistence
# ─────────────────────────────────────────────────────────


class TestResultForPersistence:
    """DB 持久化 schema（spec §3.3）— 把运行时缓冲拼回去"""

    def test_appends_runtime_buffer_to_frontend_result(self):
        """持久化版 = 收敛版 + 运行时缓冲字段"""
        frontend_result = _transform_result_for_frontend({
            "task_id": "abc",
            "output": {"full_text": "正文"},
            "usage": {"tokens_used": 100, "time_elapsed": 2.0, "model": "gpt-4o"},
        })

        persisted = _result_for_persistence(
            frontend_result,
            stages=[{"agent": "WritingAgent", "status": "completed"}],
            thinking_steps=[{"step_number": 1, "agent": "WritingAgent"}],
            llm_chunks={"WritingAgent": [{"content": "x"}]},
            reasoning_chunks={"WritingAgent|0": [{"content": "y"}]},
            draft_content="草稿",
            agent_events={"events": [{"timestamp": "..."}]},
        )

        # 收敛版字段保留
        assert persisted["task_id"] == "abc"
        assert persisted["output"]["full_text"] == "正文"
        assert persisted["usage"]["tokens_used"] == 100

        # 运行时缓冲字段全部带入
        assert persisted["_stages"] == [{"agent": "WritingAgent", "status": "completed"}]
        assert persisted["_thinkingSteps"] == [{"step_number": 1, "agent": "WritingAgent"}]
        assert persisted["_llmChunks"] == {"WritingAgent": [{"content": "x"}]}
        assert persisted["_reasoningChunks"] == {"WritingAgent|0": [{"content": "y"}]}
        assert persisted["_draftContent"] == "草稿"
        assert persisted["agent_events"] == {"events": [{"timestamp": "..."}]}

    def test_partial_persistence_only_includes_provided_fields(self):
        """只传入部分字段时，其余不入库（不写入 None）"""
        frontend_result = _transform_result_for_frontend({"task_id": "abc"})
        persisted = _result_for_persistence(
            frontend_result,
            stages=[{"agent": "x"}],
            # 其它字段不传
        )

        assert "_stages" in persisted
        for forbidden in ("_thinkingSteps", "_llmChunks",
                          "_reasoningChunks", "_draftContent", "agent_events"):
            assert forbidden not in persisted

    def test_persistence_does_not_mutate_input(self):
        """持久化不能修改入参 frontend_result"""
        frontend_result = _transform_result_for_frontend({"output": {"full_text": "x"}})
        snapshot = dict(frontend_result)
        snapshot["output"] = dict(frontend_result["output"])

        _result_for_persistence(frontend_result, stages=[{"agent": "y"}])

        assert frontend_result == snapshot
        # 特别注意：output 不应该被原地加 _stages
        assert "_stages" not in frontend_result


# ─────────────────────────────────────────────────────────
# 3. _should_skip_in_business_track
# ─────────────────────────────────────────────────────────


class TestShouldSkipInBusinessTrack:
    """业务轨过滤逻辑

    第四批修复（2026-07-01）：
        thinking_step / reasoning 不再被 _INTERNAL_EVENT_TYPES_V2 黑名单过滤。
        改为按 audience 字段判断：仅 audience='observability' 的事件不下发。
        这两个事件默认 audience='both'，业务轨下发，前端累积到 stages[].reasoning。
    """

    def test_thinking_step_passes_through(self):
        """thinking_step 默认 audience='both' → 业务轨下发（修复前是 True 跳过）"""
        evt = StreamEvent(task_id="x", event_type="thinking_step", data={})
        assert _should_skip_in_business_track(evt) is False

    def test_reasoning_passes_through(self):
        """reasoning 默认 audience='both' → 业务轨下发（修复前是 True 跳过）"""
        evt = StreamEvent(task_id="x", event_type="reasoning", data={})
        assert _should_skip_in_business_track(evt) is False

    def test_thinking_step_observability_is_skipped(self):
        """显式 audience='observability' 的 thinking_step 才不下发"""
        evt = StreamEvent(
            task_id="x",
            event_type="thinking_step",
            data={},
            audience="observability",
        )
        assert _should_skip_in_business_track(evt) is True

    def test_business_events_pass_through(self):
        for et in ("plan", "task_created", "progress", "stage",
                   "llm_chunk", "result", "error", "done"):
            evt = StreamEvent(task_id="x", event_type=et, data={})
            assert _should_skip_in_business_track(evt) is False, (
                f"{et} 应该是业务事件"
            )

    def test_explicit_audience_observability_is_skipped(self):
        """显式 audience='observability' 的事件也跳过"""
        evt = StreamEvent(
            task_id="x",
            event_type="custom_event",
            data={},
            audience="observability",
        )
        assert _should_skip_in_business_track(evt) is True

    def test_audience_both_passes_through(self):
        """audience='both' 的事件在业务轨下发（默认 publish_thinking_step 就是 both）"""
        evt = StreamEvent(
            task_id="x",
            event_type="thinking_step",
            data={},
            audience="both",
        )
        # audience='both' 时不再过滤，business 轨必须下发
        assert _should_skip_in_business_track(evt) is False


# ─────────────────────────────────────────────────────────
# 4. StreamEvent audience 字段默认值
# ─────────────────────────────────────────────────────────


class TestStreamEventAudienceDefault:
    """StreamEvent.audience 字段必须存在，默认 'frontend'"""

    def test_default_audience_is_frontend(self):
        evt = StreamEvent(task_id="x", event_type="result", data={})
        assert evt.audience == "frontend", (
            "默认 audience='frontend' 保持向后兼容"
        )

    def test_audience_can_be_set_explicitly(self):
        evt = StreamEvent(
            task_id="x",
            event_type="custom",
            data={},
            audience="observability",
        )
        assert evt.audience == "observability"

    def test_audience_can_be_both(self):
        evt = StreamEvent(
            task_id="x",
            event_type="thinking_step",
            data={},
            audience="both",
        )
        assert evt.audience == "both"

    def test_invalid_audience_rejected_by_type(self):
        """Literal 类型约束（dataclass 不强制，仅提示用），这里只验证字面量传错时仍可构造
        （dataclass 没有原生 type enforcement，留给 mypy / pyright）。"""
        # 注：dataclass 不会在构造时校验 Literal 值
        # 这是已知的 Python 限制，依赖静态类型检查
        try:
            evt = StreamEvent(
                task_id="x",
                event_type="thinking_step",
                data={},
                audience="invalid_value",
            )
            # 如果构造成功，本测试通过（dataclass 不强制）
            assert evt.audience == "invalid_value"
        except (ValueError, TypeError):
            pass  # Literal 校验生效


# ─────────────────────────────────────────────────────────
# 5. publish_thinking_step / publish_reasoning audience 默认值
# ─────────────────────────────────────────────────────────


class TestPublishMethodsAudienceDefault:
    """publish_* 方法默认 audience='both'（向后兼容）"""

    def test_publish_thinking_step_default_audience_is_both(self):
        sig = inspect.signature(task_queue.publish_thinking_step)
        audience_param = sig.parameters.get("audience")
        assert audience_param is not None, (
            "publish_thinking_step 应接受 audience 参数"
        )
        assert audience_param.default == "both", (
            "默认 audience='both'，让旧 /agent/stream 仍然能收到"
        )

    def test_publish_reasoning_default_audience_is_both(self):
        sig = inspect.signature(task_queue.publish_reasoning)
        audience_param = sig.parameters.get("audience")
        assert audience_param is not None
        assert audience_param.default == "both"


# ─────────────────────────────────────────────────────────
# 6. /agent/stream/v2 端点存在性 + deprecation 标记（路由级 smoke test）
# ─────────────────────────────────────────────────────────


class TestRouteRegistration:
    """关键端点必须注册到 router"""

    def test_legacy_stream_v1_marked_deprecated(self):
        from src.agent.api import router

        routes = {r.path: r for r in router.routes if hasattr(r, "path")}
        legacy = routes.get("/agent/stream")
        assert legacy is not None, "/agent/stream 必须存在（保留兼容）"
        assert legacy.deprecated is True, "/agent/stream 必须标记 deprecated"

    def test_v2_stream_endpoint_registered(self):
        from src.agent.api import router

        routes = {r.path: r for r in router.routes if hasattr(r, "path")}
        v2 = routes.get("/agent/stream/v2")
        assert v2 is not None, "/agent/stream/v2 必须注册"
        # v2 不应该 deprecated（FastAPI route 的 deprecated 字段：
        # 未设则为 None，显式设置才是 False/True）
        assert v2.deprecated in (False, None), (
            f"v2 不应是 deprecated，当前值: {v2.deprecated}"
        )

    def test_obs_stream_endpoint_registered(self):
        from src.agent.api import router

        routes = {r.path: r for r in router.routes if hasattr(r, "path")}
        obs = routes.get("/agent/obs/stream/{task_id}")
        assert obs is not None, "/agent/obs/stream/{task_id} 必须注册"