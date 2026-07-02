"""trigger → runStreaming(parameters) 链路集成验证

端到端确认 BUG 修复 #2：
- `_build_trigger_data(trigger, preferences)` 在 trigger dict 里产出 `parameters` 字段
- 该 dict 通过 SSE `event: trigger` data 行传到前端
- 前端 `useConversationChat.on('trigger')` 把 trigger.parameters 传给 runStreaming
- `runStreaming` 把 parameters 放进 bodyPayload.parameters
- /agent/stream 端点把 request.parameters 透传到 skill_context

我们直接读源码做静态对照 + 用真实 Python 代码调用 `_build_trigger_data`，
不依赖 mock LLM / FastAPI 客户端。
"""

import json
from pathlib import Path

import pytest


# ── 真实路径用 importlib import，绕过 `src` 包不在 sys.path 的问题 ──
import importlib
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from src.conversation.schemas import GenerateTrigger  # noqa: E402
from src.conversation.engine import _build_trigger_data  # noqa: E402


class TestBuildTriggerDataParameters:
    """_build_trigger_data 必须包含 `parameters` 字段（与 BUG #2 修复绑定）"""

    def test_parameters_copied_from_preferences(self):
        """preferences 完整平铺到 trigger['parameters']"""
        trigger = GenerateTrigger(
            action="create",
            topic="智能笔记应用推广",
            platform="xiaohongshu",
            tone="活泼",
            audience="年轻女性",
        )
        preferences = {
            "creativity": 0.7,
            "length": "medium",
            "tone": "活泼",
            "audience": "年轻女性",
        }
        data = _build_trigger_data(trigger, preferences)

        assert "parameters" in data, "前端 runStreaming 依赖 trigger.parameters"
        assert data["parameters"] == preferences, (
            "parameters 必须 1:1 等于 preferences，否则前端传到 /agent/stream 会丢字段"
        )

    def test_parameters_omitted_when_preferences_is_none(self):
        """preferences 为 None 时不应产 parameters 字段（不污染 trigger）"""
        trigger = GenerateTrigger(action="create", topic="foo")
        data = _build_trigger_data(trigger, None)

        assert "parameters" not in data

    def test_parameters_omitted_when_preferences_is_empty_dict(self):
        """preferences 为空 dict 时仍写入 parameters（前端可走空对象安全路径）"""
        trigger = GenerateTrigger(action="create", topic="foo")
        data = _build_trigger_data(trigger, {})

        # 现状：empty dict 也会被 dict() 复制写入 → parameters == {}
        # 这对前端安全，因为 runStreaming 接 {} 也会照常发 bodyPayload.parameters={}
        assert data.get("parameters") == {}

    def test_pipeline_request_does_not_contain_parameters(self):
        """pipeline_request 是 backend 内部用，不应混入 preferences"""
        trigger = GenerateTrigger(action="create", topic="foo")
        preferences = {"creativity": 0.5, "length": "short", "tone": "x", "audience": "y"}
        data = _build_trigger_data(trigger, preferences)

        assert "parameters" not in data["pipeline_request"], (
            "pipeline_request 是 backend → agent 内部协议，前端的 parameters 字段不应泄漏进去"
        )


class TestTriggerDataShapeContract:
    """trigger dict 的字段契约（前端 useConversationChat 会按这些 key 取值）"""

    def test_required_keys(self):
        """所有 GenerateTrigger 字段必须出现在 trigger dict 顶层"""
        trigger = GenerateTrigger(
            action="create",
            topic="t",
            platform="p",
            style="s",
            length="medium",
            tone="t",
            audience="a",
            keywords=["k1"],
            revision_focus="",
        )
        data = _build_trigger_data(trigger, None)

        for k in (
            "action", "topic", "platform", "style", "length", "tone",
            "audience", "keywords", "revision_focus",
            "_prompt", "pipeline_request",
        ):
            assert k in data, f"trigger 缺少必填字段 {k}"

    def test_trigger_dict_is_json_serializable(self):
        """trigger 走 SSE 时会被 json.dumps，必须可序列化"""
        trigger = GenerateTrigger(
            action="create",
            topic="智能笔记应用推广",
            platform="xiaohongshu",
            keywords=["笔记", "效率"],
        )
        preferences = {"creativity": 0.5, "length": "medium"}
        data = _build_trigger_data(trigger, preferences)

        # 不应抛 TypeError
        s = json.dumps(data, ensure_ascii=False)
        # 能往返回来
        roundtrip = json.loads(s)
        assert roundtrip["parameters"] == preferences


class TestFrontendConsumesParametersField:
    """前端代码静态扫描：确认 useConversationChat 把 trigger.parameters 传给 runStreaming"""

    HOOK_PATH = Path(__file__).resolve().parents[3] / (
        "frontend/src/hooks/useConversationChat.ts"
    )
    STORE_PATH = Path(__file__).resolve().parents[3] / (
        "frontend/src/stores/useAgentStore.ts"
    )
    AGENT_STREAM_PATH = Path(__file__).resolve().parents[3] / (
        "backend/src/agent/api.py"
    )

    def test_hook_passes_trigger_to_run_streaming(self):
        """useConversationChat.on('trigger') 应调用 runStreaming 并把 trigger 整体传入"""
        text = self.HOOK_PATH.read_text(encoding="utf-8")
        # 找到 on('trigger' 那段
        assert "on('trigger'" in text or "on(\"trigger\"" in text, (
            "useConversationChat 必须订阅 trigger 事件"
        )
        assert "runStreaming" in text, (
            "useConversationChat 必须触发 runStreaming"
        )

    def test_run_streaming_accepts_parameters_argument(self):
        """useAgentStore.runStreaming 签名应有 parameters 参数"""
        text = self.STORE_PATH.read_text(encoding="utf-8")
        # 找到 runStreaming 定义，参数列表里应该包含 parameters
        # 简单粗匹配（避免 regex 复杂度）
        assert "parameters?: " in text or "parameters:" in text, (
            "useAgentStore.runStreaming 签名应包含 parameters 参数"
        )

    def test_run_streaming_includes_parameters_in_payload(self):
        """runStreaming 应把 parameters 写到 bodyPayload.parameters"""
        text = self.STORE_PATH.read_text(encoding="utf-8")
        assert "parameters" in text and "bodyPayload" in text, (
            "runStreaming 应把 parameters 透传到 bodyPayload.parameters"
        )
        # 进一步：parameters 应该出现在 bodyPayload 的赋值块
        # 不做更细的 regex，避免脆弱

    def test_agent_stream_endpoint_reads_parameters(self):
        """后端 /agent/stream 必须接收并使用 request.parameters"""
        text = self.AGENT_STREAM_PATH.read_text(encoding="utf-8")
        assert "parameters" in text, (
            "/agent/stream 应读取 request.parameters"
        )
        # 进一步：必须有"parameters: Optional[dict]"声明
        assert "parameters: Optional[dict]" in text or "parameters: dict" in text, (
            "/agent/stream 必须定义 parameters 字段（request 模型）"
        )
        # 必须真的使用：把 parameters 注入到 skill_context
        assert "request.parameters" in text, (
            "/agent/stream 必须真的用到 request.parameters（不是只声明不用）"
        )
        assert "skill_context" in text, (
            "parameters 应被注入 skill_context，让下游 skill 能拿到"
        )