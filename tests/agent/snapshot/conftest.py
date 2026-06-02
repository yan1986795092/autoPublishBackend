"""Snapshot 测试 fixtures"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from src.agent.snapshot.models import (
    ExecutionSnapshot, PromptMessageSnapshot, ContextSnapshot,
    LlmRequestSnapshot, LlmResponseSnapshot,
)


@pytest.fixture
def snapshot_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_snapshot():
    return ExecutionSnapshot(
        snapshot_id="snap_test001",
        agent_name="WritingAgent",
        timestamp=datetime(2026, 6, 1, 12, 0, 0),
        duration_ms=1500,
        task_id="task_001",
        prompt_messages=[
            PromptMessageSnapshot(
                role="system",
                template_name="system.base",
                template_version="v2.0",
                template_hash="abc123",
                source="template",
                resolved_content="You are an expert writer...",
            ),
            PromptMessageSnapshot(
                role="user",
                template_name="writing.task",
                template_version="v2.3",
                template_hash="def456",
                source="template",
                resolved_content="Write a blog post about AI...",
                variables={"topic": "AI", "content_type": "blog"},
            ),
        ],
        context=ContextSnapshot(
            profile_snapshot={"name": "WritingAgent", "role": "内容创作者"},
            memory_ids=["mem_001", "mem_002"],
            memory_text_preview="关于 AI 写作的前 200 字",
        ),
        llm_request=LlmRequestSnapshot(
            model="gpt-4o",
            params={"temperature": 0.7},
            messages_count=2,
            total_prompt_chars=350,
        ),
        llm_response=LlmResponseSnapshot(
            content="# AI 写作\n\n这是一篇关于 AI 的文章...",
            finish_reason="stop",
            prompt_tokens=200,
            completion_tokens=500,
            total_tokens=700,
            latency_ms=1200,
        ),
        result_summary="生成了 800 字的博客文章",
        result={"status": "completed", "output": {"content": "..."}},
        tags=["test", "integration"],
    )
