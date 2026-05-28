"""Tests for handler error wrapping — each handler should raise HandlerError on failure."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.exceptions import HandlerError


@pytest.mark.asyncio
async def test_handle_writing_raises_handler_error():
    """WritingAgent handler wraps exceptions in HandlerError"""
    from src.agent.handlers.writing import handle_writing

    llm = MagicMock(spec=["model"], model="gpt-4")
    task_input = {"topic": "test", "content_type": "blog"}

    # Mock PromptBuilder.build() to fail
    with (
        patch("src.agent.handlers.writing.PromptBuilder") as mock_builder_cls,
        pytest.raises(HandlerError, match="prompt builder failed"),
    ):
        instance = mock_builder_cls.return_value
        instance.build.side_effect = RuntimeError("prompt builder failed")

        await handle_writing(task_input, llm)


@pytest.mark.asyncio
async def test_handle_review_raises_handler_error():
    """ReviewAgent handler wraps exceptions in HandlerError"""
    from src.agent.handlers.review import handle_review

    llm = MagicMock(spec=["model"], model="gpt-4")
    task_input = {"content": "some content to review"}

    # Mock PromptBuilder to fail
    with (
        patch("src.agent.handlers.review.PromptBuilder") as mock_builder_cls,
        pytest.raises(HandlerError, match="builder failed"),
    ):
        instance = mock_builder_cls.return_value
        instance.build.side_effect = RuntimeError("builder failed")

        await handle_review(task_input, llm)


@pytest.mark.asyncio
async def test_handle_seo_raises_handler_error():
    """SEOAgent handler wraps exceptions in HandlerError"""
    from src.agent.handlers.seo import handle_seo

    llm = MagicMock(spec=["model"], model="gpt-4")
    task_input = {"topic": "test topic", "content": "some content"}

    # Mock suggest_related_keywords to fail
    with (
        patch("src.agent.handlers.seo.suggest_related_keywords", side_effect=ValueError("keyword tool failed")),
        pytest.raises(HandlerError, match="keyword tool failed"),
    ):
        await handle_seo(task_input, llm)


@pytest.mark.asyncio
async def test_handle_code_raises_handler_error():
    """CodeAgent handler wraps exceptions in HandlerError"""
    from src.agent.handlers.code import handle_code

    llm = MagicMock(spec=["model"], model="gpt-4")
    task_input = {"topic": "write a function", "language": "python"}

    # Mock PromptBuilder to fail
    with (
        patch("src.agent.handlers.code.PromptBuilder") as mock_builder_cls,
        pytest.raises(HandlerError, match="prompt build failed"),
    ):
        instance = mock_builder_cls.return_value
        instance.build.side_effect = RuntimeError("prompt build failed")

        await handle_code(task_input, llm)


@pytest.mark.asyncio
async def test_handler_error_inherits_agent_error():
    """HandlerError is a subclass of AgentError"""
    from src.core.exceptions import HandlerError, AgentError

    assert issubclass(HandlerError, AgentError)
    err = HandlerError("test error", handler="WritingAgent")
    assert err.handler == "WritingAgent"
    assert str(err) == "test error"


@pytest.mark.asyncio
async def test_handler_error_with_details():
    """HandlerError carries optional details dict"""
    from src.core.exceptions import HandlerError

    details = {"step": "llm_call", "tokens": 100}
    err = HandlerError("fail", handler="WritingAgent", details=details)
    assert err.details == details
