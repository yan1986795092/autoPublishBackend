"""HotTopicsTool (LangChain BaseTool) 测试"""
import json
import pytest
from src.tools.langchain import HotTopicsTool


@pytest.mark.asyncio
async def test_mock_mode_returns_data():
    tool = HotTopicsTool(provider="mock")
    result = json.loads(await tool._arun(platform="weibo", count=5))
    assert result["success"] is True
    assert "topics" in result["data"]
    assert len(result["data"]["topics"]) == 5
    assert result["data"]["source"] == "mock"


@pytest.mark.asyncio
async def test_mock_mode_zhihu():
    tool = HotTopicsTool(provider="mock")
    result = json.loads(await tool._arun(platform="zhihu", count=3))
    assert result["success"] is True
    assert len(result["data"]["topics"]) == 3
    assert result["data"]["platform"] == "zhihu"


@pytest.mark.asyncio
async def test_invalid_platform_falls_back_to_weibo():
    tool = HotTopicsTool(provider="mock")
    result = json.loads(await tool._arun(platform="unknown", count=5))
    assert result["success"] is True
    assert len(result["data"]["topics"]) == 5


@pytest.mark.asyncio
async def test_mock_api_key_still_returns_mock():
    """即使有 api_key，当前也返回 mock"""
    tool = HotTopicsTool(api_key="test_key", provider="mock")
    result = json.loads(await tool._arun(platform="weibo", count=3))
    assert result["success"] is True
    assert result["data"]["source"] == "mock"
