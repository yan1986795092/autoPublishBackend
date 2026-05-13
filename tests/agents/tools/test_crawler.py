"""CrawlerTool (LangChain BaseTool) 测试"""
import json
import pytest
from src.tools.langchain import CrawlerTool


@pytest.mark.asyncio
async def test_mock_mode_returns_data():
    tool = CrawlerTool(provider="mock")
    result = json.loads(await tool._arun(url="https://example.com"))
    assert result["success"] is True
    assert "title" in result["data"]
    assert "content" in result["data"]


@pytest.mark.asyncio
async def test_mock_mode_max_length():
    tool = CrawlerTool(provider="mock")
    result = json.loads(await tool._arun(url="https://example.com", max_length=20))
    assert result["success"] is True


@pytest.mark.asyncio
async def test_provider_non_mock_without_key_falls_back():
    """未提供 api_key 时自动降级为 mock"""
    tool = CrawlerTool(provider="real")
    result = json.loads(await tool._arun(url="https://example.com"))
    assert result["success"] is True
