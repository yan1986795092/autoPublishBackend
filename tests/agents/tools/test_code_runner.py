"""CodeRunnerTool (LangChain BaseTool) 测试"""
import json
import pytest
from src.tools.langchain import CodeRunnerTool


@pytest.mark.asyncio
async def test_mock_mode_returns_result():
    tool = CodeRunnerTool(provider="mock")
    result = json.loads(await tool._arun(code="print('hello')", language="python"))
    assert result["success"] is True
    assert result["data"]["mock"] is True


@pytest.mark.asyncio
async def test_unsupported_language():
    tool = CodeRunnerTool(api_key="test", provider="real")
    result = json.loads(await tool._arun(code="console.log('hi')", language="javascript"))
    assert result["success"] is False
    assert "不支持的語言" in result["error"]


@pytest.mark.asyncio
async def test_static_analysis_mode():
    """非 mock 模式下进行静态分析"""
    tool = CodeRunnerTool(api_key="test", provider="real")
    result = json.loads(await tool._arun(code="def foo():\n    x = 1\n    return x", language="python"))
    assert result["success"] is True
    assert "static_analysis" in result["data"]
    assert "foo" in result["data"]["static_analysis"]["functions"]


@pytest.mark.asyncio
async def test_static_analysis_syntax_error():
    tool = CodeRunnerTool(api_key="test", provider="real")
    result = json.loads(await tool._arun(code="def foo(:", language="python"))
    assert result["success"] is False
