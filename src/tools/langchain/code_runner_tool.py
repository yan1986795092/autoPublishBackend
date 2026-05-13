"""CodeRunner — LangChain BaseTool 版本（静态分析）"""
from __future__ import annotations
import ast
import json
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class CodeRunnerInput(BaseModel):
    code: str = Field(description="代码内容")
    language: str = Field(default="python", description="编程语言")


class CodeRunnerTool(BaseTool):
    name: str = "CodeRunner"
    description: str = "分析代码片段（静态分析，不执行代码）"
    args_schema: Type[BaseModel] = CodeRunnerInput
    api_key: str = ""
    provider: str = "mock"

    async def _arun(self, code: str, language: str = "python", **kwargs) -> str:
        if self.provider == "mock":
            return json.dumps({"success": True, "data": {"mock": True, "language": language, "code_length": len(code)}}, ensure_ascii=False)

        if language != "python":
            return json.dumps({"success": False, "error": f"不支持的語言: {language}"}, ensure_ascii=False)

        return json.dumps(self._static_analyze(code), ensure_ascii=False)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async _arun")

    def _static_analyze(self, code: str) -> dict:
        try:
            tree = ast.parse(code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
            loops = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
            conditionals = sum(1 for node in ast.walk(tree) if isinstance(node, ast.If))
            return {
                "success": True,
                "data": {
                    "static_analysis": {
                        "functions": functions,
                        "calls": calls,
                        "loops": loops,
                        "conditionals": conditionals,
                        "lines": len(code.splitlines()),
                    }
                },
                "summary": f"分析完成: {len(functions)} 个函数, {len(calls)} 个调用",
            }
        except SyntaxError as e:
            return {"success": False, "error": f"语法错误: {e}"}
