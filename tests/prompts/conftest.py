"""PromptRegistry 测试 fixtures"""
import pytest
from src.prompts.registry import PromptRegistry


@pytest.fixture
def registry():
    return PromptRegistry()
