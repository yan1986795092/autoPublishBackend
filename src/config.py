from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, field_validator
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _env_file_path() -> Path:
    """返回 backend/.env 的绝对路径，从本文件所在位置解析。

    无论进程当前工作目录是什么均可正常工作。
    """
    return Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    app_name: str = "AutoPublish"
    debug: bool = True
    api_prefix: str = "/api/v1"

    database_url: str = "postgresql+psycopg2://autopublish:autopublish_dev@localhost:5432/autopublish"

    secret_key: str = "autopublish-local-dev-key-change-in-production-123456"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    refresh_token_expire_days: int = 7

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo-16k"
    openai_base_url: str = "https://api.openai.com/v1"

    # 阿里通义千问 — 兼容 QWEN_API_KEY / DASHSCOPE_API_KEY 两种环境变量名（主力，2026/08/20 到期）
    qwen_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("qwen_api_key", "QWEN_API_KEY", "DASHSCOPE_API_KEY"))
    qwen_model: str = "qwen3.7-max"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    zhipu_api_key: Optional[str] = None
    zhipu_model: str = "glm-4-flash"
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"

    # DeepSeek — 兼容 OpenAI 协议
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    bocha_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("bocha_api_key", "BOCHA_API_KEY", "BOCHA_SEARCH_API_KEY"))

    default_llm_provider: str = "qwen"
    default_temperature: float = 0.7
    default_max_tokens: int = 2000

    # ========== 按角色模型配置（可选） ==========
    # 主模型 (Manager/Orchestrator) — 任务拆解与规划
    manager_llm_provider: Optional[str] = None
    manager_model: Optional[str] = None
    manager_api_key: Optional[str] = None
    manager_base_url: Optional[str] = None

    # 生文模型 (WritingAgent) — 内容生成
    writing_llm_provider: Optional[str] = None
    writing_model: Optional[str] = None
    writing_api_key: Optional[str] = None
    writing_base_url: Optional[str] = None

    # 生图模型 (ImageAgent) — 图片生成
    image_llm_provider: Optional[str] = None
    image_model: Optional[str] = None
    image_api_key: Optional[str] = None
    image_base_url: Optional[str] = None

    # 子 Agent 模型 (ReviewAgent, SEOAgent 等)
    subagent_llm_provider: Optional[str] = None
    subagent_model: Optional[str] = None
    subagent_api_key: Optional[str] = None
    subagent_base_url: Optional[str] = None
    # ============================================================

    def get_role_config(self, role: str) -> dict:
        """返回某个角色的 {provider, model, api_key, base_url}。
        role 取值: manager / writing / image / subagent
        回退链: 角色专用配置 -> 该角色 provider 默认配置 -> 遍历全部 provider
        """
        provider = getattr(self, f"{role}_llm_provider", None) or self.default_llm_provider
        api_key = getattr(self, f"{role}_api_key", None)
        model = getattr(self, f"{role}_model", None)
        base_url = getattr(self, f"{role}_base_url", None)

        # 如果角色未设置 api_key，回退到角色的 provider 配置
        if not api_key:
            api_key = getattr(self, f"{provider}_api_key", "")
            model = model or getattr(self, f"{provider}_model", "")
            base_url = base_url or getattr(self, f"{provider}_base_url", "")

        # 如果还没有 key，遍历所有已知 provider 找第一个有效的
        if not api_key:
            for p in ["openai", "zhipu", "qwen", "deepseek"]:
                api_key = getattr(self, f"{p}_api_key", "")
                if api_key:
                    provider = p
                    model = getattr(self, f"{p}_model", "")
                    base_url = getattr(self, f"{p}_base_url", "")
                    break

        return {
            "provider": provider,
            "model": model or "",
            "api_key": api_key or "",
            "base_url": base_url or "",
        }

    # ========== 日志配置 ==========
    log_level: str = "INFO"
    log_json_format: bool = False

    # ========== Agent 编排 ==========
    worker_max_retries: int = 3
    orchestrator_timeout: float = 180.0
    task_concurrency: int = 3

    # ========== MCP (Model Context Protocol) ==========
    mcp_enabled: bool = True
    mcp_enable_search: bool = True
    mcp_enable_brave: bool = False
    mcp_enable_jina: bool = True
    mcp_enable_tavily: bool = False
    mcp_enable_pexels: bool = False
    mcp_enable_github: bool = False

    # ========== LLM 网关 ==========
    llm_gateway_url: str = "http://localhost:9010/v1"

    # ========== Agent Token 预算 ==========
    agent_token_budgets: dict = {
        "ManagerAgent": 4000,
        "WritingAgent": 4000,
        "SEOAgent": 2500,
        "ReviewAgent": 3000,
    }

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug(cls, value):
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on", "debug", "dev", "development", "local"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", "release", "prod", "production"}:
                return False
        return value

    model_config = SettingsConfigDict(
        env_file=str(_env_file_path()),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
