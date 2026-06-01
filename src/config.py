from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    app_name: str = "AutoPublish"
    debug: bool = True
    api_prefix: str = "/api/v1"

    database_url: str = "sqlite:///./autopublish.db"

    secret_key: str = "autopublish-local-dev-key-change-in-production-123456"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    refresh_token_expire_days: int = 7

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo-16k"
    openai_base_url: str = "https://api.openai.com/v1"

    # anthropic config removed — was never used at runtime

    # 阿里通义千问 — 兼容 QWEN_API_KEY / DASHSCOPE_API_KEY 两种环境变量名
    qwen_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("qwen_api_key", "QWEN_API_KEY", "DASHSCOPE_API_KEY"))
    qwen_model: str = "qwen-plus"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    zhipu_api_key: Optional[str] = None
    zhipu_model: str = "glm-4-flash"
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"

    default_llm_provider: str = "zhipu"
    default_temperature: float = 0.7
    default_max_tokens: int = 2000

    # ========== Per-role model configuration (optional) ==========
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

    # 子Agent模型 (ReviewAgent, SEOAgent, HotTopicAgent, CodeAgent 等)
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
            for p in ["openai", "zhipu", "qwen"]:
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

    # ========== Logging configuration ==========
    log_level: str = "INFO"
    log_json_format: bool = False

    # ========== Agent feature flags ==========
    enable_reflection: bool = False
    enable_message_bus: bool = False
    enable_hitl: bool = False
    enable_tools: bool = True
    enable_agent_registry: bool = True

    # ========== MCP (Model Context Protocol) ==========
    # MCP Server 总开关
    mcp_enabled: bool = True
    # 各 MCP Server 独立开关（留空则从环境变量读取）
    mcp_enable_search: bool = True
    mcp_enable_brave: bool = False
    mcp_enable_jina: bool = True
    mcp_enable_tavily: bool = False
    mcp_enable_pexels: bool = False
    mcp_enable_github: bool = False

    # ========== Worker / orchestration ==========
    worker_max_retries: int = 3
    orchestrator_timeout: float = 180.0
    task_concurrency: int = 3

    # ========== Agent token budgets ==========
    agent_token_budgets: dict = {
        "ManagerAgent": 4000,
        "WritingAgent": 4000,
        "SEOAgent": 1000,
        "ReviewAgent": 1500,
        "DiagramAgent": 1500,
        "ImageAgent": 500,
        "HotTopicAgent": 500,
        "CodeAgent": 2000,
        "WebSearch": 500,
        "PlatformAdapter": 1000,
        "TranslationAgent": 1500,
        "URLCrawler": 0,
    }

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
