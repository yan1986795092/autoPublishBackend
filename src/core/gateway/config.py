from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pydantic import ConfigDict
from functools import lru_cache
from typing import Optional


class GatewaySettings(BaseSettings):
    # ---- Rate Limiting ----
    rate_limit_backend: str = "memory"

    provider_limits: dict = Field(default_factory=lambda: {
        "openai":    {"rpm": 500,  "tpm": 1000000},
        "zhipuai":   {"rpm": 60,   "tpm": 100000},
        "qwen-plus": {"rpm": 120,  "tpm": 200000},
    })

    source_app_limits: dict = Field(default_factory=lambda: {
        "default": {"rpm": 1000},
    })

    queue_timeout_ms: int = 5000

    # ---- Security ----
    allowed_source_apps: list = Field(default_factory=lambda: ["agent-service", "web-frontend", "agent-studio"])
    allowed_models: list = Field(default_factory=lambda: [
        "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini",
        "glm-4-flash", "glm-4-plus",
        "qwen-plus", "qwen-max",
    ])
    max_content_length: int = 32768

    # ---- Router Weights ----
    router_default_weights: dict = Field(default_factory=lambda: {
        "cost": 0.4, "latency": 0.3, "actual_latency": 0.3,
    })

    router_hints: dict = Field(default_factory=lambda: {
        "simple":   {"cost": 0.7, "latency": 0.2, "actual_latency": 0.1},
        "complex":  {"cost": 0.1, "latency": 0.4, "actual_latency": 0.5},
    })

    # ---- Circuit Breaker ----
    cb_failure_threshold: int = 3
    cb_error_rate_threshold: float = 0.3
    cb_window_seconds: int = 300
    cb_cooldown_seconds: int = 30
    cb_half_open_probe_count: int = 3
    cb_health_check_interval: int = 60
    cb_health_check_timeout: int = 5
    cb_health_check_model: str = "gpt-3.5-turbo"

    # ---- Resilience ----
    global_request_timeout: int = 60
    provider_max_concurrency: int = 100

    # ---- Tracing ----
    trace_sample_rate: float = 1.0
    log_json: bool = True

    # ---- Database ----
    cost_db_path: str = "gateway_cost.db"
    cost_batch_size: int = 50
    cost_flush_interval: float = 5.0

    # ---- Pricing (per 1k tokens) ----
    pricing: dict = Field(default_factory=lambda: {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4o":         {"input": 0.005,  "output": 0.015},
        "gpt-4o-mini":    {"input": 0.00015, "output": 0.0006},
        "glm-4-flash":    {"input": 0.0005, "output": 0.0005},
        "glm-4-plus":     {"input": 0.01,   "output": 0.01},
        "qwen-plus":     {"input": 0.004,  "output": 0.012},
        "qwen-max":      {"input": 0.02,   "output": 0.06},
    })

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache()
def get_gateway_settings() -> GatewaySettings:
    return GatewaySettings()
