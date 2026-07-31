"""Proxy settings configuration using pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from nim_proxy.core.reasoning import resolve_reasoning_budget, ResolvedReasoning


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 服务网络与安全
    port: int = Field(default=8000, validation_alias="PORT")
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    proxy_api_token: str = Field(default="", validation_alias="PROXY_API_TOKEN")
    http_timeout: float = Field(default=120.0, validation_alias="HTTP_TIMEOUT")

    # NVIDIA NIM & 多 Key 配置
    nvidia_nim_api_keys: str | list[str] = Field(
        default="",
        validation_alias="NVIDIA_NIM_API_KEYS",
    )
    nvidia_nim_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        validation_alias="NVIDIA_NIM_BASE_URL",
    )
    nvidia_nim_proxy: str = Field(
        default="",
        validation_alias="NVIDIA_NIM_PROXY",
    )
    key_cooldown_seconds: float = Field(
        default=60.0,
        validation_alias="KEY_COOLDOWN_SECONDS",
    )
    rate_limit_rpm: int = Field(
        default=0,
        validation_alias="RATE_LIMIT_RPM",
    )

    # 模型路由与别名映射
    nim_default_model: str = Field(
        default="nvidia/nemotron-3-super-120b-a12b",
        validation_alias="NIM_DEFAULT_MODEL",
    )
    model_sonnet: str = Field(default="", validation_alias="MODEL_SONNET")
    model_opus: str = Field(default="", validation_alias="MODEL_OPUS")
    model_haiku: str = Field(default="", validation_alias="MODEL_HAIKU")
    model_fable: str = Field(default="", validation_alias="MODEL_FABLE")

    # 全局与 Route-Specific 推理策略控制
    reasoning_preference: str = Field(default="client", validation_alias="REASONING_PREFERENCE")
    reasoning_sonnet: str = Field(default="inherit", validation_alias="REASONING_SONNET")
    reasoning_opus: str = Field(default="inherit", validation_alias="REASONING_OPUS")
    reasoning_haiku: str = Field(default="inherit", validation_alias="REASONING_HAIKU")
    reasoning_fable: str = Field(default="inherit", validation_alias="REASONING_FABLE")
    default_reasoning_budget: int = Field(default=4096, validation_alias="DEFAULT_REASONING_BUDGET")

    # Web Tools 网络增强工具与多 Provider
    enable_web_search: bool = Field(default=True, validation_alias="ENABLE_WEB_SEARCH")
    enable_web_fetch: bool = Field(default=True, validation_alias="ENABLE_WEB_FETCH")
    web_search_provider: str = Field(default="duckduckgo", validation_alias="WEB_SEARCH_PROVIDER")
    duckduckgo_max_results: int = Field(default=5, validation_alias="DUCKDUCKGO_MAX_RESULTS")
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")
    doubao_search_api_key: str = Field(default="", validation_alias="DOUBAO_SEARCH_API_KEY")

    # 高级与诊断
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    enable_native_tool_cleaning: bool = Field(default=True, validation_alias="ENABLE_NATIVE_TOOL_CLEANING")

    # CORS 允许来源（逗号分隔；使用 * 时自动关闭凭据以符合浏览器安全策略）
    cors_origins: str = Field(
        default="http://localhost:8000,http://127.0.0.1:8000",
        validation_alias="CORS_ORIGINS",
    )

    def parsed_api_keys(self) -> list[str]:
        if isinstance(self.nvidia_nim_api_keys, list):
            return [k.strip() for k in self.nvidia_nim_api_keys if k.strip()]
        if isinstance(self.nvidia_nim_api_keys, str):
            return [k.strip() for k in self.nvidia_nim_api_keys.split(",") if k.strip()]
        return []

    def get_model_for_alias(self, model_name: str) -> str:
        """Resolve model aliases (sonnet, opus, haiku, fable) or return default/given model."""
        m = model_name.lower()
        if "sonnet" in m and self.model_sonnet.strip():
            return self.model_sonnet.strip()
        if "opus" in m and self.model_opus.strip():
            return self.model_opus.strip()
        if "haiku" in m and self.model_haiku.strip():
            return self.model_haiku.strip()
        if "fable" in m and self.model_fable.strip():
            return self.model_fable.strip()
        if "/" in model_name or "nemotron" in m or "nvidia" in m:
            return model_name
        return self.nim_default_model

    def get_reasoning_for_model(self, model_name: str, client_budget: int | None = None) -> ResolvedReasoning:
        """Get resolved reasoning policy considering route-specific overrides."""
        m = model_name.lower()
        route_pref = "inherit"
        if "sonnet" in m:
            route_pref = self.reasoning_sonnet
        elif "opus" in m:
            route_pref = self.reasoning_opus
        elif "haiku" in m:
            route_pref = self.reasoning_haiku
        elif "fable" in m:
            route_pref = self.reasoning_fable

        return resolve_reasoning_budget(
            preference_str=route_pref,
            global_preference_str=self.reasoning_preference,
            default_budget=self.default_reasoning_budget,
            client_budget=client_budget,
        )


settings = Settings()
