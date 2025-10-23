"""语言模型配置模型。

参考: graphrag/config/models/language_model_config.py
"""

from pydantic import BaseModel, Field, model_validator

from graphrag_v2.config.defaults import language_model_defaults
from graphrag_v2.config.enums import AuthType, ModelType


class LanguageModelConfig(BaseModel):
    """语言模型配置类。
    
    定义 LLM 的连接参数、认证方式、速率限制等配置。
    """

    # ========== 基础配置 ==========
    type: ModelType | str = Field(
        description="模型类型",
    )
    
    model: str = Field(
        description="模型名称",
    )
    
    model_provider: str | None = Field(
        description="模型提供商（如 openai, azure 等）",
        default=language_model_defaults.model_provider,
    )

    # ========== 认证配置 ==========
    auth_type: AuthType = Field(
        description="认证类型",
        default=language_model_defaults.auth_type,
    )
    
    api_key: str | None = Field(
        description="API 密钥",
        default=language_model_defaults.api_key,
    )
    
    api_base: str | None = Field(
        description="API 基础 URL",
        default=language_model_defaults.api_base,
    )
    
    api_version: str | None = Field(
        description="API 版本（用于 Azure OpenAI）",
        default=language_model_defaults.api_version,
    )
    
    deployment_name: str | None = Field(
        description="部署名称（用于 Azure OpenAI）",
        default=language_model_defaults.deployment_name,
    )
    
    organization: str | None = Field(
        description="组织 ID",
        default=language_model_defaults.organization,
    )
    
    proxy: str | None = Field(
        description="代理服务器地址",
        default=language_model_defaults.proxy,
    )

    # ========== 编码配置 ==========
    encoding_model: str = Field(
        description="编码模型名称（用于 token 计算）",
        default=language_model_defaults.encoding_model,
    )

    # ========== 速率限制 ==========
    tokens_per_minute: int | None = Field(
        description="每分钟 token 限制",
        default=language_model_defaults.tokens_per_minute,
    )
    
    requests_per_minute: int | None = Field(
        description="每分钟请求限制",
        default=language_model_defaults.requests_per_minute,
    )
    
    concurrent_requests: int = Field(
        description="并发请求数",
        default=language_model_defaults.concurrent_requests,
    )

    # ========== 重试配置 ==========
    max_retries: int = Field(
        description="最大重试次数",
        default=language_model_defaults.max_retries,
    )
    
    max_retry_wait: float = Field(
        description="最大重试等待时间（秒）",
        default=language_model_defaults.max_retry_wait,
    )
    
    request_timeout: float = Field(
        description="请求超时时间（秒）",
        default=language_model_defaults.request_timeout,
    )

    # ========== 生成参数 ==========
    max_tokens: int | None = Field(
        description="最大生成 token 数",
        default=language_model_defaults.max_tokens,
    )
    
    temperature: float = Field(
        description="温度参数（控制随机性）",
        default=language_model_defaults.temperature,
    )
    
    top_p: float = Field(
        description="Top-p 采样参数",
        default=language_model_defaults.top_p,
    )
    
    n: int = Field(
        description="生成的完成数量",
        default=language_model_defaults.n,
    )
    
    model_supports_json: bool | None = Field(
        description="模型是否支持 JSON 输出模式",
        default=language_model_defaults.model_supports_json,
    )

    # ========== 验证方法 ==========
    def _validate_model_name(self) -> None:
        """验证模型名称。

        模型名称不能为空。
        """
        if not self.model or self.model.strip() == "":
            raise ValueError("模型名称不能为空")

    def _validate_api_key(self) -> None:
        """验证 API 密钥。

        当使用 API Key 认证时，建议提供 API 密钥。
        注意：这里只是警告，不强制要求，因为可以通过环境变量提供。
        """
        # 注意：由于 use_enum_values=True，auth_type 可能是字符串
        auth_type_value = self.auth_type if isinstance(self.auth_type, str) else self.auth_type.value

        # 只在实际使用时才需要 API 密钥，这里不强制验证
        # 允许创建没有 API 密钥的配置对象，稍后通过环境变量或配置文件提供
        pass

    def _validate_azure_settings(self) -> None:
        """验证 Azure OpenAI 配置。

        使用 Azure OpenAI 时，必须提供 api_base 和 api_version。
        """
        # 注意：由于 use_enum_values=True，type 可能是字符串
        type_value = self.type if isinstance(self.type, str) else self.type.value

        is_azure = (
            type_value == "azure_openai_chat"
            or type_value == "azure_openai_embedding"
            or self.model_provider == "azure"
        )

        if is_azure:
            if self.api_base is None or self.api_base.strip() == "":
                raise ValueError("使用 Azure OpenAI 时必须提供 api_base")

            if self.api_version is None or self.api_version.strip() == "":
                raise ValueError("使用 Azure OpenAI 时必须提供 api_version")

    def _validate_tokens_per_minute(self) -> None:
        """验证每分钟 token 限制。"""
        if self.tokens_per_minute is not None and self.tokens_per_minute < 1:
            raise ValueError("tokens_per_minute 必须大于 0")

    def _validate_requests_per_minute(self) -> None:
        """验证每分钟请求限制。"""
        if self.requests_per_minute is not None and self.requests_per_minute < 1:
            raise ValueError("requests_per_minute 必须大于 0")

    def _validate_max_retries(self) -> None:
        """验证最大重试次数。"""
        if self.max_retries < 1:
            raise ValueError("max_retries 必须大于等于 1")

    @model_validator(mode="after")
    def _validate_model(self):
        """模型级别的验证。

        在所有字段赋值后执行验证。
        """
        self._validate_model_name()
        self._validate_api_key()
        self._validate_azure_settings()
        self._validate_tokens_per_minute()
        self._validate_requests_per_minute()
        self._validate_max_retries()
        return self

    class Config:
        """Pydantic 配置。"""
        use_enum_values = True

