"""
Pydantic models for configuration management.
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class ProviderError(Exception):
    """Exception raised for provider configuration errors."""
    pass


class ClaudeConfig(BaseModel):
    """Configuration for Claude provider."""
    enabled: bool = True
    api_key: str = Field(..., description="Anthropic API key")
    base_url: str = Field(default="https://api.anthropic.com", description="Anthropic API base URL")
    api_version: str = Field(default="2023-06-01", description="Anthropic API version")


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI provider."""
    enabled: bool = True
    api_key: str = Field(..., description="OpenAI API key")
    base_url: str = Field(default="https://api.openai.com", description="OpenAI API base URL")
    organization: Optional[str] = Field(None, description="OpenAI organization ID")


class AzureOpenAIConfig(BaseModel):
    """Configuration for Azure OpenAI provider."""
    enabled: bool = True
    api_key: str = Field(..., description="Azure OpenAI API key")
    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_version: str = Field(default="2024-10-21", description="Azure OpenAI API version")


class ServerConfig(BaseModel):
    """Configuration for the server."""
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    log_level: str = Field(default="INFO", description="Log level")
    workers: int = Field(default=1, ge=1, description="Number of worker processes")


class FeaturesConfig(BaseModel):
    """Configuration for optional features."""
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")
    enable_caching: bool = Field(default=False, description="Enable response caching")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_rate_limiting: bool = Field(default=False, description="Enable rate limiting")
    max_requests_per_minute: int = Field(default=60, ge=1, description="Max requests per minute per client")


class SecurityConfig(BaseModel):
    """Configuration for security settings."""
    enable_auth: bool = Field(default=False, description="Enable authentication")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    allowed_origins: list = Field(default=["*"], description="Allowed CORS origins")
    enable_cors: bool = Field(default=True, description="Enable CORS")


class ProvidersConfig(BaseModel):
    """Configuration for all providers."""
    claude: Optional[ClaudeConfig] = None
    openai: Optional[OpenAIConfig] = None
    azure: Optional[AzureOpenAIConfig] = None


class ProxyConfig(BaseModel):
    """Main configuration model for the LLM proxy."""
    providers: ProvidersConfig
    model_mapping: Dict[str, str] = Field(default_factory=dict, description="Model to provider mapping")
    server: ServerConfig = Field(default_factory=ServerConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @validator('model_mapping')
    def validate_model_mapping(cls, v):
        """Validate that model mapping only references valid providers."""
        valid_providers = {'claude', 'openai', 'azure'}
        for model, provider in v.items():
            if provider not in valid_providers:
                raise ValueError(f"Invalid provider '{provider}' in model mapping. Must be one of: {valid_providers}")
        return v

    class Config:
        env_prefix = "LLM_PROXY_"
        case_sensitive = False