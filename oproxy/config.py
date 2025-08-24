"""
Configuration for LLM Proxy
Supports OpenAI, Claude, and Azure OpenAI endpoints
"""

from typing import Dict, Any, Optional
import os
from oproxy.utils import get_base_url


openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
azure_openai_base_url = os.getenv("AZURE_OPENAI_BASE_URL", "")

openai_api_key = os.getenv("OPENAI_API_KEY", "")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")


# LLM Provider configurations
LLM_PROVIDERS = {
    "openai": {
        "base_url": get_base_url(openai_base_url),
        "api_key": openai_api_key,
        "headers": {
            "Authorization": "Bearer {api_key}"
        }
    },
    "anthropic": {
        "base_url": get_base_url(anthropic_base_url),
        "api_key": anthropic_api_key,
        "headers": {
            "x-api-key": "{api_key}",
        }
    },
    "azure": {
        "base_url": get_base_url(azure_openai_base_url),
        "api_key": azure_openai_api_key,
        "headers": {
            "api-key": "{api_key}"
        }
    }
}


# Proxy settings
PROXY_CONFIG = {
    "host": "0.0.0.0",
    "port": 8899,
    "log_level": "DEBUG" # DEBUG, INFO
}


def get_providers():
    """Get list of configured providers"""
    return list(LLM_PROVIDERS.keys())


def get_provider_config(provider: str) -> Dict[str, Any]:
    """Get configuration for a specific provider"""
    return LLM_PROVIDERS.get(provider, {})

def validate_provider_config(provider: str) -> bool:
    """Validate that required configuration is present"""
    config = get_provider_config(provider)
    if not config:
        return False
    
    required_fields = ["base_url", "api_key"]
    for field in required_fields:
        if not config.get(field):
            return False
    
    return True

def get_supported_providers() -> Dict[str, Dict[str, Any]]:
    """Get list of supported providers and their base URLs"""
    
    return {
        name: {
            "base_url": config.get("base_url"),
            "configured": validate_provider_config(name)
        }
        for name, config in LLM_PROVIDERS.items()
    }