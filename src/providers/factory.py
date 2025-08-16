"""
Provider factory for creating LLM provider instances.
"""
import httpx
from typing import Dict, Any, Optional

from .base import BaseProvider, ProviderError
from .claude import ClaudeProvider
from .openai import OpenAIProvider
from .azure import AzureOpenAIProvider


class ProviderFactory:
    """Factory class for creating provider instances."""
    
    _providers = {
        'claude': ClaudeProvider,
        'openai': OpenAIProvider,
        'azure': AzureOpenAIProvider,
    }
    
    def __init__(self, config: Dict[str, Any], http_client: Optional[httpx.AsyncClient] = None):
        self.config = config
        self.http_client = http_client or httpx.AsyncClient()
        self._provider_instances: Dict[str, BaseProvider] = {}
    
    def create_provider(self, provider_name: str, provider_config: Dict[str, Any]) -> BaseProvider:
        """Create a provider instance."""
        if provider_name not in self._providers:
            raise ProviderError(f"Unknown provider: {provider_name}")
        
        try:
            provider_class = self._providers[provider_name]
            return provider_class(provider_config, self.http_client)
        except Exception as e:
            raise ProviderError(f"Failed to create {provider_name} provider: {str(e)}")
    
    def get_provider(self, provider_name: str) -> BaseProvider:
        """Get a cached provider instance or create a new one."""
        if provider_name not in self._provider_instances:
            provider_config = self.config.get('providers', {}).get(provider_name, {})
            if not provider_config.get('enabled', True):
                raise ProviderError(f"Provider {provider_name} is disabled")
            
            self._provider_instances[provider_name] = self.create_provider(provider_name, provider_config)
        
        return self._provider_instances[provider_name]
    
    def get_provider_for_model(self, model: str) -> BaseProvider:
        """Get the provider for a specific model based on configuration mapping."""
        model_mapping = self.config.get('model_mapping', {})
        provider_name = model_mapping.get(model)
        
        if not provider_name:
            # Try to auto-detect provider based on model name
            provider_name = self._auto_detect_provider(model)
        
        if not provider_name:
            raise ProviderError(f"No provider configured for model: {model}")
        
        return self.get_provider(provider_name)
    
    def _auto_detect_provider(self, model: str) -> Optional[str]:
        """Auto-detect provider based on model name patterns."""
        model_lower = model.lower()
        
        if model_lower.startswith('claude'):
            return 'claude'
        elif model_lower.startswith('gpt') or model_lower.startswith('o1'):
            return 'openai'
        elif 'embedding' in model_lower:
            # Check if it's an OpenAI embedding model
            if 'ada' in model_lower or 'text-embedding' in model_lower:
                return 'openai'
            elif model_lower in ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']:
                return 'azure'
        
        return None
    
    def list_available_providers(self) -> Dict[str, bool]:
        """List all available providers and their status."""
        providers = {}
        
        for provider_name in self._providers.keys():
            provider_config = self.config.get('providers', {}).get(provider_name, {})
            if provider_config is None:
                providers[provider_name] = False
            else:
                # Only enable if explicitly configured and enabled
                is_enabled = provider_config.get('enabled', False)
                has_api_key = 'api_key' in provider_config
                providers[provider_name] = is_enabled and has_api_key
        
        return providers
    
    def get_supported_models(self) -> Dict[str, str]:
        """Get all supported models and their providers."""
        model_mapping = self.config.get('model_mapping', {})
        
        # Add auto-detected models
        auto_models = {}
        for model in ['gpt-4o', 'gpt-4', 'gpt-3.5-turbo', 'claude-3-5-sonnet', 'claude-3-haiku']:
            provider = self._auto_detect_provider(model)
            if provider and model not in model_mapping:
                auto_models[model] = provider
        
        return {**model_mapping, **auto_models}
    
    def register_provider(self, name: str, provider_class: type) -> None:
        """Register a new provider type."""
        if not issubclass(provider_class, BaseProvider):
            raise ProviderError(f"Provider class must inherit from BaseProvider")
        
        self._providers[name] = provider_class
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all enabled providers."""
        results = {}
        
        for provider_name in self.list_available_providers():
            try:
                provider = self.get_provider(provider_name)
                results[provider_name] = await provider.health_check()
            except Exception:
                results[provider_name] = False
        
        return results
    
    async def close(self) -> None:
        """Close all provider instances and HTTP client."""
        for provider in self._provider_instances.values():
            await provider.close()
        
        if self.http_client and not self.http_client.is_closed:
            await self.http_client.aclose()
        
        self._provider_instances.clear()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()