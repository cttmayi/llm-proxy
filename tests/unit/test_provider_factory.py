"""
Unit tests for provider factory.
"""
import pytest
import httpx
from unittest.mock import AsyncMock, patch
from src.providers.factory import ProviderFactory
from src.providers.base import BaseProvider, ProviderError
from src.providers.claude import ClaudeProvider
from src.providers.openai import OpenAIProvider
from src.providers.azure import AzureOpenAIProvider


@pytest.fixture
def factory_config():
    return {
        'providers': {
            'claude': {
                'enabled': True,
                'api_key': 'test-claude-key',
                'base_url': 'https://api.anthropic.com'
            },
            'openai': {
                'enabled': True,
                'api_key': 'test-openai-key',
                'base_url': 'https://api.openai.com'
            },
            'azure': {
                'enabled': True,
                'api_key': 'test-azure-key',
                'endpoint': 'https://test-resource.openai.azure.com'
            }
        },
        'model_mapping': {
            'gpt-4o': 'openai',
            'claude-3-5-sonnet': 'claude',
            'gpt-4': 'azure'
        }
    }


@pytest.fixture
def http_client():
    return httpx.AsyncClient()


@pytest.fixture
def provider_factory(factory_config, http_client):
    return ProviderFactory(factory_config, http_client)


class TestProviderFactory:
    
    def test_init(self, factory_config, http_client):
        factory = ProviderFactory(factory_config, http_client)
        assert factory.config == factory_config
        assert factory.http_client == http_client
        assert len(factory._provider_instances) == 0
    
    def test_init_without_http_client(self, factory_config):
        factory = ProviderFactory(factory_config)
        assert isinstance(factory.http_client, httpx.AsyncClient)
    
    def test_create_provider_claude(self, provider_factory):
        provider_config = {'api_key': 'test-key', 'base_url': 'https://api.anthropic.com'}
        provider = provider_factory.create_provider('claude', provider_config)
        assert isinstance(provider, ClaudeProvider)
        assert provider.api_key == 'test-key'
    
    def test_create_provider_openai(self, provider_factory):
        provider_config = {'api_key': 'test-key', 'base_url': 'https://api.openai.com'}
        provider = provider_factory.create_provider('openai', provider_config)
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == 'test-key'
    
    def test_create_provider_azure(self, provider_factory):
        provider_config = {'api_key': 'test-key', 'endpoint': 'https://test.azure.com'}
        provider = provider_factory.create_provider('azure', provider_config)
        assert isinstance(provider, AzureOpenAIProvider)
        assert provider.api_key == 'test-key'
    
    def test_create_provider_unknown(self, provider_factory):
        with pytest.raises(ProviderError) as exc_info:
            provider_factory.create_provider('unknown', {})
        assert "Unknown provider: unknown" in str(exc_info.value)
    
    def test_create_provider_failure(self, provider_factory):
        # Test with invalid config that causes provider init to fail
        invalid_config = {'invalid': 'config'}
        with pytest.raises(ProviderError) as exc_info:
            provider_factory.create_provider('claude', invalid_config)
        assert "Failed to create claude provider" in str(exc_info.value)
    
    def test_get_provider_cached(self, provider_factory):
        # First call creates and caches
        provider1 = provider_factory.get_provider('claude')
        assert isinstance(provider1, ClaudeProvider)
        
        # Second call returns cached instance
        provider2 = provider_factory.get_provider('claude')
        assert provider1 is provider2
    
    def test_get_provider_disabled(self, factory_config, http_client):
        config = factory_config.copy()
        config['providers']['claude']['enabled'] = False
        factory = ProviderFactory(config, http_client)
        
        with pytest.raises(ProviderError) as exc_info:
            factory.get_provider('claude')
        assert "Provider claude is disabled" in str(exc_info.value)
    
    def test_get_provider_for_model_mapped(self, provider_factory):
        provider = provider_factory.get_provider_for_model('gpt-4o')
        assert isinstance(provider, OpenAIProvider)
    
    def test_get_provider_for_model_claude_auto_detect(self, provider_factory):
        provider = provider_factory.get_provider_for_model('claude-3-5-sonnet')
        assert isinstance(provider, ClaudeProvider)
    
    def test_get_provider_for_model_gpt_auto_detect(self, provider_factory):
        provider = provider_factory.get_provider_for_model('gpt-4')
        assert isinstance(provider, AzureOpenAIProvider)  # Should use azure from mapping
    
    def test_get_provider_for_model_no_provider(self, provider_factory):
        with pytest.raises(ProviderError) as exc_info:
            provider_factory.get_provider_for_model('unknown-model')
        assert "No provider configured for model: unknown-model" in str(exc_info.value)
    
    def test_auto_detect_provider(self, provider_factory):
        assert provider_factory._auto_detect_provider('claude-3-sonnet') == 'claude'
        assert provider_factory._auto_detect_provider('gpt-4o') == 'openai'
        assert provider_factory._auto_detect_provider('o1-preview') == 'openai'
        assert provider_factory._auto_detect_provider('text-embedding-ada-002') == 'openai'
        assert provider_factory._auto_detect_provider('unknown-model') is None
    
    def test_list_available_providers(self, factory_config, http_client):
        factory = ProviderFactory(factory_config, http_client)
        providers = factory.list_available_providers()
        expected = {'claude': True, 'openai': True, 'azure': True}
        assert providers == expected
        
        # Test with disabled provider
        config = factory_config.copy()
        config['providers']['claude']['enabled'] = False
        factory = ProviderFactory(config, http_client)
        providers = factory.list_available_providers()
        assert providers['claude'] is False
    
    def test_get_supported_models(self, provider_factory):
        models = provider_factory.get_supported_models()
        expected = {
            'gpt-4o': 'openai',
            'claude-3-5-sonnet': 'claude',
            'gpt-4': 'azure',
            'gpt-4o': 'openai',  # auto-detected
            'gpt-4': 'openai',   # auto-detected (but overridden by mapping)
            'gpt-3.5-turbo': 'openai',
            'claude-3-5-sonnet': 'claude',
            'claude-3-haiku': 'claude'
        }
        
        # Check that mapped models take precedence
        assert models['gpt-4'] == 'azure'  # Should use azure from mapping, not openai from auto-detect
    
    def test_register_provider(self, provider_factory):
        class MockProvider(BaseProvider):
            async def list_models(self):
                return []
            async def chat_completion(self, request):
                pass
            async def chat_completion_stream(self, request):
                pass
            async def create_embeddings(self, request):
                pass
            def is_model_supported(self, model):
                return True
            async def health_check(self):
                return True
        
        provider_factory.register_provider('mock', MockProvider)
        
        mock_config = {'api_key': 'test-key'}
        provider = provider_factory.create_provider('mock', mock_config)
        assert isinstance(provider, MockProvider)
        
        # Clean up - remove mock provider
        del provider_factory._providers['mock']
    
    def test_register_invalid_provider(self, provider_factory):
        with pytest.raises(ProviderError) as exc_info:
            provider_factory.register_provider('invalid', str)
        assert "Provider class must inherit from BaseProvider" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, provider_factory):
        with patch.object(ClaudeProvider, 'health_check', new_callable=AsyncMock) as mock_claude_health:
            with patch.object(OpenAIProvider, 'health_check', new_callable=AsyncMock) as mock_openai_health:
                with patch.object(AzureOpenAIProvider, 'health_check', new_callable=AsyncMock) as mock_azure_health:
                    mock_claude_health.return_value = True
                    mock_openai_health.return_value = False
                    mock_azure_health.return_value = True
                    
                    results = await provider_factory.health_check_all()
                    
                    assert results['claude'] is True
                    assert results['openai'] is False
                    assert results['azure'] is True
    
    @pytest.mark.asyncio
    async def test_close(self, provider_factory):
        # Create some providers
        provider_factory.get_provider('claude')
        provider_factory.get_provider('openai')
        
        assert len(provider_factory._provider_instances) == 2
        
        await provider_factory.close()
        
        assert len(provider_factory._provider_instances) == 0
        # Note: We can't easily test http_client.aclose() without more complex mocking
    
    @pytest.mark.asyncio
    async def test_context_manager(self, factory_config):
        async with ProviderFactory(factory_config) as factory:
            assert isinstance(factory, ProviderFactory)
            assert isinstance(factory.http_client, httpx.AsyncClient)
        
        # After exiting context, instances should be cleared
        # (though we can't easily test http_client closure without more complex setup)


class TestProviderFactoryConfigVariations:
    
    def test_empty_config(self):
        # Ensure we use fresh factory without mock provider
        original_providers = ProviderFactory._providers.copy()
        try:
            ProviderFactory._providers = {
                'claude': ClaudeProvider,
                'openai': OpenAIProvider,
                'azure': AzureOpenAIProvider,
            }
            factory = ProviderFactory({})
            providers = factory.list_available_providers()
            expected = {'claude': False, 'openai': False, 'azure': False}
            assert providers == expected
        finally:
            ProviderFactory._providers = original_providers
    
    def test_missing_providers_section(self):
        original_providers = ProviderFactory._providers.copy()
        try:
            ProviderFactory._providers = {
                'claude': ClaudeProvider,
                'openai': OpenAIProvider,
                'azure': AzureOpenAIProvider,
            }
            factory = ProviderFactory({'model_mapping': {'test': 'claude'}})
            providers = factory.list_available_providers()
            expected = {'claude': False, 'openai': False, 'azure': False}
            assert providers == expected
        finally:
            ProviderFactory._providers = original_providers
    
    def test_all_providers_disabled(self):
        original_providers = ProviderFactory._providers.copy()
        try:
            ProviderFactory._providers = {
                'claude': ClaudeProvider,
                'openai': OpenAIProvider,
                'azure': AzureOpenAIProvider,
            }
            config = {
                'providers': {
                    'claude': {'enabled': False},
                    'openai': {'enabled': False},
                    'azure': {'enabled': False}
                }
            }
            factory = ProviderFactory(config)
            providers = factory.list_available_providers()
            expected = {'claude': False, 'openai': False, 'azure': False}
            assert providers == expected
        finally:
            ProviderFactory._providers = original_providers