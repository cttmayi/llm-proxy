"""
Unit tests for Azure OpenAI provider.
"""
import pytest
import httpx
import respx
from src.providers.azure import AzureOpenAIProvider
from src.providers.base import ChatRequest, ChatMessage, EmbeddingRequest, ProviderError


@pytest.fixture
def azure_config():
    return {
        'api_key': 'test-key',
        'endpoint': 'https://test-resource.openai.azure.com',
        'api_version': '2024-10-21'
    }


@pytest.fixture
def http_client():
    return httpx.AsyncClient()


@pytest.fixture
def azure_provider(azure_config, http_client):
    return AzureOpenAIProvider(azure_config, http_client)


class TestAzureOpenAIProvider:
    
    def test_init_with_valid_config(self, azure_config, http_client):
        provider = AzureOpenAIProvider(azure_config, http_client)
        assert provider.api_key == 'test-key'
        assert provider.endpoint == 'https://test-resource.openai.azure.com'
        assert provider.api_version == '2024-10-21'
    
    def test_init_without_api_key(self, http_client):
        config = {'endpoint': 'https://test-resource.openai.azure.com'}
        with pytest.raises(ProviderError) as exc_info:
            AzureOpenAIProvider(config, http_client)
        assert "Azure OpenAI API key is required" in str(exc_info.value)
        assert exc_info.value.provider == "azure"
    
    def test_init_without_endpoint(self, http_client):
        config = {'api_key': 'test-key'}
        with pytest.raises(ProviderError) as exc_info:
            AzureOpenAIProvider(config, http_client)
        assert "Azure OpenAI endpoint is required" in str(exc_info.value)
        assert exc_info.value.provider == "azure"
    
    def test_is_model_supported(self, azure_provider):
        assert azure_provider.is_model_supported("gpt-4o") is True
        assert azure_provider.is_model_supported("gpt-4") is True
        assert azure_provider.is_model_supported("gpt-35-turbo") is True
        assert azure_provider.is_model_supported("text-embedding-ada-002") is True
        assert azure_provider.is_model_supported("invalid-model") is False
        assert azure_provider.is_model_supported("claude-3-5-sonnet") is False
    
    def test_get_deployment_url(self, azure_provider):
        base_url = "https://test-resource.openai.azure.com"
        
        chat_url = azure_provider._get_deployment_url("gpt-4o", "chat")
        expected_chat = f"{base_url}/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        assert chat_url == expected_chat
        
        embeddings_url = azure_provider._get_deployment_url("text-embedding-ada-002", "embeddings")
        expected_embeddings = f"{base_url}/openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-10-21"
        assert embeddings_url == expected_embeddings
        
        models_url = azure_provider._get_deployment_url("", "models")
        expected_models = f"{base_url}/openai/models?api-version=2024-10-21"
        assert models_url == expected_models
    
    def test_endpoint_normalization(self, http_client):
        config = {
            'api_key': 'test-key',
            'endpoint': 'https://test-resource.openai.azure.com/',
            'api_version': '2024-10-21'
        }
        provider = AzureOpenAIProvider(config, http_client)
        assert provider.endpoint == 'https://test-resource.openai.azure.com'
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_success(self, azure_provider):
        mock_response = {
            'data': [
                {
                    'id': 'gpt-4o',
                    'object': 'model',
                    'created': 1715367049,
                    'owned_by': 'azure-openai'
                }
            ]
        }
        
        route = respx.get("https://test-resource.openai.azure.com/openai/models?api-version=2024-10-21").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        
        models = await azure_provider.list_models()
        
        assert len(models) == 1
        assert models[0].id == "gpt-4o"
        assert models[0].provider == "azure"
        assert route.called
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_api_error_fallback(self, azure_provider):
        route = respx.get("https://test-resource.openai.azure.com/openai/models?api-version=2024-10-21").mock(
            return_value=httpx.Response(500, json={"error": "Server error"})
        )
        
        models = await azure_provider.list_models()
        
        # Should return fallback models
        assert len(models) == 3
        assert models[0].provider == "azure"
        assert route.called
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, azure_provider):
        mock_response = {
            'id': 'chatcmpl-123',
            'object': 'chat.completion',
            'created': 1700000000,
            'model': 'gpt-4o',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'Hello! How can I help you?'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 25,
                'total_tokens': 35
            }
        }
        
        route = respx.post("https://test-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        
        request = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        
        response = await azure_provider.chat_completion(request)
        
        assert response.id == "chatcmpl-123"
        assert response.model == "gpt-4o"
        assert response.choices[0]['message']['content'] == "Hello! How can I help you?"
        assert route.called
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_completion_model_not_supported(self, azure_provider):
        request = ChatRequest(
            model="claude-3-5-sonnet",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        
        with pytest.raises(ProviderError) as exc_info:
            await azure_provider.chat_completion(request)
        
        assert "Model claude-3-5-sonnet not supported by Azure" in str(exc_info.value)
        assert exc_info.value.provider == "azure"
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_create_embeddings_success(self, azure_provider):
        mock_response = {
            'object': 'list',
            'data': [{
                'object': 'embedding',
                'embedding': [0.1, 0.2, 0.3, 0.4, 0.5],
                'index': 0
            }],
            'model': 'text-embedding-ada-002',
            'usage': {
                'prompt_tokens': 5,
                'total_tokens': 5
            }
        }
        
        route = respx.post("https://test-resource.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-10-21").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        
        request = EmbeddingRequest(
            model="text-embedding-ada-002",
            input="Hello world"
        )
        
        response = await azure_provider.create_embeddings(request)
        
        assert response.model == "text-embedding-ada-002"
        assert len(response.data) == 1
        assert response.data[0]['embedding'] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert route.called
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_health_check_success(self, azure_provider):
        route = respx.get("https://test-resource.openai.azure.com/openai/models?api-version=2024-10-21").mock(
            return_value=httpx.Response(200, json={'data': []})
        )
        
        result = await azure_provider.health_check()
        assert result is True
        assert route.called
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_health_check_failure(self, azure_provider):
        route = respx.get("https://test-resource.openai.azure.com/openai/models?api-version=2024-10-21").mock(
            return_value=httpx.Response(500, json={"error": "Server error"})
        )
        
        result = await azure_provider.health_check()
        assert result is False
        assert route.called
    
    def test_get_headers(self, azure_provider):
        headers = azure_provider.get_headers()
        assert 'api-key' in headers
        assert headers['api-key'] == 'test-key'
        assert 'Content-Type' in headers
        assert headers['Content-Type'] == 'application/json'
    
    def test_convert_messages_format_no_conversion(self, azure_provider):
        messages = [
            {'role': 'system', 'content': 'You are helpful'},
            {'role': 'user', 'content': 'Hello'}
        ]
        
        result = azure_provider._convert_messages_format(messages)
        assert result == messages