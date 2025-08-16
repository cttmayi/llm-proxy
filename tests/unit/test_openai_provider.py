"""
Unit tests for OpenAI provider.
"""
import pytest
import httpx
import respx
import json
from src.providers.openai import OpenAIProvider
from src.providers.base import ChatRequest, ChatMessage, EmbeddingRequest, ProviderError


@pytest.fixture
def openai_config():
    return {
        'api_key': 'test-key',
        'base_url': 'https://api.openai.com',
        'organization': 'test-org'
    }


@pytest.fixture
def http_client():
    return httpx.AsyncClient()


@pytest.fixture
def openai_provider(openai_config, http_client):
    return OpenAIProvider(openai_config, http_client)


class TestOpenAIProvider:
    
    def test_init_with_valid_config(self, openai_config, http_client):
        provider = OpenAIProvider(openai_config, http_client)
        assert provider.api_key == 'test-key'
        assert provider.base_url == 'https://api.openai.com'
        assert provider.organization == 'test-org'
    
    def test_init_without_api_key(self, http_client):
        config = {'base_url': 'https://api.openai.com'}
        with pytest.raises(ProviderError) as exc_info:
            OpenAIProvider(config, http_client)
        assert "OpenAI API key is required" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
    
    def test_is_model_supported(self, openai_provider):
        assert openai_provider.is_model_supported("gpt-4o") is True
        assert openai_provider.is_model_supported("gpt-3.5-turbo") is True
        assert openai_provider.is_model_supported("text-embedding-ada-002") is True
        assert openai_provider.is_model_supported("invalid-model") is False
        assert openai_provider.is_model_supported("claude-3-5-sonnet") is False
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_success(self, openai_provider):
        mock_response = {
            'object': 'list',
            'data': [
                {
                    'id': 'gpt-4o',
                    'object': 'model',
                    'created': 1715367049,
                    'owned_by': 'openai'
                },
                {
                    'id': 'gpt-3.5-turbo',
                    'object': 'model',
                    'created': 1677610602,
                    'owned_by': 'openai'
                }
            ]
        }
        
        route = respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        
        models = await openai_provider.list_models()
        
        assert len(models) == 2
        assert models[0].id == "gpt-4o"
        assert models[0].provider == "openai"
        assert models[0].owned_by == "openai"
        assert route.called
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_api_error(self, openai_provider):
        route = respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(500, json={"error": "Server error"})
        )
        
        with pytest.raises(ProviderError) as exc_info:
            await openai_provider.list_models()
        
        assert "OpenAI API error" in str(exc_info.value)
        assert exc_info.value.status_code == 500
        assert exc_info.value.provider == "openai"
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, openai_provider):
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
        
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        
        request = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        
        response = await openai_provider.chat_completion(request)
        
        assert response.id == "chatcmpl-123"
        assert response.model == "gpt-4o"
        assert response.choices[0]['message']['content'] == "Hello! How can I help you?"
        assert response.usage['prompt_tokens'] == 10
        assert route.called
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_completion_model_not_supported(self, openai_provider):
        request = ChatRequest(
            model="claude-3-5-sonnet",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        
        with pytest.raises(ProviderError) as exc_info:
            await openai_provider.chat_completion(request)
        
        assert "Model claude-3-5-sonnet not supported by OpenAI" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_completion_with_all_params(self, openai_provider):
        mock_response = {
            'id': 'chatcmpl-123',
            'object': 'chat.completion',
            'created': 1700000000,
            'model': 'gpt-4o',
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'Response'},
                'finish_reason': 'stop'
            }],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        }
        
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        
        request = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        response = await openai_provider.chat_completion(request)
        
        assert response.model == "gpt-4o"
        assert route.called
        
        # Verify the request payload
        request_json = route.calls[0].request.content.decode()
        request_data = json.loads(request_json)
        assert request_data['max_tokens'] == 100
        assert request_data['temperature'] == 0.7
        assert request_data['top_p'] == 0.9
        assert request_data['frequency_penalty'] == 0.1
        assert request_data['presence_penalty'] == 0.1
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_create_embeddings_success(self, openai_provider):
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
        
        route = respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        
        request = EmbeddingRequest(
            model="text-embedding-ada-002",
            input="Hello world"
        )
        
        response = await openai_provider.create_embeddings(request)
        
        assert response.model == "text-embedding-ada-002"
        assert len(response.data) == 1
        assert response.data[0]['embedding'] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert route.called
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_create_embeddings_with_list_input(self, openai_provider):
        mock_response = {
            'object': 'list',
            'data': [
                {
                    'object': 'embedding',
                    'embedding': [0.1, 0.2, 0.3],
                    'index': 0
                },
                {
                    'object': 'embedding',
                    'embedding': [0.4, 0.5, 0.6],
                    'index': 1
                }
            ],
            'model': 'text-embedding-ada-002',
            'usage': {'prompt_tokens': 10, 'total_tokens': 10}
        }
        
        route = respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        
        request = EmbeddingRequest(
            model="text-embedding-ada-002",
            input=["Hello", "World"]
        )
        
        response = await openai_provider.create_embeddings(request)
        
        assert len(response.data) == 2
        assert response.data[0]['index'] == 0
        assert response.data[1]['index'] == 1
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_health_check_success(self, openai_provider):
        route = respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(200, json={'object': 'list', 'data': []})
        )
        
        result = await openai_provider.health_check()
        assert result is True
        assert route.called
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_health_check_failure(self, openai_provider):
        route = respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(500, json={"error": "Server error"})
        )
        
        result = await openai_provider.health_check()
        assert result is False
        assert route.called
    
    def test_get_headers_with_organization(self, openai_provider):
        headers = openai_provider.get_headers()
        assert 'Authorization' in headers
        assert headers['Authorization'] == 'Bearer test-key'
        assert 'OpenAI-Organization' in headers
        assert headers['OpenAI-Organization'] == 'test-org'
    
    def test_get_headers_without_organization(self, http_client):
        config = {'api_key': 'test-key'}
        provider = OpenAIProvider(config, http_client)
        headers = provider.get_headers()
        assert 'Authorization' in headers
        assert headers['Authorization'] == 'Bearer test-key'
        assert 'OpenAI-Organization' not in headers
    
    def test_convert_messages_format_no_conversion(self, openai_provider):
        messages = [
            {'role': 'system', 'content': 'You are helpful'},
            {'role': 'user', 'content': 'Hello'}
        ]
        
        result = openai_provider._convert_messages_format(messages)
        assert result == messages