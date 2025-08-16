"""
Unit tests for Claude provider.
"""
import pytest
import httpx
import respx
from unittest.mock import AsyncMock, patch
from src.providers.claude import ClaudeProvider
from src.providers.base import ChatRequest, ChatMessage, ProviderError, EmbeddingRequest


@pytest.fixture
def claude_config():
    return {
        'api_key': 'test-key',
        'base_url': 'https://api.anthropic.com',
        'api_version': '2023-06-01'
    }


@pytest.fixture
def http_client():
    return httpx.AsyncClient()


@pytest.fixture
def claude_provider(claude_config, http_client):
    return ClaudeProvider(claude_config, http_client)


class TestClaudeProvider:
    
    def test_init_with_valid_config(self, claude_config, http_client):
        provider = ClaudeProvider(claude_config, http_client)
        assert provider.api_key == 'test-key'
        assert provider.base_url == 'https://api.anthropic.com'
        assert provider.api_version == '2023-06-01'
    
    def test_init_without_api_key(self, http_client):
        config = {'base_url': 'https://api.anthropic.com'}
        with pytest.raises(ProviderError) as exc_info:
            ClaudeProvider(config, http_client)
        assert "Claude API key is required" in str(exc_info.value)
        assert exc_info.value.provider == "claude"
    
    def test_is_model_supported(self, claude_provider):
        assert claude_provider.is_model_supported("claude-3-5-sonnet-20241022") is True
        assert claude_provider.is_model_supported("claude-3-5-haiku-20241022") is True
        assert claude_provider.is_model_supported("gpt-4") is False
        assert claude_provider.is_model_supported("invalid-model") is False
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models(self, claude_provider):
        models = await claude_provider.list_models()
        assert len(models) == 3
        assert models[0].provider == "claude"
        assert models[0].owned_by == "anthropic"
        assert "claude-3-5-sonnet" in models[0].id
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, claude_provider):
        mock_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! How can I help you?"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 25}
        }
        
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        
        request = ChatRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        
        response = await claude_provider.chat_completion(request)
        
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.choices[0]['message']['content'] == "Hello! How can I help you?"
        assert response.usage['prompt_tokens'] == 10
        assert response.usage['completion_tokens'] == 25
        assert route.called
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_completion_model_not_supported(self, claude_provider):
        request = ChatRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        
        with pytest.raises(ProviderError) as exc_info:
            await claude_provider.chat_completion(request)
        
        assert "Model gpt-4 not supported by Claude" in str(exc_info.value)
        assert exc_info.value.provider == "claude"
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_completion_api_error(self, claude_provider):
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(400, json={"error": "Invalid request"})
        )
        
        request = ChatRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        
        with pytest.raises(ProviderError) as exc_info:
            await claude_provider.chat_completion(request)
        
        assert "Claude API error" in str(exc_info.value)
        assert exc_info.value.status_code == 400
        assert exc_info.value.provider == "claude"
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_completion_stream_success(self, claude_provider):
        async def mock_stream():
            yield "data: {\"type\": \"content_block_delta\", \"delta\": {\"text\": \"Hello\"}}"
            yield "data: {\"type\": \"content_block_delta\", \"delta\": {\"text\": \" there\"}}"
            yield "data: {\"type\": \"message_stop\"}"
        
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=mock_stream())
        )
        
        request = ChatRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True
        )
        
        chunks = []
        async for chunk in claude_provider.chat_completion_stream(request):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        assert route.called
    
    @pytest.mark.asyncio
    async def test_create_embeddings_not_supported(self, claude_provider):
        request = EmbeddingRequest(model="claude-3-5-sonnet-20241022", input="Hello")
        
        with pytest.raises(ProviderError) as exc_info:
            await claude_provider.create_embeddings(request)
        
        assert "Claude does not support embeddings" in str(exc_info.value)
        assert exc_info.value.status_code == 501
        assert exc_info.value.provider == "claude"
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_health_check_success(self, claude_provider):
        route = respx.get("https://api.anthropic.com/v1/models").mock(
            return_value=httpx.Response(200)
        )
        
        result = await claude_provider.health_check()
        assert result is True
        assert route.called
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_health_check_failure(self, claude_provider):
        route = respx.get("https://api.anthropic.com/v1/models").mock(
            return_value=httpx.Response(500)
        )
        
        result = await claude_provider.health_check()
        assert result is False
        assert route.called
    
    def test_get_headers(self, claude_provider):
        headers = claude_provider.get_headers()
        assert 'x-api-key' in headers
        assert 'anthropic-version' in headers
        assert headers['x-api-key'] == 'test-key'
        assert headers['anthropic-version'] == '2023-06-01'
    
    def test_convert_messages_format(self, claude_provider):
        messages = [
            {'role': 'system', 'content': 'You are helpful'},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'},
            {'role': 'user', 'content': 'How are you?'}
        ]
        
        claude_messages, system_message = claude_provider._convert_messages_format(messages)
        
        assert system_message == 'You are helpful'
        assert len(claude_messages) == 3
        assert claude_messages[0]['role'] == 'user'
        assert claude_messages[0]['content'] == 'Hello'
        assert claude_messages[1]['role'] == 'assistant'
        assert claude_messages[1]['content'] == 'Hi there'
        assert claude_messages[2]['role'] == 'user'
        assert claude_messages[2]['content'] == 'How are you?'