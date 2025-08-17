"""
Integration tests using the Anthropic Python library to test the proxy.
"""
import pytest
import httpx
from anthropic import Anthropic, AsyncAnthropic

base_url = "http://127.0.0.1:8000"
api_key = "test-key"

class TestAnthropicClientIntegration:
    """Test proxy functionality using the official Anthropic Python client."""
    
    @pytest.fixture
    def anthropic_client(self):
        """Create an Anthropic client configured to use the proxy."""
        return Anthropic(
            base_url=base_url,
            api_key=api_key,
        )
    
    @pytest.fixture
    def async_anthropic_client(self):
        """Create an async Anthropic client."""
        return AsyncAnthropic(
            base_url=base_url,
            api_key=api_key,
        )
    
    def test_claude_messages_sync(self, anthropic_client):
        """Test synchronous Claude messages with Anthropic client."""
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Hello, Claude!"}
            ]
        )
        
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert response.content[0].text is not None
    
    def test_claude_messages_with_system_prompt(self, anthropic_client):
        """Test Claude messages with system prompt."""
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            system="You are a helpful assistant specialized in mathematics.",
            messages=[
                {"role": "user", "content": "What is 2+2?"}
            ]
        )
        
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert response.content[0].text is not None
    
    def test_claude_messages_with_temperature(self, anthropic_client):
        """Test Claude messages with temperature parameter."""
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            temperature=0.7,
            messages=[
                {"role": "user", "content": "Tell me a creative story about AI"}
            ]
        )
        
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
    
    def test_claude_messages_multi_turn(self, anthropic_client):
        """Test multi-turn conversation with Claude."""
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "What is its population?"}
            ]
        )
        
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert response.content[0].text is not None
    
    @pytest.mark.asyncio
    async def test_claude_messages_async(self, async_anthropic_client):
        """Test asynchronous Claude messages with Anthropic client."""
        response = await async_anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Hello, async Claude!"}
            ]
        )
        
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert response.content[0].text is not None
    
    @pytest.mark.asyncio
    async def test_claude_messages_streaming_async(self, async_anthropic_client):
        """Test async streaming Claude messages."""
        stream = await async_anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Tell me a short story."}
            ],
            stream=True
        )
        
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        assert len(chunks) > 0
        
        # Check that we have content chunks
        content_chunks = [chunk for chunk in chunks if chunk.delta and chunk.delta.text]
        assert len(content_chunks) > 0
    
    def test_claude_haiku_model(self, anthropic_client):
        """Test Claude Haiku model."""
        response = anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "What is AI?"}
            ]
        )
        
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert response.content[0].text is not None
    
    def test_claude_opus_model(self, anthropic_client):
        """Test Claude Opus model."""
        response = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Explain quantum computing"}
            ]
        )
        
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert response.content[0].text is not None
    
    def test_claude_parameters_handling(self, anthropic_client):
        """Test various Claude API parameters."""
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=150,
            temperature=0.8,
            top_p=0.9,
            messages=[
                {"role": "user", "content": "Generate a creative haiku about technology"}
            ]
        )
        
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
    
    def test_error_handling_invalid_model(self, anthropic_client):
        """Test error handling with invalid model."""
        with pytest.raises(Exception) as exc_info:
            anthropic_client.messages.create(
                model="invalid-claude-model",
                max_tokens=50,
                messages=[
                    {"role": "user", "content": "This should fail"}
                ]
            )
        
        # Should raise an appropriate error
        assert exc_info.value is not None