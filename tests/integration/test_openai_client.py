"""
Integration tests using the OpenAI Python library to test the proxy.
"""
import pytest

from openai import OpenAI, AsyncOpenAI
from typing import List, Dict, Any

import os


base_url = "http://127.0.0.1:8000/v1"
api_key = "test-key"

# base_url = os.getenv("OPENAI_BASE_URL")
# api_key = os.getenv("OPENAI_API_KEY")

class TestOpenAIClientIntegration:
    """Test proxy functionality using the official OpenAI Python client."""
    
    @pytest.fixture
    def openai_client(self):
        """Create an OpenAI client configured to use the proxy."""
        return OpenAI(
            base_url= base_url,
            api_key=api_key
        )
    
    @pytest.fixture
    def async_openai_client(self):
        """Create an async OpenAI client."""
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
    
    def test_chat_completions_sync(self, openai_client):
        """Test synchronous chat completions with OpenAI client."""
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello, this is a test message."}
            ],
            max_tokens=50
        )
        
        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message is not None
        assert response.choices[0].message.content is not None
    
    def test_chat_completions_streaming_sync(self, openai_client):
        """Test streaming chat completions with OpenAI client."""
        stream = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a short story."}
            ],
            max_tokens=50,
            stream=True
        )
        
        chunks = list(stream)
        assert len(chunks) > 0
        
        # Check that we have at least one chunk with content
        content_chunks = [chunk for chunk in chunks if chunk.choices[0].delta.content]
        assert len(content_chunks) > 0
    
    def test_list_models_sync(self, openai_client):
        """Test listing models with OpenAI client."""
        models = openai_client.models.list()
        
        assert models is not None
        assert len(models.data) > 0
        
        # Check that we have expected model types
        model_ids = [model.id for model in models.data]
        assert any("gpt" in model_id.lower() for model_id in model_ids)
    
    def test_embeddings_sync(self, openai_client):
        """Test embeddings with OpenAI client."""
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input="This is a test sentence for embeddings."
        )
        
        assert response is not None
        assert response.data is not None
        assert len(response.data) > 0
        assert response.data[0].embedding is not None
        assert isinstance(response.data[0].embedding, list)
        assert len(response.data[0].embedding) > 0
    
    @pytest.mark.asyncio
    async def test_chat_completions_async(self, async_openai_client):
        """Test asynchronous chat completions with OpenAI client."""
        response = await async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello, this is an async test."}
            ],
            max_tokens=50
        )
        
        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message is not None
        assert response.choices[0].message.content is not None
    
    @pytest.mark.asyncio
    async def test_chat_completions_streaming_async(self, async_openai_client):
        """Test async streaming chat completions with OpenAI client."""
        stream = await async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a short story."}
            ],
            max_tokens=50,
            stream=True
        )
        
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        assert len(chunks) > 0
        
        # Check that we have at least one chunk with content
        content_chunks = [chunk for chunk in chunks if chunk.choices[0].delta.content]
        assert len(content_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_list_models_async(self, async_openai_client):
        """Test async listing models with OpenAI client."""
        models = await async_openai_client.models.list()
        
        assert models is not None
        assert len(models.data) > 0
        
        # Check that we have expected model types
        model_ids = [model.id for model in models.data]
        assert any("gpt" in model_id.lower() for model_id in model_ids)
    
    def test_error_handling_sync(self, openai_client):
        """Test error handling with invalid model."""
        with pytest.raises(Exception) as exc_info:
            openai_client.chat.completions.create(
                model="invalid-model-name",
                messages=[
                    {"role": "user", "content": "This should fail."}
                ]
            )
        
        # Should raise an appropriate error
        assert exc_info.value is not None
    
    def test_claude_model_via_openai_client(self, openai_client):
        """Test accessing Claude models through OpenAI client format."""
        response = openai_client.chat.completions.create(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {"role": "user", "content": "Hello from Claude via OpenAI client!"}
            ],
            max_tokens=50
        )
        
        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message is not None
        assert response.choices[0].message.content is not None
    
    def test_message_roles_variations(self, openai_client):
        """Test various message role combinations."""
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "Now what is 3+3?"}
            ],
            max_tokens=50
        )
        
        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
    
    def test_parameters_handling(self, openai_client):
        """Test various OpenAI API parameters."""
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Generate a creative response"}
            ],
            max_tokens=100,
            temperature=0.8,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0