"""
Mock provider for testing purposes.
Returns immediate responses without making real API calls.
"""
import time
from typing import List, Dict, Any, AsyncGenerator
from src.providers.base import BaseProvider, ModelInfo, ChatResponse, ChatRequest, EmbeddingRequest, EmbeddingResponse


class MockProvider(BaseProvider):
    """Mock provider that returns immediate responses for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mock provider with config."""
        self.config = config
        self.api_key = config.get('api_key', 'mock-key')
        self.base_url = config.get('base_url', 'https://mock.example.com')
    
    async def list_models(self) -> List[ModelInfo]:
        """Return mock models."""
        return [
            ModelInfo(
                id="gpt-4o",
                created=int(time.time()),
                owned_by="openai"
            ),
            ModelInfo(
                id="gpt-4o-mini",
                created=int(time.time()),
                owned_by="openai"
            ),
            ModelInfo(
                id="gpt-3.5-turbo",
                created=int(time.time()),
                owned_by="openai"
            )
        ]
    
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Return mock chat completion response."""
        return ChatResponse(
            id="mock-chatcmpl-123",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Mock response for: {request.messages[-1].get('content', 'Hello')}"
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        )
    
    async def chat_completion_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Return mock streaming chat completion response."""
        chunks = [
            'data: {"id":"mock-chatcmpl-123","object":"chat.completion.chunk","created":' + str(int(time.time())) + ',"model":"' + request.model + '","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
            'data: {"id":"mock-chatcmpl-123","object":"chat.completion.chunk","created":' + str(int(time.time())) + ',"model":"' + request.model + '","choices":[{"index":0,"delta":{"content":"Mock"},"finish_reason":null}]}\n\n',
            'data: {"id":"mock-chatcmpl-123","object":"chat.completion.chunk","created":' + str(int(time.time())) + ',"model":"' + request.model + '","choices":[{"index":0,"delta":{"content":" response"},"finish_reason":null}]}\n\n',
            'data: {"id":"mock-chatcmpl-123","object":"chat.completion.chunk","created":' + str(int(time.time())) + ',"model":"' + request.model + '","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n'
        ]
        
        for chunk in chunks:
            yield chunk
    
    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Return mock embeddings response."""
        return EmbeddingResponse(
            object="list",
            data=[{
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "index": 0
            }],
            model=request.model,
            usage={
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        )
    
    def is_model_supported(self, model: str) -> bool:
        """Check if model is supported by this mock provider."""
        supported_models = {"gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"}
        return model in supported_models
    
    async def health_check(self) -> bool:
        """Always return healthy for mock provider."""
        return True