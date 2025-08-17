"""
Base provider interface and abstract classes for LLM API providers.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator, Optional, List
from pydantic import BaseModel
import httpx


class ModelInfo(BaseModel):
    """Model information structure."""
    id: str
    object: str = "model"
    created: int
    owned_by: str
    provider: str


class ChatMessage(BaseModel):
    """Chat message structure."""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat completion request."""
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class ChatResponse(BaseModel):
    """Chat completion response."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]


class ChatStreamResponse(BaseModel):
    """Chat completion streaming response chunk."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]


class EmbeddingRequest(BaseModel):
    """Embedding request."""
    model: str
    input: str | List[str]


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    object: str
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, Any]


class ProviderError(Exception):
    """Base exception for provider errors."""
    def __init__(self, message: str, status_code: int = 500, provider: str = None):
        self.message = message
        self.status_code = status_code
        self.provider = provider
        super().__init__(self.message)


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any], http_client: httpx.AsyncClient):
        self.config = config
        self.http_client = http_client
        self.name = self.__class__.__name__.lower().replace('provider', '')
    
    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """List available models from this provider."""
        pass
    
    @abstractmethod
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate chat completion."""
        pass
    
    @abstractmethod
    async def chat_completion_stream(self, request: ChatRequest) -> AsyncGenerator[ChatStreamResponse, None]:
        """Generate streaming chat completion."""
        pass
    
    @abstractmethod
    async def chat_completion_stream_fast(self, request: ChatRequest):
        """Generate streaming chat completion (fast version)."""
        pass

    @abstractmethod
    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Create embeddings for text."""
        pass
    
    @abstractmethod
    def is_model_supported(self, model: str) -> bool:
        """Check if a model is supported by this provider."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider health."""
        pass
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API calls."""
        return {
            'Content-Type': 'application/json',
            'User-Agent': 'LLM-Proxy/1.0'
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Clean up resources."""
        pass