"""
OpenAI provider implementation for OpenAI API.
"""
import json
import time
import uuid
from typing import Dict, Any, AsyncGenerator, List
import httpx

from .base import BaseProvider, ModelInfo, ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, ProviderError
from ..utils.cache import async_cache


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: Dict[str, Any], http_client: httpx.AsyncClient):
        super().__init__(config, http_client)
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url', 'https://api.openai.com')
        self.organization = config.get('organization')
        
        if not self.api_key:
            raise ProviderError("OpenAI API key is required", provider="openai")
    
    def _build_url(self, endpoint: str) -> str:
        """Build proper URL by handling both cases where base_url ends with /v1 or not."""
        base_url = self.base_url.rstrip('/')
        endpoint = endpoint.lstrip('/')
        
        # If base_url already ends with /v1, don't add another /v1
        if base_url.endswith('/v1'):
            return f"{base_url}/{endpoint}"
        else:
            return f"{base_url}/v1/{endpoint}"
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for OpenAI API calls."""
        headers = super().get_headers()
        headers.update({
            'Authorization': f'Bearer {self.api_key}',
        })
        
        if self.organization:
            headers['OpenAI-Organization'] = self.organization
        
        return headers
    
    @async_cache(ttl=300)
    async def list_models(self) -> List[ModelInfo]:
        """List available models from OpenAI."""
        try:
            url = self._build_url('models')
            response = await self.http_client.get(
                url,
                headers=self.get_headers(),
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get('data', []):
                models.append(ModelInfo(
                    id=model_data['id'],
                    created=model_data.get('created', int(time.time())),
                    owned_by=model_data.get('owned_by', 'openai'),
                    provider="openai"
                ))
            
            return models
            
        except httpx.HTTPError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=e.response.status_code if hasattr(e, 'response') else 500, provider="openai")
    
    def is_model_supported(self, model: str) -> bool:
        """Check if model is supported by OpenAI."""
        openai_models = {
            # GPT-4 models
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
            "gpt-4-turbo-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            
            # GPT-3.5 models
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125",
            
            # Embedding models
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
            
            # O1 models
            "o1-preview",
            "o1-preview-2024-09-12",
            "o1-mini",
            "o1-mini-2024-09-12",
        }
        return model in openai_models
    
    def _convert_messages_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert messages to OpenAI format (no conversion needed)."""
        return messages
    
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate chat completion using OpenAI API."""
        if not self.is_model_supported(request.model):
            raise ProviderError(f"Model {request.model} not supported by OpenAI", provider="openai")
        
        messages = self._convert_messages_format([
            {'role': msg.role, 'content': msg.content} 
            for msg in request.messages
        ])
        
        payload = {
            'model': request.model,
            'messages': messages,
        }
        
        if request.max_tokens:
            payload['max_tokens'] = request.max_tokens
        
        if request.temperature is not None:
            payload['temperature'] = max(0.0, min(2.0, request.temperature))
        
        if request.top_p is not None:
            payload['top_p'] = max(0.0, min(1.0, request.top_p))
        
        if request.frequency_penalty is not None:
            payload['frequency_penalty'] = max(-2.0, min(2.0, request.frequency_penalty))
        
        if request.presence_penalty is not None:
            payload['presence_penalty'] = max(-2.0, min(2.0, request.presence_penalty))
        
        try:
            response = await self.http_client.post(
                self._build_url('chat/completions'),
                headers=self.get_headers(),
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            return ChatResponse(**response.json())
            
        except httpx.HTTPError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=e.response.status_code if hasattr(e, 'response') else 500, provider="openai")
    
    async def chat_completion_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion using OpenAI API."""
        if not self.is_model_supported(request.model):
            raise ProviderError(f"Model {request.model} not supported by OpenAI", provider="openai")
        
        messages = self._convert_messages_format([
            {'role': msg.role, 'content': msg.content} 
            for msg in request.messages
        ])
        
        payload = {
            'model': request.model,
            'messages': messages,
            'stream': True,
        }
        
        if request.max_tokens:
            payload['max_tokens'] = request.max_tokens
        
        if request.temperature is not None:
            payload['temperature'] = max(0.0, min(2.0, request.temperature))
        
        if request.top_p is not None:
            payload['top_p'] = max(0.0, min(1.0, request.top_p))
        
        if request.frequency_penalty is not None:
            payload['frequency_penalty'] = max(-2.0, min(2.0, request.frequency_penalty))
        
        if request.presence_penalty is not None:
            payload['presence_penalty'] = max(-2.0, min(2.0, request.presence_penalty))
        
        try:
            async with self.http_client.stream(
                'POST',
                self._build_url('chat/completions'),
                headers=self.get_headers(),
                json=payload,
                timeout=30.0
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        yield line + "\n"
                
        except httpx.HTTPError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=e.response.status_code if hasattr(e, 'response') else 500, provider="openai")
    
    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Create embeddings using OpenAI API."""
        if not self.is_model_supported(request.model):
            raise ProviderError(f"Model {request.model} not supported by OpenAI", provider="openai")
        
        payload = {
            'model': request.model,
            'input': request.input,
        }
        
        try:
            response = await self.http_client.post(
                self._build_url('embeddings'),
                headers=self.get_headers(),
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            return EmbeddingResponse(**response.json())
            
        except httpx.HTTPError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=e.response.status_code if hasattr(e, 'response') else 500, provider="openai")
    
    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            response = await self.http_client.get(
                self._build_url('models'),
                headers=self.get_headers(),
                timeout=10.0
            )
            return response.status_code == 200
        except Exception:
            return False