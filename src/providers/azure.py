"""
Azure OpenAI provider implementation for Azure OpenAI Service.
"""
import json
import time
import uuid
from typing import Dict, Any, AsyncGenerator, List
import httpx

from .base import BaseProvider, ModelInfo, ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, ProviderError


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI Service provider."""
    
    def __init__(self, config: Dict[str, Any], http_client: httpx.AsyncClient):
        super().__init__(config, http_client)
        self.api_key = config.get('api_key')
        self.endpoint = config.get('endpoint')
        self.api_version = config.get('api_version', '2024-10-21')
        
        if not self.api_key:
            raise ProviderError("Azure OpenAI API key is required", provider="azure")
        if not self.endpoint:
            raise ProviderError("Azure OpenAI endpoint is required", provider="azure")
        
        # Normalize endpoint URL
        self.endpoint = self.endpoint.rstrip('/')
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for Azure OpenAI API calls."""
        headers = super().get_headers()
        headers.update({
            'api-key': self.api_key,
        })
        return headers
    
    def _get_deployment_url(self, deployment_name: str, endpoint_type: str) -> str:
        """Get the URL for a specific deployment and endpoint type."""
        base_url = f"{self.endpoint}/openai/deployments/{deployment_name}"
        
        if endpoint_type == "chat":
            return f"{base_url}/chat/completions?api-version={self.api_version}"
        elif endpoint_type == "embeddings":
            return f"{base_url}/embeddings?api-version={self.api_version}"
        elif endpoint_type == "models":
            return f"{self.endpoint}/openai/models?api-version={self.api_version}"
        else:
            raise ValueError(f"Unknown endpoint type: {endpoint_type}")
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models from Azure OpenAI."""
        try:
            response = await self.http_client.get(
                self._get_deployment_url("", "models"),
                headers=self.get_headers(),
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            # Azure returns deployments, which map to models
            for model_data in data.get('data', []):
                models.append(ModelInfo(
                    id=model_data['id'],
                    created=model_data.get('created', int(time.time())),
                    owned_by=model_data.get('owned_by', 'azure'),
                    provider="azure"
                ))
            
            return models
            
        except httpx.HTTPError as e:
            # Azure OpenAI might not support models endpoint, return common deployments
            return [
                ModelInfo(
                    id="gpt-4o",
                    created=int(time.time()),
                    owned_by="azure",
                    provider="azure"
                ),
                ModelInfo(
                    id="gpt-4",
                    created=int(time.time()),
                    owned_by="azure",
                    provider="azure"
                ),
                ModelInfo(
                    id="gpt-35-turbo",
                    created=int(time.time()),
                    owned_by="azure",
                    provider="azure"
                ),
            ]
    
    def is_model_supported(self, model: str) -> bool:
        """Check if model is supported by Azure OpenAI."""
        azure_models = {
            "gpt-4o",
            "gpt-4",
            "gpt-4-32k",
            "gpt-35-turbo",
            "gpt-35-turbo-16k",
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
        }
        return model in azure_models
    
    def _convert_messages_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert messages to Azure OpenAI format (no conversion needed)."""
        return messages
    
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate chat completion using Azure OpenAI API."""
        if not self.is_model_supported(request.model):
            raise ProviderError(f"Model {request.model} not supported by Azure", provider="azure")
        
        messages = self._convert_messages_format([
            {'role': msg.role, 'content': msg.content} 
            for msg in request.messages
        ])
        
        payload = {
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
                self._get_deployment_url(request.model, "chat"),
                headers=self.get_headers(),
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            data = response.json()
            
            return ChatResponse(**data)
            
        except httpx.HTTPError as e:
            raise ProviderError(f"Azure OpenAI API error: {str(e)}", status_code=e.response.status_code if hasattr(e, 'response') else 500, provider="azure")
    
    async def chat_completion_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion using Azure OpenAI API."""
        if not self.is_model_supported(request.model):
            raise ProviderError(f"Model {request.model} not supported by Azure", provider="azure")
        
        messages = self._convert_messages_format([
            {'role': msg.role, 'content': msg.content} 
            for msg in request.messages
        ])
        
        payload = {
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
                self._get_deployment_url(request.model, "chat"),
                headers=self.get_headers(),
                json=payload,
                timeout=30.0
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        yield line + "\n"
                
        except httpx.HTTPError as e:
            raise ProviderError(f"Azure OpenAI API error: {str(e)}", status_code=e.response.status_code if hasattr(e, 'response') else 500, provider="azure")
    
    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Create embeddings using Azure OpenAI API."""
        if not self.is_model_supported(request.model):
            raise ProviderError(f"Model {request.model} not supported by Azure", provider="azure")
        
        payload = {
            'input': request.input,
        }
        
        try:
            response = await self.http_client.post(
                self._get_deployment_url(request.model, "embeddings"),
                headers=self.get_headers(),
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            data = response.json()
            
            return EmbeddingResponse(**data)
            
        except httpx.HTTPError as e:
            raise ProviderError(f"Azure OpenAI API error: {str(e)}", status_code=e.response.status_code if hasattr(e, 'response') else 500, provider="azure")
    
    async def health_check(self) -> bool:
        """Check Azure OpenAI API health."""
        try:
            response = await self.http_client.get(
                self._get_deployment_url("", "models"),
                headers=self.get_headers(),
                timeout=10.0
            )
            return response.status_code == 200
        except Exception:
            return False