"""
OpenAI provider implementation for OpenAI API.
"""
import json
import time
import uuid
from typing import Dict, Any, AsyncGenerator, List
import asyncio
import openai
from openai import AsyncOpenAI

from .base import BaseProvider, ModelInfo, ChatRequest, ChatResponse, ChatStreamResponse, EmbeddingRequest, EmbeddingResponse, ProviderError
from ..utils.cache import async_cache


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: Dict[str, Any], http_client=None):
        super().__init__(config, http_client)
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        self.organization = config.get('organization')
        
        if not self.api_key:
            raise ProviderError("OpenAI API key is required", provider="openai")
        
        # Initialize OpenAI client
        client_kwargs = {
            'api_key': self.api_key,
        }
        
        if self.base_url:
            client_kwargs['base_url'] = self.base_url
            
        if self.organization:
            client_kwargs['organization'] = self.organization
            
        self.client = AsyncOpenAI(**client_kwargs)
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for OpenAI API calls."""
        return super().get_headers()
    
    @async_cache(ttl=300)
    async def list_models(self) -> List[ModelInfo]:
        """List available models from OpenAI."""
        try:
            response = await self.client.models.list()
            
            models = []
            for model_data in response.data:
                models.append(ModelInfo(
                    id=model_data.id,
                    created=getattr(model_data, 'created', int(time.time())),
                    owned_by=getattr(model_data, 'owned_by', 'openai'),
                    provider="openai"
                ))
            
            return models
            
        except openai.APIStatusError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=e.status_code, provider="openai")
        except openai.APIConnectionError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=500, provider="openai")
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=500, provider="openai")
    
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
        
        # Build request parameters
        chat_params = {
            'model': request.model,
            'messages': messages,
        }
        
        if request.max_tokens:
            chat_params['max_tokens'] = request.max_tokens
        
        if request.temperature is not None:
            chat_params['temperature'] = max(0.0, min(2.0, request.temperature))
        
        if request.top_p is not None:
            chat_params['top_p'] = max(0.0, min(1.0, request.top_p))
        
        if request.frequency_penalty is not None:
            chat_params['frequency_penalty'] = max(-2.0, min(2.0, request.frequency_penalty))
        
        if request.presence_penalty is not None:
            chat_params['presence_penalty'] = max(-2.0, min(2.0, request.presence_penalty))
        
        try:
            response = await self.client.chat.completions.create(**chat_params)
            
            # Convert OpenAI response to our ChatResponse format
            return ChatResponse(
                id=response.id,
                object=response.object,
                created=response.created,
                model=response.model,
                choices=[{
                    'index': choice.index,
                    'message': {
                        'role': choice.message.role,
                        'content': choice.message.content
                    },
                    'finish_reason': choice.finish_reason
                } for choice in response.choices],
                usage={
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            )
            
        except openai.APIStatusError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=e.status_code, provider="openai")
        except openai.APIConnectionError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=500, provider="openai")
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=500, provider="openai")
    
    async def chat_completion_stream(self, request: ChatRequest) -> AsyncGenerator[ChatStreamResponse, None]:
        """Generate streaming chat completion using OpenAI API."""
        if not self.is_model_supported(request.model):
            raise ProviderError(f"Model {request.model} not supported by OpenAI", provider="openai")
        
        messages = self._convert_messages_format([
            {'role': msg.role, 'content': msg.content} 
            for msg in request.messages
        ])
        
        # Build request parameters
        chat_params = {
            'model': request.model,
            'messages': messages,
            'stream': True,
        }
        
        if request.max_tokens:
            chat_params['max_tokens'] = request.max_tokens
        
        if request.temperature is not None:
            chat_params['temperature'] = max(0.0, min(2.0, request.temperature))
        
        if request.top_p is not None:
            chat_params['top_p'] = max(0.0, min(1.0, request.top_p))
        
        if request.frequency_penalty is not None:
            chat_params['frequency_penalty'] = max(-2.0, min(2.0, request.frequency_penalty))
        
        if request.presence_penalty is not None:
            chat_params['presence_penalty'] = max(-2.0, min(2.0, request.presence_penalty))
        
        try:
            stream = await self.client.chat.completions.create(**chat_params)
            
            async for chunk in stream:
                # Convert chunk to ChatStreamResponse format
                if chunk.choices and chunk.choices[0].delta is not None:
                    response_chunk = ChatStreamResponse(
                        id=chunk.id,
                        object=chunk.object,
                        created=chunk.created,
                        model=chunk.model,
                        choices=[{
                            'index': choice.index,
                            'delta': {
                                'role': choice.delta.role,
                                'content': choice.delta.content
                            } if choice.delta else {},
                            'finish_reason': choice.finish_reason
                        } for choice in chunk.choices]
                    )
                    yield response_chunk
                
                # Handle final chunk with usage
                if hasattr(chunk, 'usage') and chunk.usage:
                    final_chunk = ChatStreamResponse(
                        id=chunk.id,
                        object=chunk.object,
                        created=chunk.created,
                        model=chunk.model,
                        choices=[{
                            'index': choice.index,
                            'delta': {},
                            'finish_reason': choice.finish_reason
                        } for choice in chunk.choices]
                    )
                    yield final_chunk
                
        except openai.APIStatusError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=e.status_code, provider="openai")
        except openai.APIConnectionError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=500, provider="openai")
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=500, provider="openai")
    
    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Create embeddings using OpenAI API."""
        if not self.is_model_supported(request.model):
            raise ProviderError(f"Model {request.model} not supported by OpenAI", provider="openai")
        
        try:
            response = await self.client.embeddings.create(
                model=request.model,
                input=request.input
            )
            
            # Convert OpenAI response to our EmbeddingResponse format
            return EmbeddingResponse(
                object=response.object,
                data=[{
                    'object': embedding.object,
                    'embedding': embedding.embedding,
                    'index': embedding.index
                } for embedding in response.data],
                model=response.model,
                usage={
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            )
            
        except openai.APIStatusError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=e.status_code, provider="openai")
        except openai.APIConnectionError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=500, provider="openai")
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", status_code=500, provider="openai")
    
    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
    
    async def chat_completion_stream_fast(self, request: ChatRequest):
        """OpenAI does not have a native fast streaming mode - use regular streaming."""
        _ = request  # Mark parameter as intentionally unused
        raise ProviderError("OpenAI does not support fast streaming mode", provider="openai", status_code=501)
    
    async def close(self):
        """Clean up resources."""
        if self.client:
            await self.client.close()