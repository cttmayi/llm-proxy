"""
Claude provider implementation for Anthropic Claude API.
"""
import json
import time
import logging
from typing import Dict, Any, AsyncGenerator, List
import anthropic
from anthropic import AsyncAnthropic

from .base import BaseProvider, ChatRequest, ChatResponse, ChatStreamResponse, EmbeddingRequest, EmbeddingResponse, ProviderError, ModelInfo
from ..utils.cache import async_cache


class ClaudeProvider(BaseProvider):
    """Provider for Anthropic Claude API."""
    
    def __init__(self, config: Dict[str, Any], http_client=None):
        super().__init__(config, http_client)
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ProviderError("Claude API key is required", provider="claude")
        
        self.base_url = config.get('base_url')
        self.api_version = config.get('api_version', '2023-06-01')
        self.logger = logging.getLogger(__name__)
        
        # Initialize Anthropic client
        client_kwargs = {
            'base_url': self.base_url or "https://api.anthropic.com",
            'api_key': self.api_key,
        }
        
        self.client = AsyncAnthropic(**client_kwargs)
    
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Send a chat completion request to Claude."""
        if not self.is_model_supported(request.model):
            raise ProviderError(f"Model {request.model} not supported by Claude", provider="claude")
        
        try:
            claude_request = self._convert_request_to_claude_format(request)
            
            # Extract system prompt and messages
            system = claude_request.get("system")
            messages = claude_request["messages"]
            
            # Build request parameters for Anthropic client
            client_kwargs = {
                "model": request.model,
                "messages": messages,
                "max_tokens": claude_request.get("max_tokens", 4096),
            }
            
            if system:
                client_kwargs["system"] = system
            
            if request.temperature is not None:
                client_kwargs["temperature"] = request.temperature
            
            if request.top_p is not None:
                client_kwargs["top_p"] = request.top_p

            response = await self.client.messages.create(**client_kwargs)
            
            return self._convert_claude_response_to_openai_format(response, request.model)
            
        except anthropic.APIStatusError as e:
            self.logger.error(f"Claude API error: {e.status_code} - {e.message}")
            raise ProviderError(f"Claude API error: {e.status_code}", provider="claude", status_code=e.status_code)
        except anthropic.APIConnectionError as e:
            self.logger.error(f"Claude API connection error: {str(e)}")
            raise ProviderError(f"Claude API connection error: {str(e)}", provider="claude", status_code=500)
        except Exception as e:
            self.logger.error(f"Unexpected error in Claude chat completion: {str(e)}")
            raise ProviderError(str(e), provider="claude")
    
    async def chat_completion_stream(self, request: ChatRequest) -> AsyncGenerator[ChatStreamResponse, None]:
        """Send a streaming chat completion request to Claude."""
        try:
            claude_request = self._convert_request_to_claude_format(request)
            
            # Extract system prompt and messages
            system = claude_request.get("system")
            messages = claude_request["messages"]
            
            message_id = f"chatcmpl-{int(time.time())}"
            created = int(time.time())
            
            # Build request parameters for Anthropic client
            client_kwargs = {
                "model": request.model,
                "messages": messages,
                "max_tokens": claude_request.get("max_tokens", 4096),
            }
            
            if system:
                client_kwargs["system"] = system
            
            if request.temperature is not None:
                client_kwargs["temperature"] = request.temperature
            
            if request.top_p is not None:
                client_kwargs["top_p"] = request.top_p
            
            # Send initial response
            initial_response = ChatStreamResponse(
                id=message_id,
                object='chat.completion.chunk',
                created=created,
                model=request.model,
                choices=[{
                    'index': 0,
                    'delta': {'role': 'assistant'},
                    'finish_reason': None
                }]
            )
            yield initial_response
            
            async with self.client.messages.stream(**client_kwargs) as stream:
                async for event in stream:
                    # Handle different event types from Claude's streaming
                    try:
                        event_type = str(getattr(event, 'type', ''))
                        
                        # Content block delta - streaming text content
                        if event_type == 'content_block_delta':
                            delta = getattr(event, 'delta', None)
                            if delta and getattr(delta, 'text', None):
                                text = delta.text
                                chunk_response = ChatStreamResponse(
                                    id=message_id,
                                    object='chat.completion.chunk',
                                    created=created,
                                    model=request.model,
                                    choices=[{
                                        'index': 0,
                                        'delta': {'content': text},
                                        'finish_reason': None
                                    }]
                                )
                                yield chunk_response
                                
                        # Message delta - final metadata including stop_reason
                        elif event_type == 'message_delta':
                            delta = getattr(event, 'delta', None)
                            if delta and getattr(delta, 'stop_reason', None):
                                finish_response = ChatStreamResponse(
                                    id=message_id,
                                    object='chat.completion.chunk',
                                    created=created,
                                    model=request.model,
                                    choices=[{
                                        'index': 0,
                                        'delta': {},
                                        'finish_reason': delta.stop_reason
                                    }]
                                )
                                yield finish_response
                                
                        # Message stop - end of stream
                        elif event_type == 'message_stop':
                            final_response = ChatStreamResponse(
                                id=message_id,
                                object='chat.completion.chunk',
                                created=created,
                                model=request.model,
                                choices=[{
                                    'index': 0,
                                    'delta': {},
                                    'finish_reason': 'stop'
                                }]
                            )
                            yield final_response
                            break
                    except Exception as e:
                        self.logger.warning(f"Error processing event: {e}")
                        continue
                
                # Ensure we always send a final chunk
                final_response = ChatStreamResponse(
                    id=message_id,
                    object='chat.completion.chunk',
                    created=created,
                    model=request.model,
                    choices=[{
                        'index': 0,
                        'delta': {},
                        'finish_reason': 'stop'
                    }]
                )
                yield final_response
                
        except anthropic.APIStatusError as e:
            self.logger.error(f"Claude API error: {e.status_code} - {e.message}")
            raise ProviderError(str(e), provider="claude", status_code=e.status_code)

    async def chat_completion_stream_fast(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Send a streaming chat completion request to Claude (fast version for native format)."""
        try:
            claude_request = self._convert_request_to_claude_format(request)
            
            # Extract system prompt and messages
            system = claude_request.get("system")
            messages = claude_request["messages"]
            
            # Build request parameters for Anthropic client
            client_kwargs = {
                "model": request.model,
                "messages": messages,
                "max_tokens": claude_request.get("max_tokens", 4096),
            }
            
            if system:
                client_kwargs["system"] = system
            
            if request.temperature is not None:
                client_kwargs["temperature"] = request.temperature
            
            if request.top_p is not None:
                client_kwargs["top_p"] = request.top_p
            
            async with self.client.messages.stream(**client_kwargs) as stream:
                async for event in stream:
                    if hasattr(event, 'type'):
                        event_data = {
                            "type": event.type,
                            "data": event.model_dump() if hasattr(event, 'model_dump') else str(event)
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                
        except anthropic.APIStatusError as e:
            self.logger.error(f"Claude API error: {e.status_code} - {e.message}")
            raise ProviderError(str(e), provider="claude", status_code=e.status_code)
        except anthropic.APIConnectionError as e:
            self.logger.error(f"Claude API connection error: {str(e)}")
            raise ProviderError(str(e), provider="claude", status_code=500)
        except Exception as e:
            self.logger.error(f"Unexpected error in Claude streaming: {str(e)}")
            raise ProviderError(str(e), provider="claude")
    
    @async_cache(ttl=300)
    async def list_models(self) -> List[ModelInfo]:
        """List available models from Claude."""
        return [
            ModelInfo(
                id=model,
                object="model",
                created=int(time.time()),
                owned_by="anthropic",
                provider="claude"
            )
            for model in ["claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"]
        ]
    
    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Claude does not support embeddings."""
        _ = request  # Mark as intentionally unused
        raise ProviderError("Claude does not support embeddings", provider="claude", status_code=501)
    
    def is_model_supported(self, model: str) -> bool:
        """Check if a model is supported by Claude."""
        supported = self.get_supported_models()
        model_lower = model.lower()
        
        # Check exact match
        if model_lower in [m.lower() for m in supported]:
            return True
            
        # Check if model starts with any supported prefix
        for supported_model in supported:
            if model_lower in supported_model.lower():
                return True
                
        return False
    
    async def health_check(self) -> bool:
        """Check the health of the Claude API."""
        try:
            # Claude doesn't have a models endpoint, so we try to make a simple request
            await self.client.messages.create(
                model="claude-3-haiku-20241022",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=1
            )
            return True
        except Exception:
            return False
    
    async def close(self):
        """Clean up resources."""
        if self.client:
            await self.client.close()
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API calls."""
        return super().get_headers()
    
    def _convert_messages_format(self, messages: List[Dict[str, str]]) -> tuple:
        """Convert messages format and separate system messages."""
        claude_messages = []
        system_message = None
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                claude_messages.append(msg)
        
        return claude_messages, system_message
    

    def get_supported_models(self) -> list[str]:
        """Get list of supported Claude models."""
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]
    
    def _convert_request_to_claude_format(self, request: ChatRequest) -> Dict[str, Any]:
        """Convert OpenAI format request to Claude format."""
        messages = []
        system_prompt = None
        
        # Separate system messages from user/assistant messages
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        claude_request = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature or 0.7,
        }
        
        if system_prompt:
            claude_request["system"] = system_prompt
            
        if request.top_p:
            claude_request["top_p"] = request.top_p
            
        return claude_request
    
    def _convert_claude_response_to_openai_format(self, claude_response, model: str) -> ChatResponse:
        """Convert Claude response to OpenAI format."""
        content = ""
        if claude_response.content and len(claude_response.content) > 0:
            content = claude_response.content[0].text
            
        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": claude_response.usage.input_tokens,
                "completion_tokens": claude_response.usage.output_tokens,
                "total_tokens": claude_response.usage.input_tokens + claude_response.usage.output_tokens
            }
        )