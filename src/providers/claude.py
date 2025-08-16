"""
Claude provider implementation for Anthropic Claude API.
"""
import json
import time
import logging
from typing import Dict, Any, Optional, AsyncGenerator, List
import httpx
from pydantic import BaseModel

from .base import BaseProvider, ChatMessage, ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, ProviderError, ModelInfo


class ClaudeProvider(BaseProvider):
    """Provider for Anthropic Claude API."""
    
    def __init__(self, config: Dict[str, Any], http_client: httpx.AsyncClient):
        super().__init__(config, http_client)
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ProviderError("Claude API key is required", provider="claude")
        
        self.base_url = config.get('base_url', 'https://api.anthropic.com').rstrip('/')
        self.api_version = config.get('api_version', '2023-06-01')
        self.logger = logging.getLogger(__name__)
        
        self.headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": self.api_version,
        }
    
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Send a chat completion request to Claude."""
        if not self.is_model_supported(request.model):
            raise ProviderError(f"Model {request.model} not supported by Claude", provider="claude")
        
        try:
            claude_request = self._convert_request_to_claude_format(request)
            
            response = await self.http_client.post(
                self._build_url("/v1/messages"),
                headers=self.headers,
                json=claude_request,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            return self._convert_claude_response_to_openai_format(result, request.model)
            
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Claude API error: {e.response.status_code} - {e.response.text}")
            raise ProviderError(f"Claude API error: {e.response.status_code}", provider="claude", status_code=e.response.status_code)
        except Exception as e:
            self.logger.error(f"Unexpected error in Claude chat completion: {str(e)}")
            raise ProviderError(str(e), provider="claude")
    
    async def chat_completion_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Send a streaming chat completion request to Claude."""
        try:
            claude_request = self._convert_request_to_claude_format(request)
            claude_request["stream"] = True
            
            message_id = f"chatcmpl-{int(time.time())}"
            created = int(time.time())
            
            # Send initial response
            initial_data = {
                'id': message_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': request.model,
                'choices': [{
                    'index': 0,
                    'delta': {'role': 'assistant'},
                    'finish_reason': None
                }]
            }
            yield f"data: {json.dumps(initial_data)}\n\n"
            
            async with self.http_client.stream(
                "POST",
                self._build_url("/v1/messages"),
                headers=self.headers,
                json=claude_request,
                timeout=30.0
            ) as response:
                response.raise_for_status()
                
                content = ""
                async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if data.get('type') == 'content_block_delta':
                                    delta_text = data.get('delta', {}).get('text', '')
                                    if delta_text:
                                        content += delta_text
                                        chunk_data = {
                                            'id': message_id,
                                            'object': 'chat.completion.chunk',
                                            'created': created,
                                            'model': request.model,
                                            'choices': [{
                                                'index': 0,
                                                'delta': {'content': delta_text},
                                                'finish_reason': None
                                            }]
                                        }
                                        yield f"data: {json.dumps(chunk_data)}\n\n"
                                elif data.get('type') == 'message_stop':
                                    final_data = {
                                        'id': message_id,
                                        'object': 'chat.completion.chunk',
                                        'created': created,
                                        'model': request.model,
                                        'choices': [{
                                            'index': 0,
                                            'delta': {},
                                            'finish_reason': 'stop'
                                        }]
                                    }
                                    yield f"data: {json.dumps(final_data)}\n\n"
                                    yield "data: [DONE]\n\n"
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                                
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Claude API error: {e.response.status_code} - {e.response.text}")
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'api_error'}})}\n\n"
        except Exception as e:
            self.logger.error(f"Unexpected error in Claude streaming: {str(e)}")
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'internal_error'}})}\n\n"
    
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
            for model in self.get_supported_models()
        ]
    
    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Claude does not support embeddings."""
        raise ProviderError("Claude does not support embeddings", provider="claude", status_code=501)
    
    def is_model_supported(self, model: str) -> bool:
        """Check if a model is supported by Claude."""
        return model in self.get_supported_models()
    
    async def health_check(self) -> bool:
        """Check the health of the Claude API."""
        try:
            response = await self.http_client.get(
                self._build_url("/v1/models"),
                headers=self.headers,
                timeout=10.0
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API calls."""
        return self.headers
    
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
    
    def _build_url(self, path: str) -> str:
        """Build URL ensuring no duplicate API version paths."""
        # Remove leading slash from path
        path = path.lstrip('/')
        
        # Check if base_url already ends with /v1 or /v1/
        if self.base_url.endswith('/v1') or self.base_url.endswith('/v1/'):
            # Remove /v1 from the beginning of the path if present
            if path.startswith('v1/'):
                path = path[3:]  # Remove 'v1/'
                return f"{self.base_url}/{path}"
            else:
                return f"{self.base_url}/{path}"
        else:
            # Standard case: base_url doesn't include version
            if path.startswith('v1/'):
                return f"{self.base_url}/{path}"
            else:
                return f"{self.base_url}/v1/{path}"

    def get_supported_models(self) -> list[str]:
        """Get list of supported Claude models."""
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229"
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
    
    def _convert_claude_response_to_openai_format(self, claude_response: Dict[str, Any], model: str) -> ChatResponse:
        """Convert Claude response to OpenAI format."""
        content = ""
        if "content" in claude_response and claude_response["content"]:
            content = claude_response["content"][0].get("text", "")
            
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
                "prompt_tokens": claude_response.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": claude_response.get("usage", {}).get("output_tokens", 0),
                "total_tokens": claude_response.get("usage", {}).get("input_tokens", 0) + claude_response.get("usage", {}).get("output_tokens", 0)
            }
        )