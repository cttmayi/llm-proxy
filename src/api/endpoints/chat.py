"""
Chat completion API endpoints.
"""
import json
import uuid
from typing import Dict, Any, AsyncGenerator
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.providers.base import ChatRequest, ChatMessage, ProviderError
from src.providers.factory import ProviderFactory


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    model: str
    messages: list[Dict[str, str]]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response model."""
    id: str
    object: str
    created: int
    model: str
    choices: list[Dict[str, Any]]
    usage: Dict[str, Any]


class ChatCompletionChunk(BaseModel):
    """Chat completion streaming chunk model."""
    id: str
    object: str
    created: int
    model: str
    choices: list[Dict[str, Any]]


router = APIRouter(prefix="/v1", tags=["chat"])


def create_chat_router(provider_factory: ProviderFactory) -> APIRouter:
    """Create chat endpoints with provider factory."""
    
    @router.post("/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        """Create a chat completion (streaming or non-streaming based on request.stream)."""
        try:
            provider = provider_factory.get_provider_for_model(request.model)
            
            chat_request = ChatRequest(
                model=request.model,
                messages=[ChatMessage(role=msg["role"], content=msg["content"]) for msg in request.messages],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=request.stream,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty
            )
            
            if request.stream:
                async def generate_stream() -> AsyncGenerator[str, None]:
                    async for chunk in provider.chat_completion_stream(chat_request):
                        yield chunk
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream; charset=utf-8",
                        "X-Accel-Buffering": "no"
                    }
                )
            else:
                response = await provider.chat_completion(chat_request)
                return ChatCompletionResponse(
                    id=response.id,
                    object=response.object,
                    created=response.created,
                    model=response.model,
                    choices=response.choices,
                    usage=response.usage
                )
            
        except ProviderError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/messages")
    async def create_claude_message(request: ChatCompletionRequest):
        """Create a Claude-native message (compatible with Anthropic API)."""
        if not request.model.startswith("claude"):
            raise HTTPException(
                status_code=400,
                detail="This endpoint only supports Claude models"
            )
        
        try:
            provider = provider_factory.get_provider('claude')
            
            chat_request = ChatRequest(
                model=request.model,
                messages=[ChatMessage(role=msg["role"], content=msg["content"]) for msg in request.messages],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=request.stream,
                top_p=request.top_p
            )
            
            if request.stream:
                async def generate_stream() -> AsyncGenerator[str, None]:
                    async for chunk in provider.chat_completion_stream(chat_request):
                        yield chunk
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream; charset=utf-8",
                        "X-Accel-Buffering": "no"
                    }
                )
            else:
                response = await provider.chat_completion(chat_request)
                
                # Convert to Claude format
                return {
                    "id": response.id,
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": response.choices[0]["message"]["content"]
                        }
                    ],
                    "model": response.model,
                    "stop_reason": response.choices[0]["finish_reason"],
                    "usage": {
                        "input_tokens": response.usage["prompt_tokens"],
                        "output_tokens": response.usage["completion_tokens"]
                    }
                }
                
        except ProviderError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return router