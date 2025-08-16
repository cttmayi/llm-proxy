"""
Response capture middleware for detailed API monitoring.
"""
import time
import json
import logging
from typing import Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from src.web.monitor import api_monitor, APICall

logger = logging.getLogger(__name__)


class ResponseCaptureMiddleware(BaseHTTPMiddleware):
    """Middleware for capturing detailed request/response data for API monitoring."""
    
    async def dispatch(self, request: Request, call_next):
        """Capture and store detailed API call information."""
        start_time = time.time()
        request_id = None
        
        # Find existing request ID from LoggingMiddleware
        if hasattr(request.state, 'request_id'):
            request_id = request.state.request_id
        else:
            import uuid
            request_id = str(uuid.uuid4())
        
        # Capture request details
        request_body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    body_str = body.decode('utf-8')
                    try:
                        request_body = json.loads(body_str)
                    except json.JSONDecodeError:
                        request_body = body_str
                    
                    # Restore body for downstream handlers
                    async def receive():
                        return {"type": "http.request", "body": body}
                    request._receive = receive
            except Exception as e:
                request_body = f"[Error reading request body: {str(e)}]"
        
        # Capture model from request body if possible
        model = None
        if isinstance(request_body, dict):
            model = request_body.get('model')
        
        # Only capture specific LLM API calls
        path = str(request.url.path)
        is_llm_endpoint = path in ['/v1/chat/completions', '/v1/embeddings', '/messages'] and not path.startswith('/web')
        
        if not is_llm_endpoint:
            # Skip monitoring for non-LLM endpoints
            response = await call_next(request)
            return response
            
        # Process LLM API calls
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Capture response body
            response_body = None
            try:
                response_body_str = None
                
                # Method 1: Check if response has body attribute (JSONResponse, etc.)
                if hasattr(response, 'body') and response.body is not None:
                    if isinstance(response.body, bytes):
                        response_body_str = response.body.decode('utf-8')
                    else:
                        response_body_str = str(response.body)
                
                # Method 2: Try to collect from body_iterator
                elif hasattr(response, 'body_iterator'):
                    try:
                        body_parts = []
                        async for chunk in response.body_iterator:
                            if isinstance(chunk, bytes):
                                body_parts.append(chunk)
                            else:
                                body_parts.append(str(chunk).encode('utf-8'))
                        
                        if body_parts:
                            body = b''.join(body_parts)
                            response_body_str = body.decode('utf-8')
                            
                            # Recreate response with captured content
                            from starlette.responses import Response
                            response = Response(
                                content=body,
                                status_code=response.status_code,
                                headers=dict(response.headers)
                            )
                    except Exception as e:
                        response_body_str = f"[Error reading body: {str(e)}]"
                
                else:
                    response_body_str = "[Response body not directly accessible]"
                
                # Parse JSON if possible
                if response_body_str and response_body_str != "[Response body not directly accessible]" and not response_body_str.startswith("[Error"):
                    try:
                        response_body = json.loads(response_body_str)
                    except json.JSONDecodeError:
                        response_body = response_body_str
                else:
                    response_body = response_body_str or "[Empty response]"
                    
            except Exception as e:
                response_body = f"[Error capturing response: {str(e)}]"
            
            # Extract model from response if not in request
            if not model and isinstance(response_body, dict):
                model = response_body.get('model')
            
            # Determine provider
            # provider = None
            # if model:
            #     model_lower = str(model).lower()
            #     if 'claude' in model_lower:
            #         provider = 'claude'
            #     elif any(x in model_lower for x in ['gpt', 'o1', 'o3', 'o4']):
            #         provider = 'openai'
            #     elif 'azure' in model_lower:
            #         provider = 'azure'
            print(f"Response body: {response_body}")
            # Create API call record
            api_call = APICall(
                id=request_id,
                timestamp=time.time(),
                method=request.method,
                path=path,
                headers=dict(request.headers),
                request_body=request_body,
                response_body=response_body,
                status_code=response.status_code,
                duration_ms=duration_ms,
                model=model,
                provider=None,
                error=None if response.status_code < 400 else f"HTTP {response.status_code}"
            )
            
            # Add to monitor
            api_monitor.add_call(api_call)
            
            # Add monitoring headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-API-Monitor"] = "enabled"
            
            return response
            
        except Exception as e:
            # Log error case
            duration_ms = (time.time() - start_time) * 1000
            
            api_call = APICall(
                id=request_id,
                timestamp=time.time(),
                method=request.method,
                path=str(request.url.path),
                headers=dict(request.headers),
                request_body=request_body,
                response_body=None,
                status_code=500,
                duration_ms=duration_ms,
                model=model,
                provider=None,
                error=str(e)
            )
            
            api_monitor.add_call(api_call)
            raise