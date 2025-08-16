"""
Logging middleware for request/response logging and API monitoring.
"""
import time
import json
import logging
from typing import Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import uuid

from src.web.monitor import api_monitor, APICall

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    def __init__(self, app, log_level: str = "INFO"):
        super().__init__(app)
        self.log_level = log_level.upper()
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def dispatch(self, request: Request, call_next):
        """Process request and response logging."""
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        
        # Extract request details
        request_info = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type")
        }
        
        # Log request body for POST/PUT requests (excluding sensitive data)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Try to parse JSON and redact sensitive fields
                    try:
                        json_body = json.loads(body.decode())
                        redacted_body = self._redact_sensitive_data(json_body)
                        request_info["body"] = redacted_body
                    except json.JSONDecodeError:
                        request_info["body_size"] = len(body)
                
                # Important: Re-assign the body to the request so downstream handlers can read it
                async def receive():
                    return {"type": "http.request", "body": body}
                request._receive = receive
            except Exception:
                pass
        
        logger.info("Request started", extra={"request": request_info})
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            response_info = {
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
                "content_type": response.headers.get("content-type")
            }
            
            logger.info("Request completed", extra={"response": response_info})
            
            # Extract model from response if possible
            model = None
            provider = None
            response_body = None
            
            # For JSON responses, try to extract model info
            # if response.headers.get("content-type", "").startswith("application/json"):
            #     try:
            #         # For streaming responses, we can't easily read the body
            #         # For non-streaming, we can try to capture it
            #         pass
            #     except Exception:
            #         pass
            
            # Create API call record
            # api_call = APICall(
            #     id=request_id,
            #     timestamp=time.time(),
            #     method=request.method,
            #     path=request.url.path,
            #     headers=dict(request.headers),
            #     request_body=request_info.get("body"),
            #     response_body=response_body,
            #     status_code=response.status_code,
            #     duration_ms=round(process_time * 1000, 2),
            #     model=model,
            #     provider=provider,
            #     error=None if response.status_code < 400 else f"HTTP {response.status_code}"
            # )
            
            # Add to monitor
            # api_monitor.add_call(api_call)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time_ms": round(process_time * 1000, 2)
                },
                exc_info=True
            )
            
            # Return error response with request ID
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id
                },
                headers={"X-Request-ID": request_id}
            )
    
    def _redact_sensitive_data(self, data: Any) -> Any:
        """Redact sensitive information from request data."""
        if isinstance(data, dict):
            redacted = {}
            for key, value in data.items():
                if key.lower() in ["api_key", "authorization", "password", "token", "secret"]:
                    redacted[key] = "[REDACTED]"
                elif key.lower() == "messages" and isinstance(value, list):
                    # Redact message content for privacy
                    redacted[key] = [{k: v for k, v in msg.items() if k != "content"} for msg in value]
                else:
                    redacted[key] = self._redact_sensitive_data(value)
            return redacted
        elif isinstance(data, list):
            return [self._redact_sensitive_data(item) for item in data]
        else:
            return data


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting request metrics."""
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "request_duration": [],
            "status_codes": {}
        }
    
    async def dispatch(self, request: Request, call_next):
        """Collect metrics for each request."""
        start_time = time.time()
        
        # Increment total requests
        self.metrics["total_requests"] += 1
        
        try:
            response = await call_next(request)
            
            # Record status code
            status_code = str(response.status_code)
            if status_code not in self.metrics["status_codes"]:
                self.metrics["status_codes"][status_code] = 0
            self.metrics["status_codes"][status_code] += 1
            
            # Record duration
            duration = time.time() - start_time
            self.metrics["request_duration"].append(duration)
            
            # Keep only last 1000 durations to prevent memory growth
            if len(self.metrics["request_duration"]) > 1000:
                self.metrics["request_duration"] = self.metrics["request_duration"][-1000:]
            
            return response
            
        except Exception as e:
            # Increment error count
            self.metrics["total_errors"] += 1
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        durations = self.metrics["request_duration"]
        
        return {
            "total_requests": self.metrics["total_requests"],
            "total_errors": self.metrics["total_errors"],
            "error_rate": self.metrics["total_errors"] / max(self.metrics["total_requests"], 1),
            "avg_duration_ms": sum(durations) / len(durations) * 1000 if durations else 0,
            "status_codes": self.metrics["status_codes"]
        }