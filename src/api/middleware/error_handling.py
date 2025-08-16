"""
Error handling middleware for standardized error responses.
"""
import logging
import json
from typing import Dict, Any
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.providers.base import ProviderError

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for standardized error handling."""
    
    def __init__(self, app, debug: bool = False):
        super().__init__(app)
        self.debug = debug
    
    async def dispatch(self, request: Request, call_next):
        """Process requests and handle errors."""
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as exc:
            return await self._handle_http_exception(request, exc)
        except StarletteHTTPException as exc:
            return await self._handle_http_exception(request, HTTPException(
                status_code=exc.status_code,
                detail=exc.detail
            ))
        except ProviderError as exc:
            return await self._handle_provider_error(request, exc)
        except Exception as exc:
            return await self._handle_generic_error(request, exc)
    
    async def _handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions."""
        error_response = {
            "error": {
                "message": exc.detail,
                "type": "invalid_request_error",
                "code": None,
                "param": None
            }
        }
        
        # Add request ID if available
        if hasattr(request.state, 'request_id'):
            error_response["request_id"] = request.state.request_id
        
        logger.warning(
            f"HTTP {exc.status_code} error",
            extra={
                "request_id": getattr(request.state, 'request_id', None),
                "status_code": exc.status_code,
                "detail": exc.detail,
                "url": str(request.url),
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    async def _handle_provider_error(self, request: Request, exc: ProviderError) -> JSONResponse:
        """Handle provider-specific errors."""
        error_response = {
            "error": {
                "message": exc.message,
                "type": "provider_error",
                "code": f"{exc.provider}_error" if exc.provider else "provider_error",
                "provider": exc.provider
            }
        }
        
        # Add request ID if available
        if hasattr(request.state, 'request_id'):
            error_response["request_id"] = request.state.request_id
        
        logger.error(
            f"Provider error from {exc.provider}",
            extra={
                "request_id": getattr(request.state, 'request_id', None),
                "provider": exc.provider,
                "status_code": exc.status_code,
                "message": exc.message,
                "url": str(request.url),
                "method": request.method
            },
            exc_info=self.debug
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    async def _handle_generic_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle generic exceptions."""
        error_response = {
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": "internal_server_error"
            }
        }
        
        if self.debug:
            error_response["error"]["traceback"] = str(exc)
        
        # Add request ID if available
        if hasattr(request.state, 'request_id'):
            error_response["request_id"] = request.state.request_id
        
        logger.error(
            "Internal server error",
            extra={
                "request_id": getattr(request.state, 'request_id', None),
                "url": str(request.url),
                "method": request.method,
                "exception": str(exc)
            },
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )


class ValidationErrorMiddleware(BaseHTTPMiddleware):
    """Middleware for handling validation errors with better messages."""
    
    async def dispatch(self, request: Request, call_next):
        """Process requests and handle validation errors."""
        try:
            response = await call_next(request)
            return response
            
        except Exception as exc:
            if "pydantic" in str(type(exc)).lower():
                return await self._handle_validation_error(request, exc)
            raise
    
    async def _handle_validation_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle Pydantic validation errors."""
        error_message = str(exc)
        
        # Extract field-specific errors
        if hasattr(exc, 'errors'):
            errors = exc.errors()
            field_errors = []
            for error in errors:
                field = error.get('loc', [])
                msg = error.get('msg', 'Invalid input')
                field_errors.append({
                    "field": ".".join(str(f) for f in field),
                    "message": msg
                })
            
            error_response = {
                "error": {
                    "message": "Validation error",
                    "type": "validation_error",
                    "code": "invalid_parameters",
                    "details": field_errors
                }
            }
        else:
            error_response = {
                "error": {
                    "message": error_message,
                    "type": "validation_error",
                    "code": "invalid_parameters"
                }
            }
        
        # Add request ID if available
        if hasattr(request.state, 'request_id'):
            error_response["request_id"] = request.state.request_id
        
        logger.warning(
            "Validation error",
            extra={
                "request_id": getattr(request.state, 'request_id', None),
                "error": error_message,
                "url": str(request.url),
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response
        )


class CORSConfig:
    """CORS configuration helper."""
    
    def __init__(self, allowed_origins: list, allow_credentials: bool = True):
        self.allowed_origins = allowed_origins
        self.allow_credentials = allow_credentials
    
    def get_cors_config(self) -> dict:
        """Get CORS configuration for FastAPI."""
        return {
            "allow_origins": self.allowed_origins,
            "allow_credentials": self.allow_credentials,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["*"]
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # Simple in-memory storage
    
    async def dispatch(self, request: Request, call_next):
        """Rate limit requests based on client IP."""
        if self.max_requests <= 0:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - self.window_seconds
        self.requests = {
            ip: [req_time for req_time in times if req_time > cutoff_time]
            for ip, times in self.requests.items()
        }
        
        # Check current requests
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        if len(self.requests[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                }
            )
        
        # Record this request
        self.requests[client_ip].append(current_time)
        
        return await call_next(request)