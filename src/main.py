"""
Main FastAPI application for LLM Proxy.
"""
import os
import sys
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# import dotenv
# dotenv.load_dotenv()

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config.loader import ConfigLoader
from src.config.models import ProxyConfig
from src.providers.factory import ProviderFactory
from src.api.endpoints.chat import create_chat_router
from src.api.endpoints.embeddings import create_embeddings_router
from src.api.endpoints.models import create_models_router
from src.api.endpoints.health import create_health_router
from src.web.routes import create_web_router
from fastapi.staticfiles import StaticFiles
from src.api.middleware.logging import LoggingMiddleware, MetricsMiddleware
from src.api.middleware.error_handling import ErrorHandlingMiddleware, ValidationErrorMiddleware
from src.api.middleware.error_handling import CORSConfig
from src.api.middleware.response_capture import ResponseCaptureMiddleware


def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Load configuration
    if config is None:
        config_loader = ConfigLoader()
        app_config = config_loader.load_config()
    else:
        from src.config.models import ProxyConfig
        app_config = ProxyConfig(**config)
    
    # Create provider factory
    provider_factory = ProviderFactory(app_config.model_dump())
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan event handler for startup and shutdown."""
        logger = logging.getLogger(__name__)
        logger.info("Starting LLM Proxy...")
        
        # Log configuration summary
        config_loader = ConfigLoader()
        config_summary = config_loader.get_config_summary(app_config)
        logger.info(f"Configuration loaded: {config_summary}")
        
        # Check provider health
        try:
            health_status = await provider_factory.health_check_all()
            logger.info(f"Provider health status: {health_status}")
        except Exception as e:
            logger.warning(f"Failed to check provider health: {e}")
        
        yield
        
        # Clean up resources
        logger.info("Shutting down LLM Proxy...")
        await provider_factory.close()
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="LLM Proxy",
        description="A unified API proxy for multiple LLM providers",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Configure CORS
    if app_config.security.enable_cors:
        cors_config = CORSConfig(
            allowed_origins=app_config.security.allowed_origins,
            allow_credentials=True
        )
        cors_kwargs = cors_config.get_cors_config()
        app.add_middleware(CORSMiddleware, **cors_kwargs)
    
    # Add middleware (order matters - response capture should be last)
    app.add_middleware(LoggingMiddleware, log_level=app_config.server.log_level)
    app.add_middleware(ErrorHandlingMiddleware, debug=(app_config.server.log_level == "DEBUG"))
    app.add_middleware(ValidationErrorMiddleware)
    app.add_middleware(ResponseCaptureMiddleware)
    
    if app_config.features.enable_metrics:
        app.add_middleware(MetricsMiddleware)
    
    # Include routers
    app.include_router(create_chat_router(provider_factory))
    app.include_router(create_embeddings_router(provider_factory))
    app.include_router(create_models_router(provider_factory))
    app.include_router(create_health_router(provider_factory))
    app.include_router(create_web_router())
    
    # Mount static files for web interface
    from pathlib import Path
    web_dir = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")

    return app


def main():
    """Main entry point for running the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Proxy Server")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--log-level", default=None, help="Log level")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.load_config(args.config)
    
    # Override with command line arguments
    host = args.host or config.server.host
    port = args.port or config.server.port
    log_level = args.log_level or config.server.log_level
    
    # Create and run app
    app = create_app(config.model_dump())
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        reload=args.reload,
        workers=config.server.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()