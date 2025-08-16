"""
Health check API endpoints.
"""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.providers.factory import ProviderFactory


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str
    timestamp: str
    version: str = "1.0.0"
    uptime: float


class ProviderHealth(BaseModel):
    """Provider health status."""
    provider: str
    status: str
    last_check: str
    response_time_ms: float | None = None


class DetailedHealth(BaseModel):
    """Detailed health check response."""
    overall: HealthStatus
    providers: Dict[str, ProviderHealth]


router = APIRouter(tags=["health"])


def create_health_router(provider_factory: ProviderFactory) -> APIRouter:
    """Create health endpoints with provider factory."""
    
    @router.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        import time
        from datetime import datetime
        
        return HealthStatus(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            uptime=time.time()
        )
    
    @router.get("/health/ready")
    async def readiness_check():
        """Readiness check endpoint for Kubernetes."""
        try:
            # Check if at least one provider is available
            available_providers = provider_factory.list_available_providers()
            active_providers = [name for name, enabled in available_providers.items() if enabled]
            
            if not active_providers:
                raise HTTPException(status_code=503, detail="No providers available")
            
            return {"status": "ready", "providers": active_providers}
            
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
    
    @router.get("/health/live")
    async def liveness_check():
        """Liveness check endpoint for Kubernetes."""
        return {"status": "alive"}
    
    @router.get("/health/detailed", response_model=DetailedHealth)
    async def detailed_health_check():
        """Detailed health check with provider status."""
        import time
        from datetime import datetime
        
        try:
            # Check provider health
            provider_health = await provider_factory.health_check_all()
            
            detailed_providers = {}
            for provider_name, is_healthy in provider_health.items():
                detailed_providers[provider_name] = ProviderHealth(
                    provider=provider_name,
                    status="healthy" if is_healthy else "unhealthy",
                    last_check=datetime.utcnow().isoformat(),
                    response_time_ms=None  # Could add response time measurement
                )
            
            overall_status = "healthy" if any(provider_health.values()) else "unhealthy"
            
            return DetailedHealth(
                overall=HealthStatus(
                    status=overall_status,
                    timestamp=datetime.utcnow().isoformat(),
                    uptime=time.time()
                ),
                providers=detailed_providers
            )
            
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
    
    return router