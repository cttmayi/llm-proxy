"""
Models API endpoints for listing available models.
"""
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.providers.base import ModelInfo
from src.providers.factory import ProviderFactory


class ModelData(BaseModel):
    """Model data for OpenAI-compatible response."""
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    """Models response model."""
    object: str = "list"
    data: List[ModelData]


router = APIRouter(prefix="/v1", tags=["models"])


def create_models_router(provider_factory: ProviderFactory) -> APIRouter:
    """Create models endpoints with provider factory."""
    
    @router.get("/models", response_model=ModelsResponse)
    async def list_models():
        """List all available models across all providers."""
        try:
            all_models = []
            
            # Get models from all enabled providers
            for provider_name in provider_factory.list_available_providers():
                if provider_factory.list_available_providers()[provider_name]:
                    try:
                        provider = provider_factory.get_provider(provider_name)
                        models = await provider.list_models()
                        
                        for model in models:
                            all_models.append(ModelData(
                                id=model.id,
                                created=model.created,
                                owned_by=model.owned_by
                            ))
                    except Exception:
                        # Skip providers that fail to list models
                        continue
            
            return ModelsResponse(data=all_models)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/models/available", response_model=Dict[str, List[str]])
    async def get_available_models_by_provider():
        """Get available models grouped by provider."""
        try:
            result = {}
            
            for provider_name in provider_factory.list_available_providers():
                if provider_factory.list_available_providers()[provider_name]:
                    try:
                        provider = provider_factory.get_provider(provider_name)
                        models = await provider.list_models()
                        result[provider_name] = [model.id for model in models]
                    except Exception:
                        result[provider_name] = []
            
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/models/{model_id}", response_model=ModelData)
    async def get_model(model_id: str):
        """Get a specific model by ID."""
        try:
            # Check all providers for the model
            for provider_name in provider_factory.list_available_providers():
                if provider_factory.list_available_providers()[provider_name]:
                    try:
                        provider = provider_factory.get_provider(provider_name)
                        if provider.is_model_supported(model_id):
                            # Return mock model info since we don't have full model details
                            import time
                            return ModelData(
                                id=model_id,
                                created=int(time.time()),
                                owned_by=provider_name
                            )
                    except Exception:
                        continue
            
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return router