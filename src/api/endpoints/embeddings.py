"""
Embeddings API endpoints.
"""
from typing import Dict, Any, Union, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.providers.base import EmbeddingRequest, ProviderError
from src.providers.factory import ProviderFactory


class EmbeddingRequestModel(BaseModel):
    """Embedding request model."""
    model: str
    input: Union[str, List[str]]


class EmbeddingData(BaseModel):
    """Embedding data model."""
    object: str
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Embedding response model."""
    object: str
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]


router = APIRouter(prefix="/v1", tags=["embeddings"])


def create_embeddings_router(provider_factory: ProviderFactory) -> APIRouter:
    """Create embeddings endpoints with provider factory."""
    
    @router.post("/embeddings", response_model=EmbeddingResponse)
    async def create_embeddings(request: EmbeddingRequestModel):
        """Create embeddings for the given input."""
        try:
            provider = provider_factory.get_provider_for_model(request.model)
            
            # Check if provider supports embeddings
            if hasattr(provider, 'create_embeddings'):
                embedding_request = EmbeddingRequest(
                    model=request.model,
                    input=request.input
                )
                
                response = await provider.create_embeddings(embedding_request)
                
                # Convert to OpenAI format
                return EmbeddingResponse(
                    object=response.object,
                    data=[
                        EmbeddingData(
                            object=item["object"],
                            embedding=item["embedding"],
                            index=item["index"]
                        )
                        for item in response.data
                    ],
                    model=response.model,
                    usage=response.usage
                )
            else:
                raise HTTPException(
                    status_code=501,
                    detail=f"Provider for model {request.model} does not support embeddings"
                )
                
        except ProviderError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return router