"""
API routes for the API monitor.
"""
from typing import List, Optional, Union
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from datetime import datetime

from src.web.monitor import api_monitor

router = APIRouter(tags=["monitor"])


class APICallResponse(BaseModel):
    """Response model for API call."""
    id: str
    timestamp: float
    method: str
    path: str
    headers: Union[dict, str]
    request_body: Optional[Union[dict, str]]
    response_body: Optional[Union[dict, str]]
    status_code: int
    duration_ms: float
    model: Optional[str]
    provider: Optional[str]
    error: Optional[str]


class APICallsResponse(BaseModel):
    """Response model for API calls list."""
    calls: List[APICallResponse]
    total: int


class StatsResponse(BaseModel):
    """Response model for statistics."""
    total_calls: int
    error_rate: float
    avg_duration: float
    model_stats: dict
    provider_stats: dict
    last_call_time: Optional[float]


@router.get("/calls", response_model=APICallsResponse)
async def get_api_calls(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    model: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """Get API calls with optional filtering."""
    calls = []
    
    if model:
        calls = api_monitor.get_calls_by_model(model, limit + offset)
    elif status == "success":
        all_calls = api_monitor.get_calls(limit + offset)
        calls = [c for c in all_calls if c.status_code < 400]
    elif status == "error":
        all_calls = api_monitor.get_calls(limit + offset)
        calls = [c for c in all_calls if c.status_code >= 400]
    else:
        calls = api_monitor.get_calls(limit + offset)
    
    # Apply offset
    calls = calls[offset:offset + limit]
    
    # Convert to response model
    response_calls = []
    for call in calls:
        response_calls.append(APICallResponse(
            id=call.id,
            timestamp=call.timestamp,
            method=call.method,
            path=call.path,
            headers=call.headers,
            request_body=call.request_body,
            response_body=call.response_body,
            status_code=call.status_code,
            duration_ms=call.duration_ms,
            model=call.model,
            provider=call.provider,
            error=call.error
        ))
    
    return APICallsResponse(
        calls=response_calls,
        total=len(api_monitor.calls)
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API usage statistics."""
    stats = api_monitor.get_stats()
    return StatsResponse(**stats)


@router.delete("/calls")
async def clear_calls():
    """Clear all API call records."""
    api_monitor.clear()
    return {"message": "API calls cleared"}


@router.get("/calls/{call_id}", response_model=APICallResponse)
async def get_call_detail(call_id: str):
    """Get details of a specific API call."""
    calls = api_monitor.get_calls(limit=1000)
    call = next((c for c in calls if c.id == call_id), None)
    
    if not call:
        raise HTTPException(status_code=404, detail="API call not found")
    
    return APICallResponse(
        id=call.id,
        timestamp=call.timestamp,
        method=call.method,
        path=call.path,
        headers=call.headers,
        request_body=call.request_body,
        response_body=call.response_body,
        status_code=call.status_code,
        duration_ms=call.duration_ms,
        model=call.model,
        provider=call.provider,
        error=call.error
    )