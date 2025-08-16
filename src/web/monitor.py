"""
API monitoring system to capture and display actual request/response data.
"""
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from threading import Lock


@dataclass
class APICall:
    """Represents a single API call with request and response data."""
    id: str
    timestamp: float
    method: str
    path: str
    headers: Dict[str, str]
    request_body: Any
    response_body: Any
    status_code: int
    duration_ms: float
    model: Optional[str] = None
    provider: Optional[str] = None
    error: Optional[str] = None


class APIMonitor:
    """Singleton class to monitor and store API calls."""
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the monitor."""
        self.calls: List[APICall] = []
        self.max_calls = 1000  # Keep last 1000 calls
        self.active = True
    
    def add_call(self, call: APICall):
        """Add an API call to the monitor."""
        with self._lock:
            self.calls.append(call)
            if len(self.calls) > self.max_calls:
                self.calls.pop(0)
    
    def get_calls(self, limit: int = 50, offset: int = 0) -> List[APICall]:
        """Get recent API calls."""
        with self._lock:
            return self.calls[-(offset + limit):-offset or None]
    
    def get_calls_by_model(self, model: str, limit: int = 50) -> List[APICall]:
        """Get API calls for a specific model."""
        with self._lock:
            return [call for call in self.calls if call.model == model][-limit:]
    
    def get_calls_by_provider(self, provider: str, limit: int = 50) -> List[APICall]:
        """Get API calls for a specific provider."""
        with self._lock:
            return [call for call in self.calls if call.provider == provider][-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        with self._lock:
            if not self.calls:
                return {"total_calls": 0, "error_rate": 0, "avg_duration": 0}
            
            total_calls = len(self.calls)
            error_calls = len([call for call in self.calls if call.status_code >= 400])
            total_duration = sum(call.duration_ms for call in self.calls)
            
            model_stats = {}
            provider_stats = {}
            
            for call in self.calls:
                if call.model:
                    model_stats[call.model] = model_stats.get(call.model, 0) + 1
                if call.provider:
                    provider_stats[call.provider] = provider_stats.get(call.provider, 0) + 1
            
            return {
                "total_calls": total_calls,
                "error_rate": error_calls / total_calls,
                "avg_duration": total_duration / total_calls,
                "model_stats": model_stats,
                "provider_stats": provider_stats,
                "last_call_time": self.calls[-1].timestamp if self.calls else None
            }
    
    def get_last_call(self) -> Optional[APICall]:
        """Get the most recent API call."""
        with self._lock:
            return self.calls[-1] if self.calls else None
    
    def clear(self):
        """Clear all stored calls."""
        with self._lock:
            self.calls.clear()


# Global monitor instance
api_monitor = APIMonitor()