"""
Async caching utilities for LLM providers.
"""
import functools
import time
from typing import Any, Callable, Dict, Tuple


def async_cache(ttl: int = 300) -> Callable:
    """
    Async decorator for caching async methods with TTL (Time To Live).
    
    Args:
        ttl: Cache time to live in seconds (default: 300)
    
    Usage:
        @async_cache(ttl=600)
        async def list_models(self) -> List[ModelInfo]:
            # Your API call here
            return result
    
    Features:
        - Caches results based on method name and arguments
        - Automatic cache expiration after TTL
        - Per-instance cache storage
        - Manual cache clearing support
    """
    def decorator(func) -> Callable:
        # Use function-level storage to avoid cross-instance conflicts
        cache_attr = f"_{func.__name__}_cache"
        timestamp_attr = f"_{func.__name__}_cache_timestamp"
        
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs) -> Any:
            # Initialize cache storage if not exists
            if not hasattr(self, cache_attr):
                setattr(self, cache_attr, {})
            if not hasattr(self, timestamp_attr):
                setattr(self, timestamp_attr, {})
            
            cache = getattr(self, cache_attr)
            timestamps = getattr(self, timestamp_attr)
            
            # Create cache key from method name and arguments
            cache_key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if cache exists and is valid
            if cache_key in cache:
                if current_time - timestamps[cache_key] < ttl:
                    return cache[cache_key]
            
            # Execute function and cache result
            result = await func(self, *args, **kwargs)
            cache[cache_key] = result
            timestamps[cache_key] = current_time
            
            return result
        
        # Add cache clearing method
        def clear_cache(self) -> None:
            """Clear cache for this method."""
            if hasattr(self, cache_attr):
                getattr(self, cache_attr).clear()
            if hasattr(self, timestamp_attr):
                getattr(self, timestamp_attr).clear()
        
        wrapper.clear_cache = clear_cache
        return wrapper
    return decorator


def simple_cache(ttl: int = 300) -> Callable:
    """
    Simple cache decorator for non-async methods.
    
    Args:
        ttl: Cache time to live in seconds (default: 300)
    
    Usage:
        @simple_cache(ttl=600)
        def is_model_supported(self, model: str) -> bool:
            # Your logic here
            return result
    """
    def decorator(func) -> Callable:
        cache_attr = f"_{func.__name__}_cache"
        timestamp_attr = f"_{func.__name__}_cache_timestamp"
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # Initialize cache storage if not exists
            if not hasattr(self, cache_attr):
                setattr(self, cache_attr, {})
            if not hasattr(self, timestamp_attr):
                setattr(self, timestamp_attr, {})
            
            cache = getattr(self, cache_attr)
            timestamps = getattr(self, timestamp_attr)
            
            # Create cache key
            cache_key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check cache
            if cache_key in cache:
                if current_time - timestamps[cache_key] < ttl:
                    return cache[cache_key]
            
            # Execute and cache
            result = func(self, *args, **kwargs)
            cache[cache_key] = result
            timestamps[cache_key] = current_time
            
            return result
        
        def clear_cache(self) -> None:
            """Clear cache for this method."""
            if hasattr(self, cache_attr):
                getattr(self, cache_attr).clear()
            if hasattr(self, timestamp_attr):
                getattr(self, timestamp_attr).clear()
        
        wrapper.clear_cache = clear_cache
        return wrapper
    return decorator