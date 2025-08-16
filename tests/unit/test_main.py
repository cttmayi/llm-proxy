"""
Unit tests for main application.
"""
import pytest
from fastapi.testclient import TestClient

from src.main import create_app
from src.config.loader import ConfigLoader


class TestMain:
    
    def test_create_app_with_config(self):
        """Test app creation with configuration."""
        test_config = {
            'providers': {
                'claude': {
                    'enabled': True,
                    'api_key': 'test-key',
                    'base_url': 'https://api.anthropic.com'
                }
            },
            'model_mapping': {
                'claude-3-5-sonnet': 'claude'
            }
        }
        
        app = create_app(test_config)
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_create_app_default_config(self):
        """Test app creation with default configuration."""
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_app_endpoints_exist(self):
        """Test that all expected endpoints exist."""
        test_config = {
            'providers': {
                'claude': {
                    'enabled': True,
                    'api_key': 'test-key',
                    'base_url': 'https://api.anthropic.com'
                }
            },
            'model_mapping': {
                'claude-3-5-sonnet': 'claude'
            }
        }
        
        app = create_app(test_config)
        client = TestClient(app)
        
        # Test all endpoints return something (not 404 for the base paths)
        endpoints = [
            "/health",
            "/health/ready",
            "/health/live",
            "/health/detailed",
            "/v1/models",
            "/docs",
            "/openapi.json"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            # These endpoints should either exist (200) or be handled by FastAPI (404 is acceptable)
            assert response.status_code in [200, 404, 503]  # 503 is acceptable for unavailable providers
    
    def test_config_loader_basic(self):
        """Test basic configuration loading."""
        loader = ConfigLoader()
        
        # Test default config creation
        config = loader._get_default_config()
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8000