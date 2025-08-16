"""
Integration tests for models endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.main import create_app


@pytest.fixture
def test_config():
    return {
        'providers': {
            'claude': {
                'enabled': True,
                'api_key': 'test-claude-key',
                'base_url': 'https://api.anthropic.com'
            },
            'openai': {
                'enabled': True,
                'api_key': 'test-openai-key',
                'base_url': 'https://api.openai.com'
            }
        },
        'model_mapping': {
            'gpt-4o': 'openai',
            'claude-3-5-sonnet': 'claude'
        }
    }


@pytest.fixture
def client(test_config):
    app = create_app(test_config)
    return TestClient(app)


class TestModelsEndpoints:
    
    def test_list_models_success(self, client):
        """Test successful models listing."""
        claude_models = [
            {
                'id': 'claude-3-5-sonnet-20241022',
                'created': 1700000000,
                'owned_by': 'anthropic',
                'provider': 'claude'
            },
            {
                'id': 'claude-3-5-haiku-20241022',
                'created': 1700000001,
                'owned_by': 'anthropic',
                'provider': 'claude'
            }
        ]
        
        openai_models = [
            {
                'id': 'gpt-4o',
                'created': 1700000002,
                'owned_by': 'openai',
                'provider': 'openai'
            },
            {
                'id': 'gpt-3.5-turbo',
                'created': 1700000003,
                'owned_by': 'openai',
                'provider': 'openai'
            }
        ]
        
        with patch('src.providers.claude.ClaudeProvider.list_models') as mock_claude_list, \
             patch('src.providers.openai.OpenAIProvider.list_models') as mock_openai_list:
            
            from src.providers.base import ModelInfo
            mock_claude_list.return_value = [ModelInfo(**model) for model in claude_models]
            mock_openai_list.return_value = [ModelInfo(**model) for model in openai_models]
            
            response = client.get("/v1/models")
            
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert len(data["data"]) == 4
            
            model_ids = [model["id"] for model in data["data"]]
            assert "claude-3-5-sonnet-20241022" in model_ids
            assert "gpt-4o" in model_ids
    
    def test_get_model_success(self, client):
        """Test getting a specific model."""
        response = client.get("/v1/models/gpt-4o")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "gpt-4o"
        assert data["owned_by"] == "openai"
    
    def test_get_model_not_found(self, client):
        """Test getting a non-existent model."""
        response = client.get("/v1/models/nonexistent-model")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_get_available_models_by_provider(self, client):
        """Test getting models grouped by provider."""
        claude_models = [
            {
                'id': 'claude-3-5-sonnet-20241022',
                'created': 1700000000,
                'owned_by': 'anthropic',
                'provider': 'claude'
            }
        ]
        
        openai_models = [
            {
                'id': 'gpt-4o',
                'created': 1700000002,
                'owned_by': 'openai',
                'provider': 'openai'
            }
        ]
        
        with patch('src.providers.claude.ClaudeProvider.list_models') as mock_claude_list, \
             patch('src.providers.openai.OpenAIProvider.list_models') as mock_openai_list:
            
            from src.providers.base import ModelInfo
            mock_claude_list.return_value = [ModelInfo(**model) for model in claude_models]
            mock_openai_list.return_value = [ModelInfo(**model) for model in openai_models]
            
            response = client.get("/v1/models/available")
            
            assert response.status_code == 200
            data = response.json()
            assert "claude" in data
            assert "openai" in data
            assert data["claude"] == ["claude-3-5-sonnet-20241022"]
            assert data["openai"] == ["gpt-4o"]
    
    def test_list_models_provider_error(self, client):
        """Test models listing when providers fail."""
        with patch('src.providers.claude.ClaudeProvider.list_models') as mock_claude_list, \
             patch('src.providers.openai.OpenAIProvider.list_models') as mock_openai_list:
            
            mock_claude_list.side_effect = Exception("Provider error")
            mock_openai_list.return_value = []
            
            response = client.get("/v1/models")
            
            # Should still return successfully, just with empty data for failed providers
            assert response.status_code == 200
            data = response.json()
            assert "data" in data


class TestHealthEndpoints:
    
    def test_health_check_basic(self, client):
        """Test basic health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_readiness_check_success(self, client):
        """Test readiness check with available providers."""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "providers" in data
        assert len(data["providers"]) > 0
    
    def test_liveness_check(self, client):
        """Test liveness check."""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
    
    def test_detailed_health_check(self, client):
        """Test detailed health check."""
        with patch('src.providers.claude.ClaudeProvider.health_check') as mock_claude_health, \
             patch('src.providers.openai.OpenAIProvider.health_check') as mock_openai_health:
            
            mock_claude_health.return_value = True
            mock_openai_health.return_value = False
            
            response = client.get("/health/detailed")
            
            assert response.status_code == 200
            data = response.json()
            assert "overall" in data
            assert "providers" in data
            assert data["providers"]["claude"]["status"] == "healthy"
            assert data["providers"]["openai"]["status"] == "unhealthy"