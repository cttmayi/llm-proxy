"""
Configuration loader for loading and validating configuration from various sources.
"""
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

from .models import ProxyConfig, ProvidersConfig, ClaudeConfig, OpenAIConfig, AzureOpenAIConfig, ServerConfig, FeaturesConfig, SecurityConfig, ProviderError


class ConfigLoader:
    """Configuration loader that supports multiple sources."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
    
    def _find_config_file(self) -> str:
        """Find the configuration file from common locations."""
        possible_paths = [
            "config/config.json",
            "config.json",
            "proxy_config.json",
            os.path.expanduser("~/.llm-proxy/config.json"),
            "/etc/llm-proxy/config.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return "config/config.json"
    
    def load_from_file(self, config_path: Optional[str] = None) -> ProxyConfig:
        """Load configuration from JSON file."""
        path = config_path or self.config_path
        
        if not os.path.exists(path):
            # Return default configuration if file doesn't exist
            return self._get_default_config()
        
        try:
            with open(path, 'r') as f:
                config_data = json.load(f)
            
            return ProxyConfig(**config_data)
            
        except json.JSONDecodeError as e:
            raise ProviderError(f"Invalid JSON in config file {path}: {str(e)}")
        except Exception as e:
            raise ProviderError(f"Failed to load config from {path}: {str(e)}")
    
    def load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {
            'providers': {},
            'model_mapping': {},
            'server': {},
            'features': {},
            'security': {}
        }
        
        # Load provider configurations from environment
        if os.getenv('CLAUDE_API_KEY'):
            config['providers']['claude'] = {
                'enabled': os.getenv('CLAUDE_ENABLED', 'true').lower() != 'false',
                'api_key': os.getenv('CLAUDE_API_KEY'),
                'base_url': os.getenv('CLAUDE_BASE_URL', 'https://api.anthropic.com'),
                'api_version': os.getenv('CLAUDE_API_VERSION', '2023-06-01')
            }
        
        if os.getenv('OPENAI_API_KEY'):
            config['providers']['openai'] = {
                'enabled': os.getenv('OPENAI_ENABLED', 'true').lower() != 'false',
                'api_key': os.getenv('OPENAI_API_KEY'),
                'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com'),
                'organization': os.getenv('OPENAI_ORGANIZATION')
            }
        
        if os.getenv('AZURE_OPENAI_API_KEY') and os.getenv('AZURE_OPENAI_ENDPOINT'):
            config['providers']['azure'] = {
                'enabled': os.getenv('AZURE_ENABLED', 'true').lower() != 'false',
                'api_key': os.getenv('AZURE_OPENAI_API_KEY'),
                'endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
                'api_version': os.getenv('AZURE_API_VERSION', '2024-10-21')
            }
        
        # Load server configuration
        if os.getenv('LLM_PROXY_HOST'):
            config['server']['host'] = os.getenv('LLM_PROXY_HOST')
        if os.getenv('LLM_PROXY_PORT'):
            config['server']['port'] = int(os.getenv('LLM_PROXY_PORT'))
        if os.getenv('LLM_PROXY_LOG_LEVEL'):
            config['server']['log_level'] = os.getenv('LLM_PROXY_LOG_LEVEL')
        
        # Load model mapping from environment
        model_mapping_env = os.getenv('LLM_PROXY_MODEL_MAPPING')
        if model_mapping_env:
            try:
                config['model_mapping'] = json.loads(model_mapping_env)
            except json.JSONDecodeError:
                # Parse simple key=value pairs
                mapping = {}
                for pair in model_mapping_env.split(','):
                    if '=' in pair:
                        model, provider = pair.split('=', 1)
                        mapping[model.strip()] = provider.strip()
                config['model_mapping'] = mapping
        
        # Filter out empty sections
        return {k: v for k, v in config.items() if v}
    
    def load_config(self, config_path: Optional[str] = None) -> ProxyConfig:
        """Load configuration from all sources with proper merging."""
        # Start with file configuration
        file_config = self.load_from_file(config_path)
        
        # # Load environment configuration
        # env_config = self.load_from_env()
        
        # # Merge configurations (environment takes precedence)
        # merged_config = self._merge_configs(file_config.dict(), env_config)
        
        # return ProxyConfig(**merged_config)

        return file_config
    
    def _merge_configs(self, file_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge file and environment configurations."""
        merged = file_config.copy()
        
        # Merge providers
        if 'providers' in env_config:
            if 'providers' not in merged:
                merged['providers'] = {}
            merged['providers'].update(env_config['providers'])
        
        # Merge other sections
        for section in ['model_mapping', 'server', 'features', 'security']:
            if section in env_config:
                if section not in merged:
                    merged[section] = {}
                if isinstance(env_config[section], dict):
                    merged[section].update(env_config[section])
                else:
                    merged[section] = env_config[section]
        
        return merged
    
    def _get_default_config(self) -> ProxyConfig:
        """Get default configuration when no config file exists."""
        return ProxyConfig(
            providers=ProvidersConfig(),
            model_mapping={},
            server=ServerConfig(),
            features=FeaturesConfig(),
            security=SecurityConfig()
        )
    
    def create_config_template(self, output_path: str) -> None:
        """Create a configuration template file."""
        default_config = ProxyConfig(
            providers=ProvidersConfig(
                claude=ClaudeConfig(
                    enabled=True,
                    api_key="your-claude-api-key-here",
                    base_url="https://api.anthropic.com",
                    api_version="2023-06-01"
                ),
                openai=OpenAIConfig(
                    enabled=True,
                    api_key="your-openai-api-key-here",
                    base_url="https://api.openai.com",
                    organization=None
                ),
                azure=AzureOpenAIConfig(
                    enabled=True,
                    api_key="your-azure-openai-api-key-here",
                    endpoint="https://your-resource.openai.azure.com",
                    api_version="2024-10-21"
                )
            ),
            model_mapping={
                "gpt-4o": "openai",
                "claude-3-5-sonnet": "claude",
                "gpt-4": "azure"
            },
            server=ServerConfig(
                host="0.0.0.0",
                port=8000,
                log_level="INFO",
                workers=1
            ),
            features=FeaturesConfig(
                enable_streaming=True,
                enable_caching=False,
                enable_metrics=True,
                enable_rate_limiting=False,
                max_requests_per_minute=60
            ),
            security=SecurityConfig(
                enable_auth=False,
                api_key_header="X-API-Key",
                allowed_origins=["*"],
                enable_cors=True
            )
        )
        
        config_dict = default_config.dict()
        
        # Remove API keys from template
        if 'providers' in config_dict:
            for provider in config_dict['providers'].values():
                if provider and 'api_key' in provider:
                    provider['api_key'] = 'your-api-key-here'
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate_config(self, config: ProxyConfig) -> bool:
        """Validate the configuration."""
        try:
            # Check that at least one provider is configured
            providers = config.providers.dict()
            active_providers = [
                name for name, provider in providers.items() 
                if provider and provider.get('enabled') and provider.get('api_key')
            ]
            
            if not active_providers:
                raise ProviderError("At least one provider must be configured with valid API key")
            
            return True
            
        except Exception as e:
            raise ProviderError(f"Configuration validation failed: {str(e)}")
    
    def get_config_summary(self, config: ProxyConfig) -> Dict[str, Any]:
        """Get a summary of the configuration for logging."""
        providers = config.providers.dict()
        active_providers = [
            name for name, provider in providers.items() 
            if provider and provider.get('enabled') and provider.get('api_key')
        ]
        
        return {
            'active_providers': active_providers,
            'model_mapping_count': len(config.model_mapping),
            'server_config': {
                'host': config.server.host,
                'port': config.server.port,
                'log_level': config.server.log_level
            },
            'features': {
                'streaming': config.features.enable_streaming,
                'caching': config.features.enable_caching,
                'metrics': config.features.enable_metrics,
                'rate_limiting': config.features.enable_rate_limiting
            }
        }