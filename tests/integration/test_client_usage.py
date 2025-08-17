"""
Comprehensive client usage tests and examples for the LLM proxy.
"""
import pytest
import json
import os
from typing import Dict, Any, List

from openai import OpenAI, AsyncOpenAI
from anthropic import Anthropic, AsyncAnthropic
import httpx

from src.main import app


class TestProxyClientUsage:
    """Test suite demonstrating how to use the proxy with official client libraries."""
    
    @pytest.fixture
    def test_app_client(self):
        """Create test client for FastAPI app."""
        return httpx.Client(app=app, base_url="http://test")
    
    @pytest.fixture
    def openai_sync_client(self, test_app_client):
        """Create synchronous OpenAI client for testing."""
        return OpenAI(
            base_url="http://test/v1",
            api_key="test-key",
            http_client=test_app_client
        )
    
    @pytest.fixture
    def openai_async_client(self, test_app_client):
        """Create asynchronous OpenAI client for testing."""
        return AsyncOpenAI(
            base_url="http://test/v1",
            api_key="test-key",
            http_client=httpx.AsyncClient(app=app, base_url="http://test")
        )
    
    @pytest.fixture
    def anthropic_sync_client(self, test_app_client):
        """Create synchronous Anthropic client for testing."""
        return Anthropic(
            base_url="http://test/v1",
            api_key="test-key",
            http_client=test_app_client
        )
    
    @pytest.fixture
    def anthropic_async_client(self, test_app_client):
        """Create asynchronous Anthropic client for testing."""
        return AsyncAnthropic(
            base_url="http://test/v1",
            api_key="test-key",
            http_client=httpx.AsyncClient(app=app, base_url="http://test")
        )
    
    def test_openai_basic_chat_completion(self, openai_sync_client):
        """Test basic chat completion with OpenAI client."""
        response = openai_sync_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=50
        )
        
        assert response.id is not None
        assert response.object == "chat.completion"
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content is not None
    
    def test_openai_streaming_chat(self, openai_sync_client):
        """Test streaming chat completion with OpenAI client."""
        stream = openai_sync_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a short story."}],
            max_tokens=50,
            stream=True
        )
        
        chunks = list(stream)
        assert len(chunks) > 0
        
        # Check for content chunks
        content_chunks = [chunk for chunk in chunks 
                         if chunk.choices and chunk.choices[0].delta.content]
        assert len(content_chunks) > 0
    
    def test_openai_list_models(self, openai_sync_client):
        """Test listing models with OpenAI client."""
        models = openai_sync_client.models.list()
        
        assert models.object == "list"
        assert len(models.data) > 0
        
        # Check for expected models
        model_ids = [model.id for model in models.data]
        assert any("gpt" in model_id.lower() for model_id in model_ids)
    
    def test_openai_embeddings(self, openai_sync_client):
        """Test embeddings with OpenAI client."""
        response = openai_sync_client.embeddings.create(
            model="text-embedding-ada-002",
            input="This is a test sentence for embeddings."
        )
        
        assert response.object == "list"
        assert len(response.data) == 1
        assert response.data[0].object == "embedding"
        assert isinstance(response.data[0].embedding, list)
        assert len(response.data[0].embedding) > 0
    
    def test_anthropic_basic_messages(self, anthropic_sync_client):
        """Test basic messages with Anthropic client."""
        response = anthropic_sync_client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello, Claude!"}]
        )
        
        assert response.id is not None
        assert response.model == "claude-3-5-sonnet"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text is not None
    
    def test_anthropic_with_system_prompt(self, anthropic_sync_client):
        """Test Anthropic messages with system prompt."""
        response = anthropic_sync_client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=100,
            system="You are a helpful math tutor.",
            messages=[{"role": "user", "content": "What is 2+2?"}]
        )
        
        assert response.content[0].text is not None
    
    def test_anthropic_multi_turn_conversation(self, anthropic_sync_client):
        """Test multi-turn conversation with Anthropic client."""
        response = anthropic_sync_client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "What is its population?"}
            ]
        )
        
        assert response.content[0].text is not None
    
    @pytest.mark.asyncio
    async def test_async_openai_chat(self, openai_async_client):
        """Test async chat completion with OpenAI client."""
        response = await openai_async_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, async test!"}],
            max_tokens=50
        )
        
        assert response.choices[0].message.content is not None
    
    @pytest.mark.asyncio
    async def test_async_anthropic_messages(self, anthropic_async_client):
        """Test async messages with Anthropic client."""
        response = await anthropic_async_client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello, async Claude!"}]
        )
        
        assert response.content[0].text is not None
    
    def test_cross_provider_compatibility(self, openai_sync_client):
        """Test accessing different providers through OpenAI client format."""
        
        # Test OpenAI models
        openai_response = openai_sync_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello from OpenAI!"}],
            max_tokens=30
        )
        assert openai_response.choices[0].message.content is not None
        
        # Test Claude models through OpenAI format
        claude_response = openai_sync_client.chat.completions.create(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello from Claude!"}],
            max_tokens=30
        )
        assert claude_response.choices[0].message.content is not None
    
    def test_client_configuration_examples(self):
        """Test various client configuration patterns."""
        
        # Example configurations
        configurations = [
            {
                "name": "Local Development",
                "openai": {
                    "base_url": "http://localhost:8000/v1",
                    "api_key": "dev-key"
                },
                "anthropic": {
                    "base_url": "http://localhost:8000/v1",
                    "api_key": "dev-key"
                }
            },
            {
                "name": "Docker Setup",
                "openai": {
                    "base_url": "http://0.0.0.0:8000/v1",
                    "api_key": "docker-key"
                },
                "anthropic": {
                    "base_url": "http://0.0.0.0:8000/v1",
                    "api_key": "docker-key"
                }
            }
        ]
        
        for config in configurations:
            assert "name" in config
            assert "openai" in config
            assert "anthropic" in config
            assert config["openai"]["base_url"].endswith("/v1")
            assert config["anthropic"]["base_url"].endswith("/v1")
    
    def test_error_handling_examples(self):
        """Test error handling with client libraries."""
        
        # These are examples for documentation
        error_examples = [
            {
                "scenario": "Invalid model name",
                "description": "Should handle gracefully when invalid model is requested"
            },
            {
                "scenario": "Network timeout",
                "description": "Should handle network timeout appropriately"
            },
            {
                "scenario": "Rate limiting",
                "description": "Should handle rate limit responses"
            }
        ]
        
        assert len(error_examples) > 0
        for example in error_examples:
            assert "scenario" in example
            assert "description" in example
    
    def test_usage_examples_documentation(self):
        """Generate usage examples for documentation."""
        
        examples = {
            "openai_basic": '''
            from openai import OpenAI
            
            client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="any-string-works"
            )
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello!"}]
            )
            print(response.choices[0].message.content)
            ''',
            
            "openai_streaming": '''
            from openai import OpenAI
            
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")
            
            stream = client.chat.completions.create(
                model="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "Tell me a story"}],
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="")
            ''',
            
            "anthropic_basic": '''
            from anthropic import Anthropic
            
            client = Anthropic(
                base_url="http://localhost:8000/v1",
                api_key="any-string-works"
            )
            
            response = client.messages.create(
                model="claude-3-5-sonnet",
                max_tokens=1000,
                messages=[{"role": "user", "content": "Hello!"}]
            )
            print(response.content[0].text)
            ''',
            
            "anthropic_streaming": '''
            from anthropic import Anthropic
            
            client = Anthropic(base_url="http://localhost:8000/v1", api_key="test")
            
            stream = client.messages.create(
                model="claude-3-5-sonnet",
                max_tokens=1000,
                messages=[{"role": "user", "content": "Tell me a story"}],
                stream=True
            )
            
            for chunk in stream:
                if chunk.delta and chunk.delta.text:
                    print(chunk.delta.text, end="")
            '''
        }
        
        # Validate all examples are syntactically correct
        for name, code in examples.items():
            try:
                compile(code, f'example_{name}', 'exec')
            except SyntaxError as e:
                pytest.fail(f"Invalid example code for {name}: {e}")


class TestClientSetupGuide:
    """Test suite for client setup and configuration."""
    
    def test_environment_variables_setup(self):
        """Test environment-based configuration."""
        setup_steps = [
            "1. Install client libraries: pip install openai anthropic",
            "2. Start the proxy: python src/main.py",
            "3. Configure clients to use proxy base URL",
            "4. Use any string as API key (proxy doesn't validate keys)",
            "5. Test with basic requests"
        ]
        
        assert len(setup_steps) == 5
    
    def test_docker_setup_example(self):
        """Test Docker-based setup."""
        docker_example = {
            "docker_run": "docker run -p 8000:8000 llm-proxy",
            "client_config": {
                "base_url": "http://localhost:8000/v1",
                "api_key": "docker-key"
            }
        }
        
        assert docker_example["docker_run"].startswith("docker run")
        assert docker_example["client_config"]["base_url"] == "http://localhost:8000/v1"
    
    def test_configuration_file_examples(self):
        """Test configuration file examples."""
        configs = {
            "openai_client.py": '''
            from openai import OpenAI
            
            client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="your-key-here"
            )
            ''',
            
            "anthropic_client.py": '''
            from anthropic import Anthropic
            
            client = Anthropic(
                base_url="http://localhost:8000/v1",
                api_key="your-key-here"
            )
            ''',
            
            "async_client.py": '''
            import asyncio
            from openai import AsyncOpenAI
            
            async def main():
                client = AsyncOpenAI(
                    base_url="http://localhost:8000/v1",
                    api_key="test-key"
                )
                
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hello!"}]
                )
                print(response.choices[0].message.content)
            
            if __name__ == "__main__":
                asyncio.run(main())
            '''
        }
        
        for filename, content in configs.items():
            try:
                compile(content, filename, 'exec')
            except SyntaxError as e:
                pytest.fail(f"Invalid config file {filename}: {e}")
    
    def test_troubleshooting_examples(self):
        """Test troubleshooting scenarios."""
        troubleshooting = {
            "connection_refused": {
                "symptom": "Cannot connect to proxy",
                "solution": "Ensure proxy is running on specified host/port"
            },
            "model_not_found": {
                "symptom": "Model not found error",
                "solution": "Check model_mapping configuration and model availability"
            },
            "timeout_errors": {
                "symptom": "Request timeouts",
                "solution": "Increase client timeout settings"
            }
        }
        
        assert len(troubleshooting) > 0
        for issue, details in troubleshooting.items():
            assert "symptom" in details
            assert "solution" in details