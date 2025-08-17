"""
Example usage tests showing how to use the proxy with official client libraries.
"""
import pytest
import json
from typing import Dict, Any

from tests.integration.test_openai_client import TestOpenAIClientIntegration
from tests.integration.test_anthropic_client import TestAnthropicClientIntegration


class TestClientUsageExamples:
    """Test suite with practical usage examples for client libraries."""
    
    def test_openai_client_basic_example(self):
        """Basic OpenAI client usage example."""
        # This is an example of how to use the OpenAI client with the proxy
        # In actual tests, this would use the test fixtures
        
        example_code = '''
        from openai import OpenAI
        
        # Configure client to use the proxy
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="your-api-key-here"  # Can be any non-empty string
        )
        
        # Chat completion
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        
        print(response.choices[0].message.content)
        '''
        
        # Verify the example is valid Python
        try:
            compile(example_code, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Invalid example code: {e}")
    
    def test_openai_client_streaming_example(self):
        """OpenAI client streaming example."""
        example_code = '''
        from openai import OpenAI
        
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="your-api-key-here"
        )
        
        # Streaming response
        stream = client.chat.completions.create(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Tell me a story"}],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        '''
        
        try:
            compile(example_code, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Invalid example code: {e}")
    
    def test_anthropic_client_basic_example(self):
        """Basic Anthropic client usage example."""
        example_code = '''
        from anthropic import Anthropic
        
        # Configure client to use the proxy
        client = Anthropic(
            base_url="http://localhost:8000/v1",
            api_key="your-api-key-here"
        )
        
        # Claude message
        response = client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "Hello, Claude!"}
            ]
        )
        
        print(response.content[0].text)
        '''
        
        try:
            compile(example_code, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Invalid example code: {e}")
    
    def test_anthropic_client_streaming_example(self):
        """Anthropic client streaming example."""
        example_code = '''
        from anthropic import Anthropic
        
        client = Anthropic(
            base_url="http://localhost:8000/v1",
            api_key="your-api-key-here"
        )
        
        # Streaming response
        stream = client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Tell me a story"}],
            stream=True
        )
        
        for chunk in stream:
            if chunk.delta and chunk.delta.text:
                print(chunk.delta.text, end="")
        '''
        
        try:
            compile(example_code, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Invalid example code: {e}")
    
    def test_configuration_examples(self):
        """Test configuration examples."""
        examples = [
            {
                "name": "OpenAI Client Configuration",
                "config": {
                    "base_url": "http://localhost:8000/v1",
                    "api_key": "dummy-key",
                    "organization": None,
                    "timeout": 30.0
                }
            },
            {
                "name": "Anthropic Client Configuration",
                "config": {
                    "base_url": "http://localhost:8000/v1",
                    "api_key": "dummy-key",
                    "timeout": 30.0
                }
            }
        ]
        
        for example in examples:
            assert "base_url" in example["config"]
            assert example["config"]["base_url"].endswith("/v1")
            assert "api_key" in example["config"]
    
    def test_error_handling_examples(self):
        """Test error handling examples."""
        examples = [
            {
                "scenario": "Invalid model",
                "model": "invalid-model-name",
                "expected_error": "model_not_found"
            },
            {
                "scenario": "Missing API key",
                "model": "gpt-4o",
                "config": {"base_url": "http://localhost:8000/v1"},
                "expected_error": "authentication_error"
            }
        ]
        
        # These are examples for documentation purposes
        assert len(examples) > 0
        for example in examples:
            assert "scenario" in example
            assert "model" in example
            assert "expected_error" in example
    
    def test_environment_setup_examples(self):
        """Test environment setup examples."""
        setup_examples = [
            {
                "description": "Basic proxy setup",
                "steps": [
                    "1. Install dependencies: pip install openai anthropic",
                    "2. Start proxy: python src/main.py",
                    "3. Configure client base_url to http://localhost:8000/v1",
                    "4. Use any string as api_key (e.g., 'test-key')"
                ]
            },
            {
                "description": "Custom host/port setup",
                "steps": [
                    "1. Start proxy: python src/main.py --host 0.0.0.0 --port 8080",
                    "2. Configure client base_url to http://0.0.0.0:8080/v1",
                    "3. Use client as normal"
                ]
            }
        ]
        
        for example in setup_examples:
            assert "description" in example
            assert "steps" in example
            assert len(example["steps"]) > 0