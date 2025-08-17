# Client Library Integration Tests

This directory contains comprehensive tests demonstrating how to use the LLM proxy with official client libraries from OpenAI and Anthropic.

## Test Files

### 1. `test_openai_client.py`
Tests the proxy using the official OpenAI Python client library.

**Features tested:**
- Synchronous chat completions
- Streaming chat completions
- Model listing
- Embeddings generation
- Error handling
- Cross-provider model access

### 2. `test_anthropic_client.py`
Tests the proxy using the official Anthropic Python client library.

**Features tested:**
- Synchronous Claude messages
- Streaming Claude messages
- Multi-turn conversations
- System prompt handling
- Error handling

### 3. `test_client_usage.py`
Comprehensive client usage examples and setup guides.

**Includes:**
- Complete usage examples
- Configuration patterns
- Docker setup
- Troubleshooting guides
- Best practices

### 4. `test_client_examples.py`
Practical examples and documentation snippets.

**Features:**
- Copy-paste ready code examples
- Configuration templates
- Environment setup guides
- Error handling patterns

## Prerequisites

Install the required client libraries:

```bash
pip install openai anthropic pytest-asyncio
```

## Running the Tests

### Run all client tests:
```bash
pytest tests/integration/test_client_usage.py -v
```

### Run specific client tests:
```bash
# OpenAI client tests
pytest tests/integration/test_openai_client.py -v

# Anthropic client tests
pytest tests/integration/test_anthropic_client.py -v

# Usage examples
pytest tests/integration/test_client_examples.py -v
```

### Run async tests:
```bash
pytest tests/integration/test_client_usage.py::TestProxyClientUsage::test_async_openai_chat -v
```

## Usage Examples

### Basic OpenAI Client Usage

```python
from openai import OpenAI

# Configure client to use the proxy
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="any-string-works"  # Proxy doesn't validate keys
)

# Chat completion
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Streaming with OpenAI Client

```python
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
```

### Basic Anthropic Client Usage

```python
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
```

### Async Usage

```python
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

asyncio.run(main())
```

## Configuration Patterns

### Environment Variables

```bash
# Set proxy endpoint
export PROXY_BASE_URL=http://localhost:8000/v1

# Use in Python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("PROXY_BASE_URL"),
    api_key="test-key"
)
```

### Configuration Classes

```python
class ProxyConfig:
    def __init__(self, base_url="http://localhost:8000/v1"):
        self.base_url = base_url
        self.api_key = "test-key"

config = ProxyConfig()
client = OpenAI(base_url=config.base_url, api_key=config.api_key)
```

## Docker Usage

### Running with Docker

```bash
# Start proxy in Docker
docker run -p 8000:8000 llm-proxy

# Configure client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="docker-key"
)
```

### Docker Compose

```yaml
version: '3.8'
services:
  llm-proxy:
    image: llm-proxy
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=your-openai-key
      - CLAUDE_API_KEY=your-claude-key
```

## Testing Best Practices

### 1. Mock vs Real API Tests
- **Mock tests**: Use for unit testing without external dependencies
- **Integration tests**: Use real client libraries with test proxy
- **End-to-end tests**: Test with actual API keys (separate test suite)

### 2. Test Isolation
Each test should:
- Use separate client instances
- Clean up any state
- Not depend on test order

### 3. Error Handling
Test various error scenarios:
- Invalid model names
- Network timeouts
- Rate limiting
- Authentication errors

### 4. Performance Testing
Include tests for:
- Streaming performance
- Large payload handling
- Concurrent requests

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure proxy is running
   - Check host/port configuration
   - Verify firewall settings

2. **Model Not Found**
   - Check model_mapping configuration
   - Verify provider supports the model
   - Check API key permissions

3. **Timeout Errors**
   - Increase client timeout settings
   - Check network connectivity
   - Monitor proxy response times

4. **Authentication Issues**
   - Ensure API keys are properly configured
   - Check provider-specific requirements
   - Verify key permissions

### Debug Mode

Enable detailed logging:

```python
import logging
from openai import OpenAI

logging.basicConfig(level=logging.DEBUG)

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="test-key"
)
```

## Performance Considerations

### Connection Pooling

```python
from openai import OpenAI
import httpx

http_client = httpx.Client(
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
)

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="test-key",
    http_client=http_client
)
```

### Async Performance

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="test-key",
    max_retries=3,
    timeout=30.0
)
```

## Contributing

When adding new client tests:

1. Follow existing test patterns
2. Add comprehensive error handling
3. Include both sync and async versions
4. Update documentation
5. Add usage examples

## Quick Start Script

Create a `test_clients.py` file:

```python
#!/usr/bin/env python3
"""Quick test script for client library integration."""

import asyncio
from openai import OpenAI, AsyncOpenAI

def test_openai_sync():
    """Test OpenAI sync client."""
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=50
    )
    
    print("OpenAI Sync Test:", response.choices[0].message.content[:50] + "...")

async def test_openai_async():
    """Test OpenAI async client."""
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="test")
    
    response = await client.chat.completions.create(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=50
    )
    
    print("OpenAI Async Test:", response.choices[0].message.content[:50] + "...")

if __name__ == "__main__":
    print("Testing client library integration...")
    
    # Test sync client
    test_openai_sync()
    
    # Test async client
    asyncio.run(test_openai_async())
    
    print("All tests completed!")
```

Run the quick test:
```bash
python test_clients.py
```