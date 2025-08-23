# LLM Proxy Server

A flexible HTTP proxy server built on `proxy.py` that routes requests to different LLM providers (OpenAI, Anthropic, Azure OpenAI) based on URL patterns.

## Features

- 🎯 **Multi-Provider Support**: Routes requests to OpenAI, Anthropic, or Azure OpenAI
- 🔧 **Configurable Routing**: URL-based routing with customizable patterns
- 🔒 **API Key Management**: Secure handling of provider API keys
- 🚀 **Easy Setup**: Simple configuration via environment variables
- 📊 **Health Checks**: Built-in testing utilities for each provider

## Quick Start

### Prerequisites
- Python 3.8+
- [uv](https://astral.sh/uv) (recommended for dependency management)

### Installation

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Setup the project**:
```bash
# Clone and navigate to the project
git clone <repository-url>
cd llm-proxy-server

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Unix/macOS
# or .venv\Scripts\activate  # Windows

uv pip install -r requirements.txt
```

3. **Configure API Keys**:
```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"
export OPENAI_BASE_URL="https://api.openai.com"  # Optional

# Anthropic (Claude)
export ANTHROPIC_API_KEY="your-anthropic-key"
export ANTHROPIC_BASE_URL="https://api.anthropic.com"  # Optional

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_BASE_URL="https://your-resource.openai.azure.com"
```

### Usage

#### Start the Server
```bash
python main.py
```

The server will start on `http://localhost:8080` by default.

#### Making Requests

**OpenAI API**:
```bash
curl -X POST http://localhost:8080/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Anthropic API**:
```bash
curl -X POST http://localhost:8080/anthropic/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Azure OpenAI API**:
```bash
curl -X POST http://localhost:8080/azure/chat/completions \
  -H "Content-Type: application/json" \
  -H "api-key: $AZURE_OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Configuration

### URL Routing

The proxy routes requests based on URL patterns:

| URL Pattern | Provider |
|-------------|----------|
| `/openai/v1/chat/completions` | OpenAI |
| `/anthropic/v1/messages` | Anthropic |
| `/azure/chat/completions` | Azure OpenAI |

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes (for OpenAI) |
| `OPENAI_BASE_URL` | Custom OpenAI endpoint | No |
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes (for Anthropic) |
| `ANTHROPIC_BASE_URL` | Custom Anthropic endpoint | No |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key | Yes (for Azure) |
| `AZURE_OPENAI_BASE_URL` | Azure OpenAI endpoint | Yes (for Azure) |

## Testing

### Validate Provider Configuration

Test each provider individually:

```bash
# Test OpenAI
python tests/test_openai_direct.py

# Test Anthropic
python tests/test_anthropic_direct.py

# Test Azure OpenAI
python tests/test_azure_direct.py  # if available
```

### Health Checks

The proxy includes built-in health checks:

```bash
# Check if the proxy is running
curl http://localhost:8080/health

# Check OpenAI connectivity
curl http://localhost:8080/openai/health

# Check Anthropic connectivity
curl http://localhost:8080/anthropic/health
```

## Architecture

### Plugin System

The proxy uses a plugin-based architecture:

- **`oproxy/config.py`**: Provider configurations and routing rules
- **`oproxy/plugins.py`**: Request interception and routing logic
- **`main.py`**: Server entry point using proxy.py CLI

### Core Components

1. **LLMProxyPlugin**: Main plugin class that handles:
   - Request routing based on URL patterns
   - Header management for different providers
   - API key injection

2. **ProviderConfig**: Configuration management for:
   - API endpoints
   - Authentication headers
   - Custom base URLs

## Development

### Project Structure
```
llm-proxy-server/
├── main.py              # Server entry point
├── oproxy/
│   ├── __init__.py
│   ├── config.py        # Provider configurations
│   └── plugins.py       # Proxy plugins
├── tests/
│   ├── test_openai_direct.py
│   ├── test_anthropic_direct.py
│   └── test_azure_direct.py
├── requirements.txt     # Dependencies
├── CLAUDE.md           # Claude-specific instructions
└── README.md           # This file
```

### Adding New Providers

To add support for a new LLM provider:

1. **Update `oproxy/config.py`**:
   ```python
   # Add provider configuration
   "new_provider": {
       "base_url": "https://api.newprovider.com",
       "headers": {
           "Authorization": f"Bearer {os.getenv('NEW_PROVIDER_KEY')}"
       }
   }
   ```

2. **Update routing in `oproxy/plugins.py`**:
   ```python
   # Add new route pattern
   "/new-provider/v1/chat": "new_provider"
   ```

3. **Add environment variable**:
   ```bash
   export NEW_PROVIDER_KEY="your-key"
   ```

### Customization

#### Custom Base URLs
Override provider endpoints using environment variables:
```bash
export OPENAI_BASE_URL="https://custom-openai-endpoint.com"
export ANTHROPIC_BASE_URL="https://custom-anthropic-endpoint.com"
```

#### Custom Ports
Modify the default port (8080) in `main.py`:
```python
# Change from:
if __name__ == "__main__":
    main()

# To:
if __name__ == "__main__":
    import sys
    sys.argv = ["proxy.py", "--port", "9000"]
    main()
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if the server is running: `lsof -i :8080`
   - Verify the port isn't already in use

2. **Authentication Errors**
   - Ensure API keys are properly set in environment variables
   - Check for typos in the key names

3. **Routing Issues**
   - Verify URL patterns match exactly
   - Check the proxy logs for routing decisions

### Debug Mode

Enable verbose logging:
```bash
python main.py --log-level DEBUG
```

### Testing Connectivity

Test direct provider connectivity:
```bash
# Test OpenAI
python -c "import openai; openai.api_key='your-key'; print(openai.Model.list())"

# Test Anthropic
python -c "import anthropic; client=anthropic.Anthropic(api_key='your-key'); print(client.messages.create(model='claude-3-sonnet', max_tokens=10, messages=[{'role':'user','content':'test'}]))"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new providers
5. Run existing tests
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the [troubleshooting section](#troubleshooting)
- Review [existing issues](https://github.com/your-repo/issues)
- Create a new issue with detailed information