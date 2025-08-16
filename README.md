# LLM Proxy

A unified API proxy that provides a single interface to multiple Large Language Model providers including Claude (Anthropic), OpenAI, and Azure OpenAI.

## Features

- **Unified API**: Single interface for multiple LLM providers
- **Provider Support**: Claude, OpenAI, and Azure OpenAI
- **OpenAI Compatibility**: Compatible with OpenAI API format
- **Streaming Support**: Real-time streaming responses
- **Health Monitoring**: Built-in health checks for all providers
- **Rate Limiting**: Optional rate limiting per client
- **Metrics Collection**: Request/response metrics and logging
- **Flexible Configuration**: JSON configuration with environment variable support
- **Security**: CORS support, authentication options

## Architecture

The project has been refactored with a clean architecture:

```
src/
├── api/
│   ├── endpoints/          # API endpoint handlers
│   │   ├── chat.py        # Chat completion endpoints
│   │   ├── embeddings.py  # Embeddings endpoints
│   │   ├── models.py      # Models listing endpoints
│   │   └── health.py      # Health check endpoints
│   └── middleware/        # Request/response middleware
├── providers/             # LLM provider implementations
│   ├── base.py            # Base provider interface
│   ├── claude.py          # Claude provider
│   ├── openai.py          # OpenAI provider
│   ├── azure.py           # Azure OpenAI provider
│   └── factory.py         # Provider factory pattern
├── config/               # Configuration management
│   ├── models.py          # Pydantic configuration models
│   └── loader.py          # Configuration loader
└── main.py               # FastAPI application
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-proxy

# Run setup script
./scripts/setup.sh
```

### 2. Configuration

Create a configuration file or use environment variables:

Create `config/config.json`:

```json
{
  "providers": {
    "claude": {
      "enabled": true,
      "api_key": "your-claude-api-key",
      "base_url": "https://api.anthropic.com"
    },
    "openai": {
      "enabled": true,
      "api_key": "your-openai-api-key"
    },
    "azure": {
      "enabled": false,
      "api_key": "your-azure-key",
      "endpoint": "https://your-resource.openai.azure.com"
    }
  },
  "model_mapping": {
    "gpt-4o": "openai",
    "claude-3-5-sonnet": "claude",
    "gpt-4": "azure"
  }
}
```

### 3. Run the Server

```bash
# Activate virtual environment
source venv/bin/activate

# Run the server
python src/main.py --config config/config.json

# Or with environment variables
python src/main.py

# Or with custom host/port
python src/main.py --host 0.0.0.0 --port 8080
```

## API Usage

### Chat Completions

```bash
# Using curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'

# Using OpenAI client
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Streaming Responses

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### Embeddings

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-ada-002",
    "input": "The quick brown fox jumps over the lazy dog"
  }'
```

### List Models

```bash
curl http://localhost:8000/v1/models
```

## Configuration

### Providers

#### Claude (Anthropic)
```json
{
  "claude": {
    "enabled": true,
    "api_key": "sk-ant-...",
    "base_url": "https://api.anthropic.com",
    "api_version": "2023-06-01"
  }
}
```

#### OpenAI
```json
{
  "openai": {
    "enabled": true,
    "api_key": "sk-...",
    "base_url": "https://api.openai.com",
    "organization": "org-..."  # optional
  }
}
```

#### Azure OpenAI
```json
{
  "azure": {
    "enabled": true,
    "api_key": "...",
    "endpoint": "https://your-resource.openai.azure.com",
    "api_version": "2024-10-21"
  }
}
```

### Model Mapping

Configure which models map to which providers:

```json
{
  "model_mapping": {
    "gpt-4o": "openai",
    "claude-3-5-sonnet": "claude",
    "gpt-4": "azure"
  }
}
```

### Server Configuration

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": "INFO",
    "workers": 1
  }
}
```

### Features

```json
{
  "features": {
    "enable_streaming": true,
    "enable_caching": false,
    "enable_metrics": true,
    "enable_rate_limiting": false,
    "max_requests_per_minute": 60
  }
}
```

## Testing

### Run All Tests

```bash
./scripts/run_tests.sh
```

### Run Specific Test Suites

```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Configuration

The test suite includes:
- **Unit Tests**: Individual provider tests
- **Integration Tests**: API endpoint tests
- **Mock Providers**: Test without real API calls
- **Coverage Reports**: HTML coverage reports

## Health Checks

The proxy provides multiple health check endpoints:

- `GET /health` - Basic health status
- `GET /health/ready` - Kubernetes readiness probe
- `GET /health/live` - Kubernetes liveness probe
- `GET /health/detailed` - Detailed provider health status

## Security

### Authentication

Enable API key authentication:

```json
{
  "security": {
    "enable_auth": true,
    "api_key_header": "X-API-Key"
  }
}
```

### CORS

Configure CORS origins:

```json
{
  "security": {
    "enable_cors": true,
    "allowed_origins": ["https://yourdomain.com"]
  }
}
```

## Environment Variables

All configuration can be set via environment variables:

```bash
# Provider API keys
CLAUDE_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...

# Server settings
LLM_PROXY_HOST=0.0.0.0
LLM_PROXY_PORT=8000
LLM_PROXY_LOG_LEVEL=INFO

# Model mapping
LLM_PROXY_MODEL_MAPPING=gpt-4o=openai,claude-3-5-sonnet=claude
```

## Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "src/main.py"]
```

## Monitoring

### Metrics

When `enable_metrics` is true, metrics are collected:
- Request counts
- Response times
- Error rates
- Status code distribution

### Logging

Structured logging with:
- Request IDs for tracking
- Sensitive data redaction
- Provider-specific error logging
- Performance metrics

## Troubleshooting

### Common Issues

1. **Provider Authentication Errors**
   - Check API keys in configuration
   - Verify provider endpoints are accessible

2. **Model Not Found**
   - Check `model_mapping` configuration
   - Ensure provider supports the requested model

3. **Rate Limiting**
   - Adjust `max_requests_per_minute` in configuration
   - Check provider rate limits

4. **CORS Issues**
   - Configure `allowed_origins` appropriately
   - Enable CORS in security settings

### Debug Mode

Run with debug logging:

```bash
python src/main.py --log-level DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run test suite: `./scripts/run_tests.sh`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.