# LLM Proxy Testing Scripts

This directory contains comprehensive testing scripts for the LLM Proxy service.

## Available Scripts

### 1. run_test_quick.sh
A fast, simple test to verify basic functionality.

**Usage:**
```bash
./scripts/run_test_quick.sh
```

**Tests:**
- Server health check
- Models endpoint
- Basic chat completion
- Basic embeddings
- Lists available models

### 2. run_test_models.sh
A comprehensive test suite for all endpoints and models.

**Usage:**
```bash
./scripts/run_test_models.sh
```

**Tests:**
- All health endpoints (`/health`, `/health/detailed`, `/health/ready`, `/health/live`)
- Models endpoint with full model list
- Chat completions (GPT-4o, Claude)
- Streaming chat completions
- Embeddings API
- Error handling scenarios
- Performance benchmarking
- Documentation endpoints

## Prerequisites

Before running tests:

1. **Start the LLM Proxy server:**
   ```bash
   source ./.venv/bin/activate
   python src/main.py --config config/config.json
   ```

2. **Install dependencies:**
   ```bash
   # On macOS
   brew install jq
   
   # On Ubuntu/Debian
   sudo apt-get install jq
   
   # On CentOS/RHEL
   sudo yum install jq
   ```

## Manual Testing Examples

### Health Check
```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/detailed
```

### List Models
```bash
curl http://localhost:8000/v1/models
```

### Chat Completion
```bash
# OpenAI GPT-4o
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Claude
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming Chat
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
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

## Testing Different Providers

### OpenAI Models
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4`
- `gpt-3.5-turbo`

### Claude Models
- `claude-3-5-sonnet`
- `claude-3-5-haiku`
- `claude-3-opus`

### Embedding Models
- `text-embedding-ada-002`
- `text-embedding-3-small`
- `text-embedding-3-large`

## Troubleshooting

### Common Issues

1. **Server not running:**
   ```bash
   source ./.venv/bin/activate
   python src/main.py --config config/config.json
   ```

2. **jq not found:**
   Install jq using your package manager (see prerequisites).

3. **Connection refused:**
   Ensure the server is listening on `localhost:8000`.

4. **API key errors:**
   Check your `.env` file for correct API keys.

### Debug Mode
Run the server with debug logging:
```bash
python src/main.py --config config/config.json --log-level DEBUG
```

## Integration with CI/CD

These scripts can be integrated into CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Test LLM Proxy
  run: |
    ./scripts/run_test_quick.sh
    ./scripts/run_test_models.sh
```