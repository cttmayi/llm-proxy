# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Install dependencies
uv pip install -r requirements.txt

### Run the server
./scripts/run_server.sh

### Run tests
./scripts/run_tests.sh
pytest tests/unit -v                    # Unit tests only
pytest tests/integration -v             # Integration tests only
pytest tests/ --cov=src --cov-report=html  # With coverage

### Quick functionality test
./scripts/run_test_quick.sh

### Comprehensive testing
./scripts/run_test_models.sh

## Architecture Overview

This is a **unified API proxy** that provides a single interface to multiple LLM providers (Claude, OpenAI, Azure OpenAI) with OpenAI-compatible API format.

### Core Architecture

- **FastAPI-based** Python application with async/await patterns
- **Provider factory pattern** for dynamic provider instantiation
- **Clean architecture** with separation of concerns:
  - `src/api/endpoints/` - REST API handlers
  - `src/providers/` - LLM provider implementations
  - `src/config/` - Configuration management (Pydantic models)
  - `src/api/middleware/` - Cross-cutting concerns (logging, CORS, error handling)

### Key Components

1. **Provider System**: Factory pattern with `BaseProvider` abstract base class
   - `ClaudeProvider`, `OpenAIProvider`, `AzureOpenAIProvider`
   - Auto-detection based on model names
   - Configurable model-to-provider mapping

2. **Configuration**: JSON-based with environment variable fallback
   - `config/config.json` for provider settings
   - `config/config.dev.json` for development
   - Environment variables with `LLM_PROXY_` prefix

3. **API Endpoints** (OpenAI-compatible):
   - `/v1/chat/completions` - Chat completions (streaming supported)
   - `/v1/embeddings` - Text embeddings
   - `/v1/models` - Available models
   - `/health*` - Health check endpoints

### Development Workflow

1. **Configuration**: Edit `config/config.json` or use environment variables for API keys
2. **Testing**: Use provided scripts in `scripts/` directory
3. **Code style**: Uses black (formatting), flake8 (linting), mypy (type checking)
4. **Dependencies**: Managed via `requirements.txt` with separate dev/test dependencies

### Common Model Mappings

- `gpt-4o` → OpenAI
- `claude-3-5-sonnet` → Claude
- `text-embedding-ada-002` → OpenAI/Azure (auto-detected)

### Testing Patterns

- **Unit tests**: Mock provider interactions in `tests/unit/`
- **Integration tests**: Real API tests in `tests/integration/`
- **Test markers**: `@pytest.mark.unit`, `@pytest.mark.integration`
- **Async testing**: pytest-asyncio with `asyncio_mode = auto`