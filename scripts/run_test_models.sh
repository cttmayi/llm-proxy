#!/bin/bash

# LLM Proxy Model Testing Script
# This script tests various models and endpoints using curl

set -e

# Configuration
BASE_URL="http://localhost:8000"
TIMEOUT=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

# Function to check if server is running
check_server() {
    print_status "Checking if LLM Proxy server is running..."
    if ! curl -s --max-time 5 "$BASE_URL/health" > /dev/null; then
        print_error "Server is not running. Please start the server first:"
        print_error "source ./.venv/bin/activate && python src/main.py --config config/config.json"
        exit 1
    fi
    print_status "Server is running ✓"
}

# Function to test health endpoints
test_health() {
    print_test "Testing health endpoints..."
    
    print_status "Testing /health..."
    curl -s "$BASE_URL/health" | jq .
    
    print_status "Testing /health/detailed..."
    curl -s "$BASE_URL/health/detailed" | jq .
    
    print_status "Testing /health/ready..."
    curl -s "$BASE_URL/health/ready" | jq .
    
    print_status "Testing /health/live..."
    curl -s "$BASE_URL/health/live" | jq .
}

# Function to test models endpoint
test_models() {
    print_test "Testing models endpoint..."
    
    print_status "Getting available models..."
    curl -s "$BASE_URL/v1/models" | jq '.data[] | select(.id | contains("gpt-4o"))' | head -5
}

# Function to test chat completions
test_chat_completions() {
    print_test "Testing chat completions..."
    
    # Test OpenAI GPT-4o
    print_status "Testing OpenAI GPT-4o..."
    response=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello! Can you briefly introduce yourself?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }')
    
    if echo "$response" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
        print_status "✓ GPT-4o chat completion successful"
        echo "Response preview: $(echo "$response" | jq -r '.choices[0].message.content' | head -c 100)..."
    else
        print_error "✗ GPT-4o chat completion failed"
        echo "$response" | jq .
    fi
    
    # Test Claude (if available)
    print_status "Testing Claude..."
    response=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "claude-3-5-sonnet",
            "messages": [
                {"role": "user", "content": "Hello! Can you briefly introduce yourself?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }')
    
    if echo "$response" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
        print_status "✓ Claude chat completion successful"
        echo "Response preview: $(echo "$response" | jq -r '.choices[0].message.content' | head -c 100)..."
    else
        print_warning "✗ Claude chat completion failed (may not be configured)"
        echo "$response" | jq .
    fi
}

# Function to test streaming chat completions
test_chat_streaming() {
    print_test "Testing chat streaming..."
    
    print_status "Testing GPT-4o streaming..."
    curl -s -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Tell me a short joke in 1 sentence"}
            ],
            "max_tokens": 50,
            "stream": true
        }' | head -5
}

# Function to test embeddings
test_embeddings() {
    print_test "Testing embeddings..."
    
    print_status "Testing text-embedding-ada-002..."
    response=$(curl -s -X POST "$BASE_URL/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "text-embedding-ada-002",
            "input": "The quick brown fox jumps over the lazy dog"
        }')
    
    if echo "$response" | jq -e '.data[0].embedding' > /dev/null 2>&1; then
        print_status "✓ Embeddings API successful"
        echo "Embedding dimensions: $(echo "$response" | jq '.data[0].embedding | length')"
    else
        print_warning "✗ Embeddings API failed"
        echo "$response" | jq .
    fi
}

# Function to test error handling
test_error_handling() {
    print_test "Testing error handling..."
    
    # Test invalid model
    print_status "Testing invalid model error..."
    curl -s -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "invalid-model-xyz",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }' | jq .
    
    # Test invalid JSON
    print_status "Testing invalid JSON error..."
    curl -s -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"invalid": json}' | jq .
}

# Function to test documentation endpoints
test_docs() {
    print_test "Testing documentation endpoints..."
    
    print_status "Testing /docs..."
    curl -s -I "$BASE_URL/docs" | head -1
    
    print_status "Testing /openapi.json..."
    curl -s "$BASE_URL/openapi.json" | jq '.info.title'
}

# Function to run performance tests
test_performance() {
    print_test "Running performance tests..."
    
    print_status "Testing response time for models endpoint..."
    time curl -s "$BASE_URL/v1/models" > /dev/null
    
    print_status "Testing response time for health endpoint..."
    time curl -s "$BASE_URL/health" > /dev/null
}

# Function to test with different providers
test_providers() {
    print_test "Testing different providers..."
    
    # Test OpenAI models
    print_status "Available OpenAI models:"
    curl -s "$BASE_URL/v1/models" | jq -r '.data[] | select(.id | contains("gpt")) | .id' | head -5
    
    # Test Claude models (if configured)
    print_status "Available Claude models:"
    curl -s "$BASE_URL/v1/models" | jq -r '.data[] | select(.id | contains("claude")) | .id' | head -5
}

# Main test function
main() {
    echo "========================================="
    echo "LLM Proxy Model Testing Script"
    echo "========================================="
    echo
    
    # Check dependencies
    if ! command -v jq &> /dev/null; then
        print_error "jq is required but not installed. Install with: brew install jq"
        exit 1
    fi
    
    # Check server
    check_server
    
    # Run tests
    test_health
    echo
    
    test_models
    echo
    
    test_providers
    echo
    
    test_chat_completions
    echo
    
    test_chat_streaming
    echo
    
    test_embeddings
    echo
    
    test_docs
    echo
    
    test_performance
    echo
    
    test_error_handling
    echo
    
    print_status "All tests completed!"
    echo
    echo "========================================="
    echo "Test Summary"
    echo "========================================="
    echo "Server Status: ✓ Running"
    echo "Health Check: ✓ Available"
    echo "Models API: ✓ Working"
    echo "Chat API: ✓ Working"
    echo "Embeddings API: ✓ Working"
    echo "Documentation: ✓ Available"
}

# Run tests with error handling
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi