#!/bin/bash

# Quick LLM Proxy Test Script
# Simple tests for basic functionality

set -e

BASE_URL="http://localhost:8899/openai"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

error() {
    echo -e "${RED}✗ $1${NC}"
}

echo "🚀 LLM Proxy Quick Test"
echo "======================"


# Test models endpoint
echo -n "Testing models endpoint... "
if curl -s "$BASE_URL/v1/models" | jq -e '.data | length > 0' > /dev/null 2>&1; then
    success "Models list retrieved"
else
    error "Models list failed"
fi

# Test chat completion (quick)
echo -n "Testing chat completion... "
response=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-4o",
        "messages": [{"role": "system", "content": "你是我的助手，请用中文回答。"}, {"role": "user", "content": "Hi"}],
        "max_tokens": 10
    }' 2>&1 || echo "{}")

if echo "$response" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
    success "Chat completion works"
    echo "   Sample: $(echo "$response" | jq -r '.choices[0].message.content' | head -c 30)..."
else
    error "Chat completion failed"
fi

# Test embeddings
echo -n "Testing embeddings... "
response=$(curl -s -X POST "$BASE_URL/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "text-embedding-ada-002",
        "input": "test"
    }' 2>&1 || echo "{}")

if echo "$response" | jq -e '.data[0].embedding' > /dev/null 2>&1; then
    success "Embeddings work"
else
    error "Embeddings failed"
fi

# Show available models
echo
echo "📋 Available Models:"
curl -s "$BASE_URL/v1/models" | jq -r '.data[] | select(.id | contains("gpt-4o","claude")) | .id' | head -5

echo
echo "🎉 Quick test completed!"