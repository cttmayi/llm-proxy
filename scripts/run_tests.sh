#!/bin/bash
# Test runner script for LLM Proxy

set -e

# Activate virtual environment
source .venv/bin/activate

echo "Running LLM Proxy tests..."

# Run unit tests
echo "Running unit tests..."
pytest tests/unit -v

# Run integration tests
echo "Running integration tests..."
pytest tests/integration -v

# Run all tests with coverage
echo "Running all tests with coverage..."
pytest tests/ --cov=src --cov-report=html --cov-report=term

echo "All tests completed!"