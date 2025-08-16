#!/bin/bash
# Setup script for LLM Proxy

set -e

echo "Setting up LLM Proxy..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.11
fi

# Activate virtual environment
source ./.venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Create config directory if it doesn't exist
mkdir -p config

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit .env file with your actual API keys"
fi

# Run tests
echo "Running tests..."
pytest tests/unit -v

echo "Setup complete!"
echo ""
echo "To run the server:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Set your API keys in .env file"
echo "3. Run: python src/main.py --config config/config.json"