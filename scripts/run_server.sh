#!/bin/bash

lsof -i :8000

lsof -ti:8000 | xargs kill -9

source .venv/bin/activate

python src/main.py --config config/config.json