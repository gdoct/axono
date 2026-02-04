#!/bin/bash

# check for uv
if ! command -v uv &> /dev/null; then
    echo "uv could not be found. Please install uv first."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found."
    uv venv .venv
    uv pip install -r requirements.txt
fi

source .venv/bin/activate
python -m axono.main
