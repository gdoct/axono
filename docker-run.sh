#!/bin/bash
set -e

# Default to current directory if no argument provided
WORKSPACE="${1:-$(pwd)}"

# Create host data dir if it doesn't exist
mkdir -p "$HOME/.axono"

docker run --rm \
    --network host \
    -v "$HOME/.axono:/usr/local/share/axono" \
    -v "$WORKSPACE:/workspace" \
    axono:latest
