#!/bin/bash

# This script installs axono as a package in the system.
# It checks for uv and then uses it to install axono as a tool.

if ! command -v uv &> /dev/null; then
    echo "uv could not be found. Please install uv first."
    exit 1
fi

echo "Installing axono..."
uv tool install .

echo "Installation complete. You can now run 'axono' from anywhere."