# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Axono is a terminal-based AI coding assistant that runs entirely locally via LM Studio. It uses a Textual TUI frontend with a LangChain ReAct agent backend. Python >=3.10, MIT licensed.

## Commands

```bash
# Run the app (creates .venv if needed)
./run.sh

# Run directly
source .venv/bin/activate
python -m axono.main

# Install system-wide (requires uv)
./install.sh            # runs: uv tool install .

# Run all tests
pytest

# Run a single test file
pytest tests/test_coding.py

# Run a single test
pytest tests/test_coding.py::TestInvestigate::test_score_file_keyword_match -v

# Install dependencies
uv pip install -r requirements.txt
```

No linter or formatter is configured in pyproject.toml, but the project follows Black/isort style conventions.

## Architecture

### Agent Layer (`axono/agent.py`)
LangChain ReAct agent with two core tools:
- **`bash`**: Executes shell commands with LLM-based safety evaluation (via `safety.py`). Tracks working directory across calls using `__CWD__:` markers in output.
- **`code`**: Triggers the multi-stage coding pipeline for file creation/modification tasks.

Optional MCP (Model Context Protocol) tools are loaded from `~/.axono/mcp.json` if `langchain-mcp-adapters` is available.

### 5-Stage Coding Pipeline (`axono/coding.py` - largest module)
1. **Investigation** - Scans project files, scores them by extension weight/keyword match/recency, selects relevant files within a context budget (max files + max chars).
2. **Planning** - LLM produces a JSON plan listing files to read and patches to apply.
3. **Generation** - LLM generates complete file contents as a JSON array of patches (create/update).
4. **Application** - Writes patches to disk, creating directories as needed.
5. **Validation** - LLM reviews generated code for correctness.

### Safety (`axono/safety.py`)
LLM-based command safety judgment (not regex). Returns `{"dangerous": bool, "reason": str}`. The `bash` tool's `unsafe` flag allows user override after confirmation.

### Configuration (`axono/config.py`)
Load priority (highest wins): environment variables > `.env` file > `~/.axono/config.json` > built-in defaults. Key settings in `.env.example`.

### TUI (`axono/main.py`, `axono/ui.py`)
Textual framework. `AxonoApp` manages the chat interface. Agent streams updates via `astream(stream_mode="updates")` with structured message types: `("assistant", str)`, `("tool_call", str)`, `("tool_result", str)`, `("error", str)`.

## Key Conventions

- **Async throughout**: Agent streaming, LLM calls (`ainvoke`), and TUI workers all use async/await.
- **JSON interchange between pipeline stages**: Tolerates markdown fences in LLM output; has fallback parsing for malformed responses.
- **Minimal change principle**: When editing code, follow existing style and avoid unrelated refactors.
- **Tests mirror source structure**: Each `axono/*.py` has a corresponding `tests/test_*.py`. Tests use mocked LLM responses and isolated temp directories for config.
- **`langchain` is pinned to 1.2.8** in requirements.txt; other deps float.
