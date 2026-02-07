# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Multiple model types**: Configure separate models for instruction and reasoning tasks via `LLM_INSTRUCTION_MODEL` and `LLM_REASONING_MODEL` environment variables
- **Iterative shell pipeline** (`shell.py`): Adaptive step-by-step command execution with project type detection, safety checks, and error recovery
- **First-run onboarding** (`onboarding.py`): Setup wizard for configuring LM Studio connection on first launch or with `--onboard` flag
- **Chat history** (`history.py`): Persists last 30 prompts across sessions with arrow key navigation
- **Shared pipeline infrastructure** (`pipeline.py`): Common utilities for iterative pipelines including JSON parsing, text truncation, and LLM helpers
- **Configurable data directory**: Set `AXONO_DATA_DIR` to customize where config, history, and MCP settings are stored

### Changed
- Coding pipeline now uses configurable model types (instruction/reasoning) via shared `get_model_name()` helper
- Improved configuration system with clearer load order documentation

### Previous
- Initial project structure for Axono
- Terminal-based UI (`ui.py`) using Textual
- Local LLM integration via `agent.py`
- Multi-stage coding pipeline in `coding.py`
- Command safety evaluation in `safety.py`
