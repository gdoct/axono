# Contributing to Axono

Thanks for your interest in contributing to Axono! This document covers the basics for getting started.

## Getting Started

1. Fork the repository and clone your fork
2. Set up the development environment:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure your LM Studio endpoint
4. Run the app with `./run.sh`

## Making Changes

1. Create a branch for your change (`git checkout -b my-feature`)
2. Make your changes in the `axono/` package
3. Test your changes by running the app locally
4. Commit with a clear message describing what you changed and why
5. Push to your fork and open a pull request

## Project Structure

```
axono/
├── __init__.py   # Package marker
├── main.py       # TUI application entry point
├── agent.py      # LangChain agent, tools, and orchestration
├── coding.py     # Multi-stage coding pipeline
├── config.py     # Configuration management
├── safety.py     # LLM-based command safety evaluator
└── ui.py         # Textual TUI widgets and layout
```

## Guidelines

- Keep changes focused — one feature or fix per pull request
- Follow the existing code style and conventions
- Test with a local LM Studio instance before submitting
- Update the README if your change affects usage or setup

## Reporting Issues

Open an issue on GitHub with:
- A clear description of the problem or suggestion
- Steps to reproduce (for bugs)
- Your environment (OS, Python version, LM Studio version)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
