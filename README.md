# Axono

[![CI](https://github.com/gdoct/axono/actions/workflows/ci.yml/badge.svg)](https://github.com/gdoct/axono/actions/workflows/ci.yml)
[![Code Coverage](https://codecov.io/gh/gdoct/axono/branch/main/graph/badge.svg)](https://codecov.io/gh/gdoct/axono)
[![CodeQL](https://github.com/gdoct/axono/actions/workflows/codeql.yml/badge.svg)](https://github.com/gdoct/axono/actions/workflows/codeql.yml)

Axono is your private, fully local AI coding companion. It lives in your terminal and connects to the Large Language Models (LLMs) running on your own machine (via [LM Studio](https://lmstudio.ai/)).

Unlike cloud-based assistants, **Axono is 100% free and private**. Your code, your data, and your conversations never leave your computer. It gives you a conversational interface to your filesystem—letting you run shell commands, generate code, and edit files just by chatting. You can extend it as you wish by integrating mcp servers.

It's designed to bring agentic AI capabilities to "normal" hardware. You don't need a massive server cluster; it runs perfectly on consumer cards.

```
    _
   / \   __  __  ___   _ __    ___
  / _ \  \ \/ / / _ \ | '_ \  / _ \
 / ___ \  >  < | (_) || | | || (_) |
/_/   \_\/_/\_\ \___/ |_| |_| \___/
```

## Why Axono?

- **100% Local & Private**: No API keys, no monthly subscriptions, and no data tracking. Axono connects to tools like [LM Studio](https://lmstudio.ai/) to run entirely offline.
- **Consumer Hardware Ready**: You don't need enterprise gear. If you have a decent GPU (like an NVIDIA RTX 30-series), you can run capable coding models locally with ease.
- **Real Agentic Capability**: It doesn't just chat. Axono can explore your project structure, create plans, write code, run shell commands, and validate its own work.
- **Safety First**: Every command the AI wants to run is evaluated for safety, keeping you in control of what happens to your system.
- **Extensible**: Support for the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) lets you plug in additional tools and context servers.

## Warning

This project is not meant to be used on any serious system. Be very careful ho you use it as it is autonomous and will do anything to fulfill the task at hand, which may lead to enexpected results. There are some built-in safeguards - it will ask to confirm before formatting your system drive. But be very careful.

## Requirements

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) running with a model loaded
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
git clone https://github.com/gdoct/axono.git
cd axono

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Configure your LM Studio endpoint
cp .env.example .env
# Edit .env with your settings
```
<img width="1309" height="577" alt="image" src="https://github.com/user-attachments/assets/9fd0ecbb-ab3d-47d1-a831-97c4e68519ea" />

## Usage

```bash
./run.sh
```

Or manually:

```bash
source .venv/bin/activate
python -m axono.main
```

Once running, type natural language messages and press Enter. Axono can:

- Answer questions using the loaded LLM
- Execute shell commands when you ask it to perform system tasks
- Write and edit code files through the `code` tool
- Track your working directory across commands

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Enter` | Send message |
| `Ctrl+L` | Clear chat |
| `Ctrl+C` | Quit |

## Configuration

Axono loads configuration from multiple sources (later overrides earlier):

1. Built-in defaults
2. `~/.axono/config.json`
3. `.env` file
4. Environment variables

See [.env.example](.env.example) for available settings.

### MCP Servers

To add MCP tool servers, create `~/.axono/mcp.json`:

```json
{
  "servers": {
    "my-server": {
      "command": "npx",
      "args": ["-y", "@my/mcp-server"]
    }
  }
}
```

## Project Structure

```
axono/
├── axono/
│   ├── __init__.py   # Package marker
│   ├── main.py       # TUI application entry point
│   ├── agent.py      # LangChain agent, tools, orchestration
│   ├── coding.py     # Multi-stage coding pipeline
│   ├── config.py     # Configuration management
│   ├── safety.py     # Command safety evaluator
│   └── ui.py         # Textual TUI widgets and layout
├── requirements.txt
├── run.sh
├── .env.example
└── LICENSE
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
