"""Centralised configuration for Axono.

Load order (later sources override earlier ones):
  1. Built-in defaults
  2. ~/.axono/config.json
  3. .env file (via python-dotenv)
  4. Real environment variables
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env first so real env vars still win over it
load_dotenv()

# ── defaults ──────────────────────────────────────────────────────────
_DEFAULTS = {
    "base_url": "http://192.168.32.1:1234/v1",
    "model_name": "local-model",
    "model_provider": "openai",
    "api_key": "lm-studio",
    "command_timeout": "30",
    "investigation_enabled": "true",
    "max_context_files": "8",
    "max_context_chars": "30000",
}

# ── data directory (configurable via AXONO_DATA_DIR) ──────────────────
def _get_data_dir() -> Path:
    """Return the data directory, respecting AXONO_DATA_DIR env var."""
    env_dir = os.environ.get("AXONO_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".axono"


_config_dir = _get_data_dir()
_config_path = _config_dir / "config.json"

_file_cfg: dict = {}
if _config_path.is_file():
    try:
        _file_cfg = json.loads(_config_path.read_text(encoding="utf-8"))
        if not isinstance(_file_cfg, dict):
            _file_cfg = {}
    except (json.JSONDecodeError, OSError):
        _file_cfg = {}


def _get(key: str) -> str:
    """Return a config value using the load-order described above."""
    # Map config keys to the corresponding env-var names
    env_map = {
        "base_url": "LLM_BASE_URL",
        "model_name": "LLM_MODEL_NAME",
        "model_provider": "LLM_MODEL_PROVIDER",
        "api_key": "LLM_API_KEY",
        "command_timeout": "COMMAND_TIMEOUT",
        "investigation_enabled": "INVESTIGATION_ENABLED",
        "max_context_files": "MAX_CONTEXT_FILES",
        "max_context_chars": "MAX_CONTEXT_CHARS",
    }

    # 4) env var  (highest priority)
    env_name = env_map.get(key)
    if env_name:
        env_val = os.getenv(env_name)
        if env_val:  # non-empty string
            return env_val

    # 3) ~/.axono/config.json
    val = _file_cfg.get(key)
    if val is not None and str(val):
        return str(val)

    # 1) built-in default
    return _DEFAULTS[key]


# ── public constants ──────────────────────────────────────────────────
LLM_BASE_URL: str = _get("base_url")
LLM_MODEL_NAME: str = _get("model_name")
LLM_MODEL_PROVIDER: str = _get("model_provider")
LLM_API_KEY: str = _get("api_key")
COMMAND_TIMEOUT: int = int(_get("command_timeout"))
INVESTIGATION_ENABLED: bool = _get("investigation_enabled").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAX_CONTEXT_FILES: int = int(_get("max_context_files"))
MAX_CONTEXT_CHARS: int = int(_get("max_context_chars"))


# ── MCP server configuration ────────────────────────────────────────
def load_mcp_config() -> dict:
    """Load MCP server configuration from the data directory.

    The file path can be overridden via the ``MCP_CONFIG_PATH`` env var.
    Returns the ``"servers"`` dict suitable for ``MultiServerMCPClient``,
    or an empty dict if no config exists or it is invalid.
    """
    candidates = [
        os.environ.get("MCP_CONFIG_PATH", ""),
        str(_get_data_dir() / "mcp.json"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                servers = data.get("servers", {})
                if not isinstance(servers, dict):
                    print(
                        f"Warning: 'servers' in {path} is not a dict, "
                        "ignoring MCP config",
                        file=sys.stderr,
                    )
                    return {}
                return servers
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"Warning: Failed to load MCP config from {path}: {exc}",
                    file=sys.stderr,
                )
                return {}
    return {}


# ── onboarding helpers ─────────────────────────────────────────────


def config_dir() -> Path:
    """Return the data directory path, respecting AXONO_DATA_DIR env var."""
    return _get_data_dir()


def config_path() -> Path:
    """Return the canonical path to ``~/.axono/config.json``."""
    return config_dir() / "config.json"


def needs_onboarding() -> bool:
    """Return *True* when no config file exists."""
    return not config_path().is_file()


def save_config(settings: dict[str, str]) -> Path:
    """Write *settings* to ``~/.axono/config.json``.

    Creates the ``~/.axono`` directory if it doesn't exist.
    Returns the path written to.
    """
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = config_path()
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return path
