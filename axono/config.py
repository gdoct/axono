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
    "model_name": "",  # Empty means use the loaded model in LM Studio backend
    "model_provider": "openai",
    "api_key": "lm-studio",
    "instruction_model": "",  # Empty means use model_name (or backend default)
    "reasoning_model": "",  # Empty means use instruction_model
    "command_timeout": "30",
    "investigation_enabled": "true",
    "max_context_files": "8",
    "max_context_chars": "30000",
    "llm_investigation": "false",  # Use LLM to rank files during investigation
    "embedding_model": "",  # Empty means use default (all-MiniLM-L6-v2)
    "embedding_db_path": "",  # Empty means ~/.axono/embeddings.db
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
        "instruction_model": "LLM_INSTRUCTION_MODEL",
        "reasoning_model": "LLM_REASONING_MODEL",
        "command_timeout": "COMMAND_TIMEOUT",
        "investigation_enabled": "INVESTIGATION_ENABLED",
        "max_context_files": "MAX_CONTEXT_FILES",
        "max_context_chars": "MAX_CONTEXT_CHARS",
        "llm_investigation": "LLM_INVESTIGATION",
        "embedding_model": "EMBEDDING_MODEL",
        "embedding_db_path": "EMBEDDING_DB_PATH",
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
LLM_INSTRUCTION_MODEL: str = _get("instruction_model")
LLM_REASONING_MODEL: str = _get("reasoning_model")
COMMAND_TIMEOUT: int = int(_get("command_timeout"))


def get_model_name(model_type: str = "instruction") -> str:
    """Return the model name for the given model type.

    Model types:
      - "instruction": For general instruction-following tasks (agent, coding, etc.)
      - "reasoning": For tasks requiring deeper reasoning

    Resolution order:
      1. If reasoning_model is requested and set, use it
      2. If instruction_model is set, use it
      3. Empty string (let the backend use its currently loaded model)
    """
    if model_type == "reasoning" and LLM_REASONING_MODEL:
        return LLM_REASONING_MODEL
    if LLM_INSTRUCTION_MODEL:
        return LLM_INSTRUCTION_MODEL
    return ""  # Empty string signals to use the backend's currently loaded model


INVESTIGATION_ENABLED: bool = _get("investigation_enabled").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LLM_INVESTIGATION: bool = _get("llm_investigation").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAX_CONTEXT_FILES: int = int(_get("max_context_files"))
MAX_CONTEXT_CHARS: int = int(_get("max_context_chars"))

EMBEDDING_MODEL: str = _get("embedding_model")


def get_embedding_db_path() -> Path:
    """Return the path to the embeddings database."""
    custom_path = _get("embedding_db_path")
    if custom_path:
        return Path(custom_path)
    return _get_data_dir() / "embeddings.db"


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


# ── trusted folders configuration ─────────────────────────────────────


def _trusted_folders_path() -> Path:
    """Return the path to the trusted folders file."""
    return config_dir() / "trusted_folders.json"


def load_trusted_folders() -> list[str]:
    """Load the list of trusted workspace folders.

    Returns a list of absolute paths that the user has trusted.
    """
    path = _trusted_folders_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(f) for f in data if isinstance(f, str)]
        return []
    except (json.JSONDecodeError, OSError):
        return []


def save_trusted_folders(folders: list[str]) -> Path:
    """Save the list of trusted workspace folders.

    Args:
        folders: List of absolute paths to trusted folders.

    Returns:
        The path written to.
    """
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = _trusted_folders_path()
    path.write_text(json.dumps(folders, indent=2) + "\n", encoding="utf-8")
    return path


def add_trusted_folder(folder: str) -> list[str]:
    """Add a folder to the trusted folders list.

    Args:
        folder: Absolute path to the folder to trust.

    Returns:
        The updated list of trusted folders.
    """
    folders = load_trusted_folders()
    # Normalize the path
    normalized = os.path.abspath(os.path.expanduser(folder))
    if normalized not in folders:
        folders.append(normalized)
        save_trusted_folders(folders)
    return folders


def is_folder_trusted(folder: str) -> bool:
    """Check if a folder is in the trusted folders list.

    Args:
        folder: Absolute path to check.

    Returns:
        True if the folder is trusted.
    """
    normalized = os.path.abspath(os.path.expanduser(folder))
    return normalized in load_trusted_folders()
