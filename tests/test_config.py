"""Unit tests for axono.config."""

import io
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers – we can't simply ``import axono.config`` because the module
# executes configuration logic at import time (load_dotenv, config-file
# parsing, constant assignment).  Instead we reload it inside controlled
# environments.
# ---------------------------------------------------------------------------


def _reload_config(env=None, config=None, config_raw=None):
    """Reload ``axono.config`` with a controlled environment.

    Parameters
    ----------
    env : dict | None
        Extra environment variables to set for the duration of the import.
    config : dict | None
        Contents to write to ``~/.axono/config.json`` (serialised as JSON).
    config_raw : str | None
        Raw text to write to ``~/.axono/config.json`` (takes precedence
        over *config* when both are given).

    Returns
    -------
    module
        The freshly-imported ``axono.config`` module.
    """
    import importlib
    import tempfile

    tmpdir = tempfile.mkdtemp()

    raw = (
        config_raw
        if config_raw is not None
        else (json.dumps(config) if config is not None else None)
    )
    if raw is not None:
        config_dir = Path(tmpdir) / ".axono"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(raw)

    patch_env = {"HOME": tmpdir}
    if env:
        patch_env.update(env)

    # Remove any cached env vars that might leak between reloads
    for var in (
        "LLM_BASE_URL",
        "LLM_MODEL_NAME",
        "LLM_MODEL_PROVIDER",
        "LLM_API_KEY",
        "LLM_INSTRUCTION_MODEL",
        "LLM_REASONING_MODEL",
        "COMMAND_TIMEOUT",
        "INVESTIGATION_ENABLED",
        "MAX_CONTEXT_FILES",
        "MAX_CONTEXT_CHARS",
        "MCP_CONFIG_PATH",
    ):
        patch_env.setdefault(var, "")

    with mock.patch.dict(os.environ, patch_env, clear=False):
        if "axono.config" in sys.modules:
            del sys.modules["axono.config"]
        import axono.config as cfg

        importlib.reload(cfg)
        return cfg


# ---------------------------------------------------------------------------
# Tests for _get_data_dir (AXONO_DATA_DIR support for Docker)
# ---------------------------------------------------------------------------


class TestGetDataDir:
    """Test custom data directory via AXONO_DATA_DIR env var."""

    def test_axono_data_dir_env_var(self):
        """AXONO_DATA_DIR overrides default ~/.axono location."""
        import importlib
        from pathlib import Path

        # Reload with AXONO_DATA_DIR set
        with mock.patch.dict(os.environ, {"AXONO_DATA_DIR": "/custom/data/path"}):
            if "axono.config" in sys.modules:
                del sys.modules["axono.config"]
            import axono.config as cfg

            importlib.reload(cfg)

            # Call _get_data_dir while env var is still set
            result = cfg._get_data_dir()

        assert result == Path("/custom/data/path")


# ---------------------------------------------------------------------------
# Tests for default values
# ---------------------------------------------------------------------------


class TestDefaults:
    """When no rc file, .env, or env vars are present, defaults apply."""

    def test_default_base_url(self):
        cfg = _reload_config()
        assert cfg.LLM_BASE_URL == "http://192.168.32.1:1234/v1"

    def test_default_model_name(self):
        cfg = _reload_config()
        # Empty by default; allows LM Studio to use its loaded model
        assert cfg.LLM_MODEL_NAME == ""

    def test_default_model_provider(self):
        cfg = _reload_config()
        assert cfg.LLM_MODEL_PROVIDER == "openai"

    def test_default_api_key(self):
        cfg = _reload_config()
        assert cfg.LLM_API_KEY == "lm-studio"

    def test_default_command_timeout(self):
        cfg = _reload_config()
        assert cfg.COMMAND_TIMEOUT == 30
        assert isinstance(cfg.COMMAND_TIMEOUT, int)

    def test_default_investigation_enabled(self):
        cfg = _reload_config()
        assert cfg.INVESTIGATION_ENABLED is True

    def test_default_max_context_files(self):
        cfg = _reload_config()
        assert cfg.MAX_CONTEXT_FILES == 8

    def test_default_max_context_chars(self):
        cfg = _reload_config()
        assert cfg.MAX_CONTEXT_CHARS == 30_000


# ---------------------------------------------------------------------------
# Tests for config-file overrides
# ---------------------------------------------------------------------------


class TestConfigFile:
    """Values in ~/.axono/config.json override defaults."""

    def test_config_overrides_defaults(self):
        cfg = _reload_config(
            config={
                "base_url": "http://localhost:9999/v1",
                "model_name": "my-model",
            }
        )
        assert cfg.LLM_BASE_URL == "http://localhost:9999/v1"
        assert cfg.LLM_MODEL_NAME == "my-model"

    def test_config_partial_override(self):
        cfg = _reload_config(config={"model_name": "custom-model"})
        assert cfg.LLM_MODEL_NAME == "custom-model"
        # Others should still be defaults
        assert cfg.LLM_BASE_URL == "http://192.168.32.1:1234/v1"

    def test_config_integer_values(self):
        cfg = _reload_config(
            config={
                "command_timeout": "60",
                "max_context_files": "20",
            }
        )
        assert cfg.COMMAND_TIMEOUT == 60
        assert cfg.MAX_CONTEXT_FILES == 20

    def test_config_boolean_values(self):
        for truthy in ("true", "1", "yes", "on", "True", "YES", "ON"):
            cfg = _reload_config(config={"investigation_enabled": truthy})
            assert cfg.INVESTIGATION_ENABLED is True, f"Expected True for {truthy!r}"

        for falsy in ("false", "0", "no", "off"):
            cfg = _reload_config(config={"investigation_enabled": falsy})
            assert cfg.INVESTIGATION_ENABLED is False, f"Expected False for {falsy!r}"

    def test_invalid_json_falls_back_to_defaults(self):
        """Malformed config.json is silently ignored; defaults apply."""
        cfg = _reload_config(config_raw="not valid json{{{")
        assert cfg.LLM_BASE_URL == "http://192.168.32.1:1234/v1"
        assert cfg.LLM_MODEL_NAME == ""  # Empty default

    def test_non_dict_json_falls_back_to_defaults(self):
        """config.json containing a non-dict value is silently ignored."""
        cfg = _reload_config(config_raw=json.dumps(["a", "list"]))
        assert cfg.LLM_BASE_URL == "http://192.168.32.1:1234/v1"
        assert cfg.LLM_MODEL_NAME == ""  # Empty default


# ---------------------------------------------------------------------------
# Tests for environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvVarOverrides:
    """Environment variables have highest priority."""

    def test_env_overrides_default(self):
        cfg = _reload_config(env={"LLM_MODEL_NAME": "env-model"})
        assert cfg.LLM_MODEL_NAME == "env-model"

    def test_env_overrides_config_file(self):
        cfg = _reload_config(
            env={"LLM_MODEL_NAME": "env-model"},
            config={"model_name": "file-model"},
        )
        assert cfg.LLM_MODEL_NAME == "env-model"

    def test_all_env_vars(self):
        env = {
            "LLM_BASE_URL": "http://env:1234",
            "LLM_MODEL_NAME": "env-m",
            "LLM_MODEL_PROVIDER": "env-p",
            "LLM_API_KEY": "env-k",
            "COMMAND_TIMEOUT": "99",
            "INVESTIGATION_ENABLED": "false",
            "MAX_CONTEXT_FILES": "42",
            "MAX_CONTEXT_CHARS": "50000",
        }
        cfg = _reload_config(env=env)
        assert cfg.LLM_BASE_URL == "http://env:1234"
        assert cfg.LLM_MODEL_NAME == "env-m"
        assert cfg.LLM_MODEL_PROVIDER == "env-p"
        assert cfg.LLM_API_KEY == "env-k"
        assert cfg.COMMAND_TIMEOUT == 99
        assert cfg.INVESTIGATION_ENABLED is False
        assert cfg.MAX_CONTEXT_FILES == 42
        assert cfg.MAX_CONTEXT_CHARS == 50_000

    def test_empty_env_var_falls_through(self):
        """An empty env var should not override the default."""
        cfg = _reload_config(env={"LLM_MODEL_NAME": ""})
        # Default is now empty; empty env var still falls through to default
        assert cfg.LLM_MODEL_NAME == ""


# ---------------------------------------------------------------------------
# Tests for load_mcp_config()
# ---------------------------------------------------------------------------


class TestLoadMcpConfig:
    """Tests for the MCP server configuration loader.

    ``load_mcp_config()`` calls ``Path.home()`` at runtime, so we need
    ``HOME`` to be set both during module reload *and* when calling the
    function.  We use ``mock.patch.dict`` around the call itself.
    """

    @staticmethod
    def _call(cfg, home, extra_env=None):
        """Call ``cfg.load_mcp_config()`` with *home* as ``HOME``."""
        env = {"HOME": home, "MCP_CONFIG_PATH": ""}
        if extra_env:
            env.update(extra_env)
        with mock.patch.dict(os.environ, env, clear=False):
            return cfg.load_mcp_config()

    def test_no_config_returns_empty(self):
        cfg = _reload_config()
        assert cfg.load_mcp_config() == {}

    def test_loads_from_axono_mcp_json(self, tmp_path):
        mcp_data = {"servers": {"my-server": {"command": "echo", "args": ["hi"]}}}
        axono_dir = tmp_path / ".axono"
        axono_dir.mkdir()
        (axono_dir / "mcp.json").write_text(json.dumps(mcp_data))

        cfg = _reload_config(env={"HOME": str(tmp_path)})
        result = self._call(cfg, str(tmp_path))
        assert result == {"my-server": {"command": "echo", "args": ["hi"]}}

    def test_mcp_config_path_env_override(self, tmp_path):
        custom_path = tmp_path / "custom-mcp.json"
        custom_path.write_text(json.dumps({"servers": {"custom": {}}}))

        cfg = _reload_config(
            env={"HOME": str(tmp_path), "MCP_CONFIG_PATH": str(custom_path)}
        )
        result = self._call(
            cfg, str(tmp_path), extra_env={"MCP_CONFIG_PATH": str(custom_path)}
        )
        assert "custom" in result

    def test_invalid_json_returns_empty(self, tmp_path):
        axono_dir = tmp_path / ".axono"
        axono_dir.mkdir()
        (axono_dir / "mcp.json").write_text("not valid json{{{")

        cfg = _reload_config(env={"HOME": str(tmp_path)})
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            result = self._call(cfg, str(tmp_path))
        assert result == {}
        assert "Warning" in stderr.getvalue()

    def test_servers_not_dict_returns_empty(self, tmp_path):
        axono_dir = tmp_path / ".axono"
        axono_dir.mkdir()
        (axono_dir / "mcp.json").write_text(json.dumps({"servers": ["bad"]}))

        cfg = _reload_config(env={"HOME": str(tmp_path)})
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            result = self._call(cfg, str(tmp_path))
        assert result == {}
        assert "not a dict" in stderr.getvalue()

    def test_missing_servers_key_returns_empty(self, tmp_path):
        axono_dir = tmp_path / ".axono"
        axono_dir.mkdir()
        (axono_dir / "mcp.json").write_text(json.dumps({"other": "data"}))

        cfg = _reload_config(env={"HOME": str(tmp_path)})
        result = self._call(cfg, str(tmp_path))
        assert result == {}


# ---------------------------------------------------------------------------
# Tests for model type configuration (instruction_model / reasoning_model)
# ---------------------------------------------------------------------------


class TestModelTypes:
    """Tests for instruction_model and reasoning_model configuration."""

    def test_default_instruction_model_empty(self):
        cfg = _reload_config()
        assert cfg.LLM_INSTRUCTION_MODEL == ""

    def test_default_reasoning_model_empty(self):
        cfg = _reload_config()
        assert cfg.LLM_REASONING_MODEL == ""

    def test_get_model_name_default(self):
        """When all model configs are empty, returns empty string."""
        cfg = _reload_config()
        assert cfg.get_model_name("instruction") == ""
        assert cfg.get_model_name("reasoning") == ""

    def test_get_model_name_uses_model_name(self):
        """When only model_name is set, both types use it."""
        cfg = _reload_config(config={"model_name": "base-model"})
        assert cfg.get_model_name("instruction") == "base-model"
        assert cfg.get_model_name("reasoning") == "base-model"

    def test_get_model_name_instruction_model_overrides(self):
        """instruction_model overrides model_name for both types."""
        cfg = _reload_config(
            config={
                "model_name": "base-model",
                "instruction_model": "instruct-model",
            }
        )
        assert cfg.get_model_name("instruction") == "instruct-model"
        assert cfg.get_model_name("reasoning") == "instruct-model"

    def test_get_model_name_reasoning_model_overrides(self):
        """reasoning_model overrides for reasoning type only."""
        cfg = _reload_config(
            config={
                "model_name": "base-model",
                "instruction_model": "instruct-model",
                "reasoning_model": "reason-model",
            }
        )
        assert cfg.get_model_name("instruction") == "instruct-model"
        assert cfg.get_model_name("reasoning") == "reason-model"

    def test_get_model_name_reasoning_without_instruction(self):
        """reasoning_model with empty instruction_model falls back to model_name."""
        cfg = _reload_config(
            config={
                "model_name": "base-model",
                "reasoning_model": "reason-model",
            }
        )
        assert cfg.get_model_name("instruction") == "base-model"
        assert cfg.get_model_name("reasoning") == "reason-model"

    def test_env_vars_override(self):
        """Env vars take priority over config file."""
        cfg = _reload_config(
            config={
                "instruction_model": "file-instruct",
                "reasoning_model": "file-reason",
            },
            env={
                "LLM_INSTRUCTION_MODEL": "env-instruct",
                "LLM_REASONING_MODEL": "env-reason",
            },
        )
        assert cfg.get_model_name("instruction") == "env-instruct"
        assert cfg.get_model_name("reasoning") == "env-reason"
