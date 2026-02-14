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
        "LLM_INVESTIGATION",
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

    def test_default_llm_investigation(self):
        cfg = _reload_config()
        assert cfg.LLM_INVESTIGATION is False


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

    def test_llm_investigation_boolean(self):
        cfg = _reload_config(config={"llm_investigation": "true"})
        assert cfg.LLM_INVESTIGATION is True

        cfg = _reload_config(config={"llm_investigation": "false"})
        assert cfg.LLM_INVESTIGATION is False

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
            "LLM_INVESTIGATION": "true",
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
        assert cfg.LLM_INVESTIGATION is True
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
        """model_name is no longer used; should return empty string."""
        cfg = _reload_config(config={"model_name": "base-model"})
        assert cfg.get_model_name("instruction") == ""
        assert cfg.get_model_name("reasoning") == ""

    def test_get_model_name_instruction_model_overrides(self):
        """instruction_model is used for both types when set."""
        cfg = _reload_config(
            config={
                "instruction_model": "instruct-model",
            }
        )
        assert cfg.get_model_name("instruction") == "instruct-model"
        assert cfg.get_model_name("reasoning") == "instruct-model"

    def test_get_model_name_reasoning_model_overrides(self):
        """reasoning_model overrides for reasoning type only."""
        cfg = _reload_config(
            config={
                "instruction_model": "instruct-model",
                "reasoning_model": "reason-model",
            }
        )
        assert cfg.get_model_name("instruction") == "instruct-model"
        assert cfg.get_model_name("reasoning") == "reason-model"

    def test_get_model_name_reasoning_without_instruction(self):
        """reasoning_model alone doesn't fall back for instruction type."""
        cfg = _reload_config(
            config={
                "reasoning_model": "reason-model",
            }
        )
        assert cfg.get_model_name("instruction") == ""
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


# ---------------------------------------------------------------------------
# Tests for get_embedding_db_path()
# ---------------------------------------------------------------------------


class TestGetEmbeddingDbPath:
    """Tests for the embedding database path resolver."""

    def test_default_path(self, tmp_path):
        """Without custom config, returns <data_dir>/embeddings.db."""
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            result = cfg.get_embedding_db_path()
        assert result == tmp_path / ".axono" / "embeddings.db"

    def test_custom_path(self, tmp_path):
        """When embedding_db_path is set in config, that path is used."""
        custom = str(tmp_path / "custom" / "embed.db")
        # Write config to tmp_path so _reload_config finds it with HOME=tmp_path
        axono_dir = tmp_path / ".axono"
        axono_dir.mkdir(exist_ok=True)
        (axono_dir / "config.json").write_text(
            json.dumps({"embedding_db_path": custom})
        )
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            result = cfg.get_embedding_db_path()
        assert result == Path(custom)


# ---------------------------------------------------------------------------
# Tests for onboarding helpers (config_dir, config_path, needs_onboarding)
# ---------------------------------------------------------------------------


class TestOnboardingHelpers:
    """Tests for config_dir(), config_path(), and needs_onboarding()."""

    def test_config_dir(self, tmp_path):
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            assert cfg.config_dir() == tmp_path / ".axono"

    def test_config_path(self, tmp_path):
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            assert cfg.config_path() == tmp_path / ".axono" / "config.json"

    def test_needs_onboarding_true(self, tmp_path):
        """Returns True when config.json does not exist."""
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            assert cfg.needs_onboarding() is True

    def test_needs_onboarding_false(self, tmp_path):
        """Returns False when config.json already exists."""
        # Write config to tmp_path directly so needs_onboarding finds it
        axono_dir = tmp_path / ".axono"
        axono_dir.mkdir(exist_ok=True)
        (axono_dir / "config.json").write_text(
            json.dumps({"base_url": "http://localhost:1234/v1"})
        )
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            assert cfg.needs_onboarding() is False


# ---------------------------------------------------------------------------
# Tests for save_config()
# ---------------------------------------------------------------------------


class TestSaveConfig:
    """Tests for writing the config file."""

    def test_save_config_writes_json(self, tmp_path):
        """save_config creates the directory and writes valid JSON."""
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        settings = {"base_url": "http://example:5000/v1", "model_name": "test"}
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            result = cfg.save_config(settings)

        assert result == tmp_path / ".axono" / "config.json"
        assert result.is_file()
        content = json.loads(result.read_text(encoding="utf-8"))
        assert content == settings


# ---------------------------------------------------------------------------
# Tests for _trusted_folders_path()
# ---------------------------------------------------------------------------


class TestTrustedFoldersPath:
    """Tests for the trusted folders path helper."""

    def test_returns_correct_path(self, tmp_path):
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            result = cfg._trusted_folders_path()
        assert result == tmp_path / ".axono" / "trusted_folders.json"


# ---------------------------------------------------------------------------
# Tests for load_trusted_folders()
# ---------------------------------------------------------------------------


class TestLoadTrustedFolders:
    """Tests for loading trusted folders from disk."""

    def test_no_file_returns_empty(self, tmp_path):
        """When the file doesn't exist, returns an empty list."""
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            assert cfg.load_trusted_folders() == []

    def test_valid_file(self, tmp_path):
        """Loads a well-formed JSON list of strings."""
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        axono_dir = tmp_path / ".axono"
        axono_dir.mkdir(exist_ok=True)
        (axono_dir / "trusted_folders.json").write_text(
            json.dumps(["/home/user/project1", "/home/user/project2"])
        )
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            result = cfg.load_trusted_folders()
        assert result == ["/home/user/project1", "/home/user/project2"]

    def test_non_list_json_returns_empty(self, tmp_path):
        """When file contains a non-list JSON value, returns empty list."""
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        axono_dir = tmp_path / ".axono"
        axono_dir.mkdir(exist_ok=True)
        (axono_dir / "trusted_folders.json").write_text(json.dumps({"a": "dict"}))
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            assert cfg.load_trusted_folders() == []

    def test_invalid_json_returns_empty(self, tmp_path):
        """Malformed JSON is silently ignored."""
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        axono_dir = tmp_path / ".axono"
        axono_dir.mkdir(exist_ok=True)
        (axono_dir / "trusted_folders.json").write_text("not valid json{{{")
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            assert cfg.load_trusted_folders() == []


# ---------------------------------------------------------------------------
# Tests for save_trusted_folders()
# ---------------------------------------------------------------------------


class TestSaveTrustedFolders:
    """Tests for saving trusted folders to disk."""

    def test_save_writes_file(self, tmp_path):
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        folders = ["/home/user/project1", "/tmp/project2"]
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            result = cfg.save_trusted_folders(folders)

        assert result == tmp_path / ".axono" / "trusted_folders.json"
        assert result.is_file()
        content = json.loads(result.read_text(encoding="utf-8"))
        assert content == folders


# ---------------------------------------------------------------------------
# Tests for add_trusted_folder()
# ---------------------------------------------------------------------------


class TestAddTrustedFolder:
    """Tests for adding a folder to the trusted list."""

    def test_adds_new_folder(self, tmp_path):
        """A new folder is appended and saved."""
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            result = cfg.add_trusted_folder("/home/user/project")
        normalized = os.path.abspath("/home/user/project")
        assert normalized in result

    def test_does_not_duplicate(self, tmp_path):
        """Adding the same folder twice does not create duplicates."""
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            cfg.add_trusted_folder("/home/user/project")
            result = cfg.add_trusted_folder("/home/user/project")
        normalized = os.path.abspath("/home/user/project")
        assert result.count(normalized) == 1


# ---------------------------------------------------------------------------
# Tests for is_folder_trusted()
# ---------------------------------------------------------------------------


class TestIsFolderTrusted:
    """Tests for checking folder trust status."""

    def test_trusted_folder(self, tmp_path):
        """Returns True for a folder that has been added."""
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            cfg.add_trusted_folder("/home/user/project")
            assert cfg.is_folder_trusted("/home/user/project") is True

    def test_untrusted_folder(self, tmp_path):
        """Returns False for a folder that has not been added."""
        cfg = _reload_config(env={"HOME": str(tmp_path)})
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            assert cfg.is_folder_trusted("/some/random/path") is False
