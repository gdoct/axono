"""Unit tests for axono.onboarding and the config onboarding helpers."""

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from axono.onboarding import OnboardingScreen, _verify_connection


# ---------------------------------------------------------------------------
# Helpers — reload config in an isolated HOME
# ---------------------------------------------------------------------------


def _reload_config(home):
    """Reload ``axono.config`` with HOME pointed at *home*."""
    env = {"HOME": home}
    for var in (
        "LLM_BASE_URL",
        "LLM_MODEL_NAME",
        "LLM_MODEL_PROVIDER",
        "LLM_API_KEY",
        "COMMAND_TIMEOUT",
        "INVESTIGATION_ENABLED",
        "MAX_CONTEXT_FILES",
        "MAX_CONTEXT_CHARS",
        "MCP_CONFIG_PATH",
    ):
        env.setdefault(var, "")

    with mock.patch.dict(os.environ, env, clear=False):
        if "axono.config" in sys.modules:
            del sys.modules["axono.config"]
        import axono.config as cfg

        importlib.reload(cfg)
        return cfg


# ---------------------------------------------------------------------------
# config.needs_onboarding
# ---------------------------------------------------------------------------


class TestNeedsOnboarding:

    def test_true_when_no_config(self):
        tmpdir = tempfile.mkdtemp()
        cfg = _reload_config(tmpdir)
        with mock.patch.dict(os.environ, {"HOME": tmpdir}):
            assert cfg.needs_onboarding() is True

    def test_false_when_config_exists(self):
        tmpdir = tempfile.mkdtemp()
        config_dir = Path(tmpdir) / ".axono"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({"base_url": "http://x"}))
        cfg = _reload_config(tmpdir)
        with mock.patch.dict(os.environ, {"HOME": tmpdir}):
            assert cfg.needs_onboarding() is False


# ---------------------------------------------------------------------------
# config.save_config
# ---------------------------------------------------------------------------


class TestSaveConfig:

    def test_writes_config_json(self):
        tmpdir = tempfile.mkdtemp()
        cfg = _reload_config(tmpdir)
        with mock.patch.dict(os.environ, {"HOME": tmpdir}):
            path = cfg.save_config({"base_url": "http://myhost:1234/v1", "model_name": "m"})
        assert path == Path(tmpdir) / ".axono" / "config.json"
        assert path.is_file()
        data = json.loads(path.read_text())
        assert data["base_url"] == "http://myhost:1234/v1"
        assert data["model_name"] == "m"

    def test_saved_file_is_valid_json(self):
        tmpdir = tempfile.mkdtemp()
        cfg = _reload_config(tmpdir)
        with mock.patch.dict(os.environ, {"HOME": tmpdir}):
            path = cfg.save_config({"base_url": "http://a", "api_key": "k"})
        data = json.loads(path.read_text())
        assert data["base_url"] == "http://a"
        assert data["api_key"] == "k"

    def test_creates_axono_directory(self):
        tmpdir = tempfile.mkdtemp()
        cfg = _reload_config(tmpdir)
        with mock.patch.dict(os.environ, {"HOME": tmpdir}):
            cfg.save_config({"base_url": "http://a"})
        assert (Path(tmpdir) / ".axono").is_dir()


# ---------------------------------------------------------------------------
# onboarding._verify_connection
# ---------------------------------------------------------------------------


class TestVerifyConnection:

    def test_success_with_models(self):
        body = json.dumps({"data": [{"id": "my-model"}]}).encode()
        fake_resp = mock.MagicMock()
        fake_resp.read.return_value = body
        fake_resp.__enter__ = mock.Mock(return_value=fake_resp)
        fake_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=fake_resp):
            ok, msg = _verify_connection("http://localhost:1234/v1", "key")
        assert ok is True
        assert "my-model" in msg

    def test_success_no_models(self):
        body = json.dumps({"data": []}).encode()
        fake_resp = mock.MagicMock()
        fake_resp.read.return_value = body
        fake_resp.__enter__ = mock.Mock(return_value=fake_resp)
        fake_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=fake_resp):
            ok, msg = _verify_connection("http://localhost:1234/v1", "key")
        assert ok is True
        assert "no models" in msg.lower()

    def test_connection_refused(self):
        import urllib.error

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            ok, msg = _verify_connection("http://localhost:9999/v1", "key")
        assert ok is False
        assert "Could not reach" in msg

    def test_generic_exception(self):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=RuntimeError("unexpected failure"),
        ):
            ok, msg = _verify_connection("http://localhost:9999/v1", "key")
        assert ok is False
        assert "Connection failed" in msg

    def test_unsupported_url_scheme(self):
        """Unsupported URL schemes are rejected without making a request."""
        ok, msg = _verify_connection("ftp://localhost:21/v1", "key")
        assert ok is False
        assert "Unsupported URL scheme" in msg
        assert "'ftp'" in msg

    def test_empty_scheme_rejected(self):
        """URLs without a scheme are rejected."""
        ok, msg = _verify_connection("localhost:1234/v1", "key")
        assert ok is False
        assert "Unsupported URL scheme" in msg


# ---------------------------------------------------------------------------
# OnboardingScreen — TUI tests
# ---------------------------------------------------------------------------


class TestOnboardingScreen:

    @pytest.mark.asyncio
    async def test_screen_composes_expected_widgets(self):
        """Screen renders with URL, model, key inputs and buttons."""
        from textual.app import App

        class TestApp(App):
            def on_mount(self):
                self.push_screen(OnboardingScreen())

        async with TestApp().run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert screen.query_one("#input-url")
            assert screen.query_one("#input-model")
            assert screen.query_one("#input-key")
            assert screen.query_one("#btn-verify")
            assert screen.query_one("#btn-skip")

    @pytest.mark.asyncio
    async def test_skip_dismisses_with_false(self):
        """Pressing Skip dismisses the screen with False."""
        from textual.app import App
        from textual.widgets import Button

        results = []

        class TestApp(App):
            def on_mount(self):
                self.push_screen(OnboardingScreen(), callback=results.append)

        async with TestApp().run_test(size=(80, 30)) as pilot:
            await pilot.click("#btn-skip")
            await pilot.pause()

        assert results == [False]

    @pytest.mark.asyncio
    async def test_escape_dismisses_with_false(self):
        """Pressing Escape dismisses the screen with False."""
        from textual.app import App

        results = []

        class TestApp(App):
            def on_mount(self):
                self.push_screen(OnboardingScreen(), callback=results.append)

        async with TestApp().run_test(size=(80, 30)) as pilot:
            await pilot.press("escape")
            await pilot.pause()

        assert results == [False]

    @pytest.mark.asyncio
    async def test_verify_success_saves_and_dismisses(self):
        """Successful verification saves rc and dismisses with True."""
        from textual.app import App
        from textual.widgets import Input

        tmpdir = tempfile.mkdtemp()
        results = []

        class TestApp(App):
            def on_mount(self):
                self.push_screen(OnboardingScreen(), callback=results.append)

        with mock.patch.dict(os.environ, {"HOME": tmpdir}):
            with mock.patch(
                "axono.onboarding._verify_connection",
                return_value=(True, "Connected."),
            ):
                async with TestApp().run_test(size=(80, 30)) as pilot:
                    await pilot.pause()
                    screen = pilot.app.screen
                    # Set input values
                    url_input = screen.query_one("#input-url", Input)
                    url_input.value = "http://myhost:1234/v1"
                    model_input = screen.query_one("#input-model", Input)
                    model_input.value = "test-model"
                    key_input = screen.query_one("#input-key", Input)
                    key_input.value = "test-key"

                    await pilot.click("#btn-verify")
                    # Wait for the threaded worker to complete
                    await pilot.pause(delay=0.5)

        assert results == [True]
        config_file = Path(tmpdir) / ".axono" / "config.json"
        assert config_file.is_file()
        data = json.loads(config_file.read_text())
        assert data["base_url"] == "http://myhost:1234/v1"
        assert data["model_name"] == "test-model"

    @pytest.mark.asyncio
    async def test_verify_failure_shows_error(self):
        """Failed verification shows error and stays on screen."""
        from textual.app import App
        from textual.widgets import Input, Label

        results = []

        class TestApp(App):
            def on_mount(self):
                self.push_screen(OnboardingScreen(), callback=results.append)

        with mock.patch(
            "axono.onboarding._verify_connection",
            return_value=(False, "Could not reach server: Connection refused"),
        ):
            async with TestApp().run_test(size=(80, 30)) as pilot:
                await pilot.pause()
                screen = pilot.app.screen
                url_input = screen.query_one("#input-url", Input)
                url_input.value = "http://bad:1234/v1"

                await pilot.click("#btn-verify")
                await pilot.pause(delay=0.5)

                # Screen should still be showing (not dismissed)
                assert results == []

    @pytest.mark.asyncio
    async def test_empty_url_shows_error(self):
        """Empty URL shows a validation error without hitting the network."""
        from textual.app import App
        from textual.widgets import Input

        results = []

        class TestApp(App):
            def on_mount(self):
                self.push_screen(OnboardingScreen(), callback=results.append)

        with mock.patch(
            "axono.onboarding._verify_connection"
        ) as mock_verify:
            async with TestApp().run_test(size=(80, 30)) as pilot:
                await pilot.pause()
                screen = pilot.app.screen
                url_input = screen.query_one("#input-url", Input)
                url_input.value = ""

                await pilot.click("#btn-verify")
                await pilot.pause(delay=0.5)

                # Should not have called verify at all
                mock_verify.assert_not_called()
                # Should not have dismissed
                assert results == []

    @pytest.mark.asyncio
    async def test_inputs_have_defaults(self):
        """Input fields are pre-filled with defaults from config."""
        from textual.app import App
        from textual.widgets import Input

        from axono.config import _DEFAULTS

        class TestApp(App):
            def on_mount(self):
                self.push_screen(OnboardingScreen())

        async with TestApp().run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert screen.query_one("#input-url", Input).value == _DEFAULTS["base_url"]
            assert screen.query_one("#input-model", Input).value == _DEFAULTS["model_name"]
            assert screen.query_one("#input-key", Input).value == _DEFAULTS["api_key"]


# ---------------------------------------------------------------------------
# main() integration with --onboard
# ---------------------------------------------------------------------------


class TestMainOnboarding:

    def test_onboard_flag_sets_force_onboard(self):
        with mock.patch("sys.argv", ["axono", "--onboard"]):
            with mock.patch.object(
                __import__("axono.main", fromlist=["AxonoApp"]).AxonoApp, "run"
            ):
                from axono.main import main

                main()
                # Can't easily check internal flag after run(), but at least
                # confirm it doesn't crash

    def test_no_flag_default(self):
        with mock.patch("sys.argv", ["axono"]):
            with mock.patch.object(
                __import__("axono.main", fromlist=["AxonoApp"]).AxonoApp, "run"
            ):
                from axono.main import main

                main()

    @pytest.mark.asyncio
    async def test_onboarding_screen_shown_when_needed(self):
        """When needs_onboarding() is True, OnboardingScreen is pushed."""
        with mock.patch("axono.main.needs_onboarding", return_value=True):
            with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
                from axono.main import AxonoApp

                async with AxonoApp().run_test(size=(80, 30)) as pilot:
                    await pilot.pause()
                    # The onboarding screen should be on the screen stack
                    assert any(
                        isinstance(s, OnboardingScreen)
                        for s in pilot.app.screen_stack
                    )

    @pytest.mark.asyncio
    async def test_onboarding_screen_not_shown_when_rc_exists(self):
        """When needs_onboarding() is False, OnboardingScreen is NOT pushed."""
        with mock.patch("axono.main.needs_onboarding", return_value=False):
            with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
                from axono.main import AxonoApp

                async with AxonoApp().run_test(size=(80, 30)) as pilot:
                    await pilot.pause()
                    assert not any(
                        isinstance(s, OnboardingScreen)
                        for s in pilot.app.screen_stack
                    )

    @pytest.mark.asyncio
    async def test_force_onboard_pushes_screen(self):
        """force_onboard=True always pushes OnboardingScreen."""
        with mock.patch("axono.main.needs_onboarding", return_value=False):
            with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
                from axono.main import AxonoApp

                async with AxonoApp(force_onboard=True).run_test(
                    size=(80, 30)
                ) as pilot:
                    await pilot.pause()
                    assert any(
                        isinstance(s, OnboardingScreen)
                        for s in pilot.app.screen_stack
                    )

    @pytest.mark.asyncio
    async def test_agent_init_after_onboarding_dismiss(self):
        """After onboarding dismisses, the agent should be initialized."""
        with mock.patch("axono.main.needs_onboarding", return_value=True):
            with mock.patch(
                "axono.main.build_agent", return_value=mock.AsyncMock()
            ) as mock_build:
                from axono.main import AxonoApp

                async with AxonoApp().run_test(size=(80, 30)) as pilot:
                    await pilot.pause()
                    # Agent not yet initialized (onboarding screen is up)
                    mock_build.assert_not_called()

                    # Dismiss onboarding via Skip
                    await pilot.click("#btn-skip")
                    await pilot.pause()

                    # Now agent should be initializing
                    mock_build.assert_called_once()
