"""Onboarding wizard for first-time setup.

Provides an ``OnboardingScreen`` (Textual Screen) that prompts for
LM Studio connection details, verifies them, and writes ``~/.axono/config.json``.
Also keeps the ``_verify_connection`` helper available for direct use.
"""

import json
import urllib.error
import urllib.request

from textual import work
from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, Static

from axono.config import _DEFAULTS, save_config


# ── connection verification (pure I/O, no TUI dependency) ────────


def _verify_connection(base_url: str, api_key: str) -> tuple[bool, str]:
    """Hit the ``/models`` endpoint to verify the server is reachable.

    Returns ``(ok, message)``.
    """
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = data.get("data", [])
            if models:
                names = ", ".join(m.get("id", "?") for m in models[:5])
                extra = f" (and {len(models) - 5} more)" if len(models) > 5 else ""
                return True, f"Connected. Available models: {names}{extra}"
            return True, "Connected (no models currently loaded)."
    except urllib.error.URLError as exc:
        return False, f"Could not reach server: {exc.reason}"
    except Exception as exc:
        return False, f"Connection failed: {exc}"


# ── TUI onboarding screen ───────────────────────────────────────


class OnboardingScreen(Screen[bool]):
    """Full-screen onboarding wizard shown on first run or ``--onboard``."""

    DEFAULT_CSS = """
    OnboardingScreen {
        align: center middle;
    }

    #onboard-box {
        width: 70;
        height: auto;
        padding: 1 2;
        border: thick $accent;
    }

    #onboard-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin: 0 0 1 0;
    }

    .field-label {
        margin: 1 0 0 0;
    }

    #onboard-status {
        margin: 1 0;
        text-align: center;
        min-height: 1;
    }

    #btn-row {
        margin: 1 0 0 0;
        height: auto;
        align: center middle;
    }

    #btn-verify {
        margin: 0 1;
    }

    #btn-skip {
        margin: 0 1;
    }
    """

    BINDINGS = [("escape", "skip", "Skip")]

    def compose(self) -> ComposeResult:
        with Vertical(id="onboard-box"):
            yield Label("Welcome to Axono!", id="onboard-title")
            yield Static("Configure your LM Studio connection.\n")
            yield Label("Server URL", classes="field-label")
            yield Input(
                value=_DEFAULTS["base_url"],
                placeholder="http://localhost:1234/v1",
                id="input-url",
            )
            yield Label("Model name", classes="field-label")
            yield Input(
                value=_DEFAULTS["model_name"],
                placeholder="local-model",
                id="input-model",
            )
            yield Label("API key", classes="field-label")
            yield Input(
                value=_DEFAULTS["api_key"],
                placeholder="lm-studio",
                id="input-key",
            )
            yield Label("", id="onboard-status")
            with Center(id="btn-row"):
                yield Button("Verify & Save", variant="primary", id="btn-verify")
                yield Button("Skip", variant="default", id="btn-skip")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input-url", Input).focus()

    # -- button handlers ------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-verify":
            self._do_verify()
        elif event.button.id == "btn-skip":
            self.action_skip()

    def action_skip(self) -> None:
        self.dismiss(False)

    # -- verify & save (runs in a worker so UI stays responsive) --------

    @work(thread=True)
    def _do_verify(self) -> None:
        base_url = self.query_one("#input-url", Input).value.strip()
        api_key = self.query_one("#input-key", Input).value.strip()
        model_name = self.query_one("#input-model", Input).value.strip()

        if not base_url:
            self.app.call_from_thread(self._set_status, "Server URL is required.", True)
            return

        self.app.call_from_thread(self._set_status, "Verifying connection...", False)

        ok, message = _verify_connection(base_url, api_key)

        if not ok:
            self.app.call_from_thread(self._set_status, message, True)
            return

        # Save settings
        settings = {
            "base_url": base_url,
            "model_name": model_name or _DEFAULTS["model_name"],
            "api_key": api_key or _DEFAULTS["api_key"],
        }
        path = save_config(settings)
        self.app.call_from_thread(
            self._set_status, f"Saved to {path}. {message}", False
        )
        self.app.call_from_thread(self.dismiss, True)

    def _set_status(self, text: str, is_error: bool) -> None:
        label = self.query_one("#onboard-status", Label)
        if is_error:
            label.update(f"[bold red]{text}[/bold red]")
        else:
            label.update(f"[green]{text}[/green]")
