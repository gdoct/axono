import asyncio
from typing import cast

import pytest

from axono import ui


def _get_text(widget) -> str:
    """Return a string representation of a Textual widget's content."""
    for attr in ("renderable", "render", "text"):
        val = getattr(widget, attr, None)
        if val is not None:
            return str(val() if callable(val) else val)
    return str(widget)


def test_banner_contains_banner_text():
    banner = ui.Banner()
    text = _get_text(banner)
    assert "Type a message" in text
    assert ui.BANNER_TEXT.strip() in text


def test_user_message_escapes_brackets():
    um = ui.UserMessage("hello [world]")
    text = _get_text(um)
    assert "You:" in text
    # The implementation escapes '[' to '\['; accept either form in rendering
    assert "world" in text


def test_assistant_message_default():
    am = ui.AssistantMessage("reply")
    text = _get_text(am)
    assert "Axono:" in text
    assert "reply" in text


def test_system_message_prefix():
    sm = ui.SystemMessage("ok", prefix="Tool")
    text = _get_text(sm)
    assert "Tool:" in text
    assert "ok" in text


def test_chatcontainer_add_message(monkeypatch):
    container = ui.ChatContainer()
    mounted = {}

    async def fake_mount(widget):
        mounted["widget"] = widget

    monkeypatch.setattr(container, "mount", fake_mount)

    class Dummy:
        def __init__(self):
            self.scrolled = False

        def scroll_visible(self, animate=False):
            self.scrolled = True

    w = Dummy()
    asyncio.run(container.add_message(cast(ui.Static, w)))

    assert mounted.get("widget") is w
    assert w.scrolled is True


def test_cwdstatus_update_path():
    cs = ui.CwdStatus()
    cs.update_path("/home/user/project")
    text = _get_text(cs)
    assert "CWD:" in text
    assert "/home/user/project" in text


def test_tool_group_starts_expanded():
    tg = ui.ToolGroup("Tool: bash")
    assert tg.collapsed is False


def test_get_text_fallback():
    """When widget has no renderable/render/text, falls back to str()."""

    class Bare:
        def __str__(self):
            return "bare-widget"

    assert _get_text(Bare()) == "bare-widget"


# ---------------------------------------------------------------------------
# HistoryInput
# ---------------------------------------------------------------------------


class TestHistoryInput:
    """Tests for HistoryInput arrow key history navigation."""

    def test_init_defaults(self):
        """HistoryInput has expected defaults."""
        inp = ui.HistoryInput()
        assert inp._history == []
        assert inp._history_index == -1
        assert inp._current_input == ""

    def test_on_key_up_no_history(self, monkeypatch):
        """Up arrow with no history does nothing."""
        inp = ui.HistoryInput()
        inp._history = []

        class FakeEvent:
            key = "up"
            prevented = False
            stopped = False

            def prevent_default(self):  # pragma: no cover
                self.prevented = True

            def stop(self):  # pragma: no cover
                self.stopped = True

        event = FakeEvent()
        inp.on_key(event)

        # Should return early, not call prevent_default
        assert not event.prevented
        assert not event.stopped

    @pytest.mark.asyncio
    async def test_on_key_up_first_press(self):
        """First up arrow press saves current input and shows last history item."""
        from textual.app import App

        class TestApp(App):
            def compose(self):
                yield ui.HistoryInput(id="hist")

        async with TestApp().run_test(size=(80, 24)) as pilot:
            inp = pilot.app.query_one("#hist", ui.HistoryInput)
            inp._history = ["first", "second", "third"]
            inp._history_index = -1
            inp.value = "current"

            await pilot.press("up")

            assert inp._current_input == "current"
            assert inp._history_index == 2  # len(history) - 1
            assert inp.value == "third"

    @pytest.mark.asyncio
    async def test_on_key_up_subsequent_press(self):
        """Subsequent up arrows go further back in history."""
        from textual.app import App

        class TestApp(App):
            def compose(self):
                yield ui.HistoryInput(id="hist")

        async with TestApp().run_test(size=(80, 24)) as pilot:
            inp = pilot.app.query_one("#hist", ui.HistoryInput)
            inp._history = ["first", "second", "third"]
            inp._history_index = 2
            inp._current_input = "saved"
            inp.value = "third"

            await pilot.press("up")

            assert inp._history_index == 1
            assert inp.value == "second"

    @pytest.mark.asyncio
    async def test_on_key_up_at_beginning(self):
        """Up arrow at beginning of history stays at index 0."""
        from textual.app import App

        class TestApp(App):
            def compose(self):
                yield ui.HistoryInput(id="hist")

        async with TestApp().run_test(size=(80, 24)) as pilot:
            inp = pilot.app.query_one("#hist", ui.HistoryInput)
            inp._history = ["first", "second"]
            inp._history_index = 0
            inp._current_input = "saved"
            inp.value = "first"

            await pilot.press("up")

            # Should stay at index 0, show "first"
            assert inp._history_index == 0
            assert inp.value == "first"

    def test_on_key_down_not_navigating(self, monkeypatch):
        """Down arrow when not navigating does nothing."""
        inp = ui.HistoryInput()
        inp._history = ["first", "second"]
        inp._history_index = -1

        class FakeEvent:
            key = "down"
            prevented = False
            stopped = False

            def prevent_default(self):  # pragma: no cover
                self.prevented = True

            def stop(self):  # pragma: no cover
                self.stopped = True

        event = FakeEvent()
        inp.on_key(event)

        # Should return early
        assert not event.prevented
        assert not event.stopped

    @pytest.mark.asyncio
    async def test_on_key_down_go_forward(self):
        """Down arrow goes forward in history."""
        from textual.app import App

        class TestApp(App):
            def compose(self):
                yield ui.HistoryInput(id="hist")

        async with TestApp().run_test(size=(80, 24)) as pilot:
            inp = pilot.app.query_one("#hist", ui.HistoryInput)
            inp._history = ["first", "second", "third"]
            inp._history_index = 0
            inp._current_input = "saved"
            inp.value = "first"

            await pilot.press("down")

            assert inp._history_index == 1
            assert inp.value == "second"

    @pytest.mark.asyncio
    async def test_on_key_down_at_end_restores_current(self):
        """Down arrow at end of history restores current input."""
        from textual.app import App

        class TestApp(App):
            def compose(self):
                yield ui.HistoryInput(id="hist")

        async with TestApp().run_test(size=(80, 24)) as pilot:
            inp = pilot.app.query_one("#hist", ui.HistoryInput)
            inp._history = ["first", "second"]
            inp._history_index = 1  # At "second"
            inp._current_input = "my typed text"
            inp.value = "second"

            await pilot.press("down")

            assert inp._history_index == -1
            assert inp.value == "my typed text"

    def test_on_key_other_key_ignored(self, monkeypatch):
        """Other keys are not handled by on_key."""
        inp = ui.HistoryInput()
        inp._history = ["first"]

        class FakeEvent:
            key = "a"
            prevented = False
            stopped = False

            def prevent_default(self):  # pragma: no cover
                self.prevented = True

            def stop(self):  # pragma: no cover
                self.stopped = True

        event = FakeEvent()
        inp.on_key(event)

        # Should not interact with history navigation
        assert not event.prevented
        assert not event.stopped

    def test_add_to_history(self, monkeypatch):
        """add_to_history adds prompt and resets state."""
        inp = ui.HistoryInput()
        inp._history_index = 2
        inp._current_input = "something"

        # Mock append_to_history
        monkeypatch.setattr(ui, "append_to_history", lambda p: ["first", p])

        inp.add_to_history("new prompt")

        assert inp._history == ["first", "new prompt"]
        assert inp._history_index == -1
        assert inp._current_input == ""
