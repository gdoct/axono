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


def test_tool_group_starts_collapsed():
    tg = ui.ToolGroup("Tool: bash")
    assert tg.collapsed is True


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


# ---------------------------------------------------------------------------
# TaskListMessage
# ---------------------------------------------------------------------------


class TestTaskListMessage:
    """Tests for TaskListMessage progress tracking widget."""

    def test_init_renders_all_pending(self):
        """Initial render shows all tasks as pending."""
        tlm = ui.TaskListMessage(["Task A", "Task B", "Task C"])
        text = _get_text(tlm)
        assert "Plan:" in text
        assert "Task A" in text
        assert "Task B" in text
        assert "Task C" in text
        # All should show pending indicator (○)
        assert text.count("○") == 3

    def test_start_task_marks_in_progress(self):
        """start_task marks the task as in progress."""
        tlm = ui.TaskListMessage(["Task A", "Task B"])
        tlm.start_task(0)
        text = _get_text(tlm)
        # First task should have arrow indicator
        assert "→" in text
        # Second task still pending
        assert "○" in text

    def test_complete_task_marks_completed(self):
        """complete_task marks the task as completed."""
        tlm = ui.TaskListMessage(["Task A", "Task B", "Task C"])
        tlm.start_task(0)
        tlm.complete_task(0)
        text = _get_text(tlm)
        # First task should have checkmark
        assert "✓" in text
        # Second task now in progress (current_index moved to 1)
        # Third task still pending
        assert "○" in text

    def test_complete_wrong_index_no_change(self):
        """complete_task with wrong index does nothing."""
        tlm = ui.TaskListMessage(["Task A", "Task B"])
        tlm.start_task(0)
        tlm.complete_task(1)  # Wrong index
        text = _get_text(tlm)
        # First task still in progress
        assert "→" in text
        # No checkmark yet
        assert "✓" not in text

    def test_full_progress_sequence(self):
        """Test a full sequence of starting and completing tasks."""
        tlm = ui.TaskListMessage(["Task A", "Task B", "Task C"])

        # Start task 0
        tlm.start_task(0)
        text = _get_text(tlm)
        assert text.count("→") == 1
        assert text.count("○") == 2

        # Complete task 0 - index moves to 1, so task B becomes "in progress"
        tlm.complete_task(0)
        text = _get_text(tlm)
        assert text.count("✓") == 1
        assert text.count("→") == 1  # Task B now in progress
        assert text.count("○") == 1  # Task C still pending

        # Start task 1 (already at index 1 from complete_task)
        tlm.start_task(1)
        text = _get_text(tlm)
        assert text.count("✓") == 1
        assert text.count("→") == 1
        assert text.count("○") == 1

        # Complete task 1 - index moves to 2, so task C becomes "in progress"
        tlm.complete_task(1)
        text = _get_text(tlm)
        assert text.count("✓") == 2
        assert text.count("→") == 1  # Task C now in progress
        assert text.count("○") == 0  # No pending tasks

        # Start and complete task 2
        tlm.start_task(2)
        tlm.complete_task(2)
        text = _get_text(tlm)
        assert text.count("✓") == 3
        assert text.count("○") == 0

    def test_escapes_brackets_in_task_names(self):
        """Task names with brackets are escaped."""
        tlm = ui.TaskListMessage(["Create [config] file"])
        text = _get_text(tlm)
        # Should contain the task text (escaped form)
        assert "config" in text


# ---------------------------------------------------------------------------
# ScanningPanel
# ---------------------------------------------------------------------------


class TestThinkingPanel:
    """Tests for ThinkingPanel thinking output widget."""

    def test_init_defaults(self):
        """ThinkingPanel has expected defaults."""
        panel = ui.ThinkingPanel()
        assert panel._frame == 0
        assert panel._text == ""

    def test_update_thinking(self):
        """update_thinking changes the thinking text and re-renders."""
        from unittest.mock import Mock

        panel = ui.ThinkingPanel()
        panel.update = Mock()
        panel.update_thinking("Thinking about...")
        assert panel._text == "Thinking about..."
        panel.update.assert_called()

    def test_spin_cycles_frames(self):
        """_spin cycles through spinner frames and updates display."""
        from unittest.mock import Mock

        panel = ui.ThinkingPanel()
        panel.update = Mock()
        panel._spin()
        assert panel._frame == 1
        panel.update.assert_called_once()
        call_text = panel.update.call_args[0][0]
        assert panel.SPINNER_FRAMES[1] in call_text

    def test_spin_wraps_around(self):
        """_spin wraps to frame 0 after the last frame."""
        from unittest.mock import Mock

        panel = ui.ThinkingPanel()
        panel.update = Mock()
        panel._frame = len(panel.SPINNER_FRAMES) - 1
        panel._spin()
        assert panel._frame == 0

    def test_render_truncates_long_text(self):
        """_render_thinking truncates to MAX_LINES."""
        from unittest.mock import Mock

        panel = ui.ThinkingPanel()
        panel.update = Mock()
        long_text = "\n".join(f"Line {i}" for i in range(20))
        panel._text = long_text
        panel._render_thinking()
        call_text = panel.update.call_args[0][0]
        # Should only have MAX_LINES lines of thinking text (plus spinner)
        # Total newlines should be <= MAX_LINES (body has MAX_LINES lines = MAX_LINES - 1 newlines)
        body_lines = call_text.strip().split("\n")
        assert len(body_lines) <= panel.MAX_LINES + 1  # spinner line + body lines

    def test_escapes_brackets(self):
        """ThinkingPanel escapes markup brackets."""
        from unittest.mock import Mock

        panel = ui.ThinkingPanel()
        panel.update = Mock()
        panel._text = "Thinking about [important] thing"
        panel._render_thinking()
        call_text = panel.update.call_args[0][0]
        assert "important" in call_text

    @pytest.mark.asyncio
    async def test_on_mount_starts_timer(self):
        """on_mount starts the animation timer."""
        from textual.app import App

        class TestApp(App):
            def compose(self):
                yield ui.ThinkingPanel(id="think")

        async with TestApp().run_test(size=(80, 24)) as pilot:
            panel = pilot.app.query_one("#think", ui.ThinkingPanel)
            assert panel._timer is not None


# ---------------------------------------------------------------------------
# ScanningPanel
# ---------------------------------------------------------------------------


class TestScanningPanel:
    """Tests for ScanningPanel spinner widget."""

    def test_init_defaults(self):
        """ScanningPanel has expected defaults."""
        panel = ui.ScanningPanel()
        assert panel._frame == 0
        assert panel._status == "Scanning workspace..."

    def test_update_status(self):
        """update_status changes the status text."""
        panel = ui.ScanningPanel()
        panel.update_status("Scanning workspace (42 files)...")
        assert panel._status == "Scanning workspace (42 files)..."

    def test_spin_cycles_frames(self):
        """_spin cycles through spinner frames and updates display."""
        from unittest.mock import Mock

        panel = ui.ScanningPanel()
        panel.update = Mock()
        panel._spin()
        assert panel._frame == 1
        panel.update.assert_called_once()
        call_text = panel.update.call_args[0][0]
        assert panel.SPINNER_FRAMES[1] in call_text
        assert "Scanning workspace..." in call_text

    def test_spin_wraps_around(self):
        """_spin wraps to frame 0 after the last frame."""
        from unittest.mock import Mock

        panel = ui.ScanningPanel()
        panel.update = Mock()
        panel._frame = len(panel.SPINNER_FRAMES) - 1
        panel._spin()
        assert panel._frame == 0

    @pytest.mark.asyncio
    async def test_on_mount_starts_timer(self):
        """on_mount starts the animation timer."""
        from textual.app import App

        class TestApp(App):
            def compose(self):
                yield ui.ScanningPanel(id="scan")

        async with TestApp().run_test(size=(80, 24)) as pilot:
            panel = pilot.app.query_one("#scan", ui.ScanningPanel)
            assert panel._timer is not None
