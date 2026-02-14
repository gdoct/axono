from textual.containers import ScrollableContainer
from textual.widgets import Collapsible, Input, Static

from axono.history import append_to_history, load_history

BANNER_TEXT = r"""
_____________________________________
   / \   __  __  ___   _ __    ___
  / _ \  \ \/ / / _ \ | '_ \  / _ \
  / ___ \  >  < | (_) || | | || (_) |
/_/   \_\/_/\_\ \___/ |_| |_| \___/
"""


class Banner(Static):
    """Startup banner with app name in ASCII art."""

    DEFAULT_CSS = """
    Banner {
        text-align: center;
        color: $accent;
        padding: 1 0;
        margin: 0 0 1 0;
    }
    """

    def __init__(self) -> None:
        content = (
            f"[bold]{BANNER_TEXT.strip()}[/bold]"
            "\n\nType a message and press Enter. I can run shell commands too!"
        )
        super().__init__(content, markup=True)


class UserMessage(Static):
    """Displays a user message in the chat."""

    DEFAULT_CSS = """
    UserMessage {
        color: yellow;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, text: str) -> None:
        escaped = text.replace("[", "\\[")
        super().__init__(f"[bold]You:[/bold] {escaped}", markup=True)


class AssistantMessage(Static):
    """Displays an assistant message in the chat."""

    DEFAULT_CSS = """
    AssistantMessage {
        color: green;
        text-style: bold;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, text: str = "") -> None:
        escaped = text.replace("[", "\\[")
        super().__init__(f"[bold green]Axono:[/bold green] {escaped}", markup=True)


class SystemMessage(Static):
    """Displays system/tool output in the chat."""

    DEFAULT_CSS = """
    SystemMessage {
        color: $text-muted;
        padding: 0 0 0 1;
        margin: 0;
    }
    """

    def __init__(self, text: str, prefix: str = "System") -> None:
        escaped = text.replace("[", "\\[")
        super().__init__(f"[dim]{prefix}:[/dim] [dim]{escaped}[/dim]", markup=True)


class TaskListMessage(Static):
    """Displays a task list with progress indicators.

    Tasks can be marked as pending, in_progress, or completed.
    """

    DEFAULT_CSS = """
    TaskListMessage {
        color: $text-muted;
        padding: 0 0 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, tasks: list[str]) -> None:
        self._tasks = tasks
        self._current_index = -1  # -1 means not started
        super().__init__(self._render_tasks(), markup=True)

    def _render_tasks(self) -> str:
        """Render the task list with status indicators."""
        lines = ["[dim]Plan:[/dim]"]
        for i, task in enumerate(self._tasks):
            escaped = task.replace("[", "\\[")
            if i < self._current_index:
                # Completed
                lines.append(f"  [green]✓[/green] [dim]{escaped}[/dim]")
            elif i == self._current_index:
                # In progress
                lines.append(f"  [yellow]→[/yellow] [bold]{escaped}[/bold]")
            else:
                # Pending
                lines.append(f"  [dim]○ {escaped}[/dim]")
        return "\n".join(lines)

    def start_task(self, index: int) -> None:
        """Mark a task as in progress."""
        self._current_index = index
        self.update(self._render_tasks())

    def complete_task(self, index: int) -> None:
        """Mark a task as completed (moves current past it)."""
        if index == self._current_index:
            self._current_index = index + 1
            self.update(self._render_tasks())


class ToolGroup(Collapsible):
    """Collapsible group for tool call + result messages.

    Starts collapsed by default, auto-collapses when the next message arrives.
    """

    DEFAULT_CSS = """
    ToolGroup {
        padding: 0 0 0 1;
        margin: 0 0 1 0;
        border: none;
        background: transparent;
    }
    ToolGroup > CollapsibleTitle {
        color: $text-muted;
        padding: 0;
    }
    ToolGroup > Contents {
        padding: 0;
    }
    """

    def __init__(self, title: str) -> None:
        super().__init__(title=title, collapsed=True)


class ChatContainer(ScrollableContainer):
    """Scrollable container holding all chat messages."""

    DEFAULT_CSS = """
    ChatContainer {
        height: 1fr;
        padding: 0 1;
    }
    """

    def _collapse_tool_groups(self) -> None:
        """Collapse all existing ToolGroup widgets."""
        for group in self.query(ToolGroup):
            group.collapsed = True

    async def add_message(self, widget: Static) -> None:
        """Mount a new message widget and scroll to it.

        When a non-tool message is added, all previous ToolGroups are
        collapsed automatically.
        """
        if not isinstance(widget, SystemMessage):
            self._collapse_tool_groups()
        await self.mount(widget)
        widget.scroll_visible(animate=False)

    async def add_tool_group(self, title: str) -> None:
        """Add a collapsible tool group (title only) and scroll to it.

        Previous ToolGroups are NOT collapsed here — they stay open until
        a non-tool message arrives (assistant reply, user message, etc.).
        Use :meth:`append_to_tool_group` to add output into the last group.
        """
        group = ToolGroup(title=title)
        await self.mount(group)
        group.scroll_visible(animate=False)

    async def append_to_tool_group(self, content_widget: Static) -> None:
        """Mount *content_widget* inside the most recent ToolGroup.

        If there is no open ToolGroup, falls back to :meth:`add_message`.
        """
        groups = list(self.query(ToolGroup))
        if groups:
            group = groups[-1]
            # Mount to the Contents widget inside the Collapsible
            contents = group.query_one("Contents")
            await contents.mount(content_widget)
            content_widget.scroll_visible(animate=False)
        else:
            await self.add_message(content_widget)


class ScanningPanel(Static):
    """Small panel showing workspace scanning progress with a spinner."""

    DEFAULT_CSS = """
    ScanningPanel {
        height: auto;
        padding: 0 1;
        color: $accent;
    }
    """

    SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, **kwargs) -> None:
        super().__init__("", markup=True, **kwargs)
        self._frame = 0
        self._status = "Scanning workspace..."

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._spin)

    def _spin(self) -> None:
        self._frame = (self._frame + 1) % len(self.SPINNER_FRAMES)
        char = self.SPINNER_FRAMES[self._frame]
        self.update(f"{char} [dim]{self._status}[/dim]")

    def update_status(self, text: str) -> None:
        self._status = text


class ThinkingPanel(Static):
    """Shows LLM reasoning/thinking output with a spinner.

    Displays the last few lines of thinking text (dim). Auto-updates
    as new thinking tokens arrive. Should be removed when thinking ends.
    """

    DEFAULT_CSS = """
    ThinkingPanel {
        height: auto;
        max-height: 8;
        padding: 0 1;
        color: $text-muted;
    }
    """

    SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    MAX_LINES = 6

    def __init__(self, **kwargs) -> None:
        super().__init__("", markup=True, **kwargs)
        self._frame = 0
        self._text = ""

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._spin)

    def _spin(self) -> None:
        self._frame = (self._frame + 1) % len(self.SPINNER_FRAMES)
        self._render_thinking()

    def update_thinking(self, text: str) -> None:
        """Update the displayed thinking text."""
        self._text = text
        self._render_thinking()

    def _render_thinking(self) -> None:
        char = self.SPINNER_FRAMES[self._frame]
        escaped = self._text.replace("[", "\\[")
        lines = escaped.strip().splitlines()
        if len(lines) > self.MAX_LINES:
            lines = lines[-self.MAX_LINES :]
        body = "\n".join(lines)
        self.update(f"{char} [dim italic]{body}[/dim italic]")


class CwdStatus(Static):
    """Displays the current working directory above the prompt."""

    DEFAULT_CSS = """
    CwdStatus {
        color: $text-muted;
        padding: 0 1;
        height: auto;
    }
    """

    def update_path(self, path: str) -> None:
        escaped = path.replace("[", "\\[")
        self.update(f"[dim]CWD:[/dim] [dim]{escaped}[/dim]")


class HistoryInput(Input):
    """Input widget with arrow-key history navigation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_index: int = -1
        self._current_input: str = ""

    def on_mount(self) -> None:
        """Load history from disk on mount."""
        self._history = load_history()

    def on_key(self, event) -> None:
        """Handle arrow key navigation through history."""
        if event.key == "up":
            if not self._history:
                return
            if self._history_index == -1:
                self._current_input = self.value
                self._history_index = len(self._history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            self.value = self._history[self._history_index]
            self.cursor_position = len(self.value)
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            if self._history_index == -1:
                return
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
                self.value = self._history[self._history_index]
            else:
                self._history_index = -1
                self.value = self._current_input
            self.cursor_position = len(self.value)
            event.prevent_default()
            event.stop()

    def add_to_history(self, prompt: str) -> None:
        """Add a prompt to history and reset navigation state."""
        self._history = append_to_history(prompt)
        self._history_index = -1
        self._current_input = ""
