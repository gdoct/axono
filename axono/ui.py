from textual.containers import ScrollableContainer
from textual.widgets import Collapsible, Static

BANNER_TEXT = r"""
    _
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


class ToolGroup(Collapsible):
    """Collapsible group for tool call + result messages.

    Starts expanded, auto-collapses when the next message arrives.
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
        super().__init__(title=title, collapsed=False)


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
            await group.mount(content_widget)
            content_widget.scroll_visible(animate=False)
        else:
            await self.add_message(content_widget)


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
