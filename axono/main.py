import argparse
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Label, Static

from axono.agent import build_agent, run_agent
from axono.config import needs_onboarding
from axono.conversation import generate_conversation_id, get_checkpointer
from axono.folderembed import FileChangeEvent, embed_folder, start_watcher
from axono.onboarding import OnboardingScreen
from axono.pipeline import set_thinking_queue
from axono.ui import (
    AssistantMessage,
    Banner,
    ChatContainer,
    CwdStatus,
    HistoryInput,
    ScanningPanel,
    SystemMessage,
    TaskListMessage,
    ThinkingPanel,
    UserMessage,
)
from axono.workspace import (
    is_workspace_trusted,
    set_workspace_root,
    trust_workspace,
)

logger = logging.getLogger(__name__)


class TrustWorkspaceScreen(ModalScreen[bool]):
    """Modal screen to ask user to trust a workspace folder."""

    DEFAULT_CSS = """
    TrustWorkspaceScreen {
        align: center middle;
    }

    TrustWorkspaceScreen > #dialog {
        width: 70;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
    }

    TrustWorkspaceScreen > #dialog > Label {
        width: 100%;
        margin-bottom: 1;
    }

    TrustWorkspaceScreen > #dialog > #buttons {
        width: 100%;
        height: auto;
        align: center middle;
    }

    TrustWorkspaceScreen > #dialog > #buttons > Button {
        margin: 0 1;
    }
    """

    def __init__(self, workspace_path: str) -> None:
        super().__init__()
        self.workspace_path = workspace_path

    def compose(self) -> ComposeResult:
        with Static(id="dialog"):
            yield Label(f"[bold]Workspace Trust Verification[/bold]\n")
            yield Label(
                f"The workspace is not in your trusted folders list:\n"
                f"[cyan]{self.workspace_path}[/cyan]\n"
            )
            yield Label(
                "Axono will be restricted to operate only within this folder.\n"
                "Do you want to trust this workspace and add it to your allow list?"
            )
            with Static(id="buttons"):
                yield Button("Trust & Continue", id="trust", variant="primary")
                yield Button(
                    "Continue Without Trusting", id="continue", variant="default"
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "trust":
            trust_workspace(self.workspace_path)
            self.dismiss(True)
        else:
            self.dismiss(False)


def _friendly_tool_name(tool_call: str) -> str:
    """Convert raw tool call to friendly display name."""
    # tool_call looks like: "shell({'task': '...', 'working_dir': '...'})"
    # or "bash({'command': 'ls'})"
    if tool_call.startswith("shell("):
        return "🔧 Running shell task..."
    elif tool_call.startswith("bash("):
        # Extract command if short enough
        try:
            start = tool_call.find("'command': '") + 12
            end = tool_call.find("'", start)
            cmd = tool_call[start:end]
            if len(cmd) > 40:
                cmd = cmd[:40] + "..."
            return f"$ {cmd}"
        except Exception:
            return "$ ..."
    elif tool_call.startswith("code("):
        return "📝 Editing code..."
    elif tool_call.startswith("duckduckgo"):
        return "🔍 Searching web..."
    else:
        # MCP or unknown tool - show name only
        name = tool_call.split("(")[0] if "(" in tool_call else tool_call
        return f"⚙️ {name}..."


class AxonoApp(App):
    """Personal AI Assistant TUI Application."""

    TITLE = "Axono"
    ALLOW_SELECT = True

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear", "Clear", show=True),
    ]

    DEFAULT_CSS = """
    Screen {
        layout: vertical;
    }

    #chat-container {
        height: 1fr;
    }

    #input-area {
        dock: bottom;
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(
        self, force_onboard: bool = False, workspace: str | None = None
    ) -> None:
        super().__init__()
        self.agent_graph = None
        self.message_history: list = []
        self._is_processing = False
        # Use provided workspace or current directory as workspace root
        self._workspace = os.path.abspath(workspace) if workspace else os.getcwd()
        self._cwd = self._workspace
        self._force_onboard = force_onboard
        self._conversation_id = generate_conversation_id()
        self._checkpointer = None
        self._workspace_trusted: bool | None = None
        self._file_watcher = None  # watchdog Observer for file changes

    def compose(self) -> ComposeResult:
        yield ChatContainer(Banner(), id="chat-container")
        yield CwdStatus(id="cwd-status")
        yield HistoryInput(placeholder="Type a message...", id="input-area")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#input-area", HistoryInput).focus()
        self.query_one("#cwd-status", CwdStatus).update_path(self._cwd)
        if self._force_onboard or needs_onboarding():
            self.push_screen(OnboardingScreen(), callback=self._on_onboarding_done)
        else:
            # Check workspace trust before initializing agent
            await self._check_workspace_trust()

    def _on_onboarding_done(self, saved: bool | None) -> None:
        """Called when the onboarding screen is dismissed."""
        # Check workspace trust after onboarding
        self.run_worker(
            self._check_workspace_trust(), name="check-trust", exclusive=True
        )

    async def _check_workspace_trust(self) -> None:
        """Check if workspace is trusted and prompt user if not."""
        if is_workspace_trusted(self._workspace):
            # Workspace is already trusted, initialize
            set_workspace_root(self._workspace)
            self.run_worker(self._initialize_agent(), name="init-agent", exclusive=True)
        else:
            # Show trust dialog
            self.push_screen(
                TrustWorkspaceScreen(self._workspace),
                callback=self._on_trust_decision,
            )

    def _on_trust_decision(self, trusted: bool | None) -> None:
        """Called when user decides whether to trust the workspace."""
        # Set workspace root regardless of trust decision
        # The workspace will still be enforced as the boundary
        set_workspace_root(self._workspace)

        chat = self.query_one("#chat-container", ChatContainer)
        if trusted:
            self.call_later(
                chat.add_message,
                SystemMessage(
                    f"Workspace trusted and added to allow list: {self._workspace}"
                ),
            )
        else:
            self.call_later(
                chat.add_message,
                SystemMessage(
                    f"Operating in workspace (not added to allow list): {self._workspace}"
                ),
            )

        self.run_worker(self._initialize_agent(), name="init-agent", exclusive=True)

    def _parse_agent_data(self, content) -> str:
        if content is None:
            return ""
        return str(content)

    async def _initialize_agent(self) -> None:
        chat = self.query_one("#chat-container", ChatContainer)
        try:
            statuses = []
            self._checkpointer = await get_checkpointer()
            self.agent_graph = await build_agent(
                on_status=statuses.append, checkpointer=self._checkpointer
            )
            logger.info("Agent initialized")
            await chat.add_message(SystemMessage("Agent initialized. Ready to chat!"))
            for s in statuses:
                await chat.add_message(SystemMessage(s, prefix="MCP"))
            # Start workspace indexing in background
            self.run_worker(
                self._start_workspace_indexing(),
                name="workspace-indexing",
                exclusive=True,
            )
        except Exception as e:
            logger.error("Failed to initialize agent: %s", e)
            await chat.add_message(
                SystemMessage(f"Failed to initialize agent: {e}", prefix="Error")
            )

    async def _start_workspace_indexing(self) -> None:
        """Start indexing the workspace and set up file watcher."""
        chat = self.query_one("#chat-container", ChatContainer)
        panel = ScanningPanel()
        await chat.mount(panel)
        panel.scroll_visible(animate=False)
        logger.info("Workspace indexing started: %s", self._workspace)
        try:
            # Start embedding the workspace
            indexed_count = 0
            async for event_type, data in embed_folder(self._workspace):
                if event_type == "start":
                    total = data["total_files"]
                    panel.update_status(f"Scanning workspace ({total} files)...")
                    logger.info("Scanning %d files", total)
                elif event_type == "file_indexed":
                    indexed_count += 1
                elif event_type == "complete":
                    logger.info(
                        "Indexed %d files (%d chunks) in %ss",
                        data["files"],
                        data["chunks"],
                        data["duration"],
                    )

            # Start file watcher for automatic updates
            self._start_file_watcher()

        except Exception as e:
            logger.error("Indexing failed: %s", e)
            await chat.add_message(
                SystemMessage(f"Indexing failed: {e}", prefix="Error")
            )
        finally:
            await panel.remove()

    def _start_file_watcher(self) -> None:
        """Start the file watcher for automatic re-indexing."""
        try:
            self._file_watcher = start_watcher(
                self._workspace,
                self._on_file_change,
            )
        except RuntimeError:
            # watchdog not installed - that's fine, just skip watching
            pass

    def _on_file_change(self, event: FileChangeEvent) -> None:
        """Handle file change events from the watcher."""
        # Re-index the workspace in the background with debounce
        # Cancel any pending re-index if one is already scheduled
        existing_workers = [w for w in self.workers if w.name == "reindex"]
        if existing_workers:
            return  # Skip if re-index is already running or scheduled

        # Use call_later to schedule work on the main thread
        self.call_later(
            lambda: self.run_worker(
                self._reindex_workspace(),
                name="reindex",
                exclusive=True,
                group="reindex",
            ),
            delay=2.0,  # Debounce: wait 2s before re-indexing (multiple file changes batched)
        )

    async def _reindex_workspace(self) -> None:
        """Re-index the workspace after file changes."""
        try:
            async for event_type, data in embed_folder(self._workspace):
                # Silent re-indexing - only log errors
                if event_type == "file_error":
                    pass  # Ignore errors during background re-indexing
        except Exception:
            pass  # Silent failure for background re-indexing

    async def on_input_submitted(self, event: HistoryInput.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return
        if self._is_processing:
            self.notify("Please wait for the current response...", severity="warning")
            return

        # Handle /config command to show onboarding
        if user_text == "/config":
            input_widget = self.query_one("#input-area", HistoryInput)
            input_widget.value = ""
            input_widget.add_to_history(user_text)
            self.push_screen(OnboardingScreen(), callback=self._on_onboarding_done)
            return

        input_widget = self.query_one("#input-area", HistoryInput)
        input_widget.value = ""
        input_widget.add_to_history(user_text)

        chat = self.query_one("#chat-container", ChatContainer)
        await chat.add_message(UserMessage(user_text))

        self.message_history.append(HumanMessage(content=user_text))

        self._is_processing = True
        self.run_worker(
            self._process_message(),
            name="agent-response",
            exclusive=True,
            group="agent",
        )

    async def _drain_thinking_queue(
        self,
        queue: asyncio.Queue,
        chat: ChatContainer,
    ) -> None:
        """Drain the thinking queue and update/remove a ThinkingPanel."""
        thinking_panel: ThinkingPanel | None = None
        while True:
            try:
                text = await asyncio.wait_for(queue.get(), timeout=0.15)
                if text is None:
                    # Thinking ended — remove panel
                    if thinking_panel is not None:
                        await thinking_panel.remove()
                        thinking_panel = None
                else:
                    # Show/update thinking panel
                    if thinking_panel is None:
                        thinking_panel = ThinkingPanel()
                        await chat.mount(thinking_panel)
                        thinking_panel.scroll_visible(animate=False)
                    thinking_panel.update_thinking(text)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                if thinking_panel is not None:
                    await thinking_panel.remove()
                break

    async def _process_message(self) -> None:
        chat = self.query_one("#chat-container", ChatContainer)
        cwd_status = self.query_one("#cwd-status", CwdStatus)
        task_list_widget: TaskListMessage | None = None
        current_tasks: list[str] = []
        current_task_index = 0

        # Set up thinking queue so pipeline LLM calls can push thinking tokens
        thinking_queue: asyncio.Queue = asyncio.Queue()
        set_thinking_queue(thinking_queue)
        drainer = asyncio.create_task(self._drain_thinking_queue(thinking_queue, chat))

        try:
            if self.agent_graph is None:
                await chat.add_message(
                    SystemMessage("Agent not initialized yet.", prefix="Error")
                )
                return

            async for event_type, data in run_agent(
                self.agent_graph,
                self.message_history,
                cwd=self._cwd,
                thread_id=self._conversation_id,
            ):
                if event_type == "messages":
                    self.message_history = list(data)
                    continue
                if event_type == "intent":
                    # Intent object - skip display (task_list will show tasks)
                    continue
                if event_type == "task_list":
                    # Show the task list with progress tracking
                    current_tasks = data if isinstance(data, list) else []
                    if current_tasks:
                        task_list_widget = TaskListMessage(current_tasks)
                        await chat.add_message(task_list_widget)
                    continue
                if event_type == "task_start":
                    # Update the task list to show current task in progress
                    if task_list_widget is not None:
                        task_list_widget.start_task(current_task_index)
                    continue
                if event_type == "task_complete":
                    # Update the task list to show task completed
                    if task_list_widget is not None:
                        task_list_widget.complete_task(current_task_index)
                        current_task_index += 1
                    continue
                message_text = self._parse_agent_data(data)
                if event_type == "assistant":
                    await chat.add_message(AssistantMessage(message_text))
                elif event_type == "tool_call":
                    friendly = _friendly_tool_name(message_text)
                    await chat.add_tool_group(friendly)
                elif event_type == "tool_result":
                    output = message_text
                    if "__CWD__:" in output:
                        lines = output.splitlines()
                        remaining: list[str] = []
                        for line in lines:
                            if line.startswith("__CWD__:"):
                                self._cwd = line.split("__CWD__:", 1)[1].strip()
                                cwd_status.update_path(self._cwd)
                            else:
                                remaining.append(line)
                        output = "\n".join(remaining).strip()
                    if output:
                        await chat.append_to_tool_group(
                            SystemMessage(output, prefix="Output"),
                        )
                elif event_type == "error":
                    await chat.add_message(SystemMessage(message_text, prefix="Error"))

        except Exception as e:
            await chat.add_message(
                SystemMessage(f"Unexpected error: {e}", prefix="Error")
            )
        finally:
            drainer.cancel()
            try:
                await drainer
            except asyncio.CancelledError:
                pass
            set_thinking_queue(None)
            self._is_processing = False

    def action_clear(self) -> None:
        chat = self.query_one("#chat-container", ChatContainer)
        chat.remove_children()
        self.message_history = []
        self._conversation_id = generate_conversation_id()
        self.notify("Chat cleared")

    def on_unmount(self) -> None:
        """Clean up resources when the app is closing."""
        logger.info("Shutting down application")
        # Cancel any running background workers
        for worker in self.workers:
            worker.cancel()
        self._stop_file_watcher()

    def _stop_file_watcher(self) -> None:
        """Stop the file watcher if running."""
        if self._file_watcher is not None:
            try:
                self._file_watcher.stop()
                # Use a generous timeout for observer to shut down cleanly
                self._file_watcher.join(timeout=2.0)
                logger.info("File watcher stopped")
            except Exception as e:
                logger.warning("Error stopping file watcher: %s", e)
            finally:
                self._file_watcher = None


def setup_logging() -> str:
    """Configure file logging. Returns the log file path."""
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"axono-{timestamp}.log"
    logging.basicConfig(
        filename=str(log_file),
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return str(log_file)


def main():
    parser = argparse.ArgumentParser(description="Axono – terminal AI assistant")
    parser.add_argument(
        "--onboard",
        action="store_true",
        help="Run the onboarding wizard (re-configure LM Studio settings)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable logging to log/axono-{datetime}.log",
    )
    parser.add_argument(
        "workspace",
        nargs="?",
        default=None,
        help="Workspace directory to operate in (default: current directory)",
    )
    args = parser.parse_args()

    if args.log:
        log_file = setup_logging()
        print(f"📝 Logging to: {log_file}")

    app = AxonoApp(force_onboard=args.onboard, workspace=args.workspace)
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
