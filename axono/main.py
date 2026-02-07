import argparse
from pathlib import Path

from langchain_core.messages import HumanMessage
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer

from axono.agent import build_agent, run_agent
from axono.config import needs_onboarding
from axono.onboarding import OnboardingScreen
from axono.ui import (
    AssistantMessage,
    Banner,
    ChatContainer,
    CwdStatus,
    HistoryInput,
    SystemMessage,
    UserMessage,
)


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

    def __init__(self, force_onboard: bool = False) -> None:
        super().__init__()
        self.agent_graph = None
        self.message_history: list = []
        self._is_processing = False
        self._cwd = str(Path.home())
        self._force_onboard = force_onboard

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
            self.run_worker(self._initialize_agent(), name="init-agent", exclusive=True)

    def _on_onboarding_done(self, saved: bool | None) -> None:
        """Called when the onboarding screen is dismissed."""
        self.run_worker(self._initialize_agent(), name="init-agent", exclusive=True)

    def _parse_agent_data(self, content) -> str:
        if content is None:
            return ""
        return str(content)

    async def _initialize_agent(self) -> None:
        chat = self.query_one("#chat-container", ChatContainer)
        try:
            statuses = []
            self.agent_graph = await build_agent(on_status=statuses.append)
            await chat.add_message(SystemMessage("Agent initialized. Ready to chat!"))
            for s in statuses:
                await chat.add_message(SystemMessage(s, prefix="MCP"))
        except Exception as e:
            await chat.add_message(
                SystemMessage(f"Failed to initialize agent: {e}", prefix="Error")
            )

    async def on_input_submitted(self, event: HistoryInput.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return
        if self._is_processing:
            self.notify("Please wait for the current response...", severity="warning")
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

    async def _process_message(self) -> None:
        chat = self.query_one("#chat-container", ChatContainer)
        cwd_status = self.query_one("#cwd-status", CwdStatus)

        try:
            if self.agent_graph is None:
                await chat.add_message(
                    SystemMessage("Agent not initialized yet.", prefix="Error")
                )
                return

            async for event_type, data in run_agent(
                self.agent_graph, self.message_history, cwd=self._cwd
            ):
                if event_type == "messages":
                    self.message_history = list(data)
                    continue
                if event_type == "intent":
                    # Intent object - skip display (task_list will show tasks)
                    continue
                if event_type == "task_list":
                    # Show the task list
                    tasks = data if isinstance(data, list) else []
                    if tasks:
                        task_text = "\n".join(
                            f"  {i+1}. {t}" for i, t in enumerate(tasks)
                        )
                        await chat.add_message(
                            SystemMessage(f"Tasks:\n{task_text}", prefix="Plan")
                        )
                    continue
                if event_type == "task_start":
                    await chat.add_message(SystemMessage(str(data), prefix="→"))
                    continue
                if event_type == "task_complete":
                    # Task completed - no need to show (next task_start will show)
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
            self._is_processing = False

    def action_clear(self) -> None:
        chat = self.query_one("#chat-container", ChatContainer)
        chat.remove_children()
        self.message_history = []
        self.notify("Chat cleared")


def main():
    parser = argparse.ArgumentParser(description="Axono – terminal AI assistant")
    parser.add_argument(
        "--onboard",
        action="store_true",
        help="Run the onboarding wizard (re-configure LM Studio settings)",
    )
    args = parser.parse_args()

    app = AxonoApp(force_onboard=args.onboard)
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
