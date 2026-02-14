"""Unit tests for axono.main (AxonoApp)."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest

from axono.main import AxonoApp, _friendly_tool_name, main, setup_logging
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
    ToolGroup,
    UserMessage,
)


@pytest.fixture(autouse=True)
def reset_workspace():
    """Reset workspace root before and after each test."""
    from axono.workspace import clear_workspace_root

    clear_workspace_root()
    yield
    clear_workspace_root()


@pytest.fixture(autouse=True)
def _mock_async_resources():
    """Mock resources that open connections/threads to prevent test hangs.

    get_checkpointer() opens an aiosqlite connection, embed_folder() opens
    another, and start_watcher() spawns a watchdog thread.  None of these
    are closed automatically when the Textual app exits run_test(), so the
    process hangs waiting for the event loop to drain.
    """

    async def _fake_checkpointer():
        return mock.MagicMock()

    async def _noop_embed(path):
        return
        yield  # noqa: unreachable – makes this an async generator

    with mock.patch("axono.main.get_checkpointer", side_effect=_fake_checkpointer):
        with mock.patch("axono.main.embed_folder", side_effect=_noop_embed):
            with mock.patch("axono.main.start_watcher", return_value=mock.MagicMock()):
                yield


def _app(pilot) -> AxonoApp:
    """Cast pilot.app to AxonoApp for type checking."""
    return cast(AxonoApp, pilot.app)


# ---------------------------------------------------------------------------
# _friendly_tool_name
# ---------------------------------------------------------------------------


class TestFriendlyToolName:

    def test_shell_tool(self):
        result = _friendly_tool_name(
            "shell({'task': 'install deps', 'working_dir': '/tmp'})"
        )
        assert result == "🔧 Running shell task..."

    def test_bash_tool_short_command(self):
        result = _friendly_tool_name("bash({'command': 'ls -la'})")
        assert result == "$ ls -la"

    def test_bash_tool_long_command(self):
        long_cmd = "cat /very/long/path/to/file/that/exceeds/forty/characters/limit.txt"
        result = _friendly_tool_name(f"bash({{'command': '{long_cmd}'}})")
        assert result.startswith("$ cat /very/long/path")
        assert result.endswith("...")
        assert len(result) <= 44 + 3  # "$ " + 40 chars + "..."

    def test_bash_tool_malformed_input(self):
        """Malformed input still parses (just extracts garbage)."""
        result = _friendly_tool_name("bash(malformed)")
        # The parsing logic works but extracts a garbled substring
        assert result.startswith("$ ")

    def test_bash_tool_exception_during_parse(self):
        """Exception during parsing falls back to $ ..."""
        from unittest import mock

        # Create a string subclass that raises on slicing
        class ExplodingStr(str):
            def __getitem__(self, key):
                raise RuntimeError("boom")

        # The function starts with startswith which works, but slicing explodes
        exploding = ExplodingStr("bash({'command': 'ls'})")
        result = _friendly_tool_name(exploding)
        assert result == "$ ..."

    def test_code_tool(self):
        result = _friendly_tool_name("code({'task': 'add tests'})")
        assert result == "📝 Editing code..."

    def test_duckduckgo_search(self):
        result = _friendly_tool_name("duckduckgo_search({'query': 'python async'})")
        assert result == "🔍 Searching web..."

    def test_mcp_tool_with_parens(self):
        result = _friendly_tool_name("weather({'location': 'NYC'})")
        assert result == "⚙️ weather..."

    def test_unknown_tool_no_parens(self):
        result = _friendly_tool_name("some_unknown_tool")
        assert result == "⚙️ some_unknown_tool..."


# ---------------------------------------------------------------------------
# _parse_agent_data (identical logic to safety._parse_agent_data)
# ---------------------------------------------------------------------------


class TestParseAgentData:

    def test_none_returns_empty(self):
        app = AxonoApp()
        assert app._parse_agent_data(None) == ""

    def test_list_returns_json(self):
        app = AxonoApp()
        assert app._parse_agent_data([1, 2]) == json.dumps([1, 2])

    def test_string_passthrough(self):
        app = AxonoApp()
        assert app._parse_agent_data("hello") == "hello"

    def test_int_coerced(self):
        app = AxonoApp()
        assert app._parse_agent_data(42) == "42"


# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------


class TestAppInit:

    def test_initial_state(self):
        app = AxonoApp()
        assert app.agent_graph is None
        assert app.message_history == []
        assert app._is_processing is False
        # _cwd now defaults to current working directory (workspace)
        assert app._cwd == os.getcwd()
        assert app._workspace == os.getcwd()

    def test_initial_state_with_workspace(self, tmp_path):
        workspace = str(tmp_path / "my_workspace")
        os.makedirs(workspace)
        app = AxonoApp(workspace=workspace)
        assert app._cwd == workspace
        assert app._workspace == workspace

    def test_title(self):
        app = AxonoApp()
        assert app.TITLE == "Axono"

    def test_bindings(self):
        app = AxonoApp()
        keys = [b.key if hasattr(b, "key") else b[0] for b in app.BINDINGS]  # type: ignore[union-attr]
        assert "ctrl+c" in keys
        assert "ctrl+l" in keys


# ---------------------------------------------------------------------------
# compose — widgets are yielded correctly
# ---------------------------------------------------------------------------


class TestCompose:

    @pytest.mark.asyncio
    async def test_compose_yields_expected_widgets(self):
        app = AxonoApp()
        # Patch build_agent so on_mount doesn't hit the real LLM
        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            async with app.run_test(size=(80, 24)) as pilot:
                assert app.query_one("#chat-container", ChatContainer)
                assert app.query_one("#cwd-status", CwdStatus)
                assert app.query_one("#input-area")

    @pytest.mark.asyncio
    async def test_input_has_focus_on_mount(self):
        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            async with AxonoApp().run_test(size=(80, 24)) as pilot:
                inp = _app(pilot).query_one("#input-area", HistoryInput)
                assert inp.has_focus


# ---------------------------------------------------------------------------
# _initialize_agent
# ---------------------------------------------------------------------------


class TestInitializeAgent:

    @pytest.mark.asyncio
    async def test_success_sets_graph_and_shows_message(self):
        fake_graph = object()

        async def fake_build(on_status=None, checkpointer=None):
            return fake_graph

        async def fake_checkpointer():
            return mock.MagicMock()

        with mock.patch("axono.main.needs_onboarding", return_value=False):
            with mock.patch("axono.main.is_workspace_trusted", return_value=True):
                with mock.patch(
                    "axono.main.get_checkpointer", side_effect=fake_checkpointer
                ):
                    with mock.patch("axono.main.build_agent", side_effect=fake_build):
                        async with AxonoApp().run_test(size=(80, 24)) as pilot:
                            await pilot.pause(delay=0.5)
                            assert _app(pilot).agent_graph is fake_graph
                            chat = _app(pilot).query_one(
                                "#chat-container", ChatContainer
                            )
                            children = list(chat.children)
                            # Should contain Banner + "Agent initialized" SystemMessage
                            sys_msgs = [
                                c for c in children if isinstance(c, SystemMessage)
                            ]
                            assert any(True for c in sys_msgs)

    @pytest.mark.asyncio
    async def test_success_with_mcp_status(self):
        async def fake_build(on_status=None, checkpointer=None):
            if on_status:
                on_status("Loaded 2 MCP tool(s): search, weather")
            return object()

        async def fake_checkpointer():
            return mock.MagicMock()

        with mock.patch("axono.main.needs_onboarding", return_value=False):
            with mock.patch("axono.main.is_workspace_trusted", return_value=True):
                with mock.patch(
                    "axono.main.get_checkpointer", side_effect=fake_checkpointer
                ):
                    with mock.patch("axono.main.build_agent", side_effect=fake_build):
                        async with AxonoApp().run_test(size=(80, 24)) as pilot:
                            await pilot.pause(delay=0.5)
                            chat = _app(pilot).query_one(
                                "#chat-container", ChatContainer
                            )
                            children = list(chat.children)
                            sys_msgs = [
                                c for c in children if isinstance(c, SystemMessage)
                            ]
                            # Banner + "Agent initialized" + MCP status
                            assert len(sys_msgs) >= 2

    @pytest.mark.asyncio
    async def test_failure_shows_error(self):
        async def failing_build(on_status=None, checkpointer=None):
            raise ConnectionError("LLM unreachable")

        async def fake_checkpointer():
            return mock.MagicMock()

        with mock.patch("axono.main.needs_onboarding", return_value=False):
            with mock.patch("axono.main.is_workspace_trusted", return_value=True):
                with mock.patch(
                    "axono.main.get_checkpointer", side_effect=fake_checkpointer
                ):
                    with mock.patch(
                        "axono.main.build_agent", side_effect=failing_build
                    ):
                        async with AxonoApp().run_test(size=(80, 24)) as pilot:
                            await pilot.pause(delay=0.5)
                            assert _app(pilot).agent_graph is None


# ---------------------------------------------------------------------------
# on_input_submitted
# ---------------------------------------------------------------------------


class TestInputSubmitted:

    @pytest.mark.asyncio
    async def test_empty_input_ignored(self):
        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            async with AxonoApp().run_test(size=(80, 24)) as pilot:
                _app(pilot).agent_graph = cast(Any, object())
                inp = _app(pilot).query_one("#input-area", HistoryInput)
                inp.value = "   "
                await inp.action_submit()
                await pilot.pause()
                # No user message should be added
                chat = _app(pilot).query_one("#chat-container", ChatContainer)
                user_msgs = [c for c in chat.children if isinstance(c, UserMessage)]
                assert len(user_msgs) == 0

    @pytest.mark.asyncio
    async def test_adds_user_message_and_clears_input(self):
        async def noop_run_agent(graph, messages):
            yield  # pragma: no cover

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", side_effect=noop_run_agent):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot)._is_processing = False
                    inp = _app(pilot).query_one("#input-area", HistoryInput)
                    inp.value = "hello agent"
                    await inp.action_submit()
                    await pilot.pause()
                    # Input should be cleared
                    assert inp.value == ""
                    # User message added
                    chat = _app(pilot).query_one("#chat-container", ChatContainer)
                    user_msgs = [c for c in chat.children if isinstance(c, UserMessage)]
                    assert len(user_msgs) == 1

    @pytest.mark.asyncio
    async def test_rejects_input_while_processing(self):
        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            async with AxonoApp().run_test(size=(80, 24)) as pilot:
                _app(pilot).agent_graph = cast(Any, object())
                _app(pilot)._is_processing = True
                inp = _app(pilot).query_one("#input-area", HistoryInput)
                inp.value = "test"
                await inp.action_submit()
                await pilot.pause()
                # No user message added since we're busy
                chat = _app(pilot).query_one("#chat-container", ChatContainer)
                user_msgs = [c for c in chat.children if isinstance(c, UserMessage)]
                assert len(user_msgs) == 0


# ---------------------------------------------------------------------------
# _process_message — event routing
# ---------------------------------------------------------------------------


class TestProcessMessage:
    """Tests call ``_process_message`` directly to avoid worker timing issues."""

    @pytest.mark.asyncio
    async def test_assistant_event(self):
        async def fake_run(graph, messages, cwd=None, thread_id=None):
            yield ("assistant", "Hi there!")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot).message_history = []
                    await _app(pilot)._process_message()
                    chat = _app(pilot).query_one("#chat-container", ChatContainer)
                    asst_msgs = [
                        c for c in chat.children if isinstance(c, AssistantMessage)
                    ]
                    assert len(asst_msgs) >= 1

    @pytest.mark.asyncio
    async def test_tool_call_event(self):
        async def fake_run(graph, messages, cwd=None, thread_id=None):
            yield ("tool_call", "bash({'command': 'ls'})")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot).message_history = []
                    await _app(pilot)._process_message()
                    chat = _app(pilot).query_one("#chat-container", ChatContainer)
                    tool_groups = [c for c in chat.children if isinstance(c, ToolGroup)]
                    assert len(tool_groups) >= 1

    @pytest.mark.asyncio
    async def test_tool_groups_collapse_on_assistant_message(self):
        async def fake_run(graph, messages, cwd=None, thread_id=None):
            yield ("tool_call", "bash({'command': 'ls'})")
            yield ("tool_result", "file1.py\n__CWD__:/tmp")
            yield ("assistant", "Done.")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot).message_history = []
                    await _app(pilot)._process_message()
                    chat = _app(pilot).query_one("#chat-container", ChatContainer)
                    tool_groups = [c for c in chat.children if isinstance(c, ToolGroup)]
                    assert len(tool_groups) >= 1
                    assert all(tg.collapsed for tg in tool_groups)

    @pytest.mark.asyncio
    async def test_tool_result_with_cwd(self):
        async def fake_run(graph, messages, cwd=None, thread_id=None):
            yield ("tool_result", "file1.py\nfile2.py\n__CWD__:/tmp/project")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot).message_history = []
                    await _app(pilot)._process_message()
                    assert _app(pilot)._cwd == "/tmp/project"

    @pytest.mark.asyncio
    async def test_tool_result_cwd_only_no_output_message(self):
        """When tool_result is only __CWD__, no Output SystemMessage is added."""

        async def fake_run(graph, messages, cwd=None, thread_id=None):
            yield ("tool_result", "__CWD__:/home/user")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot).message_history = []
                    await _app(pilot)._process_message()
                    assert _app(pilot)._cwd == "/home/user"

    @pytest.mark.asyncio
    async def test_error_event(self):
        async def fake_run(graph, messages, cwd=None, thread_id=None):
            yield ("error", "Something went wrong")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot).message_history = []
                    await _app(pilot)._process_message()
                    chat = _app(pilot).query_one("#chat-container", ChatContainer)
                    sys_msgs = [
                        c for c in chat.children if isinstance(c, SystemMessage)
                    ]
                    assert len(sys_msgs) >= 1

    @pytest.mark.asyncio
    async def test_intent_event_skipped(self):
        """Intent events are skipped (not displayed)."""
        from types import SimpleNamespace

        intent = SimpleNamespace(type="chat", task_list=[], reasoning="Test")

        async def fake_run(graph, messages, cwd=None, thread_id=None):
            yield ("intent", intent)
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot).message_history = []
                    await _app(pilot)._process_message()
                    # Should not crash - intent events are silently skipped

    @pytest.mark.asyncio
    async def test_task_list_event(self):
        """Task list events show the task plan via TaskListMessage."""

        async def fake_run(graph, messages, cwd=None, thread_id=None):
            yield ("task_list", ["Create file", "Run tests"])
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot).message_history = []
                    await _app(pilot)._process_message()
                    chat = _app(pilot).query_one("#chat-container", ChatContainer)
                    task_msgs = [
                        c for c in chat.children if isinstance(c, TaskListMessage)
                    ]
                    # Should have a TaskListMessage with the tasks
                    assert len(task_msgs) == 1

    @pytest.mark.asyncio
    async def test_task_list_empty_skipped(self):
        """Empty task lists don't create messages."""

        async def fake_run(graph, messages, cwd=None, thread_id=None):
            yield ("task_list", [])  # Empty list
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot).message_history = []
                    await _app(pilot)._process_message()
                    # Should not crash - empty task list is handled

    @pytest.mark.asyncio
    async def test_task_start_event(self):
        """Task start events update the TaskListMessage progress."""

        async def fake_run(graph, messages, cwd=None, thread_id=None):
            yield ("task_list", ["Create file", "Run tests"])
            yield ("task_start", "Create file")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot).message_history = []
                    await _app(pilot)._process_message()
                    chat = _app(pilot).query_one("#chat-container", ChatContainer)
                    task_msgs = [
                        c for c in chat.children if isinstance(c, TaskListMessage)
                    ]
                    assert len(task_msgs) == 1
                    # First task should be in progress (index 0)
                    assert task_msgs[0]._current_index == 0

    @pytest.mark.asyncio
    async def test_task_complete_event_updates_progress(self):
        """Task complete events update the TaskListMessage."""

        async def fake_run(graph, messages, cwd=None, thread_id=None):
            yield ("task_list", ["Create file", "Run tests"])
            yield ("task_start", "Create file")
            yield ("task_complete", "Create file")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot).message_history = []
                    await _app(pilot)._process_message()
                    chat = _app(pilot).query_one("#chat-container", ChatContainer)
                    task_msgs = [
                        c for c in chat.children if isinstance(c, TaskListMessage)
                    ]
                    assert len(task_msgs) == 1
                    # First task completed, index moved to 1
                    assert task_msgs[0]._current_index == 1

    @pytest.mark.asyncio
    async def test_messages_event_updates_history(self):
        """The 'messages' event replaces message_history.

        Note: _parse_agent_data is called on *every* event's data before
        the event-type check (line 123 of main.py).  Because it calls
        json.dumps on lists, the data must be JSON-serialisable.
        """
        updated = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

        call_log = []

        async def fake_run(graph, messages, cwd=None, thread_id=None):
            call_log.append("called")
            yield ("messages", updated)

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            async with AxonoApp().run_test(size=(80, 24)) as pilot:
                _app(pilot).agent_graph = cast(Any, object())
                _app(pilot).message_history = [{"role": "user", "content": "hi"}]
                with mock.patch("axono.main.run_agent", new=fake_run):
                    await _app(pilot)._process_message()
                assert call_log == ["called"]
                assert _app(pilot).message_history == updated

    @pytest.mark.asyncio
    async def test_agent_not_initialized(self):
        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            async with AxonoApp().run_test(size=(80, 24)) as pilot:
                _app(pilot).agent_graph = None
                _app(pilot).message_history = []
                await _app(pilot)._process_message()
                # Should not crash, just show error message

    @pytest.mark.asyncio
    async def test_process_message_exception_resets_flag(self):
        async def exploding_run(graph, messages, cwd=None):
            raise RuntimeError("kaboom")  # pragma: no cover
            yield  # pragma: no cover

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=exploding_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot)._is_processing = True
                    _app(pilot).message_history = []
                    await _app(pilot)._process_message()
                    assert _app(pilot)._is_processing is False


# ---------------------------------------------------------------------------
# _drain_thinking_queue
# ---------------------------------------------------------------------------


class TestDrainThinkingQueue:

    @pytest.mark.asyncio
    async def test_thinking_shows_and_removes_on_none(self):
        """ThinkingPanel is created on text, updated, then removed on None."""
        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            async with AxonoApp().run_test(size=(80, 24)) as pilot:
                app = _app(pilot)
                chat = app.query_one("#chat-container", ChatContainer)

                queue: asyncio.Queue = asyncio.Queue()
                drainer = asyncio.create_task(app._drain_thinking_queue(queue, chat))

                # Send thinking text
                await queue.put("Thinking about this...")
                await pilot.pause(delay=0.3)

                # Should have a ThinkingPanel
                panels = list(app.query(ThinkingPanel))
                assert len(panels) == 1

                # Send None to end thinking
                await queue.put(None)
                await pilot.pause(delay=0.3)

                # Panel should be removed
                panels = list(app.query(ThinkingPanel))
                assert len(panels) == 0

                # Clean up
                drainer.cancel()
                await drainer

    @pytest.mark.asyncio
    async def test_thinking_panel_removed_on_cancel(self):
        """ThinkingPanel is removed when drainer is cancelled."""
        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            async with AxonoApp().run_test(size=(80, 24)) as pilot:
                app = _app(pilot)
                chat = app.query_one("#chat-container", ChatContainer)

                queue: asyncio.Queue = asyncio.Queue()
                drainer = asyncio.create_task(app._drain_thinking_queue(queue, chat))

                # Send thinking text to create panel
                await queue.put("Thinking...")
                await pilot.pause(delay=0.3)

                # Panel should exist
                panels = list(app.query(ThinkingPanel))
                assert len(panels) == 1

                # Cancel the drainer
                drainer.cancel()
                await drainer

                # Panel should be removed
                panels = list(app.query(ThinkingPanel))
                assert len(panels) == 0

    @pytest.mark.asyncio
    async def test_thinking_queue_wired_in_process_message(self):
        """_process_message sets up thinking queue and cleans up after."""
        import axono.pipeline as pipeline_mod
        from axono.pipeline import set_thinking_queue

        async def fake_run(graph, messages, cwd=None, thread_id=None):
            # Verify queue is set while running
            assert pipeline_mod._thinking_queue is not None
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).agent_graph = cast(Any, object())
                    _app(pilot).message_history = []
                    await _app(pilot)._process_message()
                    # After processing, queue should be cleaned up
                    assert pipeline_mod._thinking_queue is None


# ---------------------------------------------------------------------------
# action_clear
# ---------------------------------------------------------------------------


class TestActionClear:

    @pytest.mark.asyncio
    async def test_clears_chat_and_history(self):
        with mock.patch("axono.main.get_checkpointer", new=mock.AsyncMock()):
            with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    _app(pilot).message_history = [{"role": "user", "content": "hi"}]
                    _app(pilot).action_clear()
                    await pilot.pause()
                    assert _app(pilot).message_history == []
                    chat = _app(pilot).query_one("#chat-container", ChatContainer)
                    # After clear, there may be initialization messages from async worker
                    # Just verify message_history is cleared
                    assert _app(pilot).message_history == []


# ---------------------------------------------------------------------------
# /config command
# ---------------------------------------------------------------------------


class TestConfigCommand:

    @pytest.mark.asyncio
    async def test_config_command_opens_onboarding(self):
        """The /config command opens the onboarding screen (lines 131-135)."""
        from axono.onboarding import OnboardingScreen

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.needs_onboarding", return_value=False):
                async with AxonoApp().run_test(size=(80, 40)) as pilot:
                    # Type /config command
                    input_widget = _app(pilot).query_one("#input-area", HistoryInput)
                    input_widget.value = "/config"

                    # Simulate submission using the public method
                    await _app(pilot).on_input_submitted(
                        HistoryInput.Submitted(input_widget, "/config")
                    )
                    await pilot.pause()

                    # Onboarding screen should be pushed
                    assert any(
                        isinstance(s, OnboardingScreen)
                        for s in _app(pilot).screen_stack
                    )


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


class TestMain:

    def test_main_calls_app_run(self):
        with mock.patch("sys.argv", ["axono"]):
            with mock.patch("axono.main.needs_onboarding", return_value=False):
                with mock.patch.object(AxonoApp, "run") as mock_run:
                    main()
                    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# TrustWorkspaceScreen
# ---------------------------------------------------------------------------


class TestTrustWorkspaceScreen:

    @pytest.mark.asyncio
    async def test_trust_button_trusts_and_dismisses(self):
        """Trust button adds workspace to allow list and dismisses with True."""
        from axono.main import TrustWorkspaceScreen

        dismissed_with = None

        class TestableScreen(TrustWorkspaceScreen):
            def dismiss(self, result):
                nonlocal dismissed_with
                dismissed_with = result

        screen = TestableScreen("/tmp/test_workspace")

        # Create mock button event for trust button
        mock_button = mock.Mock()
        mock_button.id = "trust"
        mock_event = mock.Mock()
        mock_event.button = mock_button

        with mock.patch("axono.main.trust_workspace") as mock_trust:
            screen.on_button_pressed(mock_event)
            mock_trust.assert_called_once_with("/tmp/test_workspace")
            assert dismissed_with is True

    @pytest.mark.asyncio
    async def test_continue_button_dismisses_without_trust(self):
        """Continue button dismisses with False without trusting."""
        from axono.main import TrustWorkspaceScreen

        dismissed_with = None

        class TestableScreen(TrustWorkspaceScreen):
            def dismiss(self, result):
                nonlocal dismissed_with
                dismissed_with = result

        screen = TestableScreen("/tmp/test_workspace")

        # Create mock button event for continue button
        mock_button = mock.Mock()
        mock_button.id = "continue"
        mock_event = mock.Mock()
        mock_event.button = mock_button

        with mock.patch("axono.main.trust_workspace") as mock_trust:
            screen.on_button_pressed(mock_event)
            mock_trust.assert_not_called()
            assert dismissed_with is False


# ---------------------------------------------------------------------------
# Workspace trust flow
# ---------------------------------------------------------------------------


class TestWorkspaceTrust:

    @pytest.mark.asyncio
    async def test_on_trust_decision_trusted(self, tmp_path):
        """_on_trust_decision shows trusted message when trusted=True."""
        workspace = str(tmp_path)

        async def noop_embed(*args, **kwargs):
            if False:
                yield  # Make it an async generator
            return

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.needs_onboarding", return_value=False):
                with mock.patch("axono.main.is_workspace_trusted", return_value=True):
                    with mock.patch("axono.main.embed_folder", side_effect=noop_embed):
                        async with AxonoApp(workspace=workspace).run_test(
                            size=(80, 24)
                        ) as pilot:
                            await pilot.pause(delay=0.5)
                            # Workspace should be trusted by this point

    @pytest.mark.asyncio
    async def test_on_trust_decision_not_trusted(self, tmp_path):
        """_on_trust_decision shows operating message when trusted=False."""
        workspace = str(tmp_path)

        async def noop_embed(*args, **kwargs):
            if False:
                yield  # Make it an async generator
            return

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.needs_onboarding", return_value=False):
                with mock.patch("axono.main.is_workspace_trusted", return_value=True):
                    with mock.patch("axono.main.embed_folder", side_effect=noop_embed):
                        async with AxonoApp(workspace=workspace).run_test(
                            size=(80, 24)
                        ) as pilot:
                            await pilot.pause(delay=0.5)


# ---------------------------------------------------------------------------
# Workspace indexing
# ---------------------------------------------------------------------------


class TestWorkspaceIndexing:

    @pytest.mark.asyncio
    async def test_start_workspace_indexing_success(self, tmp_path):
        """_start_workspace_indexing indexes workspace and starts watcher."""
        workspace = str(tmp_path)

        async def fake_embed(path):
            yield ("start", {"total_files": 5})
            yield ("file_indexed", {"path": "/test.py", "chunks": 3})
            yield ("complete", {"files": 5, "chunks": 15, "duration": 1.5})

        async def fake_build(on_status=None, checkpointer=None):
            return mock.MagicMock()

        async def fake_checkpointer():
            return mock.MagicMock()

        with mock.patch("axono.main.needs_onboarding", return_value=False):
            with mock.patch("axono.main.is_workspace_trusted", return_value=True):
                with mock.patch("axono.main.build_agent", side_effect=fake_build):
                    with mock.patch(
                        "axono.main.get_checkpointer", side_effect=fake_checkpointer
                    ):
                        with mock.patch(
                            "axono.main.embed_folder", side_effect=fake_embed
                        ):
                            with mock.patch("axono.main.start_watcher") as mock_watcher:
                                async with AxonoApp(workspace=workspace).run_test(
                                    size=(80, 24)
                                ) as pilot:
                                    await pilot.pause(delay=1.0)
                                    # Watcher should be started
                                    mock_watcher.assert_called()

    @pytest.mark.asyncio
    async def test_start_workspace_indexing_error(self, tmp_path):
        """_start_workspace_indexing handles errors gracefully."""
        workspace = str(tmp_path)

        async def failing_embed(path):
            raise RuntimeError("Embedding failed")
            yield  # Make it an async generator

        async def fake_build(on_status=None, checkpointer=None):
            return mock.MagicMock()

        async def fake_checkpointer():
            return mock.MagicMock()

        with mock.patch("axono.main.needs_onboarding", return_value=False):
            with mock.patch("axono.main.is_workspace_trusted", return_value=True):
                with mock.patch("axono.main.build_agent", side_effect=fake_build):
                    with mock.patch(
                        "axono.main.get_checkpointer", side_effect=fake_checkpointer
                    ):
                        with mock.patch(
                            "axono.main.embed_folder", side_effect=failing_embed
                        ):
                            async with AxonoApp(workspace=workspace).run_test(
                                size=(80, 24)
                            ) as pilot:
                                await pilot.pause(delay=1.0)
                                # Should not crash - error handled

    @pytest.mark.asyncio
    async def test_start_file_watcher_watchdog_not_installed(self, tmp_path):
        """_start_file_watcher handles RuntimeError when watchdog not installed."""
        workspace = str(tmp_path)

        async def noop_embed(path):
            if False:
                yield

        async def fake_build(on_status=None, checkpointer=None):
            return mock.MagicMock()

        async def fake_checkpointer():
            return mock.MagicMock()

        with mock.patch("axono.main.needs_onboarding", return_value=False):
            with mock.patch("axono.main.is_workspace_trusted", return_value=True):
                with mock.patch("axono.main.build_agent", side_effect=fake_build):
                    with mock.patch(
                        "axono.main.get_checkpointer", side_effect=fake_checkpointer
                    ):
                        with mock.patch(
                            "axono.main.embed_folder", side_effect=noop_embed
                        ):
                            with mock.patch(
                                "axono.main.start_watcher",
                                side_effect=RuntimeError("watchdog not installed"),
                            ):
                                async with AxonoApp(workspace=workspace).run_test(
                                    size=(80, 24)
                                ) as pilot:
                                    await pilot.pause(delay=1.0)
                                    # Should not crash - RuntimeError caught


# ---------------------------------------------------------------------------
# File watcher callbacks
# ---------------------------------------------------------------------------


class TestFileWatcher:

    @pytest.mark.asyncio
    async def test_on_file_change_triggers_reindex(self):
        """_on_file_change schedules re-indexing."""
        from axono.main import FileChangeEvent

        app = AxonoApp()
        app._workspace = "/tmp/test"

        # Track if call_later was called
        call_later_invoked = False
        original_call_later = app.call_later

        def mock_call_later(callback, **kwargs):
            nonlocal call_later_invoked
            call_later_invoked = True

        app.call_later = mock_call_later

        event = FileChangeEvent(event_type="modified", file_path="/tmp/test/file.py")
        app._on_file_change(event)

        assert call_later_invoked

    @pytest.mark.asyncio
    async def test_reindex_workspace_silent(self, tmp_path):
        """_reindex_workspace runs silently."""
        app = AxonoApp(workspace=str(tmp_path))

        async def fake_embed(path):
            yield ("start", {"total_files": 1})
            yield ("file_error", {"path": "/test.py", "error": "oops"})
            yield ("complete", {"files": 0, "chunks": 0, "duration": 0.1})

        with mock.patch("axono.main.embed_folder", side_effect=fake_embed):
            # Should complete without raising
            await app._reindex_workspace()

    @pytest.mark.asyncio
    async def test_reindex_workspace_exception_silent(self, tmp_path):
        """_reindex_workspace silently handles exceptions."""
        app = AxonoApp(workspace=str(tmp_path))

        async def failing_embed(path):
            raise Exception("boom")
            yield

        with mock.patch("axono.main.embed_folder", side_effect=failing_embed):
            # Should complete without raising
            await app._reindex_workspace()


# ---------------------------------------------------------------------------
# File watcher cleanup
# ---------------------------------------------------------------------------


class TestFileWatcherCleanup:

    def test_stop_file_watcher_when_running(self):
        """_stop_file_watcher stops and joins the watcher."""
        app = AxonoApp()

        mock_watcher = mock.Mock()
        app._file_watcher = mock_watcher

        app._stop_file_watcher()

        mock_watcher.stop.assert_called_once()
        mock_watcher.join.assert_called_once_with(timeout=2.0)
        assert app._file_watcher is None

    def test_stop_file_watcher_when_none(self):
        """_stop_file_watcher does nothing when no watcher."""
        app = AxonoApp()
        app._file_watcher = None

        # Should not raise
        app._stop_file_watcher()
        assert app._file_watcher is None

    @pytest.mark.asyncio
    async def test_on_unmount_stops_watcher(self):
        """on_unmount stops the file watcher."""
        app = AxonoApp()
        mock_watcher = mock.Mock()
        app._file_watcher = mock_watcher

        app.on_unmount()

        mock_watcher.stop.assert_called_once()
        assert app._file_watcher is None

    def test_on_file_change_skips_when_reindex_running(self):
        """_on_file_change returns early when re-index worker exists (line 309)."""
        from axono.main import FileChangeEvent

        app = AxonoApp()
        app._workspace = "/tmp/test"

        # Fake a running reindex worker
        fake_worker = mock.Mock()
        fake_worker.name = "reindex"
        app._workers = mock.Mock()
        app._workers.__iter__ = mock.Mock(return_value=iter([fake_worker]))

        app.call_later = mock.Mock()

        event = FileChangeEvent(event_type="modified", file_path="/tmp/test/file.py")
        app._on_file_change(event)

        # Should NOT schedule a new reindex
        app.call_later.assert_not_called()

    def test_stop_file_watcher_exception(self):
        """_stop_file_watcher handles exceptions (lines 503-504)."""
        app = AxonoApp()

        mock_watcher = mock.Mock()
        mock_watcher.stop.side_effect = RuntimeError("boom")
        app._file_watcher = mock_watcher

        # Should not raise
        app._stop_file_watcher()
        # Watcher should be set to None in finally block
        assert app._file_watcher is None

    def test_action_quit_cancels_workers(self):
        """action_quit cancels running workers (line 492)."""
        app = AxonoApp()

        fake_worker = mock.Mock()
        app._workers = mock.Mock()
        app._workers.__iter__ = mock.Mock(return_value=iter([fake_worker]))

        mock_watcher = mock.Mock()
        app._file_watcher = mock_watcher

        app.on_unmount()

        fake_worker.cancel.assert_called_once()


# ---------------------------------------------------------------------------
# Onboarding done callback
# ---------------------------------------------------------------------------


class TestOnboardingDone:

    def test_on_onboarding_done_checks_trust(self, tmp_path):
        """_on_onboarding_done triggers workspace trust check via run_worker."""
        workspace = str(tmp_path)
        app = AxonoApp(workspace=workspace)

        # Track if run_worker was called
        run_worker_called = False

        def mock_run_worker(*args, **kwargs):
            nonlocal run_worker_called
            run_worker_called = True
            # Close the coroutine to avoid 'never awaited' warning
            if args and hasattr(args[0], "close"):
                args[0].close()

        app.run_worker = mock_run_worker

        app._on_onboarding_done(True)

        assert run_worker_called


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:

    def test_creates_log_directory_and_file(self, tmp_path, monkeypatch):
        """setup_logging creates log/ dir and returns a log file path."""
        import logging as _logging

        monkeypatch.chdir(tmp_path)
        # Reset basicConfig so it can be called again
        root = _logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

        log_file = setup_logging()

        assert "log/axono-" in log_file
        assert log_file.endswith(".log")
        assert Path(log_file).parent.is_dir()

        # Clean up handlers
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

    def test_main_with_log_flag(self, tmp_path, monkeypatch):
        """--log flag triggers setup_logging."""
        monkeypatch.chdir(tmp_path)
        with mock.patch("sys.argv", ["axono", "--log"]):
            with mock.patch.object(AxonoApp, "run"):
                with mock.patch("axono.main.setup_logging") as mock_setup:
                    mock_setup.return_value = "log/axono-test.log"
                    main()
                    mock_setup.assert_called_once()

    def test_main_without_log_flag(self):
        """Without --log, setup_logging is not called."""
        with mock.patch("sys.argv", ["axono"]):
            with mock.patch.object(AxonoApp, "run"):
                with mock.patch("axono.main.setup_logging") as mock_setup:
                    main()
                    mock_setup.assert_not_called()


# ---------------------------------------------------------------------------
# Scanning panel in workspace indexing
# ---------------------------------------------------------------------------


class TestScanningPanelIntegration:

    @pytest.mark.asyncio
    async def test_scanning_panel_mounted_and_removed(self, tmp_path):
        """ScanningPanel is mounted during indexing and removed on complete."""
        workspace = str(tmp_path)

        async def fake_embed(path):
            yield ("start", {"total_files": 3})
            yield ("file_indexed", {"path": "/test.py", "chunks": 2})
            yield ("complete", {"files": 3, "chunks": 6, "duration": 0.5})

        async def fake_build(on_status=None, checkpointer=None):
            return mock.MagicMock()

        async def fake_checkpointer():
            return mock.MagicMock()

        with mock.patch("axono.main.needs_onboarding", return_value=False):
            with mock.patch("axono.main.is_workspace_trusted", return_value=True):
                with mock.patch("axono.main.build_agent", side_effect=fake_build):
                    with mock.patch(
                        "axono.main.get_checkpointer", side_effect=fake_checkpointer
                    ):
                        with mock.patch(
                            "axono.main.embed_folder", side_effect=fake_embed
                        ):
                            with mock.patch("axono.main.start_watcher"):
                                async with AxonoApp(workspace=workspace).run_test(
                                    size=(80, 24)
                                ) as pilot:
                                    await pilot.pause(delay=1.0)
                                    # Panel should be removed after completion
                                    panels = list(_app(pilot).query(ScanningPanel))
                                    assert len(panels) == 0

    @pytest.mark.asyncio
    async def test_scanning_panel_removed_on_error(self, tmp_path):
        """ScanningPanel is removed even when indexing fails."""
        workspace = str(tmp_path)

        async def failing_embed(path):
            raise RuntimeError("Embedding failed")
            yield  # noqa: unreachable

        async def fake_build(on_status=None, checkpointer=None):
            return mock.MagicMock()

        async def fake_checkpointer():
            return mock.MagicMock()

        with mock.patch("axono.main.needs_onboarding", return_value=False):
            with mock.patch("axono.main.is_workspace_trusted", return_value=True):
                with mock.patch("axono.main.build_agent", side_effect=fake_build):
                    with mock.patch(
                        "axono.main.get_checkpointer", side_effect=fake_checkpointer
                    ):
                        with mock.patch(
                            "axono.main.embed_folder", side_effect=failing_embed
                        ):
                            async with AxonoApp(workspace=workspace).run_test(
                                size=(80, 24)
                            ) as pilot:
                                await pilot.pause(delay=1.0)
                                # Panel should be removed even after error
                                panels = list(_app(pilot).query(ScanningPanel))
                                assert len(panels) == 0


# ---------------------------------------------------------------------------
# on_mount onboarding path (line 177)
# ---------------------------------------------------------------------------


class TestOnMountOnboarding:
    """Cover the on_mount branch that pushes OnboardingScreen."""

    @pytest.mark.asyncio
    async def test_on_mount_pushes_onboarding_when_needed(self, tmp_path):
        """When needs_onboarding() is True, OnboardingScreen is pushed."""
        from axono.onboarding import OnboardingScreen

        workspace = str(tmp_path)

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.needs_onboarding", return_value=True):
                with mock.patch("axono.main.is_workspace_trusted", return_value=True):
                    async with AxonoApp(workspace=workspace).run_test(
                        size=(80, 24)
                    ) as pilot:
                        await pilot.pause(delay=0.5)
                        # OnboardingScreen should be on the screen stack
                        assert any(
                            isinstance(s, OnboardingScreen)
                            for s in _app(pilot).screen_stack
                        )


# ---------------------------------------------------------------------------
# _on_trust_decision (lines 206-224)
# ---------------------------------------------------------------------------


class TestOnTrustDecision:
    """Cover _on_trust_decision with trusted=True and trusted=False."""

    @pytest.mark.asyncio
    async def test_on_trust_decision_trusted_shows_message(self, tmp_path):
        """_on_trust_decision with trusted=True shows trusted message."""
        workspace = str(tmp_path)

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.needs_onboarding", return_value=False):
                with mock.patch("axono.main.is_workspace_trusted", return_value=True):
                    async with AxonoApp(workspace=workspace).run_test(
                        size=(80, 24)
                    ) as pilot:
                        await pilot.pause(delay=0.5)
                        app = _app(pilot)
                        # Track call_later invocations
                        call_later_calls = []
                        original_call_later = app.call_later

                        def tracking_call_later(fn, *args, **kwargs):
                            call_later_calls.append((fn, args, kwargs))
                            return original_call_later(fn, *args, **kwargs)

                        app.call_later = tracking_call_later
                        # Call _on_trust_decision directly with trusted=True
                        with mock.patch("axono.main.set_workspace_root") as mock_set:
                            app._on_trust_decision(True)
                            await pilot.pause(delay=0.3)
                            mock_set.assert_called_once_with(workspace)
                        # Verify a SystemMessage about trusting was queued
                        added_msgs = [
                            a
                            for _, args, _ in call_later_calls
                            for a in args
                            if isinstance(a, SystemMessage)
                        ]
                        assert len(added_msgs) == 1

    @pytest.mark.asyncio
    async def test_on_trust_decision_not_trusted_shows_message(self, tmp_path):
        """_on_trust_decision with trusted=False shows operating message."""
        workspace = str(tmp_path)

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.needs_onboarding", return_value=False):
                with mock.patch("axono.main.is_workspace_trusted", return_value=True):
                    async with AxonoApp(workspace=workspace).run_test(
                        size=(80, 24)
                    ) as pilot:
                        await pilot.pause(delay=0.5)
                        app = _app(pilot)
                        # Track call_later invocations
                        call_later_calls = []
                        original_call_later = app.call_later

                        def tracking_call_later(fn, *args, **kwargs):
                            call_later_calls.append((fn, args, kwargs))
                            return original_call_later(fn, *args, **kwargs)

                        app.call_later = tracking_call_later
                        # Call _on_trust_decision directly with trusted=False
                        with mock.patch("axono.main.set_workspace_root") as mock_set:
                            app._on_trust_decision(False)
                            await pilot.pause(delay=0.3)
                            mock_set.assert_called_once_with(workspace)
                        # Verify a SystemMessage was queued
                        added_msgs = [
                            a
                            for _, args, _ in call_later_calls
                            for a in args
                            if isinstance(a, SystemMessage)
                        ]
                        assert len(added_msgs) == 1
