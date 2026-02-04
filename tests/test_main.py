"""Unit tests for axono.main (AxonoApp)."""

import json
from pathlib import Path
from unittest import mock

import pytest

from axono.main import AxonoApp, main
from axono.ui import (
    AssistantMessage,
    Banner,
    ChatContainer,
    CwdStatus,
    SystemMessage,
    ToolGroup,
    UserMessage,
)

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
        assert app._cwd == str(Path.home())

    def test_title(self):
        app = AxonoApp()
        assert app.TITLE == "Axono"

    def test_bindings(self):
        app = AxonoApp()
        keys = [b.key for b in app.BINDINGS]
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
                inp = pilot.app.query_one("#input-area")
                assert inp.has_focus


# ---------------------------------------------------------------------------
# _initialize_agent
# ---------------------------------------------------------------------------


class TestInitializeAgent:

    @pytest.mark.asyncio
    async def test_success_sets_graph_and_shows_message(self):
        fake_graph = object()

        async def fake_build(on_status=None):
            return fake_graph

        with mock.patch("axono.main.needs_onboarding", return_value=False):
            with mock.patch("axono.main.build_agent", side_effect=fake_build):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    await pilot.pause(delay=0.5)
                    assert pilot.app.agent_graph is fake_graph
                    chat = pilot.app.query_one("#chat-container", ChatContainer)
                    children = list(chat.children)
                    # Should contain Banner + "Agent initialized" SystemMessage
                    sys_msgs = [c for c in children if isinstance(c, SystemMessage)]
                    assert any(True for c in sys_msgs)

    @pytest.mark.asyncio
    async def test_success_with_mcp_status(self):
        async def fake_build(on_status=None):
            if on_status:
                on_status("Loaded 2 MCP tool(s): search, weather")
            return object()

        with mock.patch("axono.main.needs_onboarding", return_value=False):
            with mock.patch("axono.main.build_agent", side_effect=fake_build):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    await pilot.pause(delay=0.5)
                    chat = pilot.app.query_one("#chat-container", ChatContainer)
                    children = list(chat.children)
                    sys_msgs = [c for c in children if isinstance(c, SystemMessage)]
                    # Banner + "Agent initialized" + MCP status
                    assert len(sys_msgs) >= 2

    @pytest.mark.asyncio
    async def test_failure_shows_error(self):
        async def failing_build(on_status=None):
            raise ConnectionError("LLM unreachable")

        with mock.patch("axono.main.needs_onboarding", return_value=False):
            with mock.patch("axono.main.build_agent", side_effect=failing_build):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    await pilot.pause(delay=0.5)
                    assert pilot.app.agent_graph is None


# ---------------------------------------------------------------------------
# on_input_submitted
# ---------------------------------------------------------------------------


class TestInputSubmitted:

    @pytest.mark.asyncio
    async def test_empty_input_ignored(self):
        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            async with AxonoApp().run_test(size=(80, 24)) as pilot:
                pilot.app.agent_graph = object()
                inp = pilot.app.query_one("#input-area")
                inp.value = "   "
                await inp.action_submit()
                await pilot.pause()
                # No user message should be added
                chat = pilot.app.query_one("#chat-container", ChatContainer)
                user_msgs = [c for c in chat.children if isinstance(c, UserMessage)]
                assert len(user_msgs) == 0

    @pytest.mark.asyncio
    async def test_adds_user_message_and_clears_input(self):
        async def noop_run_agent(graph, messages):
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", side_effect=noop_run_agent):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    pilot.app.agent_graph = object()
                    pilot.app._is_processing = False
                    inp = pilot.app.query_one("#input-area")
                    inp.value = "hello agent"
                    await inp.action_submit()
                    await pilot.pause()
                    # Input should be cleared
                    assert inp.value == ""
                    # User message added
                    chat = pilot.app.query_one("#chat-container", ChatContainer)
                    user_msgs = [c for c in chat.children if isinstance(c, UserMessage)]
                    assert len(user_msgs) == 1

    @pytest.mark.asyncio
    async def test_rejects_input_while_processing(self):
        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            async with AxonoApp().run_test(size=(80, 24)) as pilot:
                pilot.app.agent_graph = object()
                pilot.app._is_processing = True
                inp = pilot.app.query_one("#input-area")
                inp.value = "test"
                await inp.action_submit()
                await pilot.pause()
                # No user message added since we're busy
                chat = pilot.app.query_one("#chat-container", ChatContainer)
                user_msgs = [c for c in chat.children if isinstance(c, UserMessage)]
                assert len(user_msgs) == 0


# ---------------------------------------------------------------------------
# _process_message — event routing
# ---------------------------------------------------------------------------


class TestProcessMessage:
    """Tests call ``_process_message`` directly to avoid worker timing issues."""

    @pytest.mark.asyncio
    async def test_assistant_event(self):
        async def fake_run(graph, messages):
            yield ("assistant", "Hi there!")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    pilot.app.agent_graph = object()
                    pilot.app.message_history = []
                    await pilot.app._process_message()
                    chat = pilot.app.query_one("#chat-container", ChatContainer)
                    asst_msgs = [
                        c for c in chat.children if isinstance(c, AssistantMessage)
                    ]
                    assert len(asst_msgs) >= 1

    @pytest.mark.asyncio
    async def test_tool_call_event(self):
        async def fake_run(graph, messages):
            yield ("tool_call", "bash({'command': 'ls'})")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    pilot.app.agent_graph = object()
                    pilot.app.message_history = []
                    await pilot.app._process_message()
                    chat = pilot.app.query_one("#chat-container", ChatContainer)
                    tool_groups = [
                        c for c in chat.children if isinstance(c, ToolGroup)
                    ]
                    assert len(tool_groups) >= 1

    @pytest.mark.asyncio
    async def test_tool_groups_collapse_on_assistant_message(self):
        async def fake_run(graph, messages):
            yield ("tool_call", "bash({'command': 'ls'})")
            yield ("tool_result", "file1.py\n__CWD__:/tmp")
            yield ("assistant", "Done.")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    pilot.app.agent_graph = object()
                    pilot.app.message_history = []
                    await pilot.app._process_message()
                    chat = pilot.app.query_one("#chat-container", ChatContainer)
                    tool_groups = [
                        c for c in chat.children if isinstance(c, ToolGroup)
                    ]
                    assert len(tool_groups) >= 1
                    assert all(tg.collapsed for tg in tool_groups)

    @pytest.mark.asyncio
    async def test_tool_result_with_cwd(self):
        async def fake_run(graph, messages):
            yield ("tool_result", "file1.py\nfile2.py\n__CWD__:/tmp/project")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    pilot.app.agent_graph = object()
                    pilot.app.message_history = []
                    await pilot.app._process_message()
                    assert pilot.app._cwd == "/tmp/project"

    @pytest.mark.asyncio
    async def test_tool_result_cwd_only_no_output_message(self):
        """When tool_result is only __CWD__, no Output SystemMessage is added."""

        async def fake_run(graph, messages):
            yield ("tool_result", "__CWD__:/home/user")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    pilot.app.agent_graph = object()
                    pilot.app.message_history = []
                    await pilot.app._process_message()
                    assert pilot.app._cwd == "/home/user"

    @pytest.mark.asyncio
    async def test_error_event(self):
        async def fake_run(graph, messages):
            yield ("error", "Something went wrong")
            yield ("messages", list(messages))

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=fake_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    pilot.app.agent_graph = object()
                    pilot.app.message_history = []
                    await pilot.app._process_message()
                    chat = pilot.app.query_one("#chat-container", ChatContainer)
                    sys_msgs = [
                        c for c in chat.children if isinstance(c, SystemMessage)
                    ]
                    assert len(sys_msgs) >= 1

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

        async def fake_run(graph, messages):
            call_log.append("called")
            yield ("messages", updated)

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            async with AxonoApp().run_test(size=(80, 24)) as pilot:
                pilot.app.agent_graph = object()
                pilot.app.message_history = [{"role": "user", "content": "hi"}]
                with mock.patch("axono.main.run_agent", new=fake_run):
                    await pilot.app._process_message()
                assert call_log == ["called"]
                assert pilot.app.message_history == updated

    @pytest.mark.asyncio
    async def test_agent_not_initialized(self):
        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            async with AxonoApp().run_test(size=(80, 24)) as pilot:
                pilot.app.agent_graph = None
                pilot.app.message_history = []
                await pilot.app._process_message()
                # Should not crash, just show error message

    @pytest.mark.asyncio
    async def test_process_message_exception_resets_flag(self):
        async def exploding_run(graph, messages):
            raise RuntimeError("kaboom")
            yield  # noqa: unreachable

        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            with mock.patch("axono.main.run_agent", new=exploding_run):
                async with AxonoApp().run_test(size=(80, 24)) as pilot:
                    pilot.app.agent_graph = object()
                    pilot.app._is_processing = True
                    pilot.app.message_history = []
                    await pilot.app._process_message()
                    assert pilot.app._is_processing is False


# ---------------------------------------------------------------------------
# action_clear
# ---------------------------------------------------------------------------


class TestActionClear:

    @pytest.mark.asyncio
    async def test_clears_chat_and_history(self):
        with mock.patch("axono.main.build_agent", return_value=mock.AsyncMock()):
            async with AxonoApp().run_test(size=(80, 24)) as pilot:
                pilot.app.message_history = [{"role": "user", "content": "hi"}]
                pilot.app.action_clear()
                await pilot.pause()
                assert pilot.app.message_history == []
                chat = pilot.app.query_one("#chat-container", ChatContainer)
                assert len(list(chat.children)) == 0


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
