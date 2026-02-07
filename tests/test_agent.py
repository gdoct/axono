"""Unit tests for axono.agent."""

import io
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import pytest_asyncio

from axono import agent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_cwd(path=None):
    """Reset the module-level ``_CURRENT_DIR`` to a known value."""
    agent._CURRENT_DIR = path or os.path.expanduser("~")


@pytest.fixture(autouse=True)
def _restore_cwd():
    """Ensure ``_CURRENT_DIR`` is reset after every test."""
    original = agent._CURRENT_DIR
    yield
    agent._CURRENT_DIR = original


def _safe_verdict(dangerous=False, reason="ok"):
    """Return a coroutine that resolves to a safety verdict dict."""

    async def _judge(cmd):
        return {"dangerous": dangerous, "reason": reason}

    return _judge


# ---------------------------------------------------------------------------
# bash tool — safety checks
# ---------------------------------------------------------------------------


class TestBashSafety:
    """The bash tool should respect the safety judge."""

    @pytest.mark.asyncio
    async def test_dangerous_command_is_blocked(self):
        with mock.patch(
            "axono.agent.judge_command",
            side_effect=_safe_verdict(dangerous=True, reason="rm is destructive"),
        ):
            result = await agent.bash.ainvoke({"command": "rm -rf /", "unsafe": False})
        assert "BLOCKED" in result
        assert "rm is destructive" in result

    @pytest.mark.asyncio
    async def test_unsafe_flag_bypasses_judge(self):
        """When unsafe=True the judge is not consulted."""
        with mock.patch("axono.agent.judge_command") as mock_judge:
            result = await agent.bash.ainvoke(
                {"command": "echo bypass", "unsafe": True}
            )
        mock_judge.assert_not_called()
        assert "bypass" in result

    @pytest.mark.asyncio
    async def test_judge_exception_does_not_block(self):
        """If the safety judge itself crashes, the command still runs."""

        async def _explode(cmd):
            raise RuntimeError("LLM unavailable")

        with mock.patch("axono.agent.judge_command", side_effect=_explode):
            result = await agent.bash.ainvoke(
                {"command": "echo fallthrough", "unsafe": False}
            )
        assert "fallthrough" in result


# ---------------------------------------------------------------------------
# bash tool — command execution
# ---------------------------------------------------------------------------


class TestBashExecution:
    """Basic command execution behaviour."""

    @pytest.mark.asyncio
    async def test_simple_command(self):
        with mock.patch("axono.agent.judge_command", side_effect=_safe_verdict()):
            result = await agent.bash.ainvoke({"command": "echo hello"})
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_stderr_included(self):
        with mock.patch("axono.agent.judge_command", side_effect=_safe_verdict()):
            result = await agent.bash.ainvoke({"command": "echo err >&2"})
        assert "err" in result

    @pytest.mark.asyncio
    async def test_empty_output_shows_exit_code(self):
        with mock.patch("axono.agent.judge_command", side_effect=_safe_verdict()):
            result = await agent.bash.ainvoke({"command": "true"})
        assert "exit code 0" in result

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        with mock.patch("axono.agent.judge_command", side_effect=_safe_verdict()):
            # Generate output longer than 8000 chars
            result = await agent.bash.ainvoke(
                {"command": "python3 -c \"print('A' * 10000)\""}
            )
        assert "truncated" in result

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self):
        with mock.patch("axono.agent.judge_command", side_effect=_safe_verdict()):
            with mock.patch(
                "subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)
            ):
                result = await agent.bash.ainvoke({"command": "sleep 999"})
        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_generic_exception_returns_error(self):
        with mock.patch("axono.agent.judge_command", side_effect=_safe_verdict()):
            with mock.patch("subprocess.run", side_effect=PermissionError("denied")):
                result = await agent.bash.ainvoke({"command": "nope"})
        assert "PermissionError" in result
        assert "denied" in result


# ---------------------------------------------------------------------------
# bash tool — cd handling
# ---------------------------------------------------------------------------


class TestBashCd:
    """The bash tool tracks the working directory via cd commands."""

    @pytest.mark.asyncio
    async def test_cd_changes_current_dir(self, tmp_path):
        _reset_cwd(str(tmp_path))
        target = tmp_path / "sub"
        target.mkdir()

        with mock.patch("axono.agent.judge_command", side_effect=_safe_verdict()):
            result = await agent.bash.ainvoke({"command": f"cd {target}"})
        assert str(target) in result
        assert agent._CURRENT_DIR == str(target)

    @pytest.mark.asyncio
    async def test_cd_nonexistent_dir(self, tmp_path):
        _reset_cwd(str(tmp_path))
        with mock.patch("axono.agent.judge_command", side_effect=_safe_verdict()):
            result = await agent.bash.ainvoke({"command": "cd /no/such/dir"})
        assert "Directory not found" in result
        # _CURRENT_DIR should be unchanged
        assert agent._CURRENT_DIR == str(tmp_path)

    @pytest.mark.asyncio
    async def test_cd_with_chained_command(self, tmp_path):
        _reset_cwd(str(tmp_path))
        target = tmp_path / "sub"
        target.mkdir()

        with mock.patch("axono.agent.judge_command", side_effect=_safe_verdict()):
            result = await agent.bash.ainvoke(
                {"command": f"cd {target} && echo inside"}
            )
        assert "inside" in result
        assert agent._CURRENT_DIR == str(target)

    @pytest.mark.asyncio
    async def test_cd_relative(self, tmp_path):
        _reset_cwd(str(tmp_path))
        (tmp_path / "rel").mkdir()

        with mock.patch("axono.agent.judge_command", side_effect=_safe_verdict()):
            result = await agent.bash.ainvoke({"command": "cd rel"})
        assert agent._CURRENT_DIR == str(tmp_path / "rel")

    @pytest.mark.asyncio
    async def test_cd_tilde(self):
        home = os.path.expanduser("~")
        _reset_cwd("/tmp")

        with mock.patch("axono.agent.judge_command", side_effect=_safe_verdict()):
            result = await agent.bash.ainvoke({"command": "cd ~"})
        assert agent._CURRENT_DIR == home


# ---------------------------------------------------------------------------
# shell tool
# ---------------------------------------------------------------------------


class TestShellTool:
    """The shell tool delegates to run_shell_pipeline."""

    @pytest.mark.asyncio
    async def test_collects_status_and_result(self):
        async def fake_pipeline(task, working_dir, unsafe):
            yield ("status", "$ ls")
            yield ("output", "file.txt")
            yield ("result", "Task complete")
            yield ("cwd", working_dir)

        with mock.patch("axono.agent.run_shell_pipeline", side_effect=fake_pipeline):
            result = await agent.shell.ainvoke(
                {"task": "list files", "working_dir": "/tmp", "unsafe": False}
            )
        assert "$ ls" in result
        assert "Task complete" in result

    @pytest.mark.asyncio
    async def test_collects_errors(self):
        async def fake_pipeline(task, working_dir, unsafe):
            yield ("error", "command failed")
            yield ("cwd", working_dir)

        with mock.patch("axono.agent.run_shell_pipeline", side_effect=fake_pipeline):
            result = await agent.shell.ainvoke(
                {"task": "do something", "working_dir": "/tmp", "unsafe": False}
            )
        assert "Errors:" in result
        assert "command failed" in result

    @pytest.mark.asyncio
    async def test_multiple_steps_shows_count(self):
        async def fake_pipeline(task, working_dir, unsafe):
            yield ("status", "$ ls")
            yield ("output", "a.txt")
            yield ("status", "$ cat a.txt")
            yield ("output", "content")
            yield ("result", "Done")
            yield ("cwd", working_dir)

        with mock.patch("axono.agent.run_shell_pipeline", side_effect=fake_pipeline):
            result = await agent.shell.ainvoke(
                {"task": "read files", "working_dir": "/tmp", "unsafe": False}
            )
        assert "Ran 2 commands" in result

    @pytest.mark.asyncio
    async def test_cwd_change_updates_global(self, tmp_path):
        _reset_cwd(str(tmp_path))
        new_dir = tmp_path / "subdir"
        new_dir.mkdir()

        async def fake_pipeline(task, working_dir, unsafe):
            yield ("cwd", str(new_dir))

        with mock.patch("axono.agent.run_shell_pipeline", side_effect=fake_pipeline):
            result = await agent.shell.ainvoke(
                {"task": "cd subdir", "working_dir": str(tmp_path), "unsafe": False}
            )
        assert str(new_dir) in result
        assert "__CWD__:" in result
        assert agent._CURRENT_DIR == str(new_dir)

    @pytest.mark.asyncio
    async def test_same_cwd_not_appended(self):
        """If cwd doesn't change, no __CWD__ marker is added."""

        async def fake_pipeline(task, working_dir, unsafe):
            yield ("result", "Done")
            yield ("cwd", "/tmp")  # Same as working_dir

        with mock.patch("axono.agent.run_shell_pipeline", side_effect=fake_pipeline):
            result = await agent.shell.ainvoke(
                {"task": "noop", "working_dir": "/tmp", "unsafe": False}
            )
        assert "__CWD__:" not in result

    @pytest.mark.asyncio
    async def test_long_output_excluded(self):
        """Output longer than 200 chars is excluded."""

        async def fake_pipeline(task, working_dir, unsafe):
            yield ("output", "x" * 300)
            yield ("result", "Done")
            yield ("cwd", "/tmp")

        with mock.patch("axono.agent.run_shell_pipeline", side_effect=fake_pipeline):
            result = await agent.shell.ainvoke(
                {"task": "big output", "working_dir": "/tmp", "unsafe": False}
            )
        # The long output should NOT be in the result
        assert "x" * 300 not in result

    @pytest.mark.asyncio
    async def test_empty_events_returns_done(self):
        """When no events yield meaningful data, returns '(done)'."""

        async def fake_pipeline(task, working_dir, unsafe):
            yield ("cwd", "/tmp")  # Same as working_dir, doesn't add output

        with mock.patch("axono.agent.run_shell_pipeline", side_effect=fake_pipeline):
            result = await agent.shell.ainvoke(
                {"task": "noop", "working_dir": "/tmp", "unsafe": False}
            )
        assert result == "(done)"

    @pytest.mark.asyncio
    async def test_multiple_errors_shows_last_two(self):
        """Only the last 2 errors are shown."""

        async def fake_pipeline(task, working_dir, unsafe):
            yield ("error", "error1")
            yield ("error", "error2")
            yield ("error", "error3")
            yield ("cwd", "/tmp")

        with mock.patch("axono.agent.run_shell_pipeline", side_effect=fake_pipeline):
            result = await agent.shell.ainvoke(
                {"task": "fail", "working_dir": "/tmp", "unsafe": False}
            )
        # Should have error2 and error3, not error1
        assert "error2" in result
        assert "error3" in result

    @pytest.mark.asyncio
    async def test_unsafe_flag_passed(self):
        """The unsafe flag is passed to run_shell_pipeline."""
        calls = []

        async def fake_pipeline(task, working_dir, unsafe):
            calls.append((task, working_dir, unsafe))
            yield ("cwd", working_dir)

        with mock.patch("axono.agent.run_shell_pipeline", side_effect=fake_pipeline):
            await agent.shell.ainvoke(
                {"task": "dangerous", "working_dir": "/tmp", "unsafe": True}
            )
        assert calls[0][2] is True  # unsafe=True was passed


# ---------------------------------------------------------------------------
# code tool
# ---------------------------------------------------------------------------


class TestCodeTool:
    """The code tool delegates to run_coding_pipeline."""

    @pytest.mark.asyncio
    async def test_collects_status_and_result(self):
        async def fake_pipeline(task, working_dir):
            yield ("status", "Scanning…")
            yield ("result", "All done")

        with mock.patch("axono.agent.run_coding_pipeline", side_effect=fake_pipeline):
            result = await agent.code.ainvoke(
                {"task": "add tests", "working_dir": "/tmp"}
            )
        assert "[status] Scanning…" in result
        assert "All done" in result

    @pytest.mark.asyncio
    async def test_collects_errors(self):
        async def fake_pipeline(task, working_dir):
            yield ("error", "something broke")

        with mock.patch("axono.agent.run_coding_pipeline", side_effect=fake_pipeline):
            result = await agent.code.ainvoke(
                {"task": "fix bug", "working_dir": "/tmp"}
            )
        assert "[error] something broke" in result


# ---------------------------------------------------------------------------
# _load_mcp_tools
# ---------------------------------------------------------------------------


class TestLoadMcpTools:
    """MCP tool loading with various configurations."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_mcp_not_installed(self):
        with mock.patch.object(agent, "_HAS_MCP", False):
            with mock.patch.object(agent, "MultiServerMCPClient", None):
                result = await agent._load_mcp_tools()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_server_config(self):
        with mock.patch.object(agent, "_HAS_MCP", True):
            with mock.patch.object(agent, "MultiServerMCPClient", mock.MagicMock()):
                with mock.patch("axono.agent.config") as mock_config:
                    mock_config.load_mcp_config.return_value = {}
                    result = await agent._load_mcp_tools()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_tools_on_success(self):
        fake_tool = SimpleNamespace(name="my_tool")

        mock_client_instance = mock.AsyncMock()
        mock_client_instance.get_tools.return_value = [fake_tool]

        mock_client_cls = mock.MagicMock(return_value=mock_client_instance)

        with mock.patch.object(agent, "_HAS_MCP", True):
            with mock.patch.object(agent, "MultiServerMCPClient", mock_client_cls):
                with mock.patch("axono.agent.config") as mock_config:
                    mock_config.load_mcp_config.return_value = {"server1": {}}
                    result = await agent._load_mcp_tools()

        assert len(result) == 1
        assert result[0].name == "my_tool"

    @pytest.mark.asyncio
    async def test_returns_empty_on_connection_error(self):
        mock_client_cls = mock.MagicMock(side_effect=ConnectionError("refused"))

        stderr = io.StringIO()
        with mock.patch.object(agent, "_HAS_MCP", True):
            with mock.patch.object(agent, "MultiServerMCPClient", mock_client_cls):
                with mock.patch("axono.agent.config") as mock_config:
                    mock_config.load_mcp_config.return_value = {"s": {}}
                    with mock.patch("sys.stderr", stderr):
                        result = await agent._load_mcp_tools()

        assert result == []
        assert "Failed to load MCP tools" in stderr.getvalue()


# ---------------------------------------------------------------------------
# build_agent
# ---------------------------------------------------------------------------


class TestBuildAgent:
    """build_agent wires up the LLM, tools, and system prompt."""

    @pytest.mark.asyncio
    async def test_build_agent_returns_graph(self):
        sentinel = object()
        with mock.patch("axono.agent.init_chat_model") as mock_init:
            with mock.patch("axono.agent._load_mcp_tools", return_value=[]):
                with mock.patch(
                    "axono.agent.create_agent", return_value=sentinel
                ) as mock_create:
                    graph = await agent.build_agent()

        assert graph is sentinel
        mock_init.assert_called_once()
        mock_create.assert_called_once()
        # Should have bash, shell, code, and search as built-in tools
        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        assert len(tools) == 4

    @pytest.mark.asyncio
    async def test_build_agent_includes_mcp_tools(self):
        fake_mcp_tool = SimpleNamespace(name="search")

        with mock.patch("axono.agent.init_chat_model"):
            with mock.patch(
                "axono.agent._load_mcp_tools", return_value=[fake_mcp_tool]
            ):
                with mock.patch(
                    "axono.agent.create_agent", return_value=object()
                ) as mock_create:
                    await agent.build_agent()

        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        assert len(tools) == 5  # bash + shell + code + search + mcp
        prompt = call_kwargs.kwargs.get("system_prompt") or call_kwargs[1].get(
            "system_prompt"
        )
        assert "search" in prompt

    @pytest.mark.asyncio
    async def test_build_agent_on_status_callback(self):
        fake_mcp_tool = SimpleNamespace(name="weather")
        statuses = []

        with mock.patch("axono.agent.init_chat_model"):
            with mock.patch(
                "axono.agent._load_mcp_tools", return_value=[fake_mcp_tool]
            ):
                with mock.patch("axono.agent.create_agent", return_value=object()):
                    await agent.build_agent(on_status=statuses.append)

        assert len(statuses) == 1
        assert "weather" in statuses[0]

    @pytest.mark.asyncio
    async def test_build_agent_no_callback_when_no_mcp(self):
        statuses = []

        with mock.patch("axono.agent.init_chat_model"):
            with mock.patch("axono.agent._load_mcp_tools", return_value=[]):
                with mock.patch("axono.agent.create_agent", return_value=object()):
                    await agent.build_agent(on_status=statuses.append)

        assert statuses == []


# ---------------------------------------------------------------------------
# run_agent
# ---------------------------------------------------------------------------


class TestRunAgent:
    """run_agent streams events from the graph."""

    @pytest.mark.asyncio
    async def test_yields_assistant_text(self):
        from langchain_core.messages import AIMessage

        ai_msg = AIMessage(content="Hello there")

        async def fake_stream(inputs, stream_mode=None):
            yield {"model": {"messages": [ai_msg]}}

        graph = mock.AsyncMock()
        graph.astream = fake_stream

        events = []
        async for ev in agent.run_agent(graph, []):
            events.append(ev)

        types = [e[0] for e in events]
        assert "assistant" in types
        assert any("Hello there" in e[1] for e in events if e[0] == "assistant")
        assert "messages" in types

    @pytest.mark.asyncio
    async def test_yields_tool_call(self):
        from langchain_core.messages import AIMessage

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "bash", "args": {"command": "ls"}, "id": "1"}],
        )

        async def fake_stream(inputs, stream_mode=None):
            yield {"model": {"messages": [ai_msg]}}

        graph = mock.AsyncMock()
        graph.astream = fake_stream

        events = []
        async for ev in agent.run_agent(graph, []):
            events.append(ev)

        assert any(e[0] == "tool_call" and "bash" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_yields_tool_result(self):
        from langchain_core.messages import ToolMessage

        tool_msg = ToolMessage(content="file list", tool_call_id="1")

        async def fake_stream(inputs, stream_mode=None):
            yield {"tools": {"messages": [tool_msg]}}

        graph = mock.AsyncMock()
        graph.astream = fake_stream

        events = []
        async for ev in agent.run_agent(graph, []):
            events.append(ev)

        assert any(e[0] == "tool_result" and "file list" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_yields_error_on_exception(self):
        async def failing_stream(inputs, stream_mode=None):
            raise ValueError("boom")
            yield  # noqa: unreachable – makes this an async generator

        graph = mock.AsyncMock()
        graph.astream = failing_stream

        events = []
        async for ev in agent.run_agent(graph, []):
            events.append(ev)

        assert any(e[0] == "error" and "boom" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_messages_accumulate(self):
        from langchain_core.messages import AIMessage, HumanMessage

        human = HumanMessage(content="hi")
        ai = AIMessage(content="hello")

        async def fake_stream(inputs, stream_mode=None):
            yield {"model": {"messages": [ai]}}

        graph = mock.AsyncMock()
        graph.astream = fake_stream

        events = []
        async for ev in agent.run_agent(graph, [human]):
            events.append(ev)

        msg_event = [e for e in events if e[0] == "messages"]
        assert len(msg_event) == 1
        messages = msg_event[0][1]
        # Should contain the original human message + the AI response
        assert len(messages) == 2
        assert messages[0].content == "hi"
        assert messages[1].content == "hello"
