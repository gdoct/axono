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


@pytest.fixture(autouse=True)
def _reset_workspace():
    """Reset workspace root after every test."""
    from axono.workspace import clear_workspace_root

    clear_workspace_root()
    yield
    clear_workspace_root()


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
    async def test_build_agent_sets_workspace_root(self):
        """build_agent sets _CURRENT_DIR from get_workspace_root (line 314)."""
        with mock.patch(
            "axono.agent.get_workspace_root", return_value="/custom/workspace"
        ):
            with mock.patch("axono.agent.init_chat_model"):
                with mock.patch("axono.agent._load_mcp_tools", return_value=[]):
                    with mock.patch("axono.agent.create_agent", return_value=object()):
                        await agent.build_agent()

        assert agent._CURRENT_DIR == "/custom/workspace"
        # Reset
        agent._CURRENT_DIR = os.getcwd()

    @pytest.mark.asyncio
    async def test_build_agent_no_callback_when_no_mcp(self):
        statuses = []

        with mock.patch("axono.agent.init_chat_model"):
            with mock.patch("axono.agent._load_mcp_tools", return_value=[]):
                with mock.patch("axono.agent.create_agent", return_value=object()):
                    await agent.build_agent(on_status=statuses.append)

        assert statuses == []


# ---------------------------------------------------------------------------
# _run_chat
# ---------------------------------------------------------------------------


class TestRunChat:
    """_run_chat handles direct LLM responses for chat intent."""

    @pytest.mark.asyncio
    async def test_yields_assistant_response(self):
        from langchain_core.messages import AIMessage, HumanMessage

        mock_llm = mock.AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="Hello!")

        events = []
        async for ev in agent._run_chat(mock_llm, [HumanMessage(content="Hi")]):
            events.append(ev)

        assert any(e[0] == "assistant" and "Hello!" in e[1] for e in events)
        assert any(e[0] == "messages" for e in events)

    @pytest.mark.asyncio
    async def test_yields_error_on_exception(self):
        mock_llm = mock.AsyncMock()
        mock_llm.ainvoke.side_effect = ValueError("LLM error")

        events = []
        async for ev in agent._run_chat(mock_llm, []):
            events.append(ev)

        assert any(e[0] == "error" and "LLM error" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_empty_content_not_yielded(self):
        from langchain_core.messages import AIMessage

        mock_llm = mock.AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="")

        events = []
        async for ev in agent._run_chat(mock_llm, []):
            events.append(ev)

        # No assistant event for empty content
        assert not any(e[0] == "assistant" for e in events)
        # But messages should still be yielded
        assert any(e[0] == "messages" for e in events)


# ---------------------------------------------------------------------------
# _run_task_step
# ---------------------------------------------------------------------------


class TestRunTaskStep:
    """_run_task_step executes a single task using the agent."""

    @pytest.mark.asyncio
    async def test_yields_agent_events(self):
        from langchain_core.messages import AIMessage, HumanMessage

        ai_msg = AIMessage(content="Done")

        async def fake_stream(inputs, stream_mode=None, config=None):
            yield {"model": {"messages": [ai_msg]}}

        graph = mock.AsyncMock()
        graph.astream = fake_stream

        events = []
        async for ev in agent._run_task_step(
            graph, [HumanMessage(content="original")], "Create file", "/tmp"
        ):
            events.append(ev)

        assert any(e[0] == "assistant" and "Done" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_includes_project_type_in_prompt(self):
        """project_type is included in task_prompt when detected (line 386)."""
        from langchain_core.messages import AIMessage

        ai_msg = AIMessage(content="Done")
        captured_inputs = {}

        async def fake_stream(inputs, stream_mode=None, config=None):
            captured_inputs.update(inputs)
            yield {"model": {"messages": [ai_msg]}}

        graph = mock.AsyncMock()
        graph.astream = fake_stream

        with mock.patch(
            "axono.agent._get_dir_context", return_value=("file1.py", "python")
        ):
            events = []
            async for ev in agent._run_task_step(graph, [], "Create file", "/tmp"):
                events.append(ev)

        # The task prompt (last message) should include the project type
        last_msg = captured_inputs["messages"][-1]
        assert "python" in last_msg.content

    @pytest.mark.asyncio
    async def test_yields_tool_calls(self):
        from langchain_core.messages import AIMessage

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "bash", "args": {"command": "touch file"}, "id": "1"}],
        )

        async def fake_stream(inputs, stream_mode=None, config=None):
            yield {"model": {"messages": [ai_msg]}}

        graph = mock.AsyncMock()
        graph.astream = fake_stream

        events = []
        async for ev in agent._run_task_step(graph, [], "Create file", "/tmp"):
            events.append(ev)

        assert any(e[0] == "tool_call" and "bash" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_yields_error_on_exception(self):
        async def failing_stream(inputs, stream_mode=None, config=None):
            raise ValueError("agent boom")
            yield  # noqa: unreachable

        graph = mock.AsyncMock()
        graph.astream = failing_stream

        events = []
        async for ev in agent._run_task_step(graph, [], "Task", "/tmp"):
            events.append(ev)

        assert any(e[0] == "error" and "agent boom" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_yields_tool_result(self):
        """Test that tool results from the tools node are yielded."""
        from langchain_core.messages import ToolMessage

        tool_msg = ToolMessage(content="file created", tool_call_id="1")

        async def fake_stream(inputs, stream_mode=None, config=None):
            yield {"tools": {"messages": [tool_msg]}}

        graph = mock.AsyncMock()
        graph.astream = fake_stream

        events = []
        async for ev in agent._run_task_step(graph, [], "Create file", "/tmp"):
            events.append(ev)

        assert any(e[0] == "tool_result" and "file created" in e[1] for e in events)


# ---------------------------------------------------------------------------
# run_agent
# ---------------------------------------------------------------------------


class TestRunAgent:
    """run_agent orchestrates intent analysis and execution."""

    @pytest.mark.asyncio
    async def test_no_user_message_yields_error(self):
        graph = mock.AsyncMock()

        events = []
        async for ev in agent.run_agent(graph, [], cwd="/tmp"):
            events.append(ev)

        assert any(e[0] == "error" and "No user message" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_chat_intent_uses_direct_llm(self):
        from langchain_core.messages import AIMessage, HumanMessage

        mock_intent = agent.Intent(type="chat", task_list=[], reasoning="Greeting")

        mock_llm_response = AIMessage(content="Hello!")

        with mock.patch("axono.agent.analyze_intent", return_value=mock_intent):
            # Patch at the source since it's imported inside the function
            with mock.patch("langchain.chat_models.init_chat_model") as mock_init:
                mock_llm = mock.AsyncMock()
                mock_llm.ainvoke.return_value = mock_llm_response
                mock_init.return_value = mock_llm

                graph = mock.AsyncMock()
                events = []
                async for ev in agent.run_agent(
                    graph, [HumanMessage(content="Hi")], cwd="/tmp"
                ):
                    events.append(ev)

        # Should yield intent event
        assert any(e[0] == "intent" for e in events)
        # Should yield assistant response
        assert any(e[0] == "assistant" and "Hello!" in e[1] for e in events)
        # Graph should NOT be called for chat intent
        graph.astream.assert_not_called()

    @pytest.mark.asyncio
    async def test_task_intent_yields_task_list(self):
        from langchain_core.messages import AIMessage, HumanMessage

        mock_intent = agent.Intent(
            type="task",
            task_list=["Step 1", "Step 2"],
            reasoning="Creation request",
        )

        ai_msg = AIMessage(content="Done")

        async def fake_stream(inputs, stream_mode=None, config=None):
            yield {"model": {"messages": [ai_msg]}}

        with mock.patch("axono.agent.analyze_intent", return_value=mock_intent):
            graph = mock.AsyncMock()
            graph.astream = fake_stream

            events = []
            async for ev in agent.run_agent(
                graph, [HumanMessage(content="Create project")], cwd="/tmp"
            ):
                events.append(ev)

        # Should yield task_list
        task_list_events = [e for e in events if e[0] == "task_list"]
        assert len(task_list_events) == 1
        assert task_list_events[0][1] == ["Step 1", "Step 2"]

        # Should yield task_start for each task
        task_starts = [e for e in events if e[0] == "task_start"]
        assert len(task_starts) == 2

        # Should yield task_complete for each task
        task_completes = [e for e in events if e[0] == "task_complete"]
        assert len(task_completes) == 2

    @pytest.mark.asyncio
    async def test_task_intent_empty_list_fallback(self):
        from langchain_core.messages import AIMessage, HumanMessage

        mock_intent = agent.Intent(
            type="task",
            task_list=[],  # Empty task list
            reasoning="Something",
        )

        ai_msg = AIMessage(content="Fallback response")

        async def fake_stream(inputs, stream_mode=None, config=None):
            yield {"model": {"messages": [ai_msg]}}

        with mock.patch("axono.agent.analyze_intent", return_value=mock_intent):
            graph = mock.AsyncMock()
            graph.astream = fake_stream

            events = []
            async for ev in agent.run_agent(
                graph, [HumanMessage(content="Do something")], cwd="/tmp"
            ):
                events.append(ev)

        # Should yield error about empty task list
        assert any(e[0] == "error" and "no tasks" in e[1].lower() for e in events)
        # Should still execute via agent fallback
        assert any(e[0] == "assistant" for e in events)

    @pytest.mark.asyncio
    async def test_intent_analysis_error(self):
        from langchain_core.messages import HumanMessage

        with mock.patch(
            "axono.agent.analyze_intent", side_effect=ValueError("LLM down")
        ):
            graph = mock.AsyncMock()

            events = []
            async for ev in agent.run_agent(
                graph, [HumanMessage(content="Hello")], cwd="/tmp"
            ):
                events.append(ev)

        assert any(e[0] == "error" and "Intent analysis failed" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_cwd_expanded(self):
        """Test that ~ in cwd is expanded."""
        import os

        from langchain_core.messages import AIMessage, HumanMessage

        mock_intent = agent.Intent(type="chat", task_list=[], reasoning="Test")

        captured_cwd = []

        async def capture_intent(msg, cwd):
            captured_cwd.append(cwd)
            return mock_intent

        with mock.patch("axono.agent.analyze_intent", side_effect=capture_intent):
            with mock.patch("langchain.chat_models.init_chat_model") as mock_init:
                mock_llm = mock.AsyncMock()
                mock_llm.ainvoke.return_value = AIMessage(content="Hi")
                mock_init.return_value = mock_llm

                graph = mock.AsyncMock()
                events = []
                async for ev in agent.run_agent(
                    graph, [HumanMessage(content="Hi")], cwd="~"
                ):
                    events.append(ev)

        # ~ should be expanded to home directory
        assert captured_cwd[0] == os.path.expanduser("~")

    @pytest.mark.asyncio
    async def test_task_step_error_continues_to_next(self):
        """Test that an error in one task doesn't stop subsequent tasks."""
        from langchain_core.messages import AIMessage, HumanMessage

        mock_intent = agent.Intent(
            type="task",
            task_list=["Task 1", "Task 2"],
            reasoning="Multi-step",
        )

        call_count = [0]

        async def fake_stream(inputs, stream_mode=None, config=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Task 1 failed")
            yield {"model": {"messages": [AIMessage(content="Task 2 done")]}}

        with mock.patch("axono.agent.analyze_intent", return_value=mock_intent):
            graph = mock.AsyncMock()
            graph.astream = fake_stream

            events = []
            async for ev in agent.run_agent(
                graph, [HumanMessage(content="Do tasks")], cwd="/tmp"
            ):
                events.append(ev)

        # Should have error from task 1
        assert any(e[0] == "error" and "Task 1 failed" in e[1] for e in events)
        # Should still complete task 2
        assert any(e[0] == "assistant" and "Task 2 done" in e[1] for e in events)
        # Both tasks should complete
        task_completes = [e for e in events if e[0] == "task_complete"]
        assert len(task_completes) == 2

    @pytest.mark.asyncio
    async def test_message_with_type_attribute(self):
        """Test extraction of user message from object with .type attribute."""
        from types import SimpleNamespace

        mock_intent = agent.Intent(type="chat", task_list=[], reasoning="Test")

        # Create a message-like object with both content and type attributes
        msg = SimpleNamespace(content="Hello world", type="human")

        with mock.patch(
            "axono.agent.analyze_intent", return_value=mock_intent
        ) as mock_analyze:
            with mock.patch("langchain.chat_models.init_chat_model") as mock_init:
                from langchain_core.messages import AIMessage

                mock_llm = mock.AsyncMock()
                mock_llm.ainvoke.return_value = AIMessage(content="Hi!")
                mock_init.return_value = mock_llm

                graph = mock.AsyncMock()
                events = []
                async for ev in agent.run_agent(graph, [msg], cwd="/tmp"):
                    events.append(ev)

        # analyze_intent should be called with the extracted message
        mock_analyze.assert_called_once()
        assert mock_analyze.call_args[0][0] == "Hello world"

    @pytest.mark.asyncio
    async def test_fallback_yields_tool_call_and_result(self):
        """Test fallback agent execution yields tool calls and results."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        mock_intent = agent.Intent(
            type="task",
            task_list=[],  # Empty - triggers fallback
            reasoning="Something",
        )

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "bash", "args": {"command": "ls"}, "id": "1"}],
        )
        tool_msg = ToolMessage(content="file1.txt", tool_call_id="1")

        async def fake_stream(inputs, stream_mode=None, config=None):
            yield {"model": {"messages": [ai_msg]}}
            yield {"tools": {"messages": [tool_msg]}}

        with mock.patch("axono.agent.analyze_intent", return_value=mock_intent):
            graph = mock.AsyncMock()
            graph.astream = fake_stream

            events = []
            async for ev in agent.run_agent(
                graph, [HumanMessage(content="Do something")], cwd="/tmp"
            ):
                events.append(ev)

        # Should yield tool_call from fallback
        assert any(e[0] == "tool_call" and "bash" in e[1] for e in events)
        # Should yield tool_result from fallback
        assert any(e[0] == "tool_result" and "file1.txt" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_fallback_exception_yields_error(self):
        """Test fallback agent execution error handling."""
        from langchain_core.messages import HumanMessage

        mock_intent = agent.Intent(
            type="task",
            task_list=[],  # Empty - triggers fallback
            reasoning="Something",
        )

        async def failing_stream(inputs, stream_mode=None, config=None):
            raise RuntimeError("Agent crashed")
            yield  # noqa: unreachable

        with mock.patch("axono.agent.analyze_intent", return_value=mock_intent):
            graph = mock.AsyncMock()
            graph.astream = failing_stream

            events = []
            async for ev in agent.run_agent(
                graph, [HumanMessage(content="Do something")], cwd="/tmp"
            ):
                events.append(ev)

        # Should yield the initial error about no tasks
        assert any(e[0] == "error" and "no tasks" in e[1].lower() for e in events)
        # Should also yield the exception error
        assert any(e[0] == "error" and "Agent crashed" in e[1] for e in events)


# ---------------------------------------------------------------------------
# bash tool — workspace restrictions
# ---------------------------------------------------------------------------


class TestBashWorkspaceRestrictions:
    """The bash tool should respect workspace boundaries."""

    @pytest.mark.asyncio
    async def test_command_outside_workspace_blocked(self, tmp_path):
        """Commands with paths outside workspace should be blocked."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        set_workspace_root(str(workspace))
        _reset_cwd(str(workspace))

        with mock.patch("axono.agent.judge_command", _safe_verdict()):
            result = await agent.bash.ainvoke({"command": "cat /etc/passwd"})

        assert "BLOCKED" in result
        assert "outside the workspace boundary" in result

    @pytest.mark.asyncio
    async def test_cd_outside_workspace_blocked(self, tmp_path):
        """CD to directory outside workspace should be blocked."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        set_workspace_root(str(workspace))
        _reset_cwd(str(workspace))

        with mock.patch("axono.agent.judge_command", _safe_verdict()):
            result = await agent.bash.ainvoke({"command": "cd /tmp"})

        assert "BLOCKED" in result
        assert "outside the workspace boundary" in result

    @pytest.mark.asyncio
    async def test_cd_parent_escaping_workspace_blocked(self, tmp_path):
        """CD with .. that escapes workspace should be blocked."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        set_workspace_root(str(workspace))
        _reset_cwd(str(workspace))

        with mock.patch("axono.agent.judge_command", _safe_verdict()):
            result = await agent.bash.ainvoke({"command": "cd .."})

        assert "BLOCKED" in result
        # The error message mentions workspace boundary in different ways
        assert "workspace boundary" in result

    @pytest.mark.asyncio
    async def test_cd_within_workspace_allowed(self, tmp_path):
        """CD within workspace should work normally."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        subdir = workspace / "subdir"
        subdir.mkdir(parents=True)
        set_workspace_root(str(workspace))
        _reset_cwd(str(workspace))

        with mock.patch("axono.agent.judge_command", _safe_verdict()):
            result = await agent.bash.ainvoke({"command": "cd subdir"})

        assert "BLOCKED" not in result
        assert f"__CWD__:{subdir}" in result

    @pytest.mark.asyncio
    async def test_command_with_relative_path_within_workspace(self, tmp_path):
        """Commands with relative paths within workspace should work."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "test.txt").write_text("hello")
        set_workspace_root(str(workspace))
        _reset_cwd(str(workspace))

        with mock.patch("axono.agent.judge_command", _safe_verdict()):
            result = await agent.bash.ainvoke({"command": "cat test.txt"})

        assert "BLOCKED" not in result

    @pytest.mark.asyncio
    async def test_no_workspace_restriction_when_not_set(self, tmp_path):
        """When no workspace is set, all paths should be allowed."""
        # Workspace is not set (cleared by fixture)
        _reset_cwd(str(tmp_path))

        with mock.patch("axono.agent.judge_command", _safe_verdict()):
            # This would be blocked if workspace was set
            result = await agent.bash.ainvoke({"command": "cat /etc/hostname"})

        # Should not be blocked by workspace (may be blocked by safety)
        assert "outside the workspace boundary" not in result

    @pytest.mark.asyncio
    async def test_chained_cd_escaping_blocked(self, tmp_path):
        """Chained commands that result in directory escape should be detected."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        set_workspace_root(str(workspace))
        _reset_cwd(str(workspace))

        with mock.patch("axono.agent.judge_command", _safe_verdict()):
            # This command includes a path that escapes
            result = await agent.bash.ainvoke({"command": "cat ../../../etc/passwd"})

        assert "BLOCKED" in result
        assert "would escape the workspace boundary" in result

    @pytest.mark.asyncio
    async def test_pure_cd_validate_cd_target_raises_workspace_violation(
        self, tmp_path
    ):
        """When check_command_paths passes but validate_cd_target raises, cd is blocked."""
        from axono.workspace import WorkspaceViolationError, set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        set_workspace_root(str(workspace))
        _reset_cwd(str(workspace))

        with mock.patch("axono.agent.judge_command", _safe_verdict()):
            with mock.patch("axono.agent.check_command_paths", return_value=None):
                with mock.patch(
                    "axono.agent.validate_cd_target",
                    side_effect=WorkspaceViolationError("test violation"),
                ):
                    result = await agent.bash.ainvoke({"command": "cd somewhere"})

        assert "BLOCKED" in result
        assert "test violation" in result

    @pytest.mark.asyncio
    async def test_chained_cd_post_execution_outside_workspace_blocked(self, tmp_path):
        """Chained cd that ends outside workspace after execution is blocked."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        set_workspace_root(str(workspace))
        _reset_cwd(str(workspace))

        fake_result = subprocess.CompletedProcess(
            args="cmd",
            returncode=0,
            stdout=f"some output\n{outside}",
            stderr="",
        )

        with mock.patch("axono.agent.judge_command", _safe_verdict()):
            with mock.patch("axono.agent.check_command_paths", return_value=None):
                with mock.patch("subprocess.run", return_value=fake_result):
                    result = await agent.bash.ainvoke(
                        {"command": f"cd {outside} && echo hi"}
                    )

        assert "BLOCKED" in result
        assert "outside the workspace boundary" in result
        # CWD should remain unchanged
        assert agent._CURRENT_DIR == str(workspace)


# ---------------------------------------------------------------------------
# code and shell tools — workspace validation
# ---------------------------------------------------------------------------


class TestCodeToolWorkspace:
    """The code tool should validate working_dir against workspace."""

    @pytest.mark.asyncio
    async def test_working_dir_outside_workspace_blocked(self, tmp_path):
        """Code tool with working_dir outside workspace should be blocked."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        set_workspace_root(str(workspace))

        result = await agent.code.ainvoke(
            {"task": "create file", "working_dir": "/tmp/other"}
        )

        assert "BLOCKED" in result
        assert "outside the workspace boundary" in result

    @pytest.mark.asyncio
    async def test_working_dir_within_workspace_allowed(self, tmp_path):
        """Code tool with working_dir inside workspace should work."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        set_workspace_root(str(workspace))

        async def fake_pipeline(task, working_dir):
            yield ("result", f"worked in {working_dir}")

        with mock.patch("axono.agent.run_coding_pipeline", fake_pipeline):
            result = await agent.code.ainvoke(
                {"task": "create file", "working_dir": str(workspace)}
            )

        assert "BLOCKED" not in result
        assert "worked in" in result


class TestShellToolWorkspace:
    """The shell tool should validate working_dir against workspace."""

    @pytest.mark.asyncio
    async def test_working_dir_outside_workspace_blocked(self, tmp_path):
        """Shell tool with working_dir outside workspace should be blocked."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        set_workspace_root(str(workspace))

        result = await agent.shell.ainvoke(
            {"task": "install deps", "working_dir": "/tmp/other"}
        )

        assert "BLOCKED" in result
        assert "outside the workspace boundary" in result

    @pytest.mark.asyncio
    async def test_working_dir_within_workspace_allowed(self, tmp_path):
        """Shell tool with working_dir inside workspace should work."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        set_workspace_root(str(workspace))

        async def fake_pipeline(task, working_dir, unsafe):
            yield ("result", f"worked in {working_dir}")

        with mock.patch("axono.agent.run_shell_pipeline", fake_pipeline):
            result = await agent.shell.ainvoke(
                {"task": "install deps", "working_dir": str(workspace)}
            )

        assert "BLOCKED" not in result
        assert "worked in" in result

    @pytest.mark.asyncio
    async def test_cwd_update_outside_workspace_warned(self, tmp_path):
        """Shell tool should warn if pipeline returns cwd outside workspace."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        set_workspace_root(str(workspace))
        _reset_cwd(str(workspace))

        async def fake_pipeline(task, working_dir, unsafe):
            yield ("result", "done")
            yield ("cwd", "/tmp/escaped")  # Outside workspace

        with mock.patch("axono.agent.run_shell_pipeline", fake_pipeline):
            result = await agent.shell.ainvoke(
                {"task": "install deps", "working_dir": str(workspace)}
            )

        assert "outside the workspace boundary" in result
        # CWD should not have changed
        assert agent._CURRENT_DIR == str(workspace)
