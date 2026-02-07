"""Unit tests for axono.shell."""

import os
import subprocess
import sys
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import pytest

from axono import shell
from axono.shell import (
    StepResult,
    _detect_project_type,
    _get_dir_context,
    _plan_next_step,
    _run_command,
    run_shell_pipeline,
)


# ---------------------------------------------------------------------------
# StepResult dataclass
# ---------------------------------------------------------------------------


class TestStepResult:

    def test_default_values(self):
        result = StepResult(
            command="ls",
            stdout="out",
            stderr="err",
            exit_code=0,
            success=True,
        )
        assert result.blocked is False
        assert result.block_reason == ""

    def test_blocked_result(self):
        result = StepResult(
            command="rm -rf /",
            stdout="",
            stderr="",
            exit_code=-1,
            success=False,
            blocked=True,
            block_reason="Dangerous command",
        )
        assert result.blocked is True
        assert result.block_reason == "Dangerous command"


# ---------------------------------------------------------------------------
# _detect_project_type
# ---------------------------------------------------------------------------


class TestDetectProjectType:

    def test_python_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        result = _detect_project_type(str(tmp_path))
        assert "Python" in result
        assert "pip" in result

    def test_python_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests")
        result = _detect_project_type(str(tmp_path))
        assert "Python" in result

    def test_python_setup_py(self, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup")
        result = _detect_project_type(str(tmp_path))
        assert "Python" in result

    def test_rust_cargo(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'test'")
        result = _detect_project_type(str(tmp_path))
        assert "Rust" in result
        assert "cargo" in result

    def test_nodejs_package_json(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "test"}')
        result = _detect_project_type(str(tmp_path))
        assert "Node.js" in result
        assert "npm" in result

    def test_makefile(self, tmp_path):
        (tmp_path / "Makefile").write_text("all:\n\techo hello")
        result = _detect_project_type(str(tmp_path))
        assert "Make" in result

    def test_cmake(self, tmp_path):
        (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)")
        result = _detect_project_type(str(tmp_path))
        assert "CMake" in result

    def test_go_mod(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/test")
        result = _detect_project_type(str(tmp_path))
        assert "Go" in result

    def test_java_maven(self, tmp_path):
        (tmp_path / "pom.xml").write_text("<project></project>")
        result = _detect_project_type(str(tmp_path))
        assert "Maven" in result

    def test_java_gradle(self, tmp_path):
        (tmp_path / "build.gradle").write_text("plugins {}")
        result = _detect_project_type(str(tmp_path))
        assert "Gradle" in result

    def test_kotlin_gradle(self, tmp_path):
        (tmp_path / "build.gradle.kts").write_text("plugins {}")
        result = _detect_project_type(str(tmp_path))
        assert "Gradle" in result

    def test_ruby_gemfile(self, tmp_path):
        (tmp_path / "Gemfile").write_text("source 'https://rubygems.org'")
        result = _detect_project_type(str(tmp_path))
        assert "Ruby" in result

    def test_php_composer(self, tmp_path):
        (tmp_path / "composer.json").write_text('{"name": "test"}')
        result = _detect_project_type(str(tmp_path))
        assert "PHP" in result

    def test_elixir_mix(self, tmp_path):
        (tmp_path / "mix.exs").write_text("defmodule Test.MixProject")
        result = _detect_project_type(str(tmp_path))
        assert "Elixir" in result

    def test_haskell_stack(self, tmp_path):
        (tmp_path / "stack.yaml").write_text("resolver: lts-19.0")
        result = _detect_project_type(str(tmp_path))
        assert "Haskell" in result

    def test_ocaml_dune(self, tmp_path):
        (tmp_path / "dune-project").write_text("(lang dune 3.0)")
        result = _detect_project_type(str(tmp_path))
        assert "OCaml" in result

    def test_swift_package(self, tmp_path):
        (tmp_path / "Package.swift").write_text("// swift-tools-version:5.5")
        result = _detect_project_type(str(tmp_path))
        assert "Swift" in result

    def test_dotnet_csproj(self, tmp_path):
        (tmp_path / "MyApp.csproj").write_text("<Project></Project>")
        result = _detect_project_type(str(tmp_path))
        assert ".NET" in result
        assert "dotnet" in result

    def test_dotnet_sln(self, tmp_path):
        (tmp_path / "MySolution.sln").write_text("Microsoft Visual Studio Solution")
        result = _detect_project_type(str(tmp_path))
        assert ".NET" in result

    def test_no_project_files(self, tmp_path):
        (tmp_path / "random.txt").write_text("just text")
        result = _detect_project_type(str(tmp_path))
        assert result is None

    def test_oserror_returns_none(self, tmp_path):
        """If the directory can't be listed, return None."""
        with mock.patch("os.listdir", side_effect=OSError("Permission denied")):
            result = _detect_project_type(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# _get_dir_context
# ---------------------------------------------------------------------------


class TestGetDirContext:

    def test_basic_listing(self, tmp_path):
        (tmp_path / "file1.py").write_text("")
        (tmp_path / "file2.txt").write_text("")
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        listing, project_type = _get_dir_context(str(tmp_path))

        assert "subdir" in listing
        assert "file1.py" in listing
        assert "file2.txt" in listing

    def test_dirs_sorted_before_files(self, tmp_path):
        (tmp_path / "z_file.txt").write_text("")
        (tmp_path / "a_dir").mkdir()

        listing, _ = _get_dir_context(str(tmp_path))

        # a_dir should come before z_file
        assert listing.index("a_dir") < listing.index("z_file.txt")

    def test_max_files_truncation(self, tmp_path):
        for i in range(30):
            (tmp_path / f"file{i:02d}.txt").write_text("")

        listing, _ = _get_dir_context(str(tmp_path), max_files=10)

        assert "+20 more" in listing or "(+20 more)" in listing

    def test_empty_directory(self, tmp_path):
        listing, _ = _get_dir_context(str(tmp_path))
        assert listing == "(empty)"

    def test_oserror_returns_unreadable(self, tmp_path):
        with mock.patch("os.listdir", side_effect=OSError("Permission denied")):
            listing, _ = _get_dir_context(str(tmp_path))
        assert listing == "(unreadable)"

    def test_project_type_detection(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("")

        listing, project_type = _get_dir_context(str(tmp_path))

        assert project_type is not None
        assert "Rust" in project_type


# ---------------------------------------------------------------------------
# _run_command
# ---------------------------------------------------------------------------


class TestRunCommand:

    @pytest.mark.asyncio
    async def test_successful_command(self, tmp_path):
        result = await _run_command("echo hello", str(tmp_path), unsafe=True)
        assert result.success is True
        assert "hello" in result.stdout
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_failing_command(self, tmp_path):
        result = await _run_command("exit 1", str(tmp_path), unsafe=True)
        assert result.success is False
        assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_command_with_stderr(self, tmp_path):
        result = await _run_command("echo error >&2", str(tmp_path), unsafe=True)
        assert "error" in result.stderr

    @pytest.mark.asyncio
    async def test_dangerous_command_blocked(self, tmp_path):
        async def dangerous_verdict(cmd):
            return {"dangerous": True, "reason": "rm is destructive"}

        with mock.patch("axono.shell.judge_command", side_effect=dangerous_verdict):
            result = await _run_command("rm -rf /", str(tmp_path), unsafe=False)

        assert result.blocked is True
        assert result.success is False
        assert "destructive" in result.block_reason

    @pytest.mark.asyncio
    async def test_safe_command_passes(self, tmp_path):
        async def safe_verdict(cmd):
            return {"dangerous": False, "reason": "safe"}

        with mock.patch("axono.shell.judge_command", side_effect=safe_verdict):
            result = await _run_command("echo safe", str(tmp_path), unsafe=False)

        assert result.blocked is False
        assert result.success is True

    @pytest.mark.asyncio
    async def test_unsafe_flag_skips_judge(self, tmp_path):
        with mock.patch("axono.shell.judge_command") as mock_judge:
            result = await _run_command("echo bypass", str(tmp_path), unsafe=True)

        mock_judge.assert_not_called()
        assert "bypass" in result.stdout

    @pytest.mark.asyncio
    async def test_judge_exception_prints_warning(self, tmp_path):
        async def exploding_judge(cmd):
            raise RuntimeError("LLM unavailable")

        stderr = StringIO()
        with mock.patch("axono.shell.judge_command", side_effect=exploding_judge):
            with mock.patch("sys.stderr", stderr):
                result = await _run_command("echo fallback", str(tmp_path), unsafe=False)

        assert "fallback" in result.stdout
        assert "safety check failed" in stderr.getvalue()

    @pytest.mark.asyncio
    async def test_timeout_error(self, tmp_path):
        with mock.patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)
        ):
            result = await _run_command("sleep 999", str(tmp_path), unsafe=True)

        assert result.success is False
        assert "Timed out" in result.stderr

    @pytest.mark.asyncio
    async def test_generic_exception(self, tmp_path):
        with mock.patch(
            "subprocess.run", side_effect=PermissionError("Permission denied")
        ):
            result = await _run_command("cmd", str(tmp_path), unsafe=True)

        assert result.success is False
        assert "Permission denied" in result.stderr


# ---------------------------------------------------------------------------
# _plan_next_step
# ---------------------------------------------------------------------------


class TestPlanNextStep:

    @pytest.mark.asyncio
    async def test_returns_parsed_json(self, tmp_path):
        response = SimpleNamespace(content='{"done": false, "command": "ls", "reason": "list files"}')
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            result = await _plan_next_step("task", str(tmp_path), [])

        assert result["command"] == "ls"
        assert result["done"] is False

    @pytest.mark.asyncio
    async def test_includes_history_in_prompt(self, tmp_path):
        response = SimpleNamespace(content='{"done": true, "summary": "done"}')
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        history = [
            StepResult("ls", "file.txt", "", 0, True),
            StepResult("cat file.txt", "content", "", 0, True),
        ]

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            await _plan_next_step("task", str(tmp_path), history)

        call_args = fake_llm.ainvoke.call_args
        user_msg = call_args[0][0][1].content  # Second message is user prompt
        assert "ls" in user_msg
        assert "cat" in user_msg

    @pytest.mark.asyncio
    async def test_includes_error_in_history(self, tmp_path):
        response = SimpleNamespace(content='{"done": true, "summary": "done"}')
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        history = [
            StepResult("bad_cmd", "", "command not found", 127, False),
        ]

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            await _plan_next_step("task", str(tmp_path), history)

        call_args = fake_llm.ainvoke.call_args
        user_msg = call_args[0][0][1].content
        assert "command not found" in user_msg or "err" in user_msg.lower()

    @pytest.mark.asyncio
    async def test_includes_project_type(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("")
        response = SimpleNamespace(content='{"done": true, "summary": "done"}')
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            await _plan_next_step("build project", str(tmp_path), [])

        call_args = fake_llm.ainvoke.call_args
        user_msg = call_args[0][0][1].content
        assert "Rust" in user_msg or "cargo" in user_msg.lower()

    @pytest.mark.asyncio
    async def test_parse_failure_returns_done(self, tmp_path):
        response = SimpleNamespace(content="not valid json")
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            result = await _plan_next_step("task", str(tmp_path), [])

        assert result["done"] is True
        assert "parse" in result["summary"].lower()


# ---------------------------------------------------------------------------
# run_shell_pipeline
# ---------------------------------------------------------------------------


class TestRunShellPipeline:

    @pytest.mark.asyncio
    async def test_immediate_done(self, tmp_path):
        """Pipeline completes immediately when planner returns done."""
        response = SimpleNamespace(content='{"done": true, "summary": "Nothing to do"}')
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_shell_pipeline("task", str(tmp_path)):
                events.append(ev)

        types = [e[0] for e in events]
        assert "result" in types
        assert "cwd" in types

    @pytest.mark.asyncio
    async def test_single_command_success(self, tmp_path):
        """Pipeline executes a single command and completes."""
        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(content='{"done": false, "command": "echo hello", "reason": "test"}')
            return SimpleNamespace(content='{"done": true, "summary": "Done"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            with mock.patch("axono.shell.judge_command", return_value={"dangerous": False}):
                events = []
                async for ev in run_shell_pipeline("task", str(tmp_path)):
                    events.append(ev)

        types = [e[0] for e in events]
        assert "status" in types
        assert "output" in types
        assert "result" in types

    @pytest.mark.asyncio
    async def test_cd_command_changes_cwd(self, tmp_path):
        """cd commands change the working directory."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(content=f'{{"done": false, "command": "cd subdir", "reason": "navigate"}}')
            return SimpleNamespace(content='{"done": true, "summary": "Done"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_shell_pipeline("task", str(tmp_path)):
                events.append(ev)

        # Check that cwd was updated
        cwd_events = [e for e in events if e[0] == "cwd"]
        assert len(cwd_events) == 1
        assert str(subdir) in cwd_events[0][1]

    @pytest.mark.asyncio
    async def test_cd_nonexistent_dir_error(self, tmp_path):
        """cd to nonexistent directory yields error."""
        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(content='{"done": false, "command": "cd /nonexistent/path", "reason": "test"}')
            return SimpleNamespace(content='{"done": true, "summary": "Done"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_shell_pipeline("task", str(tmp_path)):
                events.append(ev)

        types = [e[0] for e in events]
        assert "error" in types
        error_msg = [e[1] for e in events if e[0] == "error"][0]
        assert "not found" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_command_blocked_yields_error(self, tmp_path):
        """Blocked commands yield error events."""
        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(content='{"done": false, "command": "rm -rf /", "reason": "clean"}')
            return SimpleNamespace(content='{"done": true, "summary": "Done"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        async def dangerous_verdict(cmd):
            return {"dangerous": True, "reason": "destructive command"}

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            with mock.patch("axono.shell.judge_command", side_effect=dangerous_verdict):
                events = []
                async for ev in run_shell_pipeline("task", str(tmp_path)):
                    events.append(ev)

        types = [e[0] for e in events]
        assert "error" in types
        error_msg = [e[1] for e in events if e[0] == "error"][0]
        assert "BLOCKED" in error_msg

    @pytest.mark.asyncio
    async def test_command_failure_yields_error(self, tmp_path):
        """Failed commands yield error events."""
        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(content='{"done": false, "command": "exit 1", "reason": "fail"}')
            return SimpleNamespace(content='{"done": true, "summary": "Done"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            with mock.patch("axono.shell.judge_command", return_value={"dangerous": False}):
                events = []
                async for ev in run_shell_pipeline("task", str(tmp_path)):
                    events.append(ev)

        types = [e[0] for e in events]
        assert "error" in types

    @pytest.mark.asyncio
    async def test_empty_output_shows_checkmark(self, tmp_path):
        """Commands with empty output show checkmark."""
        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(content='{"done": false, "command": "true", "reason": "noop"}')
            return SimpleNamespace(content='{"done": true, "summary": "Done"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            with mock.patch("axono.shell.judge_command", return_value={"dangerous": False}):
                events = []
                async for ev in run_shell_pipeline("task", str(tmp_path)):
                    events.append(ev)

        output_events = [e for e in events if e[0] == "output"]
        assert any("✓" in e[1] for e in output_events)

    @pytest.mark.asyncio
    async def test_max_steps_limit(self, tmp_path):
        """Pipeline stops after max steps."""
        # Always return a command, never done
        response = SimpleNamespace(content='{"done": false, "command": "true", "reason": "loop"}')
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            with mock.patch("axono.shell.judge_command", return_value={"dangerous": False}):
                events = []
                async for ev in run_shell_pipeline("task", str(tmp_path)):
                    events.append(ev)

        result_events = [e for e in events if e[0] == "result"]
        assert len(result_events) == 1
        assert "10 steps" in result_events[0][1]

    @pytest.mark.asyncio
    async def test_planning_exception_yields_error(self, tmp_path):
        """Planning exceptions yield error events."""
        with mock.patch("axono.shell.get_llm", side_effect=RuntimeError("LLM down")):
            events = []
            async for ev in run_shell_pipeline("task", str(tmp_path)):
                events.append(ev)

        types = [e[0] for e in events]
        assert "error" in types
        error_msg = [e[1] for e in events if e[0] == "error"][0]
        assert "Planning failed" in error_msg

    @pytest.mark.asyncio
    async def test_no_command_yields_error(self, tmp_path):
        """Empty command yields error."""
        response = SimpleNamespace(content='{"done": false, "command": "", "reason": "oops"}')
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_shell_pipeline("task", str(tmp_path)):
                events.append(ev)

        types = [e[0] for e in events]
        assert "error" in types
        error_msg = [e[1] for e in events if e[0] == "error"][0]
        assert "No command" in error_msg

    @pytest.mark.asyncio
    async def test_status_with_reason(self, tmp_path):
        """Status includes reason when provided."""
        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(content='{"done": false, "command": "ls", "reason": "list directory"}')
            return SimpleNamespace(content='{"done": true, "summary": "Done"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            with mock.patch("axono.shell.judge_command", return_value={"dangerous": False}):
                events = []
                async for ev in run_shell_pipeline("task", str(tmp_path)):
                    events.append(ev)

        status_events = [e for e in events if e[0] == "status"]
        assert any("list directory" in e[1] for e in status_events)

    @pytest.mark.asyncio
    async def test_status_without_reason(self, tmp_path):
        """Status works without reason."""
        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(content='{"done": false, "command": "ls"}')
            return SimpleNamespace(content='{"done": true, "summary": "Done"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            with mock.patch("axono.shell.judge_command", return_value={"dangerous": False}):
                events = []
                async for ev in run_shell_pipeline("task", str(tmp_path)):
                    events.append(ev)

        status_events = [e for e in events if e[0] == "status"]
        assert any("$ ls" in e[1] for e in status_events)

    @pytest.mark.asyncio
    async def test_cd_with_tilde(self, tmp_path):
        """cd ~ expands to home directory."""
        home = os.path.expanduser("~")

        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(content='{"done": false, "command": "cd ~", "reason": "go home"}')
            return SimpleNamespace(content='{"done": true, "summary": "Done"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_shell_pipeline("task", str(tmp_path)):
                events.append(ev)

        cwd_events = [e for e in events if e[0] == "cwd"]
        assert cwd_events[0][1] == home

    @pytest.mark.asyncio
    async def test_unsafe_flag_passed_to_run_command(self, tmp_path):
        """unsafe flag is passed to _run_command."""
        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(content='{"done": false, "command": "rm dangerous", "reason": "test"}')
            return SimpleNamespace(content='{"done": true, "summary": "Done"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.shell.get_llm", return_value=fake_llm):
            with mock.patch("axono.shell.judge_command") as mock_judge:
                # When unsafe=True, judge should not be called
                events = []
                async for ev in run_shell_pipeline("task", str(tmp_path), unsafe=True):
                    events.append(ev)
                mock_judge.assert_not_called()
