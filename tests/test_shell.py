"""Unit tests for axono.shell."""

import json
import os
import subprocess
import sys
from io import StringIO
from unittest import mock

import pytest

from axono import shell
from axono.pipeline import ExecutionResult, Investigation, Plan, PlanStep, StepExecution
from axono.shell import (
    StepResult,
    _detect_project_type,
    _get_dir_context,
    _run_command,
    execute_shell,
    investigate_shell,
    plan_shell,
    run_shell_pipeline,
    validate_shell_execution,
)


@pytest.fixture(autouse=True)
def _reset_workspace():
    """Reset workspace root after every test."""
    from axono.workspace import clear_workspace_root

    clear_workspace_root()
    yield
    clear_workspace_root()


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
        assert result is not None and "Python" in result
        assert "pip" in result

    def test_python_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "Python" in result

    def test_python_setup_py(self, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "Python" in result

    def test_rust_cargo(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'test'")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "Rust" in result
        assert "cargo" in result

    def test_nodejs_package_json(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "test"}')
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "Node.js" in result
        assert "npm" in result

    def test_makefile(self, tmp_path):
        (tmp_path / "Makefile").write_text("all:\n\techo hello")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "Make" in result

    def test_cmake(self, tmp_path):
        (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "CMake" in result

    def test_go_mod(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/test")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "Go" in result

    def test_java_maven(self, tmp_path):
        (tmp_path / "pom.xml").write_text("<project></project>")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "Maven" in result

    def test_java_gradle(self, tmp_path):
        (tmp_path / "build.gradle").write_text("plugins {}")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "Gradle" in result

    def test_kotlin_gradle(self, tmp_path):
        (tmp_path / "build.gradle.kts").write_text("plugins {}")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "Gradle" in result

    def test_ruby_gemfile(self, tmp_path):
        (tmp_path / "Gemfile").write_text("source 'https://rubygems.org'")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "Ruby" in result

    def test_php_composer(self, tmp_path):
        (tmp_path / "composer.json").write_text('{"name": "test"}')
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "PHP" in result

    def test_elixir_mix(self, tmp_path):
        (tmp_path / "mix.exs").write_text("defmodule Test.MixProject")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "Elixir" in result

    def test_haskell_stack(self, tmp_path):
        (tmp_path / "stack.yaml").write_text("resolver: lts-19.0")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "Haskell" in result

    def test_ocaml_dune(self, tmp_path):
        (tmp_path / "dune-project").write_text("(lang dune 3.0)")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "OCaml" in result

    def test_swift_package(self, tmp_path):
        (tmp_path / "Package.swift").write_text("// swift-tools-version:5.5")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and "Swift" in result

    def test_dotnet_csproj(self, tmp_path):
        (tmp_path / "MyApp.csproj").write_text("<Project></Project>")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and ".NET" in result
        assert "dotnet" in result

    def test_dotnet_sln(self, tmp_path):
        (tmp_path / "MySolution.sln").write_text("Microsoft Visual Studio Solution")
        result = _detect_project_type(str(tmp_path))
        assert result is not None and ".NET" in result

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
                result = await _run_command(
                    "echo fallback", str(tmp_path), unsafe=False
                )

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
# investigate_shell
# ---------------------------------------------------------------------------


class TestInvestigateShell:

    def test_basic_investigation(self, tmp_path):
        (tmp_path / "file.py").write_text("")

        result = investigate_shell(str(tmp_path))

        assert result.cwd == str(tmp_path)
        assert "file.py" in result.dir_listing
        assert result.project_type is None

    def test_with_project_type(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("")

        result = investigate_shell(str(tmp_path))

        assert result.project_type is not None
        assert "Rust" in result.project_type
        assert "Rust" in result.summary

    def test_summary_contains_info(self, tmp_path):
        result = investigate_shell(str(tmp_path))

        assert str(tmp_path) in result.summary
        assert "Directory:" in result.summary


# ---------------------------------------------------------------------------
# plan_shell
# ---------------------------------------------------------------------------


class TestPlanShell:

    @pytest.mark.asyncio
    async def test_creates_plan_with_steps(self, tmp_path):
        plan_json = json.dumps(
            {
                "summary": "Build project",
                "steps": [
                    {"description": "Install dependencies", "command": "npm install"},
                    {"description": "Run build", "command": "npm run build"},
                ],
            }
        )

        inv = Investigation(
            cwd=str(tmp_path), dir_listing="package.json", project_type="Node.js"
        )

        with mock.patch(
            "axono.shell.stream_response",
            new_callable=mock.AsyncMock,
            return_value=plan_json,
        ):
            result = await plan_shell("Build the project", inv)

        assert result.summary == "Build project"
        assert len(result.steps) == 2
        assert result.steps[0].description == "Install dependencies"
        assert result.steps[0].action["command"] == "npm install"

    @pytest.mark.asyncio
    async def test_handles_parse_failure(self, tmp_path):
        inv = Investigation(cwd=str(tmp_path), dir_listing="")

        with mock.patch(
            "axono.shell.stream_response",
            new_callable=mock.AsyncMock,
            return_value="Not valid JSON",
        ):
            result = await plan_shell("task", inv)

        assert "Not valid JSON" in result.summary
        assert result.steps == []

    @pytest.mark.asyncio
    async def test_includes_project_type_in_prompt(self, tmp_path):
        plan_json = json.dumps({"summary": "s", "steps": []})

        inv = Investigation(
            cwd=str(tmp_path), dir_listing="", project_type="Rust (use: cargo build)"
        )

        with mock.patch(
            "axono.shell.stream_response",
            new_callable=mock.AsyncMock,
            return_value=plan_json,
        ) as mock_stream:
            await plan_shell("build", inv)

        call_args = mock_stream.call_args
        user_msg = call_args[0][0][1].content
        assert "Rust" in user_msg

    @pytest.mark.asyncio
    async def test_skips_invalid_steps(self, tmp_path):
        plan_json = json.dumps(
            {
                "summary": "s",
                "steps": [
                    {"description": "Valid step", "command": "ls"},
                    {"command": "missing description"},  # No description
                    "not a dict",  # Invalid type
                ],
            }
        )

        inv = Investigation(cwd=str(tmp_path), dir_listing="")

        with mock.patch(
            "axono.shell.stream_response",
            new_callable=mock.AsyncMock,
            return_value=plan_json,
        ):
            result = await plan_shell("task", inv)

        assert len(result.steps) == 1
        assert result.steps[0].description == "Valid step"

    @pytest.mark.asyncio
    async def test_previous_issues_included_in_prompt(self, tmp_path):
        """Test that previous_issues are included in the prompt for re-planning."""
        plan_json = json.dumps(
            {
                "summary": "Revised plan",
                "steps": [{"description": "Fixed step", "command": "npm install"}],
            }
        )

        inv = Investigation(cwd=str(tmp_path), dir_listing="package.json")
        previous_issues = ["Missing install step", "Build would fail without deps"]

        with mock.patch(
            "axono.shell.stream_response",
            new_callable=mock.AsyncMock,
            return_value=plan_json,
        ) as mock_stream:
            result = await plan_shell("build", inv, previous_issues=previous_issues)

        # Verify issues were included in the prompt
        call_args = mock_stream.call_args
        user_msg = call_args[0][0][1].content
        assert "Previous Plan Issues" in user_msg
        assert "Missing install step" in user_msg
        assert "Build would fail without deps" in user_msg
        assert "DIFFERENT plan" in user_msg
        assert result.summary == "Revised plan"

    @pytest.mark.asyncio
    async def test_previous_issues_none_no_section(self, tmp_path):
        """Test that no issues section is added when previous_issues is None."""
        plan_json = json.dumps({"summary": "Plan", "steps": []})

        inv = Investigation(cwd=str(tmp_path), dir_listing="")

        with mock.patch(
            "axono.shell.stream_response",
            new_callable=mock.AsyncMock,
            return_value=plan_json,
        ) as mock_stream:
            await plan_shell("task", inv, previous_issues=None)

        call_args = mock_stream.call_args
        user_msg = call_args[0][0][1].content
        assert "Previous Plan Issues" not in user_msg

    @pytest.mark.asyncio
    async def test_previous_issues_empty_no_section(self, tmp_path):
        """Test that no issues section is added when previous_issues is empty."""
        plan_json = json.dumps({"summary": "Plan", "steps": []})

        inv = Investigation(cwd=str(tmp_path), dir_listing="")

        with mock.patch(
            "axono.shell.stream_response",
            new_callable=mock.AsyncMock,
            return_value=plan_json,
        ) as mock_stream:
            await plan_shell("task", inv, previous_issues=[])

        call_args = mock_stream.call_args
        user_msg = call_args[0][0][1].content
        assert "Previous Plan Issues" not in user_msg


# ---------------------------------------------------------------------------
# execute_shell
# ---------------------------------------------------------------------------


class TestExecuteShell:

    @pytest.mark.asyncio
    async def test_successful_execution(self, tmp_path):
        plan = Plan(
            summary="List files",
            steps=[PlanStep(description="List files", action={"command": "ls"})],
        )

        with mock.patch("axono.shell.judge_command", return_value={"dangerous": False}):
            result, cwd = await execute_shell(plan, str(tmp_path), unsafe=True)

        assert result.success is True
        assert len(result.step_results) == 1
        assert result.step_results[0].success is True

    @pytest.mark.asyncio
    async def test_cd_command(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        plan = Plan(
            summary="Navigate",
            steps=[
                PlanStep(description="Go to subdir", action={"command": "cd subdir"})
            ],
        )

        result, cwd = await execute_shell(plan, str(tmp_path), unsafe=True)

        assert result.success is True
        assert cwd == str(subdir)
        assert "Changed to" in result.step_results[0].output

    @pytest.mark.asyncio
    async def test_cd_nonexistent(self, tmp_path):
        plan = Plan(
            summary="Navigate",
            steps=[
                PlanStep(description="Go nowhere", action={"command": "cd nonexistent"})
            ],
        )

        result, cwd = await execute_shell(plan, str(tmp_path), unsafe=True)

        assert result.success is False
        assert "not found" in result.step_results[0].error.lower()

    @pytest.mark.asyncio
    async def test_cd_with_tilde(self, tmp_path):
        plan = Plan(
            summary="Go home",
            steps=[PlanStep(description="Go home", action={"command": "cd ~"})],
        )

        result, cwd = await execute_shell(plan, str(tmp_path), unsafe=True)

        assert result.success is True
        assert cwd == os.path.expanduser("~")

    @pytest.mark.asyncio
    async def test_blocked_command(self, tmp_path):
        plan = Plan(
            summary="Dangerous",
            steps=[PlanStep(description="Delete", action={"command": "rm -rf /"})],
        )

        async def dangerous(cmd):
            return {"dangerous": True, "reason": "destructive"}

        with mock.patch("axono.shell.judge_command", side_effect=dangerous):
            result, cwd = await execute_shell(plan, str(tmp_path), unsafe=False)

        assert result.success is False
        assert "BLOCKED" in result.step_results[0].error

    @pytest.mark.asyncio
    async def test_failed_command(self, tmp_path):
        plan = Plan(
            summary="Fail",
            steps=[PlanStep(description="Fail", action={"command": "exit 1"})],
        )

        with mock.patch("axono.shell.judge_command", return_value={"dangerous": False}):
            result, cwd = await execute_shell(plan, str(tmp_path), unsafe=True)

        assert result.success is False
        assert result.step_results[0].success is False

    @pytest.mark.asyncio
    async def test_empty_command_skipped(self, tmp_path):
        plan = Plan(
            summary="Empty",
            steps=[PlanStep(description="Empty", action={"command": ""})],
        )

        result, cwd = await execute_shell(plan, str(tmp_path), unsafe=True)

        assert len(result.step_results) == 0

    @pytest.mark.asyncio
    async def test_multiple_steps(self, tmp_path):
        plan = Plan(
            summary="Multi",
            steps=[
                PlanStep(description="Step 1", action={"command": "echo one"}),
                PlanStep(description="Step 2", action={"command": "echo two"}),
            ],
        )

        with mock.patch("axono.shell.judge_command", return_value={"dangerous": False}):
            result, cwd = await execute_shell(plan, str(tmp_path), unsafe=True)

        assert result.success is True
        assert len(result.step_results) == 2


# ---------------------------------------------------------------------------
# validate_shell_execution
# ---------------------------------------------------------------------------


class TestValidateShellExecution:

    @pytest.mark.asyncio
    async def test_successful_validation(self, tmp_path):
        resp = json.dumps(
            {
                "ok": True,
                "issues": [],
                "summary": "Task completed",
            }
        )

        plan = Plan(summary="Build", steps=[])
        step = PlanStep(description="Build", action={"command": "npm build"})
        execution = ExecutionResult(
            success=True,
            step_results=[StepExecution(step=step, success=True, output="Built")],
        )

        with mock.patch(
            "axono.shell.stream_response",
            new_callable=mock.AsyncMock,
            return_value=resp,
        ):
            result = await validate_shell_execution("Build project", plan, execution)

        assert result.ok is True
        assert result.summary == "Task completed"

    @pytest.mark.asyncio
    async def test_failed_validation(self, tmp_path):
        resp = json.dumps(
            {
                "ok": False,
                "issues": ["Build failed"],
                "summary": "Incomplete",
            }
        )

        plan = Plan(summary="Build", steps=[])
        step = PlanStep(description="Build", action={"command": "npm build"})
        execution = ExecutionResult(
            success=False,
            step_results=[StepExecution(step=step, success=False, error="Error")],
        )

        with mock.patch(
            "axono.shell.stream_response",
            new_callable=mock.AsyncMock,
            return_value=resp,
        ):
            result = await validate_shell_execution("Build project", plan, execution)

        assert result.ok is False
        assert "Build failed" in result.issues

    @pytest.mark.asyncio
    async def test_parse_failure_defaults_to_not_ok(self, tmp_path):
        plan = Plan(summary="Build", steps=[])
        execution = ExecutionResult(success=True, step_results=[])

        with mock.patch(
            "axono.shell.stream_response",
            new_callable=mock.AsyncMock,
            return_value="All looks good!",
        ):
            result = await validate_shell_execution("task", plan, execution)

        assert result.ok is False
        assert len(result.issues) >= 1
        assert "not valid JSON" in result.summary


# ---------------------------------------------------------------------------
# run_shell_pipeline
# ---------------------------------------------------------------------------


class TestRunShellPipeline:

    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, tmp_path):
        """Test full pipeline with all stages."""
        plan_json = json.dumps(
            {
                "summary": "List files",
                "steps": [{"description": "List directory", "command": "ls"}],
            }
        )
        plan_valid = json.dumps(
            {"valid": True, "issues": [], "suggestions": [], "summary": "OK"}
        )
        exec_valid = json.dumps({"ok": True, "issues": [], "summary": "Done"})

        call_count = 0

        async def fake_stream(messages, model_type="instruction"):
            nonlocal call_count
            call_count += 1
            system_msg = messages[0].content

            if "shell planning agent" in system_msg:
                return plan_json
            elif "plan validation agent" in system_msg:
                return plan_valid
            elif "shell task validation agent" in system_msg:
                return exec_valid

        with mock.patch("axono.shell.stream_response", side_effect=fake_stream):
            with mock.patch("axono.pipeline.stream_response", side_effect=fake_stream):
                with mock.patch(
                    "axono.shell.judge_command", return_value={"dangerous": False}
                ):
                    events = []
                    async for ev in run_shell_pipeline("list files", str(tmp_path)):
                        events.append(ev)

        types = [e[0] for e in events]
        assert "status" in types
        assert "result" in types
        assert "cwd" in types

    @pytest.mark.asyncio
    async def test_plan_validation_loop(self, tmp_path):
        """Test that plan validation can iterate."""
        # First plan invalid, second valid
        plan_json = json.dumps(
            {
                "summary": "Build",
                "steps": [{"description": "Build", "command": "npm build"}],
            }
        )
        plan_invalid = json.dumps(
            {
                "valid": False,
                "issues": ["Missing install"],
                "suggestions": [],
                "summary": "Bad",
            }
        )
        plan_valid = json.dumps(
            {"valid": True, "issues": [], "suggestions": [], "summary": "OK"}
        )
        exec_valid = json.dumps({"ok": True, "issues": [], "summary": "Done"})

        validation_call = 0

        async def fake_stream(messages, model_type="instruction"):
            nonlocal validation_call
            system_msg = messages[0].content

            if "shell planning agent" in system_msg:
                return plan_json
            elif "plan validation agent" in system_msg:
                validation_call += 1
                if validation_call == 1:
                    return plan_invalid
                return plan_valid
            elif "shell task validation agent" in system_msg:
                return exec_valid

        with mock.patch("axono.shell.stream_response", side_effect=fake_stream):
            with mock.patch("axono.pipeline.stream_response", side_effect=fake_stream):
                with mock.patch(
                    "axono.shell.judge_command", return_value={"dangerous": False}
                ):
                    events = []
                    async for ev in run_shell_pipeline("build", str(tmp_path)):
                        events.append(ev)

        # Should have seen plan issues and revision
        status_msgs = [e[1] for e in events if e[0] == "status"]
        assert any(
            "issues" in msg.lower() or "revising" in msg.lower() for msg in status_msgs
        )

    @pytest.mark.asyncio
    async def test_plan_issues_passed_to_replanner(self, tmp_path):
        """Test that plan validation issues are passed to plan_shell on retry."""
        plan_invalid = json.dumps(
            {
                "valid": False,
                "issues": ["Missing deps install", "No error handling"],
                "suggestions": [],
                "summary": "Bad",
            }
        )
        plan_valid = json.dumps(
            {"valid": True, "issues": [], "suggestions": [], "summary": "OK"}
        )
        exec_valid = json.dumps({"ok": True, "issues": [], "summary": "Done"})

        plan_call_count = 0
        captured_user_msgs = []

        async def fake_stream(messages, model_type="instruction"):
            nonlocal plan_call_count
            system_msg = messages[0].content
            user_msg = messages[1].content

            if "shell planning agent" in system_msg:
                plan_call_count += 1
                captured_user_msgs.append(user_msg)
                return json.dumps(
                    {
                        "summary": f"Plan {plan_call_count}",
                        "steps": [{"description": "Step", "command": "echo ok"}],
                    }
                )
            elif "plan validation agent" in system_msg:
                # First validation fails, second passes
                if plan_call_count == 1:
                    return plan_invalid
                return plan_valid
            elif "shell task validation agent" in system_msg:
                return exec_valid

        with mock.patch("axono.shell.stream_response", side_effect=fake_stream):
            with mock.patch("axono.pipeline.stream_response", side_effect=fake_stream):
                with mock.patch(
                    "axono.shell.judge_command", return_value={"dangerous": False}
                ):
                    events = []
                    async for ev in run_shell_pipeline("build", str(tmp_path)):
                        events.append(ev)

        # Should have called plan_shell twice
        assert plan_call_count == 2

        # First call should NOT have previous issues
        assert "Previous Plan Issues" not in captured_user_msgs[0]

        # Second call SHOULD have the previous issues
        assert "Previous Plan Issues" in captured_user_msgs[1]
        assert "Missing deps install" in captured_user_msgs[1]
        assert "No error handling" in captured_user_msgs[1]
        assert "DIFFERENT plan" in captured_user_msgs[1]

    @pytest.mark.asyncio
    async def test_empty_plan_yields_error(self, tmp_path):
        plan_json = json.dumps({"summary": "Empty", "steps": []})

        with mock.patch(
            "axono.shell.stream_response",
            new_callable=mock.AsyncMock,
            return_value=plan_json,
        ):
            events = []
            async for ev in run_shell_pipeline("task", str(tmp_path)):
                events.append(ev)

        types = [e[0] for e in events]
        assert "error" in types
        error_msg = [e[1] for e in events if e[0] == "error"][0]
        assert "No steps" in error_msg

    @pytest.mark.asyncio
    async def test_planning_exception(self, tmp_path):
        with mock.patch(
            "axono.shell.stream_response", side_effect=RuntimeError("LLM down")
        ):
            events = []
            async for ev in run_shell_pipeline("task", str(tmp_path)):
                events.append(ev)

        types = [e[0] for e in events]
        assert "error" in types
        error_msg = [e[1] for e in events if e[0] == "error"][0]
        assert "Planning failed" in error_msg

    @pytest.mark.asyncio
    async def test_validation_exception_proceeds(self, tmp_path):
        """If plan validation fails, pipeline proceeds."""
        plan_json = json.dumps(
            {
                "summary": "Build",
                "steps": [{"description": "Build", "command": "echo build"}],
            }
        )
        exec_valid = json.dumps({"ok": True, "issues": [], "summary": "Done"})

        async def fake_stream(messages, model_type="instruction"):
            system_msg = messages[0].content
            if "shell planning agent" in system_msg:
                return plan_json
            elif "plan validation agent" in system_msg:
                raise RuntimeError("Validation LLM down")
            elif "shell task validation agent" in system_msg:
                return exec_valid

        with mock.patch("axono.shell.stream_response", side_effect=fake_stream):
            with mock.patch("axono.pipeline.stream_response", side_effect=fake_stream):
                with mock.patch(
                    "axono.shell.judge_command", return_value={"dangerous": False}
                ):
                    events = []
                    async for ev in run_shell_pipeline("build", str(tmp_path)):
                        events.append(ev)

        # Should still complete
        types = [e[0] for e in events]
        assert "result" in types

    @pytest.mark.asyncio
    async def test_max_plan_iterations(self, tmp_path):
        """After max iterations, proceeds with last plan."""
        plan_json = json.dumps(
            {
                "summary": "Build",
                "steps": [{"description": "Build", "command": "echo build"}],
            }
        )
        plan_invalid = json.dumps(
            {"valid": False, "issues": ["Bad"], "suggestions": [], "summary": "Bad"}
        )
        exec_valid = json.dumps({"ok": True, "issues": [], "summary": "Done"})

        async def fake_stream(messages, model_type="instruction"):
            system_msg = messages[0].content
            if "shell planning agent" in system_msg:
                return plan_json
            elif "plan validation agent" in system_msg:
                return plan_invalid  # Always invalid
            elif "shell task validation agent" in system_msg:
                return exec_valid

        with mock.patch("axono.shell.stream_response", side_effect=fake_stream):
            with mock.patch("axono.pipeline.stream_response", side_effect=fake_stream):
                with mock.patch(
                    "axono.shell.judge_command", return_value={"dangerous": False}
                ):
                    events = []
                    async for ev in run_shell_pipeline("build", str(tmp_path)):
                        events.append(ev)

        # Should have status about proceeding after attempts
        status_msgs = [e[1] for e in events if e[0] == "status"]
        assert any(
            "5 plan attempts" in msg or "attempt" in msg.lower() for msg in status_msgs
        )

    @pytest.mark.asyncio
    async def test_detected_project_type_shown(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("")

        plan_json = json.dumps(
            {
                "summary": "Build",
                "steps": [{"description": "Build", "command": "cargo build"}],
            }
        )
        plan_valid = json.dumps(
            {"valid": True, "issues": [], "suggestions": [], "summary": "OK"}
        )
        exec_valid = json.dumps({"ok": True, "issues": [], "summary": "Done"})

        async def fake_stream(messages, model_type="instruction"):
            system_msg = messages[0].content
            if "shell planning agent" in system_msg:
                return plan_json
            elif "plan validation agent" in system_msg:
                return plan_valid
            elif "shell task validation agent" in system_msg:
                return exec_valid

        with mock.patch("axono.shell.stream_response", side_effect=fake_stream):
            with mock.patch("axono.pipeline.stream_response", side_effect=fake_stream):
                with mock.patch(
                    "axono.shell.judge_command", return_value={"dangerous": False}
                ):
                    events = []
                    async for ev in run_shell_pipeline("build", str(tmp_path)):
                        events.append(ev)

        # Should show detected Rust
        status_msgs = [e[1] for e in events if e[0] == "status"]
        assert any("Rust" in msg for msg in status_msgs)

    @pytest.mark.asyncio
    async def test_execution_failure_reported(self, tmp_path):
        plan_json = json.dumps(
            {
                "summary": "Fail",
                "steps": [{"description": "Fail", "command": "exit 1"}],
            }
        )
        plan_valid = json.dumps(
            {"valid": True, "issues": [], "suggestions": [], "summary": "OK"}
        )
        exec_valid = json.dumps(
            {"ok": False, "issues": ["Failed"], "summary": "Task failed"}
        )

        async def fake_stream(messages, model_type="instruction"):
            system_msg = messages[0].content
            if "shell planning agent" in system_msg:
                return plan_json
            elif "plan validation agent" in system_msg:
                return plan_valid
            elif "shell task validation agent" in system_msg:
                return exec_valid

        with mock.patch("axono.shell.stream_response", side_effect=fake_stream):
            with mock.patch("axono.pipeline.stream_response", side_effect=fake_stream):
                with mock.patch(
                    "axono.shell.judge_command", return_value={"dangerous": False}
                ):
                    events = []
                    async for ev in run_shell_pipeline("fail", str(tmp_path)):
                        events.append(ev)

        # Should have error event
        types = [e[0] for e in events]
        assert "error" in types

    @pytest.mark.asyncio
    async def test_cwd_updated(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()

        plan_json = json.dumps(
            {
                "summary": "Navigate",
                "steps": [{"description": "Go to sub", "command": "cd sub"}],
            }
        )
        plan_valid = json.dumps(
            {"valid": True, "issues": [], "suggestions": [], "summary": "OK"}
        )
        exec_valid = json.dumps({"ok": True, "issues": [], "summary": "Done"})

        async def fake_stream(messages, model_type="instruction"):
            system_msg = messages[0].content
            if "shell planning agent" in system_msg:
                return plan_json
            elif "plan validation agent" in system_msg:
                return plan_valid
            elif "shell task validation agent" in system_msg:
                return exec_valid

        with mock.patch("axono.shell.stream_response", side_effect=fake_stream):
            with mock.patch("axono.pipeline.stream_response", side_effect=fake_stream):
                events = []
                async for ev in run_shell_pipeline("navigate", str(tmp_path)):
                    events.append(ev)

        cwd_events = [e for e in events if e[0] == "cwd"]
        assert len(cwd_events) == 1
        assert str(subdir) in cwd_events[0][1]

    @pytest.mark.asyncio
    async def test_final_validation_exception(self, tmp_path):
        """Final validation exception doesn't break pipeline."""
        plan_json = json.dumps(
            {
                "summary": "Build",
                "steps": [{"description": "Build", "command": "echo build"}],
            }
        )
        plan_valid = json.dumps(
            {"valid": True, "issues": [], "suggestions": [], "summary": "OK"}
        )

        async def fake_stream(messages, model_type="instruction"):
            system_msg = messages[0].content
            if "shell planning agent" in system_msg:
                return plan_json
            elif "plan validation agent" in system_msg:
                return plan_valid
            elif "shell task validation agent" in system_msg:
                raise RuntimeError("Validation failed")

        with mock.patch("axono.shell.stream_response", side_effect=fake_stream):
            with mock.patch("axono.pipeline.stream_response", side_effect=fake_stream):
                with mock.patch(
                    "axono.shell.judge_command", return_value={"dangerous": False}
                ):
                    events = []
                    async for ev in run_shell_pipeline("build", str(tmp_path)):
                        events.append(ev)

        # Should still have result
        result_events = [e for e in events if e[0] == "result"]
        assert len(result_events) == 1
        assert "validation error" in result_events[0][1]

    @pytest.mark.asyncio
    async def test_execution_exception(self, tmp_path):
        """Execution exception yields error and cwd."""
        plan_json = json.dumps(
            {
                "summary": "Build",
                "steps": [{"description": "Build", "command": "echo build"}],
            }
        )
        plan_valid = json.dumps(
            {"valid": True, "issues": [], "suggestions": [], "summary": "OK"}
        )

        async def fake_stream(messages, model_type="instruction"):
            system_msg = messages[0].content
            if "shell planning agent" in system_msg:
                return plan_json
            elif "plan validation agent" in system_msg:
                return plan_valid

        with mock.patch("axono.shell.stream_response", side_effect=fake_stream):
            with mock.patch("axono.pipeline.stream_response", side_effect=fake_stream):
                with mock.patch(
                    "axono.shell.execute_shell",
                    side_effect=RuntimeError("Execution crashed"),
                ):
                    events = []
                    async for ev in run_shell_pipeline("build", str(tmp_path)):
                        events.append(ev)

        types = [e[0] for e in events]
        assert "error" in types
        assert "cwd" in types

        error_msg = [e[1] for e in events if e[0] == "error"][0]
        assert "Execution failed" in error_msg

    @pytest.mark.asyncio
    async def test_plan_shell_exception(self, tmp_path):
        """When plan_shell raises exception, yields error and returns early."""
        with mock.patch(
            "axono.shell.plan_shell",
            side_effect=RuntimeError("Plan failed"),
        ):
            events = []
            async for ev in run_shell_pipeline("task", str(tmp_path)):
                events.append(ev)

        types = [e[0] for e in events]
        assert "error" in types
        assert "cwd" in types  # Should always yield cwd at the end
        error_msgs = [e[1] for e in events if e[0] == "error"]
        assert any("Planning failed" in msg for msg in error_msgs)


# ---------------------------------------------------------------------------
# Workspace restrictions in shell pipeline
# ---------------------------------------------------------------------------


class TestShellWorkspaceRestrictions:
    """Tests for workspace boundary enforcement in shell module."""

    @pytest.mark.asyncio
    async def test_run_command_blocks_path_outside_workspace(self, tmp_path):
        """_run_command should block commands with paths outside workspace."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        set_workspace_root(str(workspace))

        result = await _run_command("cat /etc/passwd", str(workspace))

        assert result.blocked is True
        assert "outside the workspace boundary" in result.block_reason

    @pytest.mark.asyncio
    async def test_run_command_allows_path_within_workspace(self, tmp_path):
        """_run_command should allow commands within workspace."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "test.txt").write_text("hello")
        set_workspace_root(str(workspace))

        with mock.patch("axono.shell.judge_command") as mock_judge:

            async def safe_judge(cmd):
                return {"dangerous": False, "reason": "ok"}

            mock_judge.side_effect = safe_judge
            result = await _run_command("cat test.txt", str(workspace))

        assert result.blocked is False

    @pytest.mark.asyncio
    async def test_execute_shell_cd_outside_workspace_blocked(self, tmp_path):
        """execute_shell should block cd commands that escape workspace."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        set_workspace_root(str(workspace))

        plan = Plan(
            summary="test",
            steps=[PlanStep(description="Go to root", action={"command": "cd /tmp"})],
            raw="{}",
        )

        result, final_cwd = await execute_shell(plan, str(workspace))

        assert result.success is False
        assert any("BLOCKED" in (sr.error or "") for sr in result.step_results)
        # CWD should remain unchanged
        assert final_cwd == str(workspace)

    @pytest.mark.asyncio
    async def test_execute_shell_cd_parent_escape_blocked(self, tmp_path):
        """execute_shell should block cd .. that escapes workspace."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        set_workspace_root(str(workspace))

        plan = Plan(
            summary="test",
            steps=[PlanStep(description="Go up", action={"command": "cd .."})],
            raw="{}",
        )

        result, final_cwd = await execute_shell(plan, str(workspace))

        assert result.success is False
        assert any("BLOCKED" in (sr.error or "") for sr in result.step_results)

    @pytest.mark.asyncio
    async def test_execute_shell_cd_within_workspace_allowed(self, tmp_path):
        """execute_shell should allow cd within workspace."""
        from axono.workspace import set_workspace_root

        workspace = tmp_path / "workspace"
        subdir = workspace / "subdir"
        subdir.mkdir(parents=True)
        set_workspace_root(str(workspace))

        plan = Plan(
            summary="test",
            steps=[
                PlanStep(description="Go to subdir", action={"command": "cd subdir"})
            ],
            raw="{}",
        )

        result, final_cwd = await execute_shell(plan, str(workspace))

        assert result.success is True
        assert final_cwd == str(subdir)

    @pytest.mark.asyncio
    async def test_no_workspace_allows_all(self, tmp_path):
        """Without workspace set, all paths should be allowed."""
        # No workspace is set (cleared by autouse fixture)

        with mock.patch("axono.shell.judge_command") as mock_judge:

            async def safe_judge(cmd):
                return {"dangerous": False, "reason": "ok"}

            mock_judge.side_effect = safe_judge
            result = await _run_command("cat /etc/hostname", str(tmp_path))

        # Should not be blocked by workspace (may succeed or fail based on file existence)
        assert result.blocked is False
