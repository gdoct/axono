"""Unit tests for axono.pipeline."""

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from axono.pipeline import (
    ActionResult,
    ExecutionResult,
    FinalValidation,
    PipelineContext,
    PlanStep,
    PlanValidation,
    StepExecution,
    coerce_response_text,
    parse_json,
    plan_next_action,
    truncate,
    validate_plan,
)

# ---------------------------------------------------------------------------
# coerce_response_text
# ---------------------------------------------------------------------------


class TestCoerceResponseText:

    def test_none_returns_empty(self):
        assert coerce_response_text(None) == ""

    def test_list_returns_json(self):
        result = coerce_response_text([{"a": 1}])
        assert result == json.dumps([{"a": 1}])

    def test_string_unchanged(self):
        assert coerce_response_text("hello") == "hello"

    def test_int_converted(self):
        assert coerce_response_text(42) == "42"


# ---------------------------------------------------------------------------
# parse_json
# ---------------------------------------------------------------------------


class TestParseJson:

    def test_valid_json(self):
        result = parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_strips_markdown_fences(self):
        result = parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_strips_plain_fences(self):
        result = parse_json('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_invalid_json_returns_none(self):
        assert parse_json("not valid json") is None

    def test_handles_whitespace(self):
        result = parse_json('  \n{"key": "value"}\n  ')
        assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# truncate
# ---------------------------------------------------------------------------


class TestTruncate:

    def test_short_text_unchanged(self):
        assert truncate("hello", 100) == "hello"

    def test_long_text_truncated(self):
        result = truncate("a" * 200, 10)
        assert result == "a" * 10 + "..."

    def test_strips_whitespace(self):
        assert truncate("  hello  ", 100) == "hello"

    def test_default_max_len(self):
        short = "x" * 100
        long_text = "y" * 600
        assert truncate(short) == short
        assert len(truncate(long_text)) == 503  # 500 + "..."


# ---------------------------------------------------------------------------
# ActionResult
# ---------------------------------------------------------------------------


class TestActionResult:

    def test_defaults(self):
        result = ActionResult(action="test", success=True)
        assert result.output == ""
        assert result.error == ""
        assert result.data == {}

    def test_with_values(self):
        result = ActionResult(
            action="run",
            success=False,
            output="out",
            error="err",
            data={"key": "val"},
        )
        assert result.action == "run"
        assert result.success is False
        assert result.output == "out"
        assert result.error == "err"
        assert result.data == {"key": "val"}


# ---------------------------------------------------------------------------
# PipelineContext
# ---------------------------------------------------------------------------


class TestPipelineContext:

    def test_defaults(self):
        ctx = PipelineContext(task="test", cwd="/tmp")
        assert ctx.history == []
        assert ctx.state == {}

    def test_add_result(self):
        ctx = PipelineContext(task="test", cwd="/tmp")
        result = ActionResult(action="a", success=True)
        ctx.add_result(result)
        assert len(ctx.history) == 1
        assert ctx.history[0] is result

    def test_last_n_results(self):
        ctx = PipelineContext(task="test", cwd="/tmp")
        for i in range(10):
            ctx.add_result(ActionResult(action=f"a{i}", success=True))

        last_3 = ctx.last_n_results(3)
        assert len(last_3) == 3
        assert last_3[0].action == "a7"
        assert last_3[2].action == "a9"

    def test_last_n_results_empty(self):
        ctx = PipelineContext(task="test", cwd="/tmp")
        assert ctx.last_n_results(5) == []


# ---------------------------------------------------------------------------
# plan_next_action
# ---------------------------------------------------------------------------


class TestPlanNextAction:

    @pytest.mark.asyncio
    async def test_returns_parsed_json(self):
        response = SimpleNamespace(content='{"action": "test", "reason": "because"}')
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        result = await plan_next_action("System", "User", llm=fake_llm)
        assert result == {"action": "test", "reason": "because"}

    @pytest.mark.asyncio
    async def test_handles_parse_failure(self):
        response = SimpleNamespace(content="not valid json")
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        result = await plan_next_action("System", "User", llm=fake_llm)
        assert result["done"] is True
        assert "parse" in result["summary"].lower()

    @pytest.mark.asyncio
    async def test_handles_markdown_fences(self):
        response = SimpleNamespace(
            content='```json\n{"done": true, "summary": "ok"}\n```'
        )
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        result = await plan_next_action("System", "User", llm=fake_llm)
        assert result["done"] is True
        assert result["summary"] == "ok"

    @pytest.mark.asyncio
    async def test_creates_llm_when_none_provided(self):
        """When llm is None, get_llm is called to create one."""
        response = SimpleNamespace(content='{"action": "test"}')
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        with mock.patch(
            "axono.pipeline.get_llm", return_value=fake_llm
        ) as mock_get_llm:
            result = await plan_next_action("System", "User")

        mock_get_llm.assert_called_once()
        assert result == {"action": "test"}


# ---------------------------------------------------------------------------
# PlanStep
# ---------------------------------------------------------------------------


class TestPlanStep:

    def test_defaults(self):
        step = PlanStep(description="Install deps")
        assert step.description == "Install deps"
        assert step.action == {}

    def test_with_action(self):
        step = PlanStep(
            description="Run build",
            action={"command": "npm run build"},
        )
        assert step.description == "Run build"
        assert step.action == {"command": "npm run build"}


# ---------------------------------------------------------------------------
# PlanValidation
# ---------------------------------------------------------------------------


class TestPlanValidation:

    def test_defaults(self):
        val = PlanValidation(valid=True)
        assert val.valid is True
        assert val.issues == []
        assert val.suggestions == []
        assert val.summary == ""

    def test_with_issues(self):
        val = PlanValidation(
            valid=False,
            issues=["missing step"],
            suggestions=["add cleanup"],
            summary="incomplete",
        )
        assert val.valid is False
        assert val.issues == ["missing step"]
        assert val.suggestions == ["add cleanup"]
        assert val.summary == "incomplete"


# ---------------------------------------------------------------------------
# StepExecution
# ---------------------------------------------------------------------------


class TestStepExecution:

    def test_defaults(self):
        step = PlanStep(description="test")
        exec_result = StepExecution(step=step, success=True)
        assert exec_result.step is step
        assert exec_result.success is True
        assert exec_result.output == ""
        assert exec_result.error == ""

    def test_with_error(self):
        step = PlanStep(description="test")
        exec_result = StepExecution(
            step=step,
            success=False,
            output="partial",
            error="command failed",
        )
        assert exec_result.success is False
        assert exec_result.output == "partial"
        assert exec_result.error == "command failed"


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


class TestExecutionResult:

    def test_defaults(self):
        result = ExecutionResult(success=True)
        assert result.success is True
        assert result.step_results == []
        assert result.summary == ""

    def test_with_steps(self):
        step1 = PlanStep(description="step1")
        step2 = PlanStep(description="step2")
        exec1 = StepExecution(step=step1, success=True)
        exec2 = StepExecution(step=step2, success=True)

        result = ExecutionResult(
            success=True,
            step_results=[exec1, exec2],
            summary="all done",
        )
        assert len(result.step_results) == 2
        assert result.summary == "all done"


# ---------------------------------------------------------------------------
# FinalValidation
# ---------------------------------------------------------------------------


class TestFinalValidation:

    def test_defaults(self):
        val = FinalValidation(ok=True)
        assert val.ok is True
        assert val.issues == []
        assert val.summary == ""

    def test_with_issues(self):
        val = FinalValidation(
            ok=False,
            issues=["syntax error", "missing import"],
            summary="code has problems",
        )
        assert val.ok is False
        assert len(val.issues) == 2
        assert val.summary == "code has problems"


# ---------------------------------------------------------------------------
# validate_plan
# ---------------------------------------------------------------------------


def _fake_llm(content):
    """Return a mock LLM whose ``ainvoke`` resolves to the given content."""
    response = SimpleNamespace(content=content)
    llm = mock.AsyncMock()
    llm.ainvoke.return_value = response
    return llm


class TestValidatePlan:

    @pytest.mark.asyncio
    async def test_valid_plan(self):
        resp = json.dumps(
            {
                "valid": True,
                "issues": [],
                "suggestions": [],
                "summary": "Plan looks good",
            }
        )
        llm = _fake_llm(resp)

        with mock.patch("axono.pipeline.get_llm", return_value=llm):
            result = await validate_plan(
                task="Build the project",
                plan_summary="Install deps and build",
                steps=[
                    PlanStep(description="Install dependencies"),
                    PlanStep(description="Run build"),
                ],
            )

        assert result.valid is True
        assert result.summary == "Plan looks good"

    @pytest.mark.asyncio
    async def test_invalid_plan(self):
        resp = json.dumps(
            {
                "valid": False,
                "issues": ["Missing test step"],
                "suggestions": ["Add a test command after build"],
                "summary": "Plan incomplete",
            }
        )
        llm = _fake_llm(resp)

        with mock.patch("axono.pipeline.get_llm", return_value=llm):
            result = await validate_plan(
                task="Build and test the project",
                plan_summary="Just build",
                steps=[
                    PlanStep(description="Run build"),
                ],
            )

        assert result.valid is False
        assert "Missing test step" in result.issues
        assert len(result.suggestions) == 1

    @pytest.mark.asyncio
    async def test_with_context(self):
        resp = json.dumps(
            {
                "valid": True,
                "issues": [],
                "suggestions": [],
                "summary": "OK",
            }
        )
        llm = _fake_llm(resp)

        with mock.patch("axono.pipeline.get_llm", return_value=llm):
            result = await validate_plan(
                task="Build project",
                plan_summary="Build with cargo",
                steps=[PlanStep(description="Run cargo build")],
                context="Project type: Rust",
            )

        # Verify context was included in prompt
        call_args = llm.ainvoke.call_args
        user_msg = call_args[0][0][1].content
        assert "Rust" in user_msg
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_parse_failure_assumes_valid(self):
        """If JSON can't be parsed, assume valid to avoid blocking."""
        llm = _fake_llm("This looks like a good plan!")

        with mock.patch("axono.pipeline.get_llm", return_value=llm):
            result = await validate_plan(
                task="Task",
                plan_summary="Summary",
                steps=[PlanStep(description="Step")],
            )

        assert result.valid is True
        assert "good plan" in result.summary

    @pytest.mark.asyncio
    async def test_handles_markdown_fences(self):
        resp = '```json\n{"valid": true, "issues": [], "suggestions": [], "summary": "OK"}\n```'
        llm = _fake_llm(resp)

        with mock.patch("axono.pipeline.get_llm", return_value=llm):
            result = await validate_plan(
                task="Task",
                plan_summary="Summary",
                steps=[PlanStep(description="Step")],
            )

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_missing_fields_default(self):
        """Missing fields in response use defaults."""
        resp = json.dumps({"valid": False})
        llm = _fake_llm(resp)

        with mock.patch("axono.pipeline.get_llm", return_value=llm):
            result = await validate_plan(
                task="Task",
                plan_summary="Summary",
                steps=[PlanStep(description="Step")],
            )

        assert result.valid is False
        assert result.issues == []
        assert result.suggestions == []
        assert result.summary == ""
