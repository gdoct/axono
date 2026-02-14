"""Unit tests for axono.pipeline."""

import asyncio
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from axono.pipeline import (
    ActionResult,
    ExecutionResult,
    FinalValidation,
    Investigation,
    PipelineConfig,
    PipelineContext,
    Plan,
    PlanStep,
    PlanValidation,
    StepExecution,
    _emit_thinking,
    _get_openai_client,
    coerce_response_text,
    llm_plan,
    llm_validate,
    parse_json,
    plan_next_action,
    run_pipeline,
    set_thinking_queue,
    stream_response,
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

    def test_json_after_preamble_text(self):
        result = parse_json('Let me think about this...\n\n{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_fenced_block_with_surrounding_text(self):
        text = (
            "Here is my reasoning.\n"
            "```json\n"
            '{"key": "value"}\n'
            "```\n"
            "That should work."
        )
        result = parse_json(text)
        assert result == {"key": "value"}

    def test_json_array_in_text(self):
        result = parse_json('Here is the plan:\n[{"path": "a.py"}]')
        assert result == [{"path": "a.py"}]

    def test_json_with_trailing_text(self):
        result = parse_json('{"ok": true} -- that is my answer')
        assert result == {"ok": True}

    def test_json_object_after_reasoning(self):
        text = (
            "I need to consider the files.\n"
            "The auth module is relevant.\n\n"
            '{"action": "investigate", "reason": "find auth files"}'
        )
        result = parse_json(text)
        assert result == {"action": "investigate", "reason": "find auth files"}

    def test_fenced_invalid_json_falls_through(self):
        """Fenced block with invalid JSON falls through (lines 184-185)."""
        # The fenced block is found but content is not valid JSON,
        # so it falls through to subsequent strategies
        result = parse_json("```json\n{invalid json here\n```")
        assert result is None

    def test_closing_bracket_invalid_json(self):
        """Matching closing bracket found but content not valid JSON (lines 201-202)."""
        # Text has { and matching } but the content between isn't valid JSON
        result = parse_json("prefix {not: valid, json} suffix {also bad}")
        assert result is None

    def test_no_json_returns_none(self):
        assert parse_json("just some text with no json at all") is None


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
# Investigation
# ---------------------------------------------------------------------------


class TestInvestigation:

    def test_defaults(self):
        inv = Investigation(cwd="/tmp", dir_listing=[])
        assert inv.cwd == "/tmp"
        assert inv.dir_listing == []
        assert inv.files == []
        assert inv.project_type is None
        assert inv.summary == ""
        assert inv.skipped == []

    def test_with_all_fields(self):
        inv = Investigation(
            cwd="/project",
            dir_listing=["file1.py", "file2.py"],
            files=[{"path": "file1.py", "content": "code"}],
            project_type="Python",
            summary="Found 2 files",
            skipped=["big_file.py"],
        )
        assert inv.cwd == "/project"
        assert len(inv.dir_listing) == 2
        assert len(inv.files) == 1
        assert inv.project_type == "Python"
        assert inv.summary == "Found 2 files"
        assert inv.skipped == ["big_file.py"]

    def test_string_dir_listing(self):
        """Shell pipeline uses string for dir_listing."""
        inv = Investigation(cwd="/tmp", dir_listing="file1, file2, file3")
        assert inv.dir_listing == "file1, file2, file3"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class TestPlan:

    def test_defaults(self):
        plan = Plan(summary="Do something")
        assert plan.summary == "Do something"
        assert plan.steps == []
        assert plan.files_to_read == []
        assert plan.patches == []
        assert plan.raw == ""

    def test_with_steps(self):
        steps = [
            PlanStep(description="Step 1", action={"command": "echo 1"}),
            PlanStep(description="Step 2", action={"command": "echo 2"}),
        ]
        plan = Plan(summary="Two steps", steps=steps)
        assert len(plan.steps) == 2

    def test_with_coding_fields(self):
        plan = Plan(
            summary="Coding plan",
            files_to_read=["main.py", "utils.py"],
            patches=[{"path": "new.py", "action": "create"}],
        )
        assert plan.files_to_read == ["main.py", "utils.py"]
        assert len(plan.patches) == 1


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:

    def test_defaults(self):
        config = PipelineConfig()
        assert config.max_plan_iterations == 5
        assert config.max_action_steps == 15
        assert config.iterative is False

    def test_custom_values(self):
        config = PipelineConfig(
            max_plan_iterations=3,
            max_action_steps=10,
            iterative=True,
        )
        assert config.max_plan_iterations == 3
        assert config.max_action_steps == 10
        assert config.iterative is True


# ---------------------------------------------------------------------------
# Thinking queue
# ---------------------------------------------------------------------------


class TestThinkingQueue:

    def test_set_thinking_queue(self):
        """set_thinking_queue updates the global queue."""
        import axono.pipeline as p

        original = p._thinking_queue
        try:
            q = asyncio.Queue()
            set_thinking_queue(q)
            assert p._thinking_queue is q
            set_thinking_queue(None)
            assert p._thinking_queue is None
        finally:
            p._thinking_queue = original

    def test_emit_thinking_with_queue(self):
        """_emit_thinking pushes to queue when set."""
        q = asyncio.Queue()
        set_thinking_queue(q)
        try:
            _emit_thinking("hello")
            assert q.get_nowait() == "hello"
            _emit_thinking(None)
            assert q.get_nowait() is None
        finally:
            set_thinking_queue(None)

    def test_emit_thinking_without_queue(self):
        """_emit_thinking does nothing when no queue is set."""
        set_thinking_queue(None)
        _emit_thinking("hello")  # Should not raise
        _emit_thinking(None)  # Should not raise


# ---------------------------------------------------------------------------
# _get_openai_client
# ---------------------------------------------------------------------------


class TestGetOpenaiClient:

    def test_creates_client_with_config(self):
        with mock.patch("axono.pipeline.AsyncOpenAI") as mock_cls:
            with mock.patch("axono.pipeline.config") as mock_config:
                mock_config.LLM_BASE_URL = "http://localhost:1234/v1"
                mock_config.LLM_API_KEY = "test-key"
                _get_openai_client()
                mock_cls.assert_called_once_with(
                    base_url="http://localhost:1234/v1",
                    api_key="test-key",
                )


# ---------------------------------------------------------------------------
# stream_response
# ---------------------------------------------------------------------------


def _make_chunk(reasoning_content=None, content=None):
    """Create a mock streaming chunk."""
    delta = SimpleNamespace(content=content)
    if reasoning_content is not None:
        delta.reasoning_content = reasoning_content
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _FakeStream:
    """Async iterable that yields chunks."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class TestStreamResponse:

    @pytest.mark.asyncio
    async def test_content_only_response(self):
        """stream_response returns content text for non-reasoning models."""
        from langchain_core.messages import HumanMessage, SystemMessage

        chunks = [_make_chunk(content="Hello "), _make_chunk(content="world")]
        mock_client = mock.AsyncMock()
        mock_client.chat.completions.create.return_value = _FakeStream(chunks)

        with mock.patch("axono.pipeline._get_openai_client", return_value=mock_client):
            result = await stream_response(
                [SystemMessage(content="System"), HumanMessage(content="Hi")]
            )

        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_thinking_then_content(self):
        """stream_response handles reasoning_content then content."""
        from langchain_core.messages import HumanMessage

        chunks = [
            _make_chunk(reasoning_content="Let me think..."),
            _make_chunk(reasoning_content=" about this"),
            _make_chunk(content="The answer"),
            _make_chunk(content=" is 42"),
        ]
        mock_client = mock.AsyncMock()
        mock_client.chat.completions.create.return_value = _FakeStream(chunks)

        queue = asyncio.Queue()
        set_thinking_queue(queue)

        try:
            with mock.patch(
                "axono.pipeline._get_openai_client", return_value=mock_client
            ):
                result = await stream_response([HumanMessage(content="User")])

            assert result == "The answer is 42"

            # Queue should have received thinking updates and None sentinel
            items = []
            while not queue.empty():
                items.append(queue.get_nowait())

            assert items[0] == "Let me think..."
            assert items[1] == "Let me think... about this"
            assert items[2] is None  # End of thinking signal
        finally:
            set_thinking_queue(None)

    @pytest.mark.asyncio
    async def test_thinking_only_no_content(self):
        """stream_response handles thinking with no following content."""
        from langchain_core.messages import HumanMessage

        chunks = [_make_chunk(reasoning_content="thinking...")]
        mock_client = mock.AsyncMock()
        mock_client.chat.completions.create.return_value = _FakeStream(chunks)

        queue = asyncio.Queue()
        set_thinking_queue(queue)

        try:
            with mock.patch(
                "axono.pipeline._get_openai_client", return_value=mock_client
            ):
                result = await stream_response([HumanMessage(content="Hi")])

            assert result == ""

            # Queue should have thinking update and None sentinel
            items = []
            while not queue.empty():
                items.append(queue.get_nowait())

            assert items[-1] is None
        finally:
            set_thinking_queue(None)

    @pytest.mark.asyncio
    async def test_no_thinking_queue(self):
        """stream_response works without a thinking queue set."""
        from langchain_core.messages import HumanMessage

        chunks = [
            _make_chunk(reasoning_content="think"),
            _make_chunk(content="answer"),
        ]
        mock_client = mock.AsyncMock()
        mock_client.chat.completions.create.return_value = _FakeStream(chunks)

        # Ensure no queue is set
        set_thinking_queue(None)

        with mock.patch("axono.pipeline._get_openai_client", return_value=mock_client):
            result = await stream_response([HumanMessage(content="Hi")])

        assert result == "answer"

    @pytest.mark.asyncio
    async def test_assistant_message_conversion(self):
        """stream_response converts non-System/Human messages to assistant role."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        chunks = [_make_chunk(content="ok")]
        mock_client = mock.AsyncMock()
        mock_client.chat.completions.create.return_value = _FakeStream(chunks)

        with mock.patch("axono.pipeline._get_openai_client", return_value=mock_client):
            await stream_response(
                [
                    SystemMessage(content="sys"),
                    HumanMessage(content="user"),
                    AIMessage(content="prev"),
                ]
            )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """stream_response returns empty string for no chunks."""
        from langchain_core.messages import HumanMessage

        mock_client = mock.AsyncMock()
        mock_client.chat.completions.create.return_value = _FakeStream([])

        with mock.patch("axono.pipeline._get_openai_client", return_value=mock_client):
            result = await stream_response([HumanMessage(content="Hi")])

        assert result == ""


# ---------------------------------------------------------------------------
# plan_next_action
# ---------------------------------------------------------------------------


class TestPlanNextAction:

    @pytest.mark.asyncio
    async def test_returns_parsed_json(self):
        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value='{"action": "test", "reason": "because"}',
        ):
            result = await plan_next_action("System", "User")

        assert result == {"action": "test", "reason": "because"}

    @pytest.mark.asyncio
    async def test_handles_parse_failure(self):
        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value="not valid json",
        ):
            result = await plan_next_action("System", "User")

        assert result["done"] is True
        assert "parse" in result["summary"].lower()

    @pytest.mark.asyncio
    async def test_handles_markdown_fences(self):
        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value='```json\n{"done": true, "summary": "ok"}\n```',
        ):
            result = await plan_next_action("System", "User")

        assert result["done"] is True
        assert result["summary"] == "ok"

    @pytest.mark.asyncio
    async def test_stream_response_is_called(self):
        """stream_response is called with correct messages."""
        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value='{"action": "test"}',
        ) as mock_stream:
            result = await plan_next_action("System", "User")

        mock_stream.assert_called_once()
        assert result == {"action": "test"}


# ---------------------------------------------------------------------------
# validate_plan
# ---------------------------------------------------------------------------


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

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=resp,
        ):
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

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=resp,
        ):
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

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=resp,
        ) as mock_stream:
            result = await validate_plan(
                task="Build project",
                plan_summary="Build with cargo",
                steps=[PlanStep(description="Run cargo build")],
                context="Project type: Rust",
            )

        # Verify context was included in prompt
        call_args = mock_stream.call_args
        user_msg = call_args[0][0][1].content
        assert "Rust" in user_msg
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_parse_failure_defaults_invalid(self):
        """If JSON can't be parsed, default to invalid."""
        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value="This looks like a good plan!",
        ):
            result = await validate_plan(
                task="Task",
                plan_summary="Summary",
                steps=[PlanStep(description="Step")],
            )

        assert result.valid is False
        assert len(result.issues) >= 1
        assert "not valid JSON" in result.summary

    @pytest.mark.asyncio
    async def test_handles_markdown_fences(self):
        resp = '```json\n{"valid": true, "issues": [], "suggestions": [], "summary": "OK"}\n```'

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=resp,
        ):
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

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=resp,
        ):
            result = await validate_plan(
                task="Task",
                plan_summary="Summary",
                steps=[PlanStep(description="Step")],
            )

        assert result.valid is False
        assert result.issues == []
        assert result.suggestions == []
        assert result.summary == ""


# ---------------------------------------------------------------------------
# llm_plan
# ---------------------------------------------------------------------------


class TestLlmPlan:

    @pytest.mark.asyncio
    async def test_creates_plan_with_steps(self):
        plan_json = json.dumps(
            {
                "summary": "Build project",
                "steps": [
                    {"description": "Install dependencies", "command": "npm install"},
                    {"description": "Run build", "command": "npm run build"},
                ],
            }
        )

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=plan_json,
        ):
            result = await llm_plan(
                task="Build the project",
                context="CWD: /project\nContents: package.json",
                system_prompt="You are a planner.",
            )

        assert result.summary == "Build project"
        assert len(result.steps) == 2
        assert result.steps[0].description == "Install dependencies"

    @pytest.mark.asyncio
    async def test_handles_parse_failure(self):
        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value="Not valid JSON",
        ):
            result = await llm_plan(
                task="task",
                context="context",
                system_prompt="prompt",
            )

        assert "Not valid JSON" in result.summary
        assert result.steps == []

    @pytest.mark.asyncio
    async def test_includes_files_to_read(self):
        plan_json = json.dumps(
            {
                "summary": "s",
                "steps": [],
                "files_to_read": ["main.py", "config.py"],
            }
        )

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=plan_json,
        ):
            result = await llm_plan("task", "context", "prompt")

        assert result.files_to_read == ["main.py", "config.py"]

    @pytest.mark.asyncio
    async def test_includes_patches(self):
        plan_json = json.dumps(
            {
                "summary": "s",
                "steps": [],
                "patches": [{"path": "new.py", "action": "create"}],
            }
        )

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=plan_json,
        ):
            result = await llm_plan("task", "context", "prompt")

        assert len(result.patches) == 1

    @pytest.mark.asyncio
    async def test_skips_invalid_steps(self):
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

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=plan_json,
        ):
            result = await llm_plan("task", "context", "prompt")

        assert len(result.steps) == 1
        assert result.steps[0].description == "Valid step"


# ---------------------------------------------------------------------------
# llm_validate
# ---------------------------------------------------------------------------


class TestLlmValidate:

    @pytest.mark.asyncio
    async def test_successful_validation(self):
        resp = json.dumps(
            {
                "ok": True,
                "issues": [],
                "summary": "Task completed successfully",
            }
        )

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=resp,
        ):
            result = await llm_validate(
                task="Build project",
                plan_summary="Build with npm",
                execution_summary="npm build: success",
            )

        assert result.ok is True
        assert result.summary == "Task completed successfully"

    @pytest.mark.asyncio
    async def test_failed_validation(self):
        resp = json.dumps(
            {
                "ok": False,
                "issues": ["Build failed", "Missing files"],
                "summary": "Task incomplete",
            }
        )

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=resp,
        ):
            result = await llm_validate(
                task="Build project",
                plan_summary="Build",
                execution_summary="build: failed",
            )

        assert result.ok is False
        assert len(result.issues) == 2

    @pytest.mark.asyncio
    async def test_parse_failure_defaults_to_not_ok(self):
        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value="Everything looks good!",
        ):
            result = await llm_validate(
                task="task",
                plan_summary="plan",
                execution_summary="exec",
            )

        assert result.ok is False
        assert len(result.issues) >= 1
        assert "not valid JSON" in result.summary

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self):
        resp = json.dumps({"ok": True, "issues": [], "summary": "OK"})
        custom_prompt = "You are a custom validator."

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=resp,
        ) as mock_stream:
            await llm_validate(
                task="task",
                plan_summary="plan",
                execution_summary="exec",
                system_prompt=custom_prompt,
            )

        call_args = mock_stream.call_args
        system_msg = call_args[0][0][0].content
        assert system_msg == custom_prompt


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:

    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, tmp_path):
        """Test full pipeline with all stages."""
        config = PipelineConfig()

        # Mock investigate function
        def investigate_fn(cwd):
            return Investigation(
                cwd=cwd,
                dir_listing="file1.py, file2.py",
                project_type="Python",
                summary="Found Python project",
            )

        # Mock plan function
        async def plan_fn(task, investigation, issues):
            return Plan(
                summary="Run tests",
                steps=[
                    PlanStep(description="Run pytest", action={"command": "pytest"})
                ],
            )

        # Mock execute function
        async def execute_fn(plan, cwd):
            step = plan.steps[0]
            return (
                ExecutionResult(
                    success=True,
                    step_results=[
                        StepExecution(step=step, success=True, output="3 tests passed")
                    ],
                    summary="Done",
                ),
                cwd,
            )

        # Mock validate function
        async def validate_fn(task, plan, execution):
            return FinalValidation(ok=True, summary="All tests passed")

        # Mock validate_plan
        with mock.patch("axono.pipeline.validate_plan") as mock_validate_plan:
            mock_validate_plan.return_value = PlanValidation(
                valid=True, summary="Plan OK"
            )

            events = []
            async for ev in run_pipeline(
                task="Run tests",
                working_dir=str(tmp_path),
                config=config,
                investigate_fn=investigate_fn,
                plan_fn=plan_fn,
                execute_fn=execute_fn,
                validate_fn=validate_fn,
            ):
                events.append(ev)

        types = [e[0] for e in events]
        assert "status" in types
        assert "result" in types
        assert "cwd" in types

        # Check for project type detection
        assert any("Python" in e[1] for e in events if e[0] == "status")

        # Check for result
        result_events = [e for e in events if e[0] == "result"]
        assert len(result_events) == 1
        assert "passed" in result_events[0][1]

    @pytest.mark.asyncio
    async def test_empty_plan_yields_error(self, tmp_path):
        """Pipeline yields error when plan has no steps."""
        config = PipelineConfig()

        def investigate_fn(cwd):
            return Investigation(cwd=cwd, dir_listing="")

        async def plan_fn(task, investigation, issues):
            return Plan(summary="Empty plan", steps=[])

        events = []
        async for ev in run_pipeline(
            task="task",
            working_dir=str(tmp_path),
            config=config,
            investigate_fn=investigate_fn,
            plan_fn=plan_fn,
            execute_fn=mock.AsyncMock(),
        ):
            events.append(ev)

        types = [e[0] for e in events]
        assert "error" in types
        error_msg = [e[1] for e in events if e[0] == "error"][0]
        assert "No steps" in error_msg

    @pytest.mark.asyncio
    async def test_planning_exception(self, tmp_path):
        """Pipeline yields error when planning fails."""
        config = PipelineConfig()

        def investigate_fn(cwd):
            return Investigation(cwd=cwd, dir_listing="")

        async def plan_fn(task, investigation, issues):
            raise RuntimeError("LLM unavailable")

        events = []
        async for ev in run_pipeline(
            task="task",
            working_dir=str(tmp_path),
            config=config,
            investigate_fn=investigate_fn,
            plan_fn=plan_fn,
            execute_fn=mock.AsyncMock(),
        ):
            events.append(ev)

        types = [e[0] for e in events]
        assert "error" in types
        assert "cwd" in types

    @pytest.mark.asyncio
    async def test_plan_validation_loop(self, tmp_path):
        """Test that plan validation can iterate."""
        config = PipelineConfig(max_plan_iterations=3)

        def investigate_fn(cwd):
            return Investigation(cwd=cwd, dir_listing="")

        plan_call_count = 0

        async def plan_fn(task, investigation, issues):
            nonlocal plan_call_count
            plan_call_count += 1
            return Plan(
                summary=f"Plan {plan_call_count}",
                steps=[PlanStep(description="Step 1")],
            )

        async def execute_fn(plan, cwd):
            step = plan.steps[0]
            return (
                ExecutionResult(
                    success=True,
                    step_results=[StepExecution(step=step, success=True)],
                ),
                cwd,
            )

        validation_call_count = 0

        async def mock_validate(*args, **kwargs):
            nonlocal validation_call_count
            validation_call_count += 1
            if validation_call_count == 1:
                return PlanValidation(
                    valid=False, issues=["Missing step"], summary="Bad"
                )
            return PlanValidation(valid=True, summary="OK")

        with mock.patch("axono.pipeline.validate_plan", side_effect=mock_validate):
            events = []
            async for ev in run_pipeline(
                task="task",
                working_dir=str(tmp_path),
                config=config,
                investigate_fn=investigate_fn,
                plan_fn=plan_fn,
                execute_fn=execute_fn,
            ):
                events.append(ev)

        # Should have called plan twice (first failed validation, second passed)
        assert plan_call_count == 2

        # Should have status about issues
        status_msgs = [e[1] for e in events if e[0] == "status"]
        assert any("issues" in msg.lower() for msg in status_msgs)

    @pytest.mark.asyncio
    async def test_max_plan_iterations(self, tmp_path):
        """Pipeline proceeds after max iterations even if validation fails."""
        config = PipelineConfig(max_plan_iterations=2)

        def investigate_fn(cwd):
            return Investigation(cwd=cwd, dir_listing="")

        async def plan_fn(task, investigation, issues):
            return Plan(summary="Plan", steps=[PlanStep(description="Step")])

        async def execute_fn(plan, cwd):
            step = plan.steps[0]
            return (
                ExecutionResult(
                    success=True,
                    step_results=[StepExecution(step=step, success=True)],
                ),
                cwd,
            )

        # Always return invalid
        with mock.patch("axono.pipeline.validate_plan") as mock_validate:
            mock_validate.return_value = PlanValidation(
                valid=False, issues=["Bad"], summary="Bad"
            )

            events = []
            async for ev in run_pipeline(
                task="task",
                working_dir=str(tmp_path),
                config=config,
                investigate_fn=investigate_fn,
                plan_fn=plan_fn,
                execute_fn=execute_fn,
            ):
                events.append(ev)

        # Should have status about proceeding after attempts
        status_msgs = [e[1] for e in events if e[0] == "status"]
        assert any("2 plan attempts" in msg for msg in status_msgs)

    @pytest.mark.asyncio
    async def test_validation_exception_proceeds(self, tmp_path):
        """Pipeline proceeds if plan validation raises exception."""
        config = PipelineConfig()

        def investigate_fn(cwd):
            return Investigation(cwd=cwd, dir_listing="")

        async def plan_fn(task, investigation, issues):
            return Plan(summary="Plan", steps=[PlanStep(description="Step")])

        async def execute_fn(plan, cwd):
            step = plan.steps[0]
            return (
                ExecutionResult(
                    success=True,
                    step_results=[StepExecution(step=step, success=True)],
                ),
                cwd,
            )

        with mock.patch(
            "axono.pipeline.validate_plan", side_effect=RuntimeError("LLM down")
        ):
            events = []
            async for ev in run_pipeline(
                task="task",
                working_dir=str(tmp_path),
                config=config,
                investigate_fn=investigate_fn,
                plan_fn=plan_fn,
                execute_fn=execute_fn,
            ):
                events.append(ev)

        # Should complete despite validation error
        types = [e[0] for e in events]
        assert "result" in types

    @pytest.mark.asyncio
    async def test_execution_exception(self, tmp_path):
        """Pipeline yields error when execution fails."""
        config = PipelineConfig()

        def investigate_fn(cwd):
            return Investigation(cwd=cwd, dir_listing="")

        async def plan_fn(task, investigation, issues):
            return Plan(summary="Plan", steps=[PlanStep(description="Step")])

        async def execute_fn(plan, cwd):
            raise RuntimeError("Execution crashed")

        with mock.patch("axono.pipeline.validate_plan") as mock_validate:
            mock_validate.return_value = PlanValidation(valid=True, summary="OK")

            events = []
            async for ev in run_pipeline(
                task="task",
                working_dir=str(tmp_path),
                config=config,
                investigate_fn=investigate_fn,
                plan_fn=plan_fn,
                execute_fn=execute_fn,
            ):
                events.append(ev)

        types = [e[0] for e in events]
        assert "error" in types
        assert "cwd" in types

        error_msg = [e[1] for e in events if e[0] == "error"][0]
        assert "Execution failed" in error_msg

    @pytest.mark.asyncio
    async def test_final_validation_exception(self, tmp_path):
        """Pipeline reports completion even if final validation fails."""
        config = PipelineConfig()

        def investigate_fn(cwd):
            return Investigation(cwd=cwd, dir_listing="")

        async def plan_fn(task, investigation, issues):
            return Plan(summary="Plan", steps=[PlanStep(description="Step")])

        async def execute_fn(plan, cwd):
            step = plan.steps[0]
            return (
                ExecutionResult(
                    success=True,
                    step_results=[StepExecution(step=step, success=True)],
                ),
                cwd,
            )

        async def validate_fn(task, plan, execution):
            raise RuntimeError("Validation crashed")

        with mock.patch("axono.pipeline.validate_plan") as mock_validate:
            mock_validate.return_value = PlanValidation(valid=True, summary="OK")

            events = []
            async for ev in run_pipeline(
                task="task",
                working_dir=str(tmp_path),
                config=config,
                investigate_fn=investigate_fn,
                plan_fn=plan_fn,
                execute_fn=execute_fn,
                validate_fn=validate_fn,
            ):
                events.append(ev)

        # Should still have result
        result_events = [e for e in events if e[0] == "result"]
        assert len(result_events) == 1
        assert "validation error" in result_events[0][1]

    @pytest.mark.asyncio
    async def test_no_validate_fn(self, tmp_path):
        """Pipeline works without validate_fn."""
        config = PipelineConfig()

        def investigate_fn(cwd):
            return Investigation(cwd=cwd, dir_listing="")

        async def plan_fn(task, investigation, issues):
            return Plan(summary="Plan", steps=[PlanStep(description="Step")])

        async def execute_fn(plan, cwd):
            step = plan.steps[0]
            return (
                ExecutionResult(
                    success=True,
                    step_results=[StepExecution(step=step, success=True)],
                ),
                cwd,
            )

        with mock.patch("axono.pipeline.validate_plan") as mock_validate:
            mock_validate.return_value = PlanValidation(valid=True, summary="OK")

            events = []
            async for ev in run_pipeline(
                task="task",
                working_dir=str(tmp_path),
                config=config,
                investigate_fn=investigate_fn,
                plan_fn=plan_fn,
                execute_fn=execute_fn,
                validate_fn=None,
            ):
                events.append(ev)

        result_events = [e for e in events if e[0] == "result"]
        assert len(result_events) == 1
        assert result_events[0][1] == "Task completed"

    @pytest.mark.asyncio
    async def test_step_output_yielded(self, tmp_path):
        """Step output is yielded."""
        config = PipelineConfig()

        def investigate_fn(cwd):
            return Investigation(cwd=cwd, dir_listing="")

        async def plan_fn(task, investigation, issues):
            return Plan(summary="Plan", steps=[PlanStep(description="Step")])

        async def execute_fn(plan, cwd):
            step = plan.steps[0]
            return (
                ExecutionResult(
                    success=True,
                    step_results=[
                        StepExecution(step=step, success=True, output="Step output")
                    ],
                ),
                cwd,
            )

        with mock.patch("axono.pipeline.validate_plan") as mock_validate:
            mock_validate.return_value = PlanValidation(valid=True, summary="OK")

            events = []
            async for ev in run_pipeline(
                task="task",
                working_dir=str(tmp_path),
                config=config,
                investigate_fn=investigate_fn,
                plan_fn=plan_fn,
                execute_fn=execute_fn,
            ):
                events.append(ev)

        output_events = [e for e in events if e[0] == "output"]
        assert len(output_events) == 1
        assert output_events[0][1] == "Step output"

    @pytest.mark.asyncio
    async def test_step_error_yielded(self, tmp_path):
        """Step errors are yielded."""
        config = PipelineConfig()

        def investigate_fn(cwd):
            return Investigation(cwd=cwd, dir_listing="")

        async def plan_fn(task, investigation, issues):
            return Plan(summary="Plan", steps=[PlanStep(description="Step")])

        async def execute_fn(plan, cwd):
            step = plan.steps[0]
            return (
                ExecutionResult(
                    success=False,
                    step_results=[
                        StepExecution(step=step, success=False, error="Step failed")
                    ],
                ),
                cwd,
            )

        with mock.patch("axono.pipeline.validate_plan") as mock_validate:
            mock_validate.return_value = PlanValidation(valid=True, summary="OK")

            events = []
            async for ev in run_pipeline(
                task="task",
                working_dir=str(tmp_path),
                config=config,
                investigate_fn=investigate_fn,
                plan_fn=plan_fn,
                execute_fn=execute_fn,
            ):
                events.append(ev)

        error_events = [e for e in events if e[0] == "error"]
        assert any("Step failed" in e[1] for e in error_events)

    @pytest.mark.asyncio
    async def test_validation_issues_in_result(self, tmp_path):
        """Validation issues appear in final result."""
        config = PipelineConfig()

        def investigate_fn(cwd):
            return Investigation(cwd=cwd, dir_listing="")

        async def plan_fn(task, investigation, issues):
            return Plan(summary="Plan", steps=[PlanStep(description="Step")])

        async def execute_fn(plan, cwd):
            step = plan.steps[0]
            return (
                ExecutionResult(
                    success=True,
                    step_results=[StepExecution(step=step, success=True)],
                ),
                cwd,
            )

        async def validate_fn(task, plan, execution):
            return FinalValidation(
                ok=False, issues=["Issue 1", "Issue 2"], summary="Problems found"
            )

        with mock.patch("axono.pipeline.validate_plan") as mock_validate:
            mock_validate.return_value = PlanValidation(valid=True, summary="OK")

            events = []
            async for ev in run_pipeline(
                task="task",
                working_dir=str(tmp_path),
                config=config,
                investigate_fn=investigate_fn,
                plan_fn=plan_fn,
                execute_fn=execute_fn,
                validate_fn=validate_fn,
            ):
                events.append(ev)

        result_events = [e for e in events if e[0] == "result"]
        assert len(result_events) == 1
        assert "Problems found" in result_events[0][1]
        assert "Issue 1" in result_events[0][1]
