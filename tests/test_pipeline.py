"""Unit tests for axono.pipeline."""

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from axono.pipeline import (
    ActionResult,
    PipelineContext,
    coerce_response_text,
    parse_json,
    plan_next_action,
    truncate,
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
        response = SimpleNamespace(content='```json\n{"done": true, "summary": "ok"}\n```')
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        result = await plan_next_action("System", "User", llm=fake_llm)
        assert result["done"] is True
        assert result["summary"] == "ok"
