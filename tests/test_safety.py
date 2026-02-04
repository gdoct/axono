"""Unit tests for axono.safety."""

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from axono import safety

# ---------------------------------------------------------------------------
# _parse_agent_data
# ---------------------------------------------------------------------------


class TestParseAgentData:
    """Covers all branches of the content normaliser."""

    def test_none_returns_empty_string(self):
        assert safety._parse_agent_data(None) == ""

    def test_list_returns_json(self):
        result = safety._parse_agent_data([{"text": "hi"}])
        assert result == json.dumps([{"text": "hi"}])

    def test_string_returned_as_is(self):
        assert safety._parse_agent_data("hello") == "hello"

    def test_integer_coerced_to_string(self):
        assert safety._parse_agent_data(42) == "42"

    def test_empty_list(self):
        assert safety._parse_agent_data([]) == "[]"


# ---------------------------------------------------------------------------
# _get_judge_llm — lazy singleton
# ---------------------------------------------------------------------------


class TestGetJudgeLlm:
    """The LLM is created lazily and then cached."""

    def test_creates_llm_on_first_call(self):
        safety._judge_llm = None
        sentinel = object()
        with mock.patch("axono.safety.init_chat_model", return_value=sentinel) as m:
            result = safety._get_judge_llm()
        assert result is sentinel
        m.assert_called_once()

    def test_returns_cached_on_second_call(self):
        sentinel = object()
        safety._judge_llm = sentinel
        try:
            with mock.patch("axono.safety.init_chat_model") as m:
                result = safety._get_judge_llm()
            assert result is sentinel
            m.assert_not_called()
        finally:
            safety._judge_llm = None

    def test_passes_config_values(self):
        safety._judge_llm = None
        with mock.patch("axono.safety.init_chat_model", return_value=object()) as m:
            with mock.patch.multiple(
                "axono.safety.config",
                LLM_MODEL_NAME="test-model",
                LLM_MODEL_PROVIDER="test-provider",
                LLM_BASE_URL="http://test:1234",
                LLM_API_KEY="test-key",
            ):
                safety._get_judge_llm()
        m.assert_called_once_with(
            model="test-model",
            model_provider="test-provider",
            base_url="http://test:1234",
            api_key="test-key",
        )
        safety._judge_llm = None


# ---------------------------------------------------------------------------
# judge_command — various LLM responses
# ---------------------------------------------------------------------------


def _fake_llm(content):
    """Return a mock LLM whose ``ainvoke`` resolves to *content*."""
    response = SimpleNamespace(content=content)
    llm = mock.AsyncMock()
    llm.ainvoke.return_value = response
    return llm


class TestJudgeCommand:
    """judge_command parses the LLM response and handles edge cases."""

    @pytest.mark.asyncio
    async def test_safe_command(self):
        llm = _fake_llm('{"dangerous": false, "reason": "read-only"}')
        with mock.patch("axono.safety._get_judge_llm", return_value=llm):
            result = await safety.judge_command("ls -la")
        assert result["dangerous"] is False
        assert result["reason"] == "read-only"

    @pytest.mark.asyncio
    async def test_dangerous_command(self):
        llm = _fake_llm('{"dangerous": true, "reason": "deletes files"}')
        with mock.patch("axono.safety._get_judge_llm", return_value=llm):
            result = await safety.judge_command("rm -rf /")
        assert result["dangerous"] is True
        assert "deletes files" in result["reason"]

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        llm = _fake_llm('```json\n{"dangerous": true, "reason": "bad"}\n```')
        with mock.patch("axono.safety._get_judge_llm", return_value=llm):
            result = await safety.judge_command("dd if=/dev/zero of=/dev/sda")
        assert result["dangerous"] is True

    @pytest.mark.asyncio
    async def test_strips_bare_backtick_fences(self):
        llm = _fake_llm('```\n{"dangerous": false, "reason": "ok"}\n```')
        with mock.patch("axono.safety._get_judge_llm", return_value=llm):
            result = await safety.judge_command("echo hi")
        assert result["dangerous"] is False

    @pytest.mark.asyncio
    async def test_fence_without_newline(self):
        llm = _fake_llm('```{"dangerous": false, "reason": "ok"}```')
        with mock.patch("axono.safety._get_judge_llm", return_value=llm):
            result = await safety.judge_command("echo hi")
        assert result["dangerous"] is False

    @pytest.mark.asyncio
    async def test_unparseable_response_falls_through(self):
        llm = _fake_llm("I think this command is fine!")
        with mock.patch("axono.safety._get_judge_llm", return_value=llm):
            result = await safety.judge_command("echo hello")
        assert result["dangerous"] is False
        assert "unparseable" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_empty_response_falls_through(self):
        llm = _fake_llm("")
        with mock.patch("axono.safety._get_judge_llm", return_value=llm):
            result = await safety.judge_command("echo hello")
        assert result["dangerous"] is False

    @pytest.mark.asyncio
    async def test_none_content_falls_through(self):
        llm = _fake_llm(None)
        with mock.patch("axono.safety._get_judge_llm", return_value=llm):
            result = await safety.judge_command("echo hello")
        assert result["dangerous"] is False

    @pytest.mark.asyncio
    async def test_list_content_returns_parsed_json(self):
        """When content is a list, _parse_agent_data json.dumps it.

        json.loads then parses that back into a Python list — *not* a dict.
        The caller (agent.py) may get unexpected types here, but
        judge_command itself just returns whatever json.loads produces.
        """
        llm = _fake_llm([{"dangerous": True, "reason": "rm"}])
        with mock.patch("axono.safety._get_judge_llm", return_value=llm):
            result = await safety.judge_command("rm -rf /")
        # json.dumps([...]) -> "[...]" -> json.loads -> list
        assert isinstance(result, list)
        assert result[0]["dangerous"] is True

    @pytest.mark.asyncio
    async def test_messages_sent_to_llm(self):
        llm = _fake_llm('{"dangerous": false, "reason": "ok"}')
        with mock.patch("axono.safety._get_judge_llm", return_value=llm):
            await safety.judge_command("whoami")
        args = llm.ainvoke.call_args[0][0]
        assert len(args) == 2
        # First message is the system prompt
        assert "security analyst" in args[0].content
        # Second message contains the command
        assert "whoami" in args[1].content

    @pytest.mark.asyncio
    async def test_whitespace_around_json(self):
        llm = _fake_llm('  \n {"dangerous": false, "reason": "safe"} \n ')
        with mock.patch("axono.safety._get_judge_llm", return_value=llm):
            result = await safety.judge_command("pwd")
        assert result["dangerous"] is False
        assert result["reason"] == "safe"
