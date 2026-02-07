"""Unit tests for axono.intent."""

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from axono.intent import (
    INTENT_ANALYZER_SYSTEM,
    Intent,
    analyze_intent,
)

# ---------------------------------------------------------------------------
# Intent dataclass
# ---------------------------------------------------------------------------


class TestIntent:

    def test_chat_intent_defaults(self):
        intent = Intent(type="chat")
        assert intent.type == "chat"
        assert intent.task_list == []
        assert intent.reasoning == ""

    def test_task_intent_with_tasks(self):
        intent = Intent(
            type="task",
            task_list=["Create directory", "Initialize project"],
            reasoning="User wants to create something",
        )
        assert intent.type == "task"
        assert len(intent.task_list) == 2
        assert intent.task_list[0] == "Create directory"

    def test_intent_type_literal(self):
        # Both valid types
        chat = Intent(type="chat")
        task = Intent(type="task")
        assert chat.type == "chat"
        assert task.type == "task"


# ---------------------------------------------------------------------------
# analyze_intent
# ---------------------------------------------------------------------------


class TestAnalyzeIntent:

    @pytest.mark.asyncio
    async def test_chat_intent_greeting(self):
        """Test that greetings are classified as chat."""
        mock_response = SimpleNamespace(
            content=json.dumps(
                {
                    "type": "chat",
                    "task_list": [],
                    "reasoning": "Greeting",
                }
            )
        )

        with mock.patch("axono.intent.get_llm") as mock_get_llm:
            mock_llm = mock.AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            intent = await analyze_intent("Hello, how are you?", "/home/user")

            assert intent.type == "chat"
            assert intent.task_list == []
            assert intent.reasoning == "Greeting"

    @pytest.mark.asyncio
    async def test_chat_intent_question(self):
        """Test that questions are classified as chat."""
        mock_response = SimpleNamespace(
            content=json.dumps(
                {
                    "type": "chat",
                    "task_list": [],
                    "reasoning": "Asking for explanation",
                }
            )
        )

        with mock.patch("axono.intent.get_llm") as mock_get_llm:
            mock_llm = mock.AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            intent = await analyze_intent("What is a Python decorator?", "/tmp")

            assert intent.type == "chat"
            assert intent.task_list == []

    @pytest.mark.asyncio
    async def test_task_intent_with_tasks(self):
        """Test that actionable requests produce task lists."""
        mock_response = SimpleNamespace(
            content=json.dumps(
                {
                    "type": "task",
                    "task_list": [
                        "Create project directory",
                        "Initialize git repository",
                        "Create pyproject.toml",
                    ],
                    "reasoning": "Request to create project",
                }
            )
        )

        with mock.patch("axono.intent.get_llm") as mock_get_llm:
            mock_llm = mock.AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            intent = await analyze_intent("Create a new Python project", "/projects")

            assert intent.type == "task"
            assert len(intent.task_list) == 3
            assert "pyproject.toml" in intent.task_list[2]

    @pytest.mark.asyncio
    async def test_invalid_json_defaults_to_chat(self):
        """Test that invalid JSON response defaults to chat."""
        mock_response = SimpleNamespace(content="This is not valid JSON")

        with mock.patch("axono.intent.get_llm") as mock_get_llm:
            mock_llm = mock.AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            intent = await analyze_intent("Do something", "/tmp")

            assert intent.type == "chat"
            assert "Could not parse" in intent.reasoning

    @pytest.mark.asyncio
    async def test_invalid_intent_type_defaults_to_chat(self):
        """Test that invalid intent type defaults to chat."""
        mock_response = SimpleNamespace(
            content=json.dumps(
                {
                    "type": "invalid_type",
                    "task_list": [],
                    "reasoning": "Something",
                }
            )
        )

        with mock.patch("axono.intent.get_llm") as mock_get_llm:
            mock_llm = mock.AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            intent = await analyze_intent("Something", "/tmp")

            assert intent.type == "chat"

    @pytest.mark.asyncio
    async def test_task_list_not_a_list_defaults_to_empty(self):
        """Test that non-list task_list is converted to empty list."""
        mock_response = SimpleNamespace(
            content=json.dumps(
                {
                    "type": "task",
                    "task_list": "not a list",
                    "reasoning": "Something",
                }
            )
        )

        with mock.patch("axono.intent.get_llm") as mock_get_llm:
            mock_llm = mock.AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            intent = await analyze_intent("Do task", "/tmp")

            assert intent.type == "task"
            assert intent.task_list == []

    @pytest.mark.asyncio
    async def test_task_list_filters_empty_items(self):
        """Test that empty items are filtered from task list."""
        mock_response = SimpleNamespace(
            content=json.dumps(
                {
                    "type": "task",
                    "task_list": ["Task 1", "", None, "Task 2"],
                    "reasoning": "Tasks",
                }
            )
        )

        with mock.patch("axono.intent.get_llm") as mock_get_llm:
            mock_llm = mock.AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            intent = await analyze_intent("Do tasks", "/tmp")

            assert intent.task_list == ["Task 1", "Task 2"]

    @pytest.mark.asyncio
    async def test_task_list_converts_to_strings(self):
        """Test that task list items are converted to strings."""
        mock_response = SimpleNamespace(
            content=json.dumps(
                {
                    "type": "task",
                    "task_list": [123, "Task", {"nested": "obj"}],
                    "reasoning": "Tasks",
                }
            )
        )

        with mock.patch("axono.intent.get_llm") as mock_get_llm:
            mock_llm = mock.AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            intent = await analyze_intent("Do tasks", "/tmp")

            assert all(isinstance(t, str) for t in intent.task_list)
            assert "123" in intent.task_list

    @pytest.mark.asyncio
    async def test_response_with_markdown_fences(self):
        """Test that markdown-fenced JSON is parsed correctly."""
        mock_response = SimpleNamespace(
            content='```json\n{"type": "chat", "task_list": [], "reasoning": "Test"}\n```'
        )

        with mock.patch("axono.intent.get_llm") as mock_get_llm:
            mock_llm = mock.AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            intent = await analyze_intent("Hello", "/tmp")

            assert intent.type == "chat"
            assert intent.reasoning == "Test"

    @pytest.mark.asyncio
    async def test_missing_reasoning_defaults_to_empty(self):
        """Test that missing reasoning defaults to empty string."""
        mock_response = SimpleNamespace(
            content=json.dumps(
                {
                    "type": "chat",
                    "task_list": [],
                }
            )
        )

        with mock.patch("axono.intent.get_llm") as mock_get_llm:
            mock_llm = mock.AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            intent = await analyze_intent("Hi", "/tmp")

            assert intent.reasoning == ""

    @pytest.mark.asyncio
    async def test_llm_receives_cwd_context(self):
        """Test that the LLM receives the current working directory."""
        mock_response = SimpleNamespace(
            content=json.dumps(
                {
                    "type": "chat",
                    "task_list": [],
                    "reasoning": "Test",
                }
            )
        )

        with mock.patch("axono.intent.get_llm") as mock_get_llm:
            mock_llm = mock.AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            await analyze_intent("Hello", "/my/custom/path")

            # Check that ainvoke was called with messages containing the cwd
            call_args = mock_llm.ainvoke.call_args
            messages = call_args[0][0]
            user_message = messages[1].content
            assert "/my/custom/path" in user_message

    @pytest.mark.asyncio
    async def test_none_content_handled(self):
        """Test that None content is handled gracefully."""
        mock_response = SimpleNamespace(content=None)

        with mock.patch("axono.intent.get_llm") as mock_get_llm:
            mock_llm = mock.AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            intent = await analyze_intent("Hello", "/tmp")

            assert intent.type == "chat"

    @pytest.mark.asyncio
    async def test_list_content_handled(self):
        """Test that list content (from some LLMs) is handled."""
        # Some LLMs return content as a list
        mock_response = SimpleNamespace(
            content=[
                {
                    "type": "text",
                    "text": '{"type": "chat", "task_list": [], "reasoning": "Test"}',
                }
            ]
        )

        with mock.patch("axono.intent.get_llm") as mock_get_llm:
            mock_llm = mock.AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            intent = await analyze_intent("Hello", "/tmp")

            # coerce_response_text converts list to JSON string, which won't parse as intent
            # so it should default to chat
            assert intent.type == "chat"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:

    def test_system_prompt_contains_examples(self):
        """Test that the system prompt has examples for both types."""
        assert '"type": "chat"' in INTENT_ANALYZER_SYSTEM
        assert '"type": "task"' in INTENT_ANALYZER_SYSTEM

    def test_system_prompt_explains_intent_types(self):
        """Test that intent types are explained."""
        assert "chat" in INTENT_ANALYZER_SYSTEM.lower()
        assert "task" in INTENT_ANALYZER_SYSTEM.lower()
        assert "greeting" in INTENT_ANALYZER_SYSTEM.lower()
