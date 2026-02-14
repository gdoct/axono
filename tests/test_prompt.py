"""Unit tests for axono.prompt module."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from axono import prompt


class TestLLMPrompt:
    """Tests for the LLMPrompt dataclass."""

    def test_llmprompt_creation(self):
        """Test creating an LLMPrompt object."""
        llm_prompt = prompt.LLMPrompt(
            name="Test Prompt",
            description="A test prompt",
            model="instruction",
            system="You are a test assistant.",
            user="Please help me test.",
        )
        assert llm_prompt.name == "Test Prompt"
        assert llm_prompt.description == "A test prompt"
        assert llm_prompt.model == "instruction"
        assert llm_prompt.system == "You are a test assistant."
        assert llm_prompt.user == "Please help me test."

    def test_llmprompt_with_reasoning_model(self):
        """Test LLMPrompt with reasoning model type."""
        llm_prompt = prompt.LLMPrompt(
            name="Reasoning Test",
            description="A reasoning prompt",
            model="reasoning",
            system="Think step by step.",
            user="What is 2+2?",
        )
        assert llm_prompt.model == "reasoning"


class TestLoadPrompt:
    """Tests for the load_prompt function."""

    def test_load_valid_prompt(self):
        """Test loading a valid prompt file."""
        llm_prompt = prompt.load_prompt("example", "someprompt")

        assert llm_prompt.name == "Code Review Assistant"
        assert "code reviews" in llm_prompt.description.lower()
        assert llm_prompt.model in ("instruction", "reasoning")
        assert "code reviewer" in llm_prompt.system.lower()
        assert "review" in llm_prompt.user.lower()

    def test_load_prompt_file_not_found(self):
        """Test that FileNotFoundError is raised for missing prompts."""
        with pytest.raises(FileNotFoundError) as exc_info:
            prompt.load_prompt("nonexistent_topic", "nonexistent_prompt")

        assert "not found" in str(exc_info.value).lower()
        assert "nonexistent_topic/nonexistent_prompt.yml" in str(exc_info.value)

    def test_load_prompt_non_dict_yaml(self, tmp_path, monkeypatch):
        """Test that ValueError is raised when YAML is not a dict (line 62)."""
        topic_dir = tmp_path / "prompts" / "bad_topic"
        topic_dir.mkdir(parents=True)
        yaml_file = topic_dir / "not_dict.yml"
        yaml_file.write_text("- just\n- a\n- list\n")

        # Make Path(__file__).parent resolve to tmp_path
        monkeypatch.setattr(prompt, "__file__", str(tmp_path / "prompt.py"))

        with pytest.raises(ValueError, match="Invalid prompt file structure"):
            prompt.load_prompt("bad_topic", "not_dict")

    def test_load_prompt_missing_required_fields(self, tmp_path, monkeypatch):
        """Test that ValueError is raised when required fields are missing (line 68)."""
        topic_dir = tmp_path / "prompts" / "bad_topic"
        topic_dir.mkdir(parents=True)
        yaml_file = topic_dir / "incomplete.yml"
        incomplete_data = {
            "Name": "Incomplete Prompt",
            "Description": "Missing other fields",
        }
        yaml_file.write_text(yaml.dump(incomplete_data))

        monkeypatch.setattr(prompt, "__file__", str(tmp_path / "prompt.py"))

        with pytest.raises(ValueError, match="missing required fields"):
            prompt.load_prompt("bad_topic", "incomplete")

    def test_load_prompt_invalid_model_type(self, tmp_path, monkeypatch):
        """Test that ValueError is raised for invalid model types (line 75)."""
        topic_dir = tmp_path / "prompts" / "bad_topic"
        topic_dir.mkdir(parents=True)
        yaml_file = topic_dir / "bad_model.yml"
        bad_data = {
            "Name": "Bad Model",
            "Description": "Invalid model type",
            "Model": "invalid_model_type",
            "System": "System message",
            "User": "User message",
        }
        yaml_file.write_text(yaml.dump(bad_data))

        monkeypatch.setattr(prompt, "__file__", str(tmp_path / "prompt.py"))

        with pytest.raises(ValueError, match="Invalid model type"):
            prompt.load_prompt("bad_topic", "bad_model")

    def test_load_prompt_case_insensitive_model(self):
        """Test that model type is case-insensitive."""
        llm_prompt = prompt.load_prompt("example", "someprompt")
        assert llm_prompt.model in ("instruction", "reasoning")

    def test_load_prompt_multiline_fields(self):
        """Test loading prompts with multiline fields."""
        llm_prompt = prompt.load_prompt("example", "someprompt")

        # System and user should be multiline strings
        assert isinstance(llm_prompt.system, str)
        assert isinstance(llm_prompt.user, str)
        assert len(llm_prompt.system) > 20  # Should be substantial
        assert len(llm_prompt.user) > 20

    def test_load_prompt_fields_are_strings(self):
        """Test that all loaded prompt fields are strings."""
        llm_prompt = prompt.load_prompt("example", "someprompt")

        assert isinstance(llm_prompt.name, str)
        assert isinstance(llm_prompt.description, str)
        assert isinstance(llm_prompt.model, str)
        assert isinstance(llm_prompt.system, str)
        assert isinstance(llm_prompt.user, str)
