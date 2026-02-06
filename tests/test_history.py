"""Unit tests for axono.history."""

import os
from pathlib import Path
from unittest import mock

import pytest


class TestHistory:
    """Tests for the history module functions."""

    @pytest.fixture
    def temp_home(self, tmp_path):
        """Create a temporary HOME directory for testing."""
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            # Reimport history module with new HOME
            import importlib
            import sys

            if "axono.history" in sys.modules:
                del sys.modules["axono.history"]

            import axono.history as hist

            # Patch the module constants for this test
            hist.HISTORY_FILE = tmp_path / ".axono" / "history"
            yield hist, tmp_path

    def test_load_history_empty_when_no_file(self, temp_home):
        """load_history returns empty list when no history file exists."""
        hist, _ = temp_home
        assert hist.load_history() == []

    def test_save_and_load_history(self, temp_home):
        """History can be saved and loaded."""
        hist, _ = temp_home
        prompts = ["first prompt", "second prompt", "third prompt"]
        hist.save_history(prompts)
        assert hist.load_history() == prompts

    def test_save_history_creates_directory(self, temp_home):
        """save_history creates ~/.axono directory if needed."""
        hist, tmp_path = temp_home
        axono_dir = tmp_path / ".axono"
        assert not axono_dir.exists()

        hist.save_history(["test"])
        assert axono_dir.exists()
        assert (axono_dir / "history").is_file()

    def test_save_history_trims_to_max(self, temp_home):
        """save_history keeps only the last MAX_HISTORY entries."""
        hist, _ = temp_home
        prompts = [f"prompt {i}" for i in range(50)]
        hist.save_history(prompts)

        loaded = hist.load_history()
        assert len(loaded) == hist.MAX_HISTORY
        assert loaded == prompts[-hist.MAX_HISTORY:]

    def test_load_history_trims_to_max(self, temp_home):
        """load_history returns at most MAX_HISTORY entries."""
        hist, tmp_path = temp_home
        # Manually write more than MAX_HISTORY lines
        axono_dir = tmp_path / ".axono"
        axono_dir.mkdir(parents=True)
        prompts = [f"prompt {i}" for i in range(50)]
        (axono_dir / "history").write_text("\n".join(prompts) + "\n")

        loaded = hist.load_history()
        assert len(loaded) == hist.MAX_HISTORY
        assert loaded == prompts[-hist.MAX_HISTORY:]

    def test_append_to_history(self, temp_home):
        """append_to_history adds a prompt and saves."""
        hist, _ = temp_home
        hist.append_to_history("first")
        hist.append_to_history("second")

        assert hist.load_history() == ["first", "second"]

    def test_append_to_history_skips_empty(self, temp_home):
        """append_to_history ignores empty prompts."""
        hist, _ = temp_home
        hist.append_to_history("first")
        hist.append_to_history("")
        hist.append_to_history("   ")

        assert hist.load_history() == ["first"]

    def test_append_to_history_skips_duplicate_last(self, temp_home):
        """append_to_history doesn't add duplicate of last entry."""
        hist, _ = temp_home
        hist.append_to_history("first")
        hist.append_to_history("first")
        hist.append_to_history("second")
        hist.append_to_history("second")

        assert hist.load_history() == ["first", "second"]

    def test_append_to_history_allows_non_consecutive_duplicates(self, temp_home):
        """append_to_history allows duplicates if not consecutive."""
        hist, _ = temp_home
        hist.append_to_history("first")
        hist.append_to_history("second")
        hist.append_to_history("first")

        assert hist.load_history() == ["first", "second", "first"]

    def test_append_to_history_returns_updated_list(self, temp_home):
        """append_to_history returns the current history list."""
        hist, _ = temp_home
        result1 = hist.append_to_history("first")
        result2 = hist.append_to_history("second")

        assert result1 == ["first"]
        assert result2 == ["first", "second"]

    def test_history_preserves_whitespace_in_prompts(self, temp_home):
        """History preserves internal whitespace in prompts."""
        hist, _ = temp_home
        prompt = "hello   world  with  spaces"
        hist.append_to_history(prompt)
        assert hist.load_history() == [prompt]

    def test_load_history_handles_empty_lines(self, temp_home):
        """load_history filters out empty lines in the history file."""
        hist, tmp_path = temp_home
        axono_dir = tmp_path / ".axono"
        axono_dir.mkdir(parents=True)
        # Write history with some empty lines
        (axono_dir / "history").write_text("first\n\nsecond\n\nthird\n")

        assert hist.load_history() == ["first", "second", "third"]

    def test_max_history_constant(self, temp_home):
        """MAX_HISTORY is set to 30."""
        hist, _ = temp_home
        assert hist.MAX_HISTORY == 30
