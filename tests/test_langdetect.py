"""Unit tests for axono.langdetect."""

import os
from unittest import mock

import pytest

from axono.langdetect import detect_project_type_from_folderpath


class TestDetectProjectTypeDirectMatch:
    """Tests for direct file-based detection in the root folder."""

    def test_detects_python_by_file(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[build-system]")
        assert detect_project_type_from_folderpath(str(tmp_path)) == "python"

    def test_detects_nodejs_by_file(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        assert detect_project_type_from_folderpath(str(tmp_path)) == "nodejs"

    def test_detects_java_by_file(self, tmp_path):
        (tmp_path / "pom.xml").write_text("<project/>")
        assert detect_project_type_from_folderpath(str(tmp_path)) == "java"

    def test_detects_go_by_file(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example")
        assert detect_project_type_from_folderpath(str(tmp_path)) == "go"

    def test_detects_rust_by_file(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]")
        assert detect_project_type_from_folderpath(str(tmp_path)) == "rust"

    def test_detects_cmake_by_file(self, tmp_path):
        (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required()")
        assert detect_project_type_from_folderpath(str(tmp_path)) == "cmake"

    def test_detects_make_by_file(self, tmp_path):
        (tmp_path / "Makefile").write_text("all:")
        assert detect_project_type_from_folderpath(str(tmp_path)) == "make"


class TestDetectProjectTypeByExtension:
    """Tests for extension-based detection (line 57, 63-64)."""

    def test_detects_python_by_extension(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hi')")
        result = detect_project_type_from_folderpath(str(tmp_path))
        # .py extension maps to python (but only after file-based checks fail)
        assert result == "python"

    def test_detects_go_by_extension(self, tmp_path):
        (tmp_path / "main.go").write_text("package main")
        result = detect_project_type_from_folderpath(str(tmp_path))
        assert result == "go"


class TestDetectProjectTypeUnknown:
    """Tests for the 'unknown' fallback."""

    def test_empty_folder_returns_unknown(self, tmp_path):
        assert detect_project_type_from_folderpath(str(tmp_path)) == "unknown"

    def test_no_indicators_returns_unknown(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b,c")
        assert detect_project_type_from_folderpath(str(tmp_path)) == "unknown"


class TestCheckFolderOSError:
    """Cover the OSError/PermissionError handler in check_folder (lines 49-50)."""

    def test_oserror_in_root_falls_through_to_bfs(self, tmp_path):
        # Make the folder unlistable
        with mock.patch("os.listdir", side_effect=OSError("Permission denied")):
            result = detect_project_type_from_folderpath(str(tmp_path))
        assert result == "unknown"


class TestBFSSrcPriority:
    """Cover the 'src' directory priority in BFS (lines 87-94)."""

    def test_finds_type_in_src_subfolder(self, tmp_path):
        """Project type found in src/ subfolder is returned (lines 89-91)."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("print('hi')")
        result = detect_project_type_from_folderpath(str(tmp_path))
        assert result == "python"

    def test_src_subfolder_no_type_continues_bfs(self, tmp_path):
        """src/ with no type is enqueued for deeper search (lines 92-94)."""
        src = tmp_path / "src"
        src.mkdir()
        inner = src / "inner"
        inner.mkdir()
        (inner / "app.js").write_text("console.log('hi')")

        result = detect_project_type_from_folderpath(str(tmp_path))
        # Should find nodejs via BFS in src/inner
        assert result == "nodejs"


class TestBFSOtherDirectories:
    """Cover the non-src directory BFS loop (lines 98-104)."""

    def test_finds_type_in_non_src_subfolder(self, tmp_path):
        """Project type found in a regular subfolder (lines 101-103)."""
        lib = tmp_path / "lib"
        lib.mkdir()
        (lib / "Cargo.toml").write_text("[package]")

        result = detect_project_type_from_folderpath(str(tmp_path))
        assert result == "rust"

    def test_deeper_bfs_finds_type(self, tmp_path):
        """BFS enqueues subfolder and finds type deeper (line 104)."""
        sub = tmp_path / "sub"
        sub.mkdir()
        deep = sub / "deep"
        deep.mkdir()
        (deep / "go.mod").write_text("module example")

        result = detect_project_type_from_folderpath(str(tmp_path))
        assert result == "go"


class TestBFSOSError:
    """Cover the OSError in BFS listing (lines 82-83)."""

    def test_bfs_oserror_continues(self, tmp_path):
        """BFS gracefully skips folders that raise OSError."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "main.py").write_text("code")

        original_listdir = os.listdir
        call_count = 0

        def patched_listdir(path):
            nonlocal call_count
            call_count += 1
            # First call is for check_folder on root, second is BFS root listing.
            # Let the root succeed but make the 'sub' folder's BFS check fail.
            if call_count == 3:
                raise OSError("Permission denied")
            return original_listdir(path)

        with mock.patch("os.listdir", side_effect=patched_listdir):
            # Even with error on one folder, should not crash
            detect_project_type_from_folderpath(str(tmp_path))
