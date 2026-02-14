"""Unit tests for axono.workspace."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from axono import workspace
from axono.workspace import (
    WorkspaceViolationError,
    check_command_paths,
    clear_workspace_root,
    get_workspace_root,
    get_workspace_trust_prompt,
    is_path_within_workspace,
    is_workspace_trusted,
    set_workspace_root,
    trust_workspace,
    validate_cd_target,
    validate_path,
)


@pytest.fixture(autouse=True)
def reset_workspace():
    """Reset workspace root before and after each test."""
    clear_workspace_root()
    yield
    clear_workspace_root()


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "subdir").mkdir()
    (workspace_dir / "file.txt").write_text("test")
    return str(workspace_dir)


# ---------------------------------------------------------------------------
# Tests for set_workspace_root / get_workspace_root / clear_workspace_root
# ---------------------------------------------------------------------------


class TestWorkspaceRoot:
    """Tests for workspace root management."""

    def test_set_workspace_root_valid(self, temp_workspace):
        """Setting a valid directory as workspace root should work."""
        result = set_workspace_root(temp_workspace)
        assert result == temp_workspace
        assert get_workspace_root() == temp_workspace

    def test_set_workspace_root_normalizes_path(self, tmp_path):
        """Workspace root should be normalized to absolute path."""
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()

        # Use a path with ~ that gets expanded
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            result = set_workspace_root("~/workspace")
            assert result == str(workspace_dir)
            assert get_workspace_root() == str(workspace_dir)

    def test_set_workspace_root_invalid_directory(self, tmp_path):
        """Setting a non-existent directory should raise ValueError."""
        fake_path = str(tmp_path / "nonexistent")
        with pytest.raises(ValueError, match="not a valid directory"):
            set_workspace_root(fake_path)

    def test_set_workspace_root_file_not_directory(self, tmp_path):
        """Setting a file path should raise ValueError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")
        with pytest.raises(ValueError, match="not a valid directory"):
            set_workspace_root(str(file_path))

    def test_get_workspace_root_not_set(self):
        """Getting workspace root when not set should return None."""
        assert get_workspace_root() is None

    def test_clear_workspace_root(self, temp_workspace):
        """Clearing workspace root should reset to None."""
        set_workspace_root(temp_workspace)
        assert get_workspace_root() is not None
        clear_workspace_root()
        assert get_workspace_root() is None


# ---------------------------------------------------------------------------
# Tests for is_path_within_workspace
# ---------------------------------------------------------------------------


class TestIsPathWithinWorkspace:
    """Tests for workspace boundary checking."""

    def test_no_workspace_set_allows_all(self, tmp_path):
        """When no workspace is set, all paths are allowed."""
        assert is_path_within_workspace("/any/path")
        assert is_path_within_workspace(str(tmp_path))
        assert is_path_within_workspace("relative/path")

    def test_path_within_workspace(self, temp_workspace):
        """Paths inside workspace should be allowed."""
        set_workspace_root(temp_workspace)

        assert is_path_within_workspace(temp_workspace)
        assert is_path_within_workspace(os.path.join(temp_workspace, "subdir"))
        assert is_path_within_workspace(os.path.join(temp_workspace, "file.txt"))
        assert is_path_within_workspace(os.path.join(temp_workspace, "new", "deep"))

    def test_path_outside_workspace(self, temp_workspace, tmp_path):
        """Paths outside workspace should be denied."""
        set_workspace_root(temp_workspace)

        # Parent directory
        assert not is_path_within_workspace(str(tmp_path))
        # Sibling directory
        sibling = tmp_path / "other"
        sibling.mkdir()
        assert not is_path_within_workspace(str(sibling))
        # Root path
        assert not is_path_within_workspace("/etc")
        # Home directory (unless it's the workspace)
        assert not is_path_within_workspace(os.path.expanduser("~"))

    def test_relative_path_resolved_from_workspace(self, temp_workspace):
        """Relative paths should be resolved from workspace root."""
        set_workspace_root(temp_workspace)

        # Relative paths within workspace
        assert is_path_within_workspace("subdir")
        assert is_path_within_workspace("./file.txt")

        # Relative paths that escape
        assert not is_path_within_workspace("..")
        assert not is_path_within_workspace("../other")
        assert not is_path_within_workspace("../../..")

    def test_path_with_prefix_similarity(self, tmp_path):
        """Test that /home/user is not allowed when workspace is /home/username."""
        # Create workspace at /tmp/xxx/username
        user_home = tmp_path / "username"
        user_home.mkdir()
        set_workspace_root(str(user_home))

        # Path /tmp/xxx/user (prefix of username but different dir)
        other_user = tmp_path / "user"
        other_user.mkdir()
        assert not is_path_within_workspace(str(other_user))

    def test_commonpath_value_error_returns_false(self, temp_workspace):
        """ValueError from os.path.commonpath returns False (lines 86-88)."""
        set_workspace_root(temp_workspace)

        with mock.patch(
            "os.path.commonpath", side_effect=ValueError("different drives")
        ):
            assert not is_path_within_workspace(
                os.path.join(temp_workspace, "file.txt")
            )


# ---------------------------------------------------------------------------
# Tests for validate_path
# ---------------------------------------------------------------------------


class TestValidatePath:
    """Tests for path validation with workspace enforcement."""

    def test_no_workspace_returns_normalized(self, tmp_path):
        """When no workspace is set, returns normalized path."""
        result = validate_path(str(tmp_path))
        assert result == str(tmp_path)

    def test_valid_path_returns_normalized(self, temp_workspace):
        """Valid paths within workspace return normalized path."""
        set_workspace_root(temp_workspace)

        result = validate_path(os.path.join(temp_workspace, "subdir"))
        assert result == os.path.join(temp_workspace, "subdir")

    def test_relative_path_resolved(self, temp_workspace):
        """Relative paths are resolved from workspace root."""
        set_workspace_root(temp_workspace)

        result = validate_path("subdir")
        assert result == os.path.join(temp_workspace, "subdir")

    def test_invalid_path_raises_error(self, temp_workspace):
        """Paths outside workspace raise WorkspaceViolationError."""
        set_workspace_root(temp_workspace)

        with pytest.raises(WorkspaceViolationError) as exc_info:
            validate_path("/etc/passwd")

        assert "outside the workspace boundary" in str(exc_info.value)
        assert temp_workspace in str(exc_info.value)

    def test_context_in_error_message(self, temp_workspace):
        """Error message includes context description."""
        set_workspace_root(temp_workspace)

        with pytest.raises(WorkspaceViolationError) as exc_info:
            validate_path("/etc/passwd", context="target file")

        assert "target file" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests for validate_cd_target
# ---------------------------------------------------------------------------


class TestValidateCdTarget:
    """Tests for cd command target validation."""

    def test_no_workspace_returns_resolved(self, tmp_path):
        """When no workspace is set, returns resolved path."""
        current = str(tmp_path)
        target = "subdir"
        expected = os.path.join(current, "subdir")

        result = validate_cd_target(target, current)
        assert result == expected

    def test_valid_cd_within_workspace(self, temp_workspace):
        """CD within workspace should succeed."""
        set_workspace_root(temp_workspace)

        result = validate_cd_target("subdir", temp_workspace)
        assert result == os.path.join(temp_workspace, "subdir")

    def test_cd_to_parent_within_workspace(self, temp_workspace):
        """CD to parent that's still in workspace should succeed."""
        set_workspace_root(temp_workspace)
        subdir = os.path.join(temp_workspace, "subdir")

        result = validate_cd_target("..", subdir)
        assert result == temp_workspace

    def test_cd_escaping_workspace_raises(self, temp_workspace, tmp_path):
        """CD that would escape workspace should raise error."""
        set_workspace_root(temp_workspace)

        with pytest.raises(WorkspaceViolationError) as exc_info:
            validate_cd_target("..", temp_workspace)

        assert "Cannot change to directory" in str(exc_info.value)
        assert "outside the workspace boundary" in str(exc_info.value)

    def test_cd_to_absolute_outside_workspace_raises(self, temp_workspace):
        """CD to absolute path outside workspace should raise error."""
        set_workspace_root(temp_workspace)

        with pytest.raises(WorkspaceViolationError):
            validate_cd_target("/etc", temp_workspace)

    def test_cd_with_tilde_expansion(self, temp_workspace, tmp_path):
        """CD with ~ should expand and validate."""
        set_workspace_root(temp_workspace)

        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            with pytest.raises(WorkspaceViolationError):
                validate_cd_target("~", temp_workspace)


# ---------------------------------------------------------------------------
# Tests for check_command_paths
# ---------------------------------------------------------------------------


class TestCheckCommandPaths:
    """Tests for command path checking."""

    def test_no_workspace_allows_all(self):
        """When no workspace is set, all commands are allowed."""
        assert check_command_paths("cat /etc/passwd", "/home/user") is None
        assert check_command_paths("cd /tmp", "/home/user") is None

    def test_absolute_path_outside_workspace(self, temp_workspace):
        """Commands with absolute paths outside workspace are blocked."""
        set_workspace_root(temp_workspace)

        result = check_command_paths("cat /etc/passwd", temp_workspace)
        assert result is not None
        assert "outside the workspace boundary" in result
        assert "/etc/passwd" in result

    def test_absolute_path_inside_workspace(self, temp_workspace):
        """Commands with absolute paths inside workspace are allowed."""
        set_workspace_root(temp_workspace)

        result = check_command_paths(f"cat {temp_workspace}/file.txt", temp_workspace)
        assert result is None

    def test_relative_path_with_parent_escape(self, temp_workspace, tmp_path):
        """Commands with .. that escape are blocked."""
        set_workspace_root(temp_workspace)

        result = check_command_paths("cat ../../../etc/passwd", temp_workspace)
        assert result is not None
        assert "would escape the workspace boundary" in result

    def test_relative_path_within_workspace(self, temp_workspace):
        """Relative paths that stay within workspace are allowed."""
        set_workspace_root(temp_workspace)

        result = check_command_paths("cat ./subdir/../file.txt", temp_workspace)
        assert result is None

    def test_tilde_path_outside_workspace(self, temp_workspace, tmp_path):
        """Commands with ~ paths outside workspace are blocked."""
        set_workspace_root(temp_workspace)

        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            result = check_command_paths("cat ~/.bashrc", temp_workspace)
            assert result is not None
            assert "outside the workspace boundary" in result

    def test_flags_are_ignored(self, temp_workspace):
        """Command flags starting with - are not treated as paths."""
        set_workspace_root(temp_workspace)

        result = check_command_paths("ls -la", temp_workspace)
        assert result is None

    def test_safe_relative_commands(self, temp_workspace):
        """Regular relative commands are allowed."""
        set_workspace_root(temp_workspace)

        assert check_command_paths("ls", temp_workspace) is None
        assert check_command_paths("git status", temp_workspace) is None
        assert check_command_paths("python script.py", temp_workspace) is None

    def test_malformed_command_passes(self, temp_workspace):
        """Malformed commands (can't be parsed) pass through."""
        set_workspace_root(temp_workspace)

        # Unbalanced quotes - shlex can't parse
        result = check_command_paths("echo 'unclosed", temp_workspace)
        assert result is None  # Let it fail at execution


# ---------------------------------------------------------------------------
# Tests for trusted folders (integration with config)
# ---------------------------------------------------------------------------


class TestTrustedFolders:
    """Tests for workspace trust functionality."""

    def test_is_workspace_trusted_empty_list(self, tmp_path):
        """New workspace is not trusted when list is empty."""
        with mock.patch("axono.workspace.is_folder_trusted", return_value=False):
            assert is_workspace_trusted(str(tmp_path)) is False

    def test_is_workspace_trusted_in_list(self, tmp_path):
        """Workspace in trusted list returns True."""
        with mock.patch("axono.workspace.is_folder_trusted", return_value=True):
            assert is_workspace_trusted(str(tmp_path)) is True

    def test_trust_workspace_adds_to_list(self, tmp_path):
        """Trusting a workspace adds it to the list."""
        mock_folders = []

        def mock_add(folder):
            mock_folders.append(folder)
            return mock_folders

        with mock.patch("axono.workspace.add_trusted_folder", side_effect=mock_add):
            trust_workspace(str(tmp_path))
            assert str(tmp_path) in mock_folders

    def test_get_workspace_trust_prompt(self, tmp_path):
        """Trust prompt contains workspace path."""
        prompt = get_workspace_trust_prompt(str(tmp_path))

        assert str(tmp_path) in prompt
        assert "not in your trusted folders list" in prompt
        assert "add it to your allow list" in prompt


# ---------------------------------------------------------------------------
# Tests for config.py trusted folders functions
# ---------------------------------------------------------------------------


class TestConfigTrustedFolders:
    """Tests for trusted folder functions in config.py."""

    def test_load_trusted_folders_no_file(self, tmp_path):
        """Loading from non-existent file returns empty list."""
        with mock.patch("axono.config._trusted_folders_path") as mock_path:
            mock_path.return_value = tmp_path / "nonexistent.json"

            from axono.config import load_trusted_folders

            result = load_trusted_folders()
            assert result == []

    def test_load_trusted_folders_valid_file(self, tmp_path):
        """Loading from valid file returns folder list."""
        folders_file = tmp_path / "trusted.json"
        folders_file.write_text(
            json.dumps(["/home/user/project1", "/home/user/project2"])
        )

        with mock.patch("axono.config._trusted_folders_path") as mock_path:
            mock_path.return_value = folders_file

            from axono.config import load_trusted_folders

            result = load_trusted_folders()
            assert result == ["/home/user/project1", "/home/user/project2"]

    def test_load_trusted_folders_invalid_json(self, tmp_path):
        """Loading from invalid JSON returns empty list."""
        folders_file = tmp_path / "trusted.json"
        folders_file.write_text("not valid json{{{")

        with mock.patch("axono.config._trusted_folders_path") as mock_path:
            mock_path.return_value = folders_file

            from axono.config import load_trusted_folders

            result = load_trusted_folders()
            assert result == []

    def test_load_trusted_folders_non_list(self, tmp_path):
        """Loading from JSON that's not a list returns empty list."""
        folders_file = tmp_path / "trusted.json"
        folders_file.write_text(json.dumps({"key": "value"}))

        with mock.patch("axono.config._trusted_folders_path") as mock_path:
            mock_path.return_value = folders_file

            from axono.config import load_trusted_folders

            result = load_trusted_folders()
            assert result == []

    def test_load_trusted_folders_filters_non_strings(self, tmp_path):
        """Non-string entries in the list are filtered out."""
        folders_file = tmp_path / "trusted.json"
        folders_file.write_text(json.dumps(["/valid/path", 123, None, "/another/path"]))

        with mock.patch("axono.config._trusted_folders_path") as mock_path:
            mock_path.return_value = folders_file

            from axono.config import load_trusted_folders

            result = load_trusted_folders()
            assert result == ["/valid/path", "/another/path"]

    def test_save_trusted_folders(self, tmp_path):
        """Saving trusted folders writes to file."""
        folders_file = tmp_path / ".axono" / "trusted_folders.json"

        with mock.patch("axono.config.config_dir") as mock_dir:
            mock_dir.return_value = tmp_path / ".axono"

            from axono.config import save_trusted_folders

            result = save_trusted_folders(["/path/one", "/path/two"])

            assert result == folders_file
            assert folders_file.exists()
            content = json.loads(folders_file.read_text())
            assert content == ["/path/one", "/path/two"]

    def test_add_trusted_folder_new(self, tmp_path):
        """Adding a new folder to empty list."""
        folders_file = tmp_path / ".axono" / "trusted_folders.json"

        with mock.patch("axono.config._trusted_folders_path") as mock_path:
            mock_path.return_value = folders_file
            with mock.patch("axono.config.config_dir") as mock_dir:
                mock_dir.return_value = tmp_path / ".axono"

                from axono.config import add_trusted_folder

                result = add_trusted_folder("/new/folder")

                assert "/new/folder" in result

    def test_add_trusted_folder_duplicate(self, tmp_path):
        """Adding duplicate folder doesn't add twice."""
        folders_file = tmp_path / ".axono" / "trusted_folders.json"
        (tmp_path / ".axono").mkdir(parents=True)
        folders_file.write_text(json.dumps(["/existing/folder"]))

        with mock.patch("axono.config._trusted_folders_path") as mock_path:
            mock_path.return_value = folders_file
            with mock.patch("axono.config.config_dir") as mock_dir:
                mock_dir.return_value = tmp_path / ".axono"

                from axono.config import add_trusted_folder

                result = add_trusted_folder("/existing/folder")

                assert result.count("/existing/folder") == 1

    def test_add_trusted_folder_normalizes_path(self, tmp_path):
        """Adding a folder normalizes the path."""
        folders_file = tmp_path / ".axono" / "trusted_folders.json"

        # Create the folder to add
        folder_to_add = tmp_path / "my_folder"
        folder_to_add.mkdir()

        with mock.patch("axono.config._trusted_folders_path") as mock_path:
            mock_path.return_value = folders_file
            with mock.patch("axono.config.config_dir") as mock_dir:
                mock_dir.return_value = tmp_path / ".axono"

                from axono.config import add_trusted_folder

                # Add with trailing slash and weird path
                result = add_trusted_folder(str(folder_to_add) + "/./")

                # Should be normalized
                assert str(folder_to_add) in result

    def test_is_folder_trusted(self, tmp_path):
        """Checking if folder is trusted."""
        folders_file = tmp_path / ".axono" / "trusted_folders.json"
        (tmp_path / ".axono").mkdir(parents=True)
        folders_file.write_text(json.dumps(["/trusted/folder"]))

        with mock.patch("axono.config._trusted_folders_path") as mock_path:
            mock_path.return_value = folders_file

            from axono.config import is_folder_trusted

            assert is_folder_trusted("/trusted/folder") is True
            assert is_folder_trusted("/untrusted/folder") is False
