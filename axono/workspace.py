"""Workspace security and path validation.

This module enforces workspace boundaries to ensure the agent cannot operate
outside the designated workspace root folder. It also handles trust verification
for new workspaces.
"""

import os
from pathlib import Path

from axono.config import add_trusted_folder, is_folder_trusted

# Module-level workspace root (set once at startup)
_workspace_root: str | None = None


class WorkspaceViolationError(Exception):
    """Raised when an operation would escape the workspace boundary."""

    pass


def set_workspace_root(path: str) -> str:
    """Set the workspace root directory.

    This should be called once at startup. All file operations and directory
    changes will be constrained to this directory and its subdirectories.

    Args:
        path: The absolute path to the workspace root.

    Returns:
        The normalized absolute path.

    Raises:
        ValueError: If the path is not a valid directory.
    """
    global _workspace_root
    normalized = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(normalized):
        raise ValueError(f"Workspace root is not a valid directory: {normalized}")
    _workspace_root = normalized
    return normalized


def get_workspace_root() -> str | None:
    """Get the current workspace root directory.

    Returns:
        The workspace root path, or None if not set.
    """
    return _workspace_root


def clear_workspace_root() -> None:
    """Clear the workspace root (for testing purposes)."""
    global _workspace_root
    _workspace_root = None


def is_path_within_workspace(path: str) -> bool:
    """Check if a path is within the workspace boundary.

    Args:
        path: The path to check (can be relative or absolute).

    Returns:
        True if the path is within the workspace, False otherwise.
        Returns True if no workspace root is set (unrestricted mode).
    """
    if _workspace_root is None:
        return True

    # Normalize the path
    if not os.path.isabs(path):
        # Relative paths are resolved from the workspace root
        normalized = os.path.abspath(os.path.join(_workspace_root, path))
    else:
        normalized = os.path.abspath(os.path.expanduser(path))

    # Check if the normalized path starts with the workspace root
    # Use os.path.commonpath to handle edge cases like /home/user vs /home/username
    try:
        common = os.path.commonpath([_workspace_root, normalized])
        return common == _workspace_root
    except ValueError:
        # Different drives on Windows
        return False


def validate_path(path: str, context: str = "path") -> str:
    """Validate that a path is within the workspace and return the normalized path.

    Args:
        path: The path to validate.
        context: Description of the path for error messages.

    Returns:
        The normalized absolute path.

    Raises:
        WorkspaceViolationError: If the path is outside the workspace.
    """
    if _workspace_root is None:
        return os.path.abspath(os.path.expanduser(path))

    # Normalize the path
    if not os.path.isabs(path):
        normalized = os.path.abspath(os.path.join(_workspace_root, path))
    else:
        normalized = os.path.abspath(os.path.expanduser(path))

    if not is_path_within_workspace(normalized):
        raise WorkspaceViolationError(
            f"Access denied: {context} '{path}' is outside the workspace boundary. "
            f"All operations must remain within: {_workspace_root}"
        )

    return normalized


def validate_cd_target(target: str, current_dir: str) -> str:
    """Validate a cd command target and return the resolved path.

    Args:
        target: The target directory for cd command.
        current_dir: The current working directory.

    Returns:
        The validated absolute path.

    Raises:
        WorkspaceViolationError: If the target is outside the workspace.
    """
    # Expand user home
    expanded = os.path.expanduser(target)

    # Resolve relative paths from current directory
    if not os.path.isabs(expanded):
        resolved = os.path.abspath(os.path.join(current_dir, expanded))
    else:
        resolved = os.path.abspath(expanded)

    # Validate against workspace
    if not is_path_within_workspace(resolved):
        raise WorkspaceViolationError(
            f"Access denied: Cannot change to directory '{target}' - "
            f"it is outside the workspace boundary. "
            f"You must remain within: {_workspace_root}"
        )

    return resolved


def check_command_paths(command: str, current_dir: str) -> str | None:
    """Check if a shell command references paths outside the workspace.

    This performs a best-effort static analysis of the command to detect
    obvious escapes like 'cd /etc' or 'cat /etc/passwd'.

    Args:
        command: The shell command to check.
        current_dir: The current working directory.

    Returns:
        An error message if a violation is detected, None otherwise.
    """
    if _workspace_root is None:
        return None

    # Check for absolute paths that escape the workspace
    # This is a heuristic - it won't catch everything, but covers common cases
    import shlex

    try:
        tokens = shlex.split(command)
    except ValueError:
        # Malformed command - let it pass and fail at execution
        return None

    for token in tokens:
        # Skip flags
        if token.startswith("-"):
            continue

        # Check absolute paths
        if token.startswith("/"):
            if not is_path_within_workspace(token):
                return (
                    f"Access denied: Command references path '{token}' "
                    f"which is outside the workspace boundary. "
                    f"All operations must remain within: {_workspace_root}"
                )

        # Check paths starting with ~
        if token.startswith("~"):
            expanded = os.path.expanduser(token)
            if not is_path_within_workspace(expanded):
                return (
                    f"Access denied: Command references path '{token}' "
                    f"which is outside the workspace boundary. "
                    f"All operations must remain within: {_workspace_root}"
                )

        # Check relative paths that go up (../../../)
        if ".." in token:
            resolved = os.path.abspath(os.path.join(current_dir, token))
            if not is_path_within_workspace(resolved):
                return (
                    f"Access denied: Command references path '{token}' "
                    f"which would escape the workspace boundary. "
                    f"All operations must remain within: {_workspace_root}"
                )

    return None


def is_workspace_trusted(workspace_path: str) -> bool:
    """Check if a workspace is trusted by the user.

    Args:
        workspace_path: The workspace path to check.

    Returns:
        True if the workspace is in the trusted folders list.
    """
    return is_folder_trusted(workspace_path)


def trust_workspace(workspace_path: str) -> list[str]:
    """Add a workspace to the trusted folders list.

    Args:
        workspace_path: The workspace path to trust.

    Returns:
        The updated list of trusted folders.
    """
    return add_trusted_folder(workspace_path)


def get_workspace_trust_prompt(workspace_path: str) -> str:
    """Get the prompt to show when asking user to trust a workspace.

    Args:
        workspace_path: The workspace path.

    Returns:
        A formatted prompt string.
    """
    return (
        f"The workspace '{workspace_path}' is not in your trusted folders list.\n"
        f"Axono will be restricted to operate only within this folder.\n\n"
        f"Do you trust this workspace and want to add it to your allow list?"
    )
