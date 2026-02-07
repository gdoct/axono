"""Prompt history management for Axono.

Stores up to MAX_HISTORY prompts in the data directory, one per line.
"""

from axono.config import config_dir

MAX_HISTORY = 30


def _history_file():
    """Return the path to the history file."""
    return config_dir() / "history"


def load_history() -> list[str]:
    """Load prompt history from disk.

    Returns a list of prompts (oldest first), or empty list if no history.
    """
    history_file = _history_file()
    if not history_file.is_file():
        return []
    try:
        text = history_file.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        return lines[-MAX_HISTORY:]
    except OSError:
        return []


def save_history(history: list[str]) -> None:
    """Save prompt history to disk.

    Keeps only the last MAX_HISTORY entries.
    """
    history_file = _history_file()
    history_file.parent.mkdir(parents=True, exist_ok=True)
    trimmed = history[-MAX_HISTORY:]
    history_file.write_text(
        "\n".join(trimmed) + "\n" if trimmed else "", encoding="utf-8"
    )


def append_to_history(prompt: str) -> list[str]:
    """Add a prompt to history and save.

    Returns the updated history list.
    """
    prompt = prompt.strip()
    if not prompt:
        return load_history()

    history = load_history()
    # Don't add duplicates of the last entry
    if history and history[-1] == prompt:
        return history

    history.append(prompt)
    history = history[-MAX_HISTORY:]
    save_history(history)
    return history
