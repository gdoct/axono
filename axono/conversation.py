"""Conversation persistence using LangGraph's SQLite checkpointer."""

import os
import uuid
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Default database location in ~/.axono/
DEFAULT_DB_PATH = Path.home() / ".axono" / "conversations.db"


def get_db_path() -> Path:
    """Get the path to the SQLite database, creating the directory if needed."""
    db_path = Path(os.environ.get("AXONO_DB_PATH", str(DEFAULT_DB_PATH)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def generate_conversation_id() -> str:
    """Generate a new unique conversation ID."""
    return str(uuid.uuid4())


async def get_checkpointer() -> AsyncSqliteSaver:
    """Create and return an AsyncSqliteSaver checkpointer.

    The checkpointer persists conversation state to SQLite, enabling
    conversation history to survive restarts.
    """
    db_path = get_db_path()
    conn = await aiosqlite.connect(str(db_path))
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver
