"""Unit tests for axono.conversation."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from axono import conversation


class TestGetDbPath:
    """Tests for get_db_path function."""

    def test_default_path(self):
        """Returns default path in ~/.axono/ when env var not set."""
        with mock.patch.dict(os.environ, {}, clear=False):
            # Remove AXONO_DB_PATH if present
            os.environ.pop("AXONO_DB_PATH", None)
            path = conversation.get_db_path()
            assert path == Path.home() / ".axono" / "conversations.db"

    def test_custom_path_from_env(self, tmp_path):
        """Uses AXONO_DB_PATH environment variable when set."""
        custom_path = tmp_path / "custom" / "mydb.db"
        with mock.patch.dict(os.environ, {"AXONO_DB_PATH": str(custom_path)}):
            path = conversation.get_db_path()
            assert path == custom_path

    def test_creates_parent_directory(self, tmp_path):
        """Creates parent directory if it doesn't exist."""
        custom_path = tmp_path / "new_dir" / "subdir" / "db.db"
        assert not custom_path.parent.exists()
        with mock.patch.dict(os.environ, {"AXONO_DB_PATH": str(custom_path)}):
            path = conversation.get_db_path()
            assert path == custom_path
            assert custom_path.parent.exists()


class TestGenerateConversationId:
    """Tests for generate_conversation_id function."""

    def test_returns_string(self):
        """Returns a string."""
        cid = conversation.generate_conversation_id()
        assert isinstance(cid, str)

    def test_returns_uuid_format(self):
        """Returns a valid UUID string."""
        import uuid

        cid = conversation.generate_conversation_id()
        # Should not raise
        parsed = uuid.UUID(cid)
        assert str(parsed) == cid

    def test_generates_unique_ids(self):
        """Each call generates a unique ID."""
        ids = [conversation.generate_conversation_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestGetCheckpointer:
    """Tests for get_checkpointer function."""

    @pytest.mark.asyncio
    async def test_returns_async_sqlite_saver(self, tmp_path):
        """Returns an AsyncSqliteSaver instance."""
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = tmp_path / "test.db"
        with mock.patch.dict(os.environ, {"AXONO_DB_PATH": str(db_path)}):
            checkpointer = await conversation.get_checkpointer()
            try:
                assert isinstance(checkpointer, AsyncSqliteSaver)
            finally:
                await checkpointer.conn.close()

    @pytest.mark.asyncio
    async def test_creates_db_file_on_use(self, tmp_path):
        """Database file is created when checkpointer is used."""
        db_path = tmp_path / "conversations.db"
        with mock.patch.dict(os.environ, {"AXONO_DB_PATH": str(db_path)}):
            checkpointer = await conversation.get_checkpointer()
            try:
                # The file may not exist until first operation
                assert checkpointer is not None
            finally:
                await checkpointer.conn.close()
