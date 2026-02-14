"""Unit tests for axono.folderembed."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from axono import folderembed
from axono.folderembed import (
    Chunk,
    ChunkMetadata,
    FileChangeEvent,
    _build_enriched_text,
    _chunk_by_lines,
    _collect_files,
    _compute_md5,
    _detect_language,
    _is_binary,
    _load_gitignore,
    _parse_file,
    _should_watch_file,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_folder(tmp_path):
    """Create a temporary folder with sample files."""
    # Python file
    py_file = tmp_path / "example.py"
    py_file.write_text('''"""Module docstring."""

import os
from pathlib import Path


@decorator
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"


class Greeter:
    """A greeter class."""

    def greet(self, name: str) -> str:
        """Greet someone."""
        return hello(name)
''')

    # JavaScript file
    js_file = tmp_path / "app.js"
    js_file.write_text("""import { something } from 'somewhere';

function greet(name) {
    return `Hello, ${name}!`;
}

class App {
    constructor() {
        this.name = 'App';
    }

    run() {
        console.log(greet(this.name));
    }
}

const arrowFunc = (x) => x * 2;
""")

    # Text file (no grammar)
    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("This is a readme file.\nWith multiple lines.\n" * 30)

    # Binary file
    bin_file = tmp_path / "binary.dat"
    bin_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")

    # Hidden file
    hidden_file = tmp_path / ".hidden"
    hidden_file.write_text("hidden content")

    # Nested directory
    subdir = tmp_path / "src"
    subdir.mkdir()
    (subdir / "utils.py").write_text("def util(): pass\n")

    # Skip directory
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "package.js").write_text("// should be skipped")

    return tmp_path


@pytest.fixture
def temp_folder_with_gitignore(temp_folder):
    """Add a .gitignore to the temp folder."""
    gitignore = temp_folder / ".gitignore"
    gitignore.write_text("*.txt\nsrc/\n")
    return temp_folder


# ---------------------------------------------------------------------------
# Tests for _is_binary
# ---------------------------------------------------------------------------


class TestIsBinary:
    """Tests for binary file detection."""

    def test_text_file_not_binary(self, tmp_path):
        f = tmp_path / "text.txt"
        f.write_text("Hello, world!")
        assert _is_binary(str(f)) is False

    def test_binary_file_is_binary(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"Hello\x00World")
        assert _is_binary(str(f)) is True

    def test_empty_file_not_binary(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert _is_binary(str(f)) is False

    def test_nonexistent_file_treated_as_binary(self):
        assert _is_binary("/nonexistent/path/file.txt") is True

    def test_null_byte_at_end_is_binary(self, tmp_path):
        f = tmp_path / "null_end.bin"
        f.write_bytes(b"Hello World\x00")
        assert _is_binary(str(f)) is True


# ---------------------------------------------------------------------------
# Tests for _compute_md5
# ---------------------------------------------------------------------------


class TestComputeMd5:
    """Tests for MD5 hash computation."""

    def test_computes_md5_hash(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!")
        md5 = _compute_md5(str(f))
        assert len(md5) == 32
        assert md5 == "6cd3556deb0da54bca060b4c39479839"

    def test_empty_file_hash(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        md5 = _compute_md5(str(f))
        assert md5 == "d41d8cd98f00b204e9800998ecf8427e"

    def test_nonexistent_file_returns_empty(self):
        md5 = _compute_md5("/nonexistent/path/file.txt")
        assert md5 == ""

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "file1.txt"
        f2 = tmp_path / "file2.txt"
        f1.write_text("content1")
        f2.write_text("content2")
        assert _compute_md5(str(f1)) != _compute_md5(str(f2))


# ---------------------------------------------------------------------------
# Tests for _load_gitignore
# ---------------------------------------------------------------------------


class TestLoadGitignore:
    """Tests for .gitignore loading."""

    def test_loads_gitignore_patterns(self, tmp_path):
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__/\n")
        spec = _load_gitignore(str(tmp_path))
        if spec is not None:  # Only if pathspec is installed
            assert spec.match_file("test.pyc")
            assert spec.match_file("__pycache__/cache.py")
            assert not spec.match_file("test.py")

    def test_no_gitignore_returns_none(self, tmp_path):
        spec = _load_gitignore(str(tmp_path))
        assert spec is None

    def test_unreadable_gitignore_returns_none(self, tmp_path):
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc")
        # Make unreadable (skip on Windows)
        if os.name != "nt":
            gitignore.chmod(0o000)
            try:
                spec = _load_gitignore(str(tmp_path))
                assert spec is None
            finally:
                gitignore.chmod(0o644)


# ---------------------------------------------------------------------------
# Tests for _collect_files
# ---------------------------------------------------------------------------


class TestCollectFiles:
    """Tests for file collection."""

    def test_collects_source_files(self, temp_folder):
        files = _collect_files(str(temp_folder))
        filenames = {Path(f).name for f in files}
        assert "example.py" in filenames
        assert "app.js" in filenames
        assert "readme.txt" in filenames
        assert "utils.py" in filenames

    def test_skips_binary_files(self, temp_folder):
        files = _collect_files(str(temp_folder))
        filenames = {Path(f).name for f in files}
        assert "binary.dat" not in filenames

    def test_skips_hidden_files(self, temp_folder):
        files = _collect_files(str(temp_folder))
        filenames = {Path(f).name for f in files}
        assert ".hidden" not in filenames

    def test_skips_node_modules(self, temp_folder):
        files = _collect_files(str(temp_folder))
        filenames = {Path(f).name for f in files}
        assert "package.js" not in filenames

    def test_respects_gitignore(self, temp_folder_with_gitignore):
        files = _collect_files(str(temp_folder_with_gitignore))
        filenames = {Path(f).name for f in files}
        # .txt files should be ignored
        assert "readme.txt" not in filenames
        # src/ should be ignored
        assert "utils.py" not in filenames
        # Python files not in src should remain
        assert "example.py" in filenames

    def test_returns_sorted_paths(self, temp_folder):
        files = _collect_files(str(temp_folder))
        assert files == sorted(files)

    def test_skips_common_extensions(self, tmp_path):
        # These should all be skipped by extension
        (tmp_path / "file.pyc").write_bytes(b"bytecode")
        (tmp_path / "file.map").write_text("sourcemap")
        (tmp_path / "file.lock").write_text("lockfile")
        files = _collect_files(str(tmp_path))
        # Note: .min.js files are text and get detected, but we skip by extension
        filenames = {Path(f).name for f in files}
        assert "file.pyc" not in filenames
        assert "file.map" not in filenames
        assert "file.lock" not in filenames

    def test_skips_symlinks(self, tmp_path):
        """Test that symlinked files are skipped."""
        # Create a real file
        real_file = tmp_path / "real.py"
        real_file.write_text("# real file")

        # Create a symlink to it
        link_file = tmp_path / "link.py"
        link_file.symlink_to(real_file)

        files = _collect_files(str(tmp_path))
        filenames = {Path(f).name for f in files}
        assert "real.py" in filenames
        assert "link.py" not in filenames

    def test_skips_symlinked_directories(self, tmp_path):
        """Test that symlinked directories are skipped."""
        # Create a real directory with a file
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        (real_dir / "file.py").write_text("# in real dir")

        # Create a symlink to the directory
        link_dir = tmp_path / "link_dir"
        link_dir.symlink_to(real_dir)

        files = _collect_files(str(tmp_path))
        # Should find file in real_dir but not through link_dir
        paths = [Path(f).parts for f in files]
        assert any("real_dir" in p for p in paths)
        assert not any("link_dir" in p for p in paths)

    def test_max_files_limit(self, tmp_path):
        """Test that file collection respects MAX_FILES limit."""
        # Create more files than the limit
        for i in range(100):
            (tmp_path / f"file{i:03d}.py").write_text(f"# file {i}")

        # Temporarily reduce MAX_FILES for testing
        with mock.patch.object(folderembed, "MAX_FILES", 50):
            files = _collect_files(str(tmp_path))

        assert len(files) <= 50

    def test_max_files_limit_across_directories(self, tmp_path):
        """Test that file limit works across multiple directories."""
        # Create multiple directories with files to trigger the outer loop break
        for d in range(10):
            subdir = tmp_path / f"dir{d:02d}"
            subdir.mkdir()
            for i in range(20):
                (subdir / f"file{i:02d}.py").write_text(f"# file {i}")

        # Temporarily reduce MAX_FILES for testing
        with mock.patch.object(folderembed, "MAX_FILES", 50):
            files = _collect_files(str(tmp_path))

        assert len(files) == 50

    def test_skips_system_directories(self, tmp_path):
        """Test that system directories are skipped."""
        # Create directories that match system excludes
        for sys_dir in ["usr", "var", "proc", "Windows"]:
            d = tmp_path / sys_dir
            d.mkdir()
            (d / "file.py").write_text("# system file")

        files = _collect_files(str(tmp_path))
        filenames = {Path(f).name for f in files}
        # None of the system directory files should be included
        assert "file.py" not in filenames

    def test_skips_default_excludes(self, tmp_path):
        """Test that default exclude directories are skipped."""
        # Create directories that match default excludes
        for exclude_dir in [".next", "vendor", ".cache"]:
            d = tmp_path / exclude_dir
            d.mkdir()
            (d / "file.py").write_text("# excluded file")

        # Also create a regular file
        (tmp_path / "main.py").write_text("# main file")

        files = _collect_files(str(tmp_path))
        filenames = {Path(f).name for f in files}
        assert "main.py" in filenames
        # The files in excluded dirs should not be there
        # (only main.py should be collected)
        assert len(files) == 1


# ---------------------------------------------------------------------------
# Tests for _detect_language
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    """Tests for language detection by extension."""

    def test_python_extensions(self):
        assert _detect_language("file.py") == "python"
        assert _detect_language("file.pyw") == "python"
        assert _detect_language("file.pyi") == "python"

    def test_javascript_extensions(self):
        assert _detect_language("file.js") == "javascript"
        assert _detect_language("file.mjs") == "javascript"
        assert _detect_language("file.jsx") == "javascript"

    def test_typescript_extensions(self):
        assert _detect_language("file.ts") == "typescript"
        assert _detect_language("file.tsx") == "tsx"  # TSX has special parser

    def test_other_languages(self):
        assert _detect_language("file.go") == "go"
        assert _detect_language("file.rs") == "rust"

    def test_unknown_extension(self):
        assert _detect_language("file.xyz") is None
        assert _detect_language("file.txt") is None

    def test_case_insensitive(self):
        assert _detect_language("file.PY") == "python"
        assert _detect_language("file.Js") == "javascript"


# ---------------------------------------------------------------------------
# Tests for _chunk_by_lines
# ---------------------------------------------------------------------------


class TestChunkByLines:
    """Tests for line-based chunking fallback."""

    def test_chunks_by_lines(self):
        content = "\n".join([f"line {i}" for i in range(100)])
        chunks = _chunk_by_lines(content, "/path/file.txt", "abc123", chunk_size=50)
        assert len(chunks) == 2
        assert chunks[0].line_start == 1
        assert chunks[0].line_end == 50
        assert chunks[1].line_start == 51
        assert chunks[1].line_end == 100

    def test_single_chunk_for_small_file(self):
        content = "line 1\nline 2\nline 3"
        chunks = _chunk_by_lines(content, "/path/file.txt", "abc123", chunk_size=50)
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "file"

    def test_chunk_metadata(self):
        content = "line 1\nline 2"
        chunks = _chunk_by_lines(content, "/path/file.txt", "abc123")
        assert chunks[0].file_path == "/path/file.txt"
        assert chunks[0].file_md5 == "abc123"
        assert chunks[0].symbol_name is None
        assert chunks[0].symbol_type == "file"


# ---------------------------------------------------------------------------
# Tests for _parse_file (tree-sitter integration)
# ---------------------------------------------------------------------------


class TestParseFile:
    """Tests for file parsing with tree-sitter."""

    def test_parses_python_file(self, temp_folder):
        py_file = temp_folder / "example.py"
        chunks = _parse_file(str(py_file), "test_md5")

        # Should find: hello function, Greeter class, greet method
        symbol_names = {c.symbol_name for c in chunks if c.symbol_name}
        assert "hello" in symbol_names
        assert "Greeter" in symbol_names
        assert "greet" in symbol_names

    def test_extracts_python_decorators(self, temp_folder):
        py_file = temp_folder / "example.py"
        chunks = _parse_file(str(py_file), "test_md5")

        hello_chunk = next(c for c in chunks if c.symbol_name == "hello")
        assert "@decorator" in hello_chunk.metadata.decorators

    def test_extracts_python_imports(self, temp_folder):
        py_file = temp_folder / "example.py"
        chunks = _parse_file(str(py_file), "test_md5")

        # All chunks should have file-level imports
        for chunk in chunks:
            assert any("os" in imp for imp in chunk.metadata.imports)

    def test_identifies_methods_vs_functions(self, temp_folder):
        py_file = temp_folder / "example.py"
        chunks = _parse_file(str(py_file), "test_md5")

        hello_chunk = next(c for c in chunks if c.symbol_name == "hello")
        greet_chunk = next(c for c in chunks if c.symbol_name == "greet")

        assert hello_chunk.symbol_type == "function"
        assert greet_chunk.symbol_type == "method"
        assert greet_chunk.metadata.parent_class == "Greeter"

    def test_parses_javascript_file(self, temp_folder):
        js_file = temp_folder / "app.js"
        chunks = _parse_file(str(js_file), "test_md5")

        symbol_names = {c.symbol_name for c in chunks if c.symbol_name}
        assert "greet" in symbol_names
        assert "App" in symbol_names

    def test_fallback_for_unknown_language(self, temp_folder):
        txt_file = temp_folder / "readme.txt"
        chunks = _parse_file(str(txt_file), "test_md5")

        # Should use line-based chunking
        assert all(c.symbol_type == "file" for c in chunks)

    def test_handles_unreadable_file(self, tmp_path):
        chunks = _parse_file("/nonexistent/file.py", "test_md5")
        assert chunks == []

    def test_handles_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")
        chunks = _parse_file(str(empty_file), "test_md5")
        # Should create a file-level chunk
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "file"


# ---------------------------------------------------------------------------
# Tests for _build_enriched_text
# ---------------------------------------------------------------------------


class TestBuildEnrichedText:
    """Tests for enriched text building."""

    def test_includes_file_path(self):
        chunk = Chunk(
            file_path="/path/to/file.py",
            file_md5="abc",
            symbol_name="my_func",
            symbol_type="function",
            line_start=1,
            line_end=5,
            source_code="def my_func(): pass",
            embedded_text="",
        )
        text = _build_enriched_text(chunk)
        assert "File: /path/to/file.py" in text

    def test_includes_symbol_info(self):
        chunk = Chunk(
            file_path="/path/to/file.py",
            file_md5="abc",
            symbol_name="my_func",
            symbol_type="function",
            line_start=1,
            line_end=5,
            source_code="def my_func(): pass",
            embedded_text="",
        )
        text = _build_enriched_text(chunk)
        assert "Function: my_func" in text

    def test_includes_imports(self):
        chunk = Chunk(
            file_path="/path/to/file.py",
            file_md5="abc",
            symbol_name="my_func",
            symbol_type="function",
            line_start=1,
            line_end=5,
            source_code="def my_func(): pass",
            embedded_text="",
            metadata=ChunkMetadata(imports=["import os", "from pathlib import Path"]),
        )
        text = _build_enriched_text(chunk)
        assert "Imports:" in text
        assert "os" in text
        assert "pathlib" in text

    def test_includes_decorators(self):
        chunk = Chunk(
            file_path="/path/to/file.py",
            file_md5="abc",
            symbol_name="my_func",
            symbol_type="function",
            line_start=1,
            line_end=5,
            source_code="def my_func(): pass",
            embedded_text="",
            metadata=ChunkMetadata(decorators=["@decorator", "@another"]),
        )
        text = _build_enriched_text(chunk)
        assert "Decorators:" in text
        assert "@decorator" in text

    def test_includes_parent_class(self):
        chunk = Chunk(
            file_path="/path/to/file.py",
            file_md5="abc",
            symbol_name="my_method",
            symbol_type="method",
            line_start=1,
            line_end=5,
            source_code="def my_method(self): pass",
            embedded_text="",
            metadata=ChunkMetadata(parent_class="MyClass"),
        )
        text = _build_enriched_text(chunk)
        assert "Class: MyClass" in text

    def test_includes_source_code(self):
        chunk = Chunk(
            file_path="/path/to/file.py",
            file_md5="abc",
            symbol_name="my_func",
            symbol_type="function",
            line_start=1,
            line_end=5,
            source_code="def my_func(): pass",
            embedded_text="",
        )
        text = _build_enriched_text(chunk)
        assert "def my_func(): pass" in text


# ---------------------------------------------------------------------------
# Tests for database operations
# ---------------------------------------------------------------------------


class TestDatabaseOperations:
    """Tests for SQLite database operations."""

    @pytest.mark.asyncio
    async def test_init_db_creates_tables(self, tmp_path):
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            conn = await folderembed._init_db()
            try:
                # Check tables exist
                async with conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ) as cursor:
                    tables = {row[0] async for row in cursor}
                assert "files" in tables
                assert "chunks" in tables
                assert "metadata" in tables
            finally:
                await conn.close()

    @pytest.mark.asyncio
    async def test_save_and_get_file_md5(self, tmp_path):
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            conn = await folderembed._init_db()
            try:
                # Save a file
                await conn.execute(
                    "INSERT INTO files (file_path, file_md5) VALUES (?, ?)",
                    ("/path/to/file.py", "abc123"),
                )
                await conn.commit()

                # Get the MD5
                md5 = await folderembed._get_file_md5(conn, "/path/to/file.py")
                assert md5 == "abc123"

                # Non-existent file
                md5 = await folderembed._get_file_md5(conn, "/nonexistent")
                assert md5 is None
            finally:
                await conn.close()

    @pytest.mark.asyncio
    async def test_save_chunks(self, tmp_path):
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            conn = await folderembed._init_db()
            try:
                chunks = [
                    Chunk(
                        file_path="/path/to/file.py",
                        file_md5="abc123",
                        symbol_name="my_func",
                        symbol_type="function",
                        line_start=1,
                        line_end=5,
                        source_code="def my_func(): pass",
                        embedded_text="enriched text",
                        embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                        metadata=ChunkMetadata(
                            imports=["import os"],
                            calls=["print"],
                            decorators=["@decorator"],
                        ),
                    )
                ]
                await folderembed._save_chunks(conn, chunks)

                # Verify file was saved
                async with conn.execute("SELECT file_md5 FROM files") as cursor:
                    row = await cursor.fetchone()
                    assert row[0] == "abc123"

                # Verify chunk was saved
                async with conn.execute("SELECT symbol_name FROM chunks") as cursor:
                    row = await cursor.fetchone()
                    assert row[0] == "my_func"

                # Verify metadata was saved
                async with conn.execute("SELECT imports FROM metadata") as cursor:
                    row = await cursor.fetchone()
                    assert json.loads(row[0]) == ["import os"]
            finally:
                await conn.close()

    @pytest.mark.asyncio
    async def test_delete_file_chunks(self, tmp_path):
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            conn = await folderembed._init_db()
            try:
                # Insert a file and chunk
                await conn.execute(
                    "INSERT INTO files (file_path, file_md5) VALUES (?, ?)",
                    ("/path/to/file.py", "abc123"),
                )
                await conn.execute(
                    "INSERT INTO chunks (file_path, symbol_name, symbol_type, "
                    "line_start, line_end, source_code, embedded_text, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "/path/to/file.py",
                        "func",
                        "function",
                        1,
                        5,
                        "code",
                        "text",
                        b"emb",
                    ),
                )
                await conn.commit()

                # Delete
                await folderembed._delete_file_chunks(conn, "/path/to/file.py")

                # Verify deleted
                async with conn.execute("SELECT COUNT(*) FROM files") as cursor:
                    row = await cursor.fetchone()
                    assert row[0] == 0
            finally:
                await conn.close()

    @pytest.mark.asyncio
    async def test_cleanup_stale_files(self, tmp_path):
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            conn = await folderembed._init_db()
            try:
                # Insert files
                await conn.execute(
                    "INSERT INTO files (file_path, file_md5) VALUES (?, ?)",
                    ("/path/to/file1.py", "abc123"),
                )
                await conn.execute(
                    "INSERT INTO files (file_path, file_md5) VALUES (?, ?)",
                    ("/path/to/file2.py", "def456"),
                )
                await conn.commit()

                # Cleanup - only file1 exists
                removed = await folderembed._cleanup_stale_files(
                    conn, {"/path/to/file1.py"}
                )

                assert removed == 1

                # Verify only file1 remains
                async with conn.execute("SELECT file_path FROM files") as cursor:
                    paths = {row[0] async for row in cursor}
                assert paths == {"/path/to/file1.py"}
            finally:
                await conn.close()


# ---------------------------------------------------------------------------
# Tests for embedding
# ---------------------------------------------------------------------------


class TestEmbedding:
    """Tests for embedding generation."""

    def test_embed_text_returns_numpy_array(self):
        # Mock the sentence transformer model
        mock_model = mock.Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        with mock.patch.object(folderembed, "_embedding_model", mock_model):
            with mock.patch.object(
                folderembed, "_get_embedding_model", return_value=mock_model
            ):
                result = folderembed._embed_text("test text")

                assert isinstance(result, np.ndarray)
                assert result.dtype == np.float32
                mock_model.encode.assert_called_once()

    def test_embed_texts_batch_returns_list(self):
        mock_model = mock.Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        with mock.patch.object(
            folderembed, "_get_embedding_model", return_value=mock_model
        ):
            result = folderembed._embed_texts_batch(["text1", "text2"])

            assert len(result) == 2
            assert all(isinstance(r, np.ndarray) for r in result)

    def test_embed_texts_batch_empty_list(self):
        result = folderembed._embed_texts_batch([])
        assert result == []

    def test_get_embedding_model_caches(self):
        # Reset the cached model
        folderembed._embedding_model = None

        mock_st = mock.Mock()
        mock_model = mock.Mock()
        mock_st.return_value = mock_model

        with mock.patch.dict(
            "sys.modules",
            {"sentence_transformers": mock.Mock(SentenceTransformer=mock_st)},
        ):
            with mock.patch("axono.config.EMBEDDING_MODEL", ""):
                # First call should create the model
                model1 = folderembed._get_embedding_model()
                # Second call should return cached
                model2 = folderembed._get_embedding_model()

                assert model1 is model2

        # Reset for other tests
        folderembed._embedding_model = None


# ---------------------------------------------------------------------------
# Tests for FAISS index
# ---------------------------------------------------------------------------


class TestFaissIndex:
    """Tests for FAISS index management."""

    def test_get_faiss_index_path(self, tmp_path):
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "embeddings.db",
        ):
            path = folderembed._get_faiss_index_path()
            assert path == tmp_path / "embeddings.faiss"

    def test_load_faiss_index_returns_none_when_missing(self, tmp_path):
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "embeddings.db",
        ):
            index, ids = folderembed._load_faiss_index()
            assert index is None
            assert ids == []

    @pytest.mark.asyncio
    async def test_rebuild_faiss_index(self, tmp_path):
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            conn = await folderembed._init_db()
            try:
                # Insert a file and chunks with embeddings
                await conn.execute(
                    "INSERT INTO files (file_path, file_md5) VALUES (?, ?)",
                    ("/path/to/file.py", "abc123"),
                )
                embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
                await conn.execute(
                    "INSERT INTO chunks (file_path, symbol_name, symbol_type, "
                    "line_start, line_end, source_code, embedded_text, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "/path/to/file.py",
                        "func",
                        "function",
                        1,
                        5,
                        "code",
                        "text",
                        embedding.tobytes(),
                    ),
                )
                await conn.commit()

                # Rebuild index
                index, ids = await folderembed._rebuild_faiss_index(conn)

                assert index is not None
                assert len(ids) == 1

                # Verify index was saved
                index_path = tmp_path / "test.faiss"
                ids_path = tmp_path / "test.ids.json"
                assert index_path.exists()
                assert ids_path.exists()
            finally:
                await conn.close()


# ---------------------------------------------------------------------------
# Tests for embed_folder (integration)
# ---------------------------------------------------------------------------


class TestEmbedFolder:
    """Integration tests for embed_folder."""

    @pytest.mark.asyncio
    async def test_embed_folder_yields_events(self, temp_folder, tmp_path):
        events = []

        # Mock the embedding batch function
        mock_embeddings = [np.array([0.1] * 384, dtype=np.float32) for _ in range(10)]

        with mock.patch(
            "axono.config.get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            with mock.patch.object(
                folderembed, "_embed_texts_batch", return_value=mock_embeddings
            ):
                async for event_type, data in folderembed.embed_folder(
                    str(temp_folder)
                ):
                    events.append((event_type, data))

        # Check expected events
        event_types = [e[0] for e in events]
        assert "start" in event_types
        assert "complete" in event_types
        assert any(t in event_types for t in ["file_indexed", "file_skipped"])

    @pytest.mark.asyncio
    async def test_embed_folder_skips_unchanged_files(self, temp_folder, tmp_path):
        def mock_batch(texts):
            return [np.array([0.1] * 384, dtype=np.float32) for _ in texts]

        with mock.patch(
            "axono.config.get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            with mock.patch.object(
                folderembed, "_embed_texts_batch", side_effect=mock_batch
            ):
                # First run
                events1 = []
                async for event_type, data in folderembed.embed_folder(
                    str(temp_folder)
                ):
                    events1.append((event_type, data))

                # Second run - files should be skipped
                events2 = []
                async for event_type, data in folderembed.embed_folder(
                    str(temp_folder)
                ):
                    events2.append((event_type, data))

        # Second run should have more skipped files
        skipped1 = sum(1 for e, _ in events1 if e == "file_skipped")
        skipped2 = sum(1 for e, _ in events2 if e == "file_skipped")
        assert skipped2 > skipped1

    @pytest.mark.asyncio
    async def test_embed_folder_handles_embedding_errors(self, temp_folder, tmp_path):
        with mock.patch(
            "axono.config.get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            with mock.patch.object(
                folderembed,
                "_embed_texts_batch",
                side_effect=Exception("Embedding error"),
            ):
                events = []
                async for event_type, data in folderembed.embed_folder(
                    str(temp_folder)
                ):
                    events.append((event_type, data))

        # Should have error events
        error_events = [e for e in events if e[0] == "file_error"]
        assert len(error_events) > 0


# ---------------------------------------------------------------------------
# Tests for query_similar
# ---------------------------------------------------------------------------


class TestQuerySimilar:
    """Tests for query_similar function."""

    @pytest.mark.asyncio
    async def test_query_similar_returns_empty_when_no_index(self, tmp_path):
        with mock.patch(
            "axono.config.get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            results = await folderembed.query_similar("test query")
            assert results == []

    @pytest.mark.asyncio
    async def test_query_similar_returns_chunks(self, temp_folder, tmp_path):
        def mock_batch(texts):
            return [np.array([0.1] * 384, dtype=np.float32) for _ in texts]

        mock_single = np.array([0.1] * 384, dtype=np.float32)

        with mock.patch(
            "axono.config.get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            with mock.patch.object(
                folderembed, "_embed_texts_batch", side_effect=mock_batch
            ):
                # Index files first
                async for _ in folderembed.embed_folder(str(temp_folder)):
                    pass

            with mock.patch.object(
                folderembed, "_embed_text", return_value=mock_single
            ):
                # Query
                results = await folderembed.query_similar("hello function", top_k=5)

        assert len(results) > 0
        assert all(isinstance(r, Chunk) for r in results)
        assert all(r.source_code for r in results)


# ---------------------------------------------------------------------------
# Tests for optional dependency handling
# ---------------------------------------------------------------------------


class TestOptionalDependencies:
    """Tests for graceful handling of missing dependencies."""

    def test_has_faiss_flag(self):
        # Just verify the flag exists and is a bool
        assert isinstance(folderembed._HAS_FAISS, bool)

    def test_has_tree_sitter_flag(self):
        assert isinstance(folderembed._HAS_TREE_SITTER, bool)

    def test_has_pathspec_flag(self):
        assert isinstance(folderembed._HAS_PATHSPEC, bool)


# ---------------------------------------------------------------------------
# Tests for additional edge cases (for 100% coverage)
# ---------------------------------------------------------------------------


class TestAdditionalEdgeCases:
    """Tests for edge cases to achieve 100% coverage."""

    def test_get_parser_typescript(self, tmp_path):
        """Test TypeScript parser."""
        ts_file = tmp_path / "app.ts"
        ts_file.write_text("function foo(): void { console.log('hi'); }")
        chunks = _parse_file(str(ts_file), "test_md5")
        assert len(chunks) > 0

    def test_get_parser_tsx(self, tmp_path):
        """Test TSX parser."""
        tsx_file = tmp_path / "component.tsx"
        tsx_file.write_text("function Component() { return <div>Hello</div>; }")
        chunks = _parse_file(str(tsx_file), "test_md5")
        assert len(chunks) > 0

    def test_get_parser_go(self, tmp_path):
        """Test Go parser - falls back to line chunking."""
        go_file = tmp_path / "main.go"
        go_file.write_text('package main\n\nfunc main() {\n\tfmt.Println("hi")\n}')
        chunks = _parse_file(str(go_file), "test_md5")
        # Go falls back to line chunking since we don't have specific extraction
        assert len(chunks) > 0
        assert chunks[0].symbol_type == "file"

    def test_get_parser_rust(self, tmp_path):
        """Test Rust parser - falls back to line chunking."""
        rs_file = tmp_path / "main.rs"
        rs_file.write_text('fn main() {\n    println!("hi");\n}')
        chunks = _parse_file(str(rs_file), "test_md5")
        assert len(chunks) > 0
        assert chunks[0].symbol_type == "file"

    def test_get_parser_unknown_language(self):
        """Test that unknown language returns None."""
        parser = folderembed._get_parser("cobol")
        assert parser is None

    def test_get_parser_caches(self):
        """Test that parsers are cached."""
        folderembed._parsers.clear()
        parser1 = folderembed._get_parser("python")
        parser2 = folderembed._get_parser("python")
        assert parser1 is parser2

    def test_python_class_with_decorators(self, tmp_path):
        """Test Python class with decorators."""
        py_file = tmp_path / "decorated.py"
        py_file.write_text('''@dataclass
@frozen
class MyClass:
    """A decorated class."""
    value: int
''')
        chunks = _parse_file(str(py_file), "test_md5")
        class_chunk = next(c for c in chunks if c.symbol_type == "class")
        assert "@dataclass" in class_chunk.metadata.decorators
        assert "@frozen" in class_chunk.metadata.decorators
        assert "@dataclass" in class_chunk.source_code

    @pytest.mark.asyncio
    async def test_save_chunks_empty_list(self, tmp_path):
        """Test _save_chunks with empty list does nothing."""
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            conn = await folderembed._init_db()
            try:
                # This should return early without error
                await folderembed._save_chunks(conn, [])
                # Verify no files were inserted
                async with conn.execute("SELECT COUNT(*) FROM files") as cursor:
                    row = await cursor.fetchone()
                    assert row[0] == 0
            finally:
                await conn.close()

    @pytest.mark.asyncio
    async def test_save_chunks_replaces_existing(self, tmp_path):
        """Test that _save_chunks removes old chunks when re-saving a file."""
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            conn = await folderembed._init_db()
            try:
                # Save initial chunks
                chunks1 = [
                    Chunk(
                        file_path="/path/to/file.py",
                        file_md5="abc123",
                        symbol_name="func1",
                        symbol_type="function",
                        line_start=1,
                        line_end=5,
                        source_code="def func1(): pass",
                        embedded_text="text1",
                        embedding=np.array([0.1, 0.2], dtype=np.float32),
                    ),
                    Chunk(
                        file_path="/path/to/file.py",
                        file_md5="abc123",
                        symbol_name="func2",
                        symbol_type="function",
                        line_start=6,
                        line_end=10,
                        source_code="def func2(): pass",
                        embedded_text="text2",
                        embedding=np.array([0.3, 0.4], dtype=np.float32),
                    ),
                ]
                await folderembed._save_chunks(conn, chunks1)

                # Verify 2 chunks exist
                async with conn.execute("SELECT COUNT(*) FROM chunks") as cursor:
                    row = await cursor.fetchone()
                    assert row[0] == 2

                # Save new chunks for same file (simulating file modification)
                chunks2 = [
                    Chunk(
                        file_path="/path/to/file.py",
                        file_md5="def456",
                        symbol_name="new_func",
                        symbol_type="function",
                        line_start=1,
                        line_end=3,
                        source_code="def new_func(): pass",
                        embedded_text="new text",
                        embedding=np.array([0.5, 0.6], dtype=np.float32),
                    ),
                ]
                await folderembed._save_chunks(conn, chunks2)

                # Verify only 1 chunk exists now (old ones removed)
                async with conn.execute("SELECT COUNT(*) FROM chunks") as cursor:
                    row = await cursor.fetchone()
                    assert row[0] == 1

                # Verify it's the new chunk
                async with conn.execute("SELECT symbol_name FROM chunks") as cursor:
                    row = await cursor.fetchone()
                    assert row[0] == "new_func"

                # Verify file MD5 was updated
                async with conn.execute("SELECT file_md5 FROM files") as cursor:
                    row = await cursor.fetchone()
                    assert row[0] == "def456"
            finally:
                await conn.close()

    @pytest.mark.asyncio
    async def test_load_all_embeddings_empty_db(self, tmp_path):
        """Test _load_all_embeddings with empty database."""
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            conn = await folderembed._init_db()
            try:
                ids, embeddings = await folderembed._load_all_embeddings(conn)
                assert ids == []
                assert embeddings is None
            finally:
                await conn.close()

    @pytest.mark.asyncio
    async def test_load_all_embeddings_empty_embeddings(self, tmp_path):
        """Test _load_all_embeddings when embeddings are empty bytes."""
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            conn = await folderembed._init_db()
            try:
                # Insert file and chunk with empty embedding
                await conn.execute(
                    "INSERT INTO files (file_path, file_md5) VALUES (?, ?)",
                    ("/path/file.py", "abc123"),
                )
                await conn.execute(
                    "INSERT INTO chunks (file_path, symbol_name, symbol_type, "
                    "line_start, line_end, source_code, embedded_text, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("/path/file.py", "func", "function", 1, 5, "code", "text", b""),
                )
                await conn.commit()

                ids, embeddings = await folderembed._load_all_embeddings(conn)
                # With empty embeddings, dim will be 0
                assert ids == []
                assert embeddings is None
            finally:
                await conn.close()

    def test_load_faiss_index_error_handling(self, tmp_path):
        """Test _load_faiss_index handles corrupted files."""
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            # Create corrupted index files
            index_path = tmp_path / "test.faiss"
            ids_path = tmp_path / "test.ids.json"
            index_path.write_text("not a valid faiss index")
            ids_path.write_text("[1, 2, 3]")

            index, ids = folderembed._load_faiss_index()
            assert index is None
            assert ids == []

    @pytest.mark.asyncio
    async def test_embed_folder_file_read_error(self, tmp_path):
        """Test embed_folder handles file read errors."""
        # Create a file then make it unreadable
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        def mock_md5(path):
            return ""  # Simulate read error

        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "embed.db",
        ):
            with mock.patch.object(folderembed, "_compute_md5", side_effect=mock_md5):
                events = []
                async for event_type, data in folderembed.embed_folder(str(tmp_path)):
                    events.append((event_type, data))

        error_events = [e for e in events if e[0] == "file_error"]
        assert len(error_events) > 0

    @pytest.mark.asyncio
    async def test_embed_folder_file_changed(self, tmp_path):
        """Test embed_folder re-indexes changed files."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        def mock_batch(texts):
            return [np.array([0.1] * 384, dtype=np.float32) for _ in texts]

        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "embed.db",
        ):
            with mock.patch.object(
                folderembed, "_embed_texts_batch", side_effect=mock_batch
            ):
                # First index
                async for _ in folderembed.embed_folder(str(tmp_path)):
                    pass

                # Modify file
                test_file.write_text("def foo(): pass\ndef bar(): pass")

                # Re-index
                events = []
                async for event_type, data in folderembed.embed_folder(str(tmp_path)):
                    events.append((event_type, data))

        # Should have indexed the changed file
        indexed = [e for e in events if e[0] == "file_indexed"]
        assert len(indexed) > 0

    @pytest.mark.asyncio
    async def test_embed_folder_parse_error(self, tmp_path):
        """Test embed_folder handles parse errors gracefully."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "embed.db",
        ):
            with mock.patch.object(
                folderembed, "_parse_file", side_effect=Exception("Parse error")
            ):
                events = []
                async for event_type, data in folderembed.embed_folder(str(tmp_path)):
                    events.append((event_type, data))

        error_events = [e for e in events if e[0] == "file_error"]
        assert len(error_events) > 0

    @pytest.mark.asyncio
    async def test_embed_folder_no_chunks(self, tmp_path):
        """Test embed_folder handles files with no chunks."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "embed.db",
        ):
            with mock.patch.object(folderembed, "_parse_file", return_value=[]):
                events = []
                async for event_type, data in folderembed.embed_folder(str(tmp_path)):
                    events.append((event_type, data))

        skipped = [
            e
            for e in events
            if e[0] == "file_skipped" and e[1].get("reason") == "no chunks"
        ]
        assert len(skipped) > 0

    @pytest.mark.asyncio
    async def test_embed_folder_cleanup_event(self, tmp_path):
        """Test embed_folder yields cleanup event when files are removed."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        def mock_batch(texts):
            return [np.array([0.1] * 384, dtype=np.float32) for _ in texts]

        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "embed.db",
        ):
            with mock.patch.object(
                folderembed, "_embed_texts_batch", side_effect=mock_batch
            ):
                # First index
                async for _ in folderembed.embed_folder(str(tmp_path)):
                    pass

                # Remove file
                test_file.unlink()

                # Re-index - should trigger cleanup
                events = []
                async for event_type, data in folderembed.embed_folder(str(tmp_path)):
                    events.append((event_type, data))

        cleanup = [e for e in events if e[0] == "cleanup"]
        assert len(cleanup) == 1
        assert cleanup[0][1]["removed_files"] == 1

    @pytest.mark.asyncio
    async def test_query_similar_no_faiss(self):
        """Test query_similar returns empty when FAISS not available."""
        with mock.patch.object(folderembed, "_HAS_FAISS", False):
            results = await folderembed.query_similar("test query")
            assert results == []

    @pytest.mark.asyncio
    async def test_query_similar_no_results(self, tmp_path):
        """Test query_similar returns empty when search yields no results."""
        mock_index = mock.Mock()
        mock_index.search.return_value = (
            np.array([[0.5]]),
            np.array([[100]]),
        )  # 100 > len(ids)

        embedding = np.array([0.1] * 384, dtype=np.float32).reshape(1, -1)

        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            with mock.patch.object(
                folderembed, "_load_faiss_index", return_value=(mock_index, [1, 2, 3])
            ):
                with mock.patch.object(
                    folderembed, "_embed_text", return_value=embedding.flatten()
                ):
                    # Mock faiss.normalize_L2 to avoid the actual call
                    with mock.patch.object(folderembed.faiss, "normalize_L2"):
                        results = await folderembed.query_similar("test query")
                        # Result id 100 > len(ids)=3, so filtered out
                        assert results == []

    @pytest.mark.asyncio
    async def test_query_similar_index_none(self):
        """Test query_similar returns empty when _load_faiss_index returns None."""
        with mock.patch.object(
            folderembed, "_load_faiss_index", return_value=(None, [])
        ):
            results = await folderembed.query_similar("test query")
            assert results == []

    def test_save_faiss_index_no_faiss(self):
        """Test _save_faiss_index does nothing when FAISS not available."""
        with mock.patch.object(folderembed, "_HAS_FAISS", False):
            # Should not raise
            folderembed._save_faiss_index(None, [])

    def test_save_faiss_index_write_error(self, tmp_path, capsys):
        """Test _save_faiss_index handles write errors gracefully."""
        mock_index = mock.Mock()

        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            with mock.patch.object(
                folderembed.faiss, "write_index", side_effect=Exception("Write error")
            ):
                # Should not raise, just print warning
                folderembed._save_faiss_index(mock_index, [1, 2, 3])

        captured = capsys.readouterr()
        assert "Warning" in captured.out

    @pytest.mark.asyncio
    async def test_rebuild_faiss_index_no_faiss(self, tmp_path):
        """Test _rebuild_faiss_index returns None when FAISS not available."""
        with mock.patch.object(
            folderembed.config,
            "get_embedding_db_path",
            return_value=tmp_path / "test.db",
        ):
            conn = await folderembed._init_db()
            try:
                with mock.patch.object(folderembed, "_HAS_FAISS", False):
                    index, ids = await folderembed._rebuild_faiss_index(conn)
                    assert index is None
                    assert ids == []
            finally:
                await conn.close()

    def test_load_faiss_index_no_faiss(self):
        """Test _load_faiss_index returns None when FAISS not available."""
        with mock.patch.object(folderembed, "_HAS_FAISS", False):
            index, ids = folderembed._load_faiss_index()
            assert index is None
            assert ids == []

    def test_get_parser_no_tree_sitter(self):
        """Test _get_parser returns None when tree-sitter not available."""
        with mock.patch.object(folderembed, "_HAS_TREE_SITTER", False):
            parser = folderembed._get_parser("python")
            assert parser is None

    def test_load_gitignore_no_pathspec(self, tmp_path):
        """Test _load_gitignore returns None when pathspec not available."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc")
        with mock.patch.object(folderembed, "_HAS_PATHSPEC", False):
            result = folderembed._load_gitignore(str(tmp_path))
            assert result is None


# ---------------------------------------------------------------------------
# Tests for import-time fallbacks (requires module reload)
# ---------------------------------------------------------------------------


class TestImportFallbacks:
    """Tests for import-time fallback handling when dependencies are missing.

    These tests reload the module with mocked imports to cover the except
    ImportError blocks that execute during module import.
    """

    def test_faiss_import_error(self):
        """Test that _HAS_FAISS is False when faiss import fails."""
        import importlib
        import sys

        # Save original module state
        original_module = sys.modules.get("axono.folderembed")
        original_faiss = sys.modules.get("faiss")

        try:
            # Remove the module and faiss from cache
            if "axono.folderembed" in sys.modules:
                del sys.modules["axono.folderembed"]
            # Make faiss import fail
            sys.modules["faiss"] = None  # This causes import to get None, not fail

            # Need to actually make the import raise ImportError
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "faiss":
                    raise ImportError("mocked faiss import error")
                return original_import(name, *args, **kwargs)

            with mock.patch.object(builtins, "__import__", side_effect=mock_import):
                # This should catch the ImportError and set _HAS_FAISS = False
                import axono.folderembed as fe

                assert fe._HAS_FAISS is False
        finally:
            # Restore original state
            if original_faiss is not None:
                sys.modules["faiss"] = original_faiss
            if original_module is not None:
                sys.modules["axono.folderembed"] = original_module

    def test_tree_sitter_import_error(self):
        """Test that _HAS_TREE_SITTER is False when tree-sitter imports fail."""
        import builtins
        import sys

        original_module = sys.modules.get("axono.folderembed")
        original_import = builtins.__import__

        tree_sitter_modules = [
            "tree_sitter_python",
            "tree_sitter_javascript",
            "tree_sitter_typescript",
            "tree_sitter_go",
            "tree_sitter_rust",
            "tree_sitter",
        ]

        def mock_import(name, *args, **kwargs):
            if name in tree_sitter_modules:
                raise ImportError(f"mocked {name} import error")
            return original_import(name, *args, **kwargs)

        try:
            if "axono.folderembed" in sys.modules:
                del sys.modules["axono.folderembed"]

            with mock.patch.object(builtins, "__import__", side_effect=mock_import):
                import axono.folderembed as fe

                assert fe._HAS_TREE_SITTER is False
        finally:
            if original_module is not None:
                sys.modules["axono.folderembed"] = original_module

    def test_pathspec_import_error(self):
        """Test that _HAS_PATHSPEC is False when pathspec import fails."""
        import builtins
        import sys

        original_module = sys.modules.get("axono.folderembed")
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pathspec":
                raise ImportError("mocked pathspec import error")
            return original_import(name, *args, **kwargs)

        try:
            if "axono.folderembed" in sys.modules:
                del sys.modules["axono.folderembed"]

            with mock.patch.object(builtins, "__import__", side_effect=mock_import):
                import axono.folderembed as fe

                assert fe._HAS_PATHSPEC is False
        finally:
            if original_module is not None:
                sys.modules["axono.folderembed"] = original_module


# ---------------------------------------------------------------------------
# Tests for file watcher functionality
# ---------------------------------------------------------------------------


class TestShouldWatchFile:
    """Tests for _should_watch_file function."""

    def test_skips_hidden_files(self, tmp_path):
        hidden = tmp_path / ".hidden.py"
        hidden.write_text("# hidden")
        result = _should_watch_file(str(hidden), str(tmp_path), None)
        assert result is False

    def test_skips_files_with_excluded_extensions(self, tmp_path):
        for ext in [".pyc", ".lock", ".map", ".png"]:
            f = tmp_path / f"file{ext}"
            f.write_bytes(b"content")
            result = _should_watch_file(str(f), str(tmp_path), None)
            assert result is False, f"Should skip {ext}"

    def test_skips_symlinks(self, tmp_path):
        real_file = tmp_path / "real.py"
        real_file.write_text("# real")
        link_file = tmp_path / "link.py"
        link_file.symlink_to(real_file)
        result = _should_watch_file(str(link_file), str(tmp_path), None)
        assert result is False

    def test_skips_files_in_skip_directories(self, tmp_path):
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        f = node_modules / "package.js"
        f.write_text("// js")
        result = _should_watch_file(str(f), str(tmp_path), None)
        assert result is False

    def test_skips_files_in_hidden_directories(self, tmp_path):
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        f = hidden_dir / "file.py"
        f.write_text("# py")
        result = _should_watch_file(str(f), str(tmp_path), None)
        assert result is False

    def test_skips_binary_files(self, tmp_path):
        binary_file = tmp_path / "binary.dat"
        binary_file.write_bytes(b"\x00\x01\x02")
        result = _should_watch_file(str(binary_file), str(tmp_path), None)
        assert result is False

    def test_respects_gitignore(self, tmp_path):
        # Create gitignore
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.txt\n")
        spec = folderembed._load_gitignore(str(tmp_path))

        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("content")
        result = _should_watch_file(str(txt_file), str(tmp_path), spec)
        assert result is False

    def test_accepts_normal_source_files(self, tmp_path):
        py_file = tmp_path / "main.py"
        py_file.write_text("# python")
        result = _should_watch_file(str(py_file), str(tmp_path), None)
        assert result is True

    def test_handles_file_outside_folder(self, tmp_path):
        # Path not relative to folder
        result = _should_watch_file("/some/other/path.py", str(tmp_path), None)
        assert result is False


class TestFileChangeEvent:
    """Tests for FileChangeEvent dataclass."""

    def test_creates_event_with_defaults(self):
        event = FileChangeEvent(
            event_type="created",
            file_path="/path/to/file.py",
        )
        assert event.event_type == "created"
        assert event.file_path == "/path/to/file.py"
        assert event.dest_path is None

    def test_creates_moved_event_with_dest(self):
        event = FileChangeEvent(
            event_type="moved",
            file_path="/path/to/old.py",
            dest_path="/path/to/new.py",
        )
        assert event.event_type == "moved"
        assert event.dest_path == "/path/to/new.py"


class TestFolderEventHandler:
    """Tests for _FolderEventHandler class."""

    def test_handles_created_event(self, tmp_path):
        import asyncio

        queue = asyncio.Queue()
        handler = folderembed._FolderEventHandler(str(tmp_path), None, queue)

        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("# test")

        # Create mock event
        mock_event = mock.Mock()
        mock_event.is_directory = False
        mock_event.src_path = str(test_file)

        handler.on_created(mock_event)

        # Check event was queued
        event = queue.get_nowait()
        assert event.event_type == "created"
        assert event.file_path == str(test_file)

    def test_handles_modified_event(self, tmp_path):
        import asyncio

        queue = asyncio.Queue()
        handler = folderembed._FolderEventHandler(str(tmp_path), None, queue)

        test_file = tmp_path / "test.py"
        test_file.write_text("# test")

        mock_event = mock.Mock()
        mock_event.is_directory = False
        mock_event.src_path = str(test_file)

        handler.on_modified(mock_event)

        event = queue.get_nowait()
        assert event.event_type == "modified"

    def test_handles_deleted_event(self, tmp_path):
        import asyncio

        queue = asyncio.Queue()
        handler = folderembed._FolderEventHandler(str(tmp_path), None, queue)

        mock_event = mock.Mock()
        mock_event.is_directory = False
        mock_event.src_path = str(tmp_path / "test.py")

        handler.on_deleted(mock_event)

        event = queue.get_nowait()
        assert event.event_type == "deleted"

    def test_handles_moved_event(self, tmp_path):
        import asyncio

        queue = asyncio.Queue()
        handler = folderembed._FolderEventHandler(str(tmp_path), None, queue)

        test_file = tmp_path / "test.py"
        test_file.write_text("# test")
        dest_file = tmp_path / "test_new.py"

        mock_event = mock.Mock()
        mock_event.is_directory = False
        mock_event.src_path = str(test_file)
        mock_event.dest_path = str(dest_file)

        handler.on_moved(mock_event)

        event = queue.get_nowait()
        assert event.event_type == "moved"
        assert event.dest_path == str(dest_file)

    def test_ignores_directory_events(self, tmp_path):
        import asyncio

        queue = asyncio.Queue()
        handler = folderembed._FolderEventHandler(str(tmp_path), None, queue)

        mock_event = mock.Mock()
        mock_event.is_directory = True
        mock_event.src_path = str(tmp_path / "subdir")

        handler.on_created(mock_event)

        assert queue.empty()

    def test_ignores_excluded_files(self, tmp_path):
        import asyncio

        queue = asyncio.Queue()
        handler = folderembed._FolderEventHandler(str(tmp_path), None, queue)

        mock_event = mock.Mock()
        mock_event.is_directory = False
        mock_event.src_path = str(tmp_path / ".hidden.py")

        handler.on_created(mock_event)

        assert queue.empty()

    def test_moved_to_valid_location_creates_event(self, tmp_path):
        """Test that moving from excluded to valid location triggers created event."""
        import asyncio

        queue = asyncio.Queue()
        handler = folderembed._FolderEventHandler(str(tmp_path), None, queue)

        # Create dest file for existence check
        dest_file = tmp_path / "valid.py"
        dest_file.write_text("# valid")

        mock_event = mock.Mock()
        mock_event.is_directory = False
        mock_event.src_path = str(tmp_path / ".hidden.py")  # Excluded source
        mock_event.dest_path = str(dest_file)  # Valid destination

        handler.on_moved(mock_event)

        # Should create a "created" event for the valid destination
        event = queue.get_nowait()
        assert event.event_type == "created"
        assert event.file_path == str(dest_file)


class TestFolderEventHandlerLogSkip:
    """Test log file skip path in _FolderEventHandler (line 1282)."""

    def test_skips_log_directory_file(self, tmp_path):
        import asyncio

        queue = asyncio.Queue()
        handler = folderembed._FolderEventHandler(str(tmp_path), None, queue)

        mock_event = mock.Mock()
        mock_event.is_directory = False
        mock_event.src_path = str(tmp_path / "log" / "app.txt")

        handler.on_modified(mock_event)

        assert queue.empty()

    def test_skips_dot_log_file(self, tmp_path):
        import asyncio

        queue = asyncio.Queue()
        handler = folderembed._FolderEventHandler(str(tmp_path), None, queue)

        mock_event = mock.Mock()
        mock_event.is_directory = False
        mock_event.src_path = str(tmp_path / "debug.log")

        handler.on_created(mock_event)

        assert queue.empty()


class TestEmbeddingModelLoadFailure:
    """Test _get_embedding_model failure path (lines 744-745)."""

    def test_raises_runtime_error_on_failure(self):
        folderembed._embedding_model = None

        mock_st = mock.Mock(side_effect=OSError("model not found"))

        with mock.patch.dict(
            "sys.modules",
            {"sentence_transformers": mock.Mock(SentenceTransformer=mock_st)},
        ):
            with mock.patch("axono.config.EMBEDDING_MODEL", ""):
                with pytest.raises(
                    RuntimeError, match="Failed to load embedding model"
                ):
                    folderembed._get_embedding_model()

        folderembed._embedding_model = None


class TestWatchFolder:
    """Tests for watch_folder async generator."""

    @pytest.mark.asyncio
    async def test_raises_when_watchdog_not_installed(self, tmp_path):
        with mock.patch.object(folderembed, "_HAS_WATCHDOG", False):
            with pytest.raises(RuntimeError, match="watchdog is not installed"):
                async for _ in folderembed.watch_folder(str(tmp_path)):
                    pass  # pragma: no cover

    @pytest.mark.asyncio
    async def test_raises_when_not_a_directory(self, tmp_path):
        test_file = tmp_path / "file.py"
        test_file.write_text("# py")

        with pytest.raises(ValueError, match="Not a directory"):
            async for _ in folderembed.watch_folder(str(test_file)):
                pass  # pragma: no cover

    @pytest.mark.asyncio
    async def test_yields_events_on_file_changes(self, tmp_path):
        import asyncio

        events_received = []

        async def collect_events():
            async for event in folderembed.watch_folder(str(tmp_path)):
                events_received.append(event)
                if len(events_received) >= 1:
                    break

        # Start watcher in background
        task = asyncio.create_task(collect_events())

        # Give watcher time to start
        await asyncio.sleep(0.5)

        # Create a file
        test_file = tmp_path / "new_file.py"
        test_file.write_text("# new file")

        # Wait for event
        await asyncio.wait_for(task, timeout=5.0)

        # Should have received at least one event
        assert len(events_received) >= 1
        assert events_received[0].event_type in ("created", "modified")

    @pytest.mark.asyncio
    async def test_yields_events_on_file_delete(self, tmp_path):
        import asyncio

        # Create a file first
        test_file = tmp_path / "to_delete.py"
        test_file.write_text("# will be deleted")

        events_received = []

        async def collect_events():
            async for event in folderembed.watch_folder(str(tmp_path)):
                events_received.append(event)
                if event.event_type == "deleted":
                    break

        # Start watcher in background
        task = asyncio.create_task(collect_events())

        # Give watcher time to start
        await asyncio.sleep(0.5)

        # Delete the file
        test_file.unlink()

        # Wait for event
        await asyncio.wait_for(task, timeout=5.0)

        # Should have received delete event
        delete_events = [e for e in events_received if e.event_type == "deleted"]
        assert len(delete_events) >= 1

    @pytest.mark.asyncio
    async def test_yields_events_on_file_move(self, tmp_path):
        import asyncio

        # Create a file first
        test_file = tmp_path / "to_move.py"
        test_file.write_text("# will be moved")
        dest_file = tmp_path / "moved.py"

        events_received = []

        async def collect_events():
            async for event in folderembed.watch_folder(str(tmp_path)):
                events_received.append(event)
                if event.event_type == "moved":
                    break

        # Start watcher in background
        task = asyncio.create_task(collect_events())

        # Give watcher time to start
        await asyncio.sleep(0.5)

        # Move/rename the file
        test_file.rename(dest_file)

        # Wait for event
        await asyncio.wait_for(task, timeout=5.0)

        # Should have received move event
        move_events = [e for e in events_received if e.event_type == "moved"]
        assert len(move_events) >= 1

    @pytest.mark.asyncio
    async def test_cancellation_stops_watcher(self, tmp_path):
        import asyncio

        async def run_watcher():
            async for _ in folderembed.watch_folder(str(tmp_path)):
                pass  # pragma: no cover

        task = asyncio.create_task(run_watcher())
        await asyncio.sleep(0.5)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:  # pragma: no cover
            pass

        # Task should be done
        assert task.done()

    @pytest.mark.asyncio
    async def test_watcher_handles_timeout(self, tmp_path):
        """Test that the watcher handles timeout and continues waiting."""
        import asyncio

        events_received = []
        timeout_occurred = False

        async def collect_events():
            nonlocal timeout_occurred
            async for event in folderembed.watch_folder(str(tmp_path)):
                events_received.append(event)
                if len(events_received) >= 1:
                    break

        # Start watcher in background
        task = asyncio.create_task(collect_events())

        # Wait longer than the 1 second timeout to ensure timeout loop is hit
        await asyncio.sleep(1.5)

        # Now create a file
        test_file = tmp_path / "delayed_file.py"
        test_file.write_text("# delayed")

        # Wait for event
        await asyncio.wait_for(task, timeout=5.0)

        # Should have received event after the timeout loop
        assert len(events_received) >= 1


class TestStartWatcher:
    """Tests for start_watcher callback-based function."""

    def test_raises_when_watchdog_not_installed(self, tmp_path):
        with mock.patch.object(folderembed, "_HAS_WATCHDOG", False):
            with pytest.raises(RuntimeError, match="watchdog is not installed"):
                folderembed.start_watcher(str(tmp_path), lambda e: None)

    def test_raises_when_not_a_directory(self, tmp_path):
        test_file = tmp_path / "file.py"
        test_file.write_text("# py")

        with pytest.raises(ValueError, match="Not a directory"):
            folderembed.start_watcher(str(test_file), lambda e: None)

    def test_returns_observer(self, tmp_path):
        events = []
        observer = folderembed.start_watcher(str(tmp_path), events.append)

        try:
            assert observer is not None
            # Observer should be running
            assert observer.is_alive()
        finally:
            observer.stop()
            observer.join(timeout=5.0)

    def test_calls_callback_on_file_change(self, tmp_path):
        import time

        events = []
        observer = folderembed.start_watcher(str(tmp_path), events.append)

        try:
            # Give observer time to start
            time.sleep(0.5)

            # Create a file
            test_file = tmp_path / "callback_test.py"
            test_file.write_text("# test")

            # Wait for event
            time.sleep(1.0)

            # Should have received events
            assert len(events) >= 1
        finally:
            observer.stop()
            observer.join(timeout=5.0)

    def test_ignores_excluded_files(self, tmp_path):
        import time

        events = []
        observer = folderembed.start_watcher(str(tmp_path), events.append)

        try:
            time.sleep(0.5)

            # Create an excluded file
            hidden_file = tmp_path / ".hidden"
            hidden_file.write_text("hidden")

            time.sleep(1.0)

            # Should not have received events for hidden file
            hidden_events = [e for e in events if ".hidden" in e.file_path]
            assert len(hidden_events) == 0
        finally:
            observer.stop()
            observer.join(timeout=5.0)

    def test_moved_to_valid_location_in_callback(self, tmp_path):
        """Test moved event from excluded to valid location in callback mode."""
        import time

        events = []

        # Create the destination file first
        dest_file = tmp_path / "valid.py"
        dest_file.write_text("# valid")

        observer = folderembed.start_watcher(str(tmp_path), events.append)

        try:
            time.sleep(0.5)

            # Create a hidden file then move it to valid location
            hidden = tmp_path / ".temp"
            hidden.write_text("# temp")
            time.sleep(0.2)
            # Simulate move by renaming
            hidden.rename(tmp_path / "renamed.py")

            time.sleep(1.0)

            # Check for events
            assert len(events) >= 1
        finally:
            observer.stop()
            observer.join(timeout=5.0)

    def test_callback_on_file_delete(self, tmp_path):
        """Test that callback receives delete events."""
        import time

        # Create a file first
        test_file = tmp_path / "to_delete.py"
        test_file.write_text("# will be deleted")

        events = []
        observer = folderembed.start_watcher(str(tmp_path), events.append)

        try:
            time.sleep(0.5)

            # Delete the file
            test_file.unlink()

            time.sleep(1.0)

            # Should have received delete event
            delete_events = [e for e in events if e.event_type == "deleted"]
            assert len(delete_events) >= 1
        finally:
            observer.stop()
            observer.join(timeout=5.0)

    def test_callback_on_file_move(self, tmp_path):
        """Test that callback receives move events."""
        import time

        # Create a file first
        test_file = tmp_path / "to_move.py"
        test_file.write_text("# will be moved")

        events = []
        observer = folderembed.start_watcher(str(tmp_path), events.append)

        try:
            time.sleep(0.5)

            # Move the file
            test_file.rename(tmp_path / "moved.py")

            time.sleep(1.0)

            # Should have received move event
            move_events = [e for e in events if e.event_type == "moved"]
            assert len(move_events) >= 1
        finally:
            observer.stop()
            observer.join(timeout=5.0)


class TestStartWatcherLogSkip:
    """Test log file skip in start_watcher CallbackHandler (line 1449)."""

    def test_ignores_log_files(self, tmp_path):
        import time

        events = []
        observer = folderembed.start_watcher(str(tmp_path), events.append)

        try:
            time.sleep(0.5)

            # Create a .log file — should be ignored
            log_file = tmp_path / "debug.log"
            log_file.write_text("log output")

            # Also create a file in a log/ directory — should be ignored
            log_dir = tmp_path / "log"
            log_dir.mkdir(exist_ok=True)
            (log_dir / "app.txt").write_text("log data")

            time.sleep(1.0)

            # No events should have been generated for log files
            log_events = [
                e
                for e in events
                if "debug.log" in e.file_path or "/log/" in e.file_path
            ]
            assert len(log_events) == 0
        finally:
            observer.stop()
            observer.join(timeout=5.0)


class TestWatchdogImportFallback:
    """Tests for watchdog import fallback."""

    def test_watchdog_import_error(self):
        """Test that _HAS_WATCHDOG is False when watchdog import fails."""
        import builtins
        import sys

        original_module = sys.modules.get("axono.folderembed")
        original_import = builtins.__import__

        watchdog_modules = [
            "watchdog",
            "watchdog.events",
            "watchdog.observers",
        ]

        def mock_import(name, *args, **kwargs):
            if any(name.startswith(m) for m in watchdog_modules):
                raise ImportError(f"mocked {name} import error")
            return original_import(name, *args, **kwargs)

        try:
            if "axono.folderembed" in sys.modules:
                del sys.modules["axono.folderembed"]

            with mock.patch.object(builtins, "__import__", side_effect=mock_import):
                import axono.folderembed as fe

                assert fe._HAS_WATCHDOG is False
        finally:
            if original_module is not None:
                sys.modules["axono.folderembed"] = original_module
