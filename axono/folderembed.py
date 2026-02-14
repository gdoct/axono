"""Semantic code search via embeddings with tree-sitter AST parsing.

This module provides functionality to:
1. Index a folder by parsing source files with tree-sitter
2. Extract symbols (functions, classes, methods) with metadata
3. Generate embeddings via LM Studio
4. Store embeddings in FAISS for fast similarity search
5. Query the index to find relevant code

Steps:
    1. Collect filenames to scan, respecting .gitignore if present.
       Exclude binary files (detect via null bytes in first 8KB),
       log files, and build artifacts.
    2. Compute MD5 for each file. Skip files already in DB with matching hash.
    3. Parse files with tree-sitter to extract AST. When a file changes,
       delete all its old chunks before re-embedding.
    4. Embed chunks (functions/classes/methods) using enriched text
       (path + imports + docstring + code). For files without a grammar,
       chunk by lines. Store both source_code (raw) and embedded_text
       alongside the embedding vector.
    5. Cleanup: remove DB entries for files that no longer exist.

Chunk schema:
    - file_path, file_md5: identity and cache key
    - symbol_name, symbol_type: function/class/method/file
    - line_start, line_end: location
    - source_code: raw code returned to LLM
    - embedded_text: enriched text used for embedding
    - embedding: vector
    - imports, calls, decorators, parent_class: structural metadata
      stored in SQLite for graph queries (not in vector DB)

Also store a file-level chunk per file (imports + signatures + docstring)
for quick overviews.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator

import aiosqlite
import numpy as np

from axono import config

if TYPE_CHECKING:
    import faiss as faiss_module
    import tree_sitter

# Optional dependencies with graceful fallback
try:
    import faiss

    _HAS_FAISS = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    _HAS_FAISS = False

try:
    import tree_sitter_go
    import tree_sitter_javascript
    import tree_sitter_python
    import tree_sitter_rust
    import tree_sitter_typescript
    from tree_sitter import Language, Parser

    _HAS_TREE_SITTER = True
except ImportError:
    tree_sitter_python = None  # type: ignore[assignment]
    tree_sitter_javascript = None  # type: ignore[assignment]
    tree_sitter_typescript = None  # type: ignore[assignment]
    tree_sitter_go = None  # type: ignore[assignment]
    tree_sitter_rust = None  # type: ignore[assignment]
    Language = None  # type: ignore[assignment, misc]
    Parser = None  # type: ignore[assignment, misc]
    _HAS_TREE_SITTER = False

try:
    import pathspec

    _HAS_PATHSPEC = True
except ImportError:
    pathspec = None  # type: ignore[assignment]
    _HAS_PATHSPEC = False

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    _HAS_WATCHDOG = True
except ImportError:
    FileSystemEvent = None  # type: ignore[assignment, misc]
    FileSystemEventHandler = None  # type: ignore[assignment, misc]
    Observer = None  # type: ignore[assignment, misc]
    _HAS_WATCHDOG = False


# ── Data Classes ─────────────────────────────────────────────────────────


@dataclass
class ChunkMetadata:
    """Structural metadata extracted from AST."""

    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    parent_class: str | None = None


@dataclass
class Chunk:
    """A code chunk with its embedding and metadata."""

    file_path: str
    file_md5: str
    symbol_name: str | None
    symbol_type: str  # function, class, method, file
    line_start: int
    line_end: int
    source_code: str
    embedded_text: str
    embedding: np.ndarray | None = None
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    chunk_id: int | None = None  # Set after DB insertion


# ── File Collection ──────────────────────────────────────────────────────

# Maximum number of files to index
MAX_FILES = 5000

# Default directories to skip (project-level)
_DEFAULT_EXCLUDES = {
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    "dist",
    "build",
    ".next",
    "target",
    "bin",
    "obj",
    ".idea",
    ".vscode",
    "vendor",
    ".tox",
    "coverage",
    ".cache",
    "env",
    ".env",
    ".pytest_cache",
    ".mypy_cache",
    ".coverage",
    ".eggs",
    "*.egg-info",
    # Data and config directories
    "log",
    "logs",
    "content",
    "tracks",
    "assets",
    "data",
}

# System directories to always exclude
_SYSTEM_EXCLUDES = {
    # Unix/Linux
    "bin",
    "sbin",
    "lib",
    "lib64",
    "tmp",
    "var",
    "usr",
    "opt",
    "proc",
    "sys",
    "dev",
    "boot",
    "mnt",
    "snap",
    "root",
    "run",
    "srv",
    # Windows
    "Windows",
    "Program Files",
    "Program Files (x86)",
    "ProgramData",
    "$Recycle.Bin",
    "System Volume Information",
}

# Combined skip dirs
_SKIP_DIRS = _DEFAULT_EXCLUDES | _SYSTEM_EXCLUDES

# File extensions to skip (binary/generated)
_SKIP_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".o",
    ".a",
    ".lib",
    ".class",
    ".jar",
    ".war",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".wav",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".lock",
    ".min.js",
    ".min.css",
    ".map",
    ".log",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".xml",
    ".csv",
    ".sql",
}


def _load_gitignore(folder: str) -> Any:
    """Load .gitignore patterns from folder if pathspec is available.

    Returns a pathspec.PathSpec object or None.
    """
    if not _HAS_PATHSPEC or pathspec is None:
        return None
    gitignore_path = Path(folder) / ".gitignore"
    if not gitignore_path.is_file():
        return None
    try:
        patterns = gitignore_path.read_text(encoding="utf-8").splitlines()
        return pathspec.PathSpec.from_lines("gitignore", patterns)
    except OSError:
        return None


def _is_binary(path: str) -> bool:
    """Check if file is binary by looking for null bytes in first 8KB."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except OSError:
        return True  # Can't read = treat as binary


def _compute_md5(path: str) -> str:
    """Compute MD5 hash of file contents."""
    hasher = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return ""


def _collect_files(folder: str) -> list[str]:
    """Collect all indexable files from folder, respecting .gitignore.

    Skips:
    - Hidden files and directories
    - Symlinks (files and directories)
    - Binary files
    - Files matching .gitignore patterns (or default excludes if no .gitignore)
    - System directories

    Limits output to MAX_FILES (5000) files.
    """
    folder_path = Path(folder).resolve()
    gitignore = _load_gitignore(str(folder_path))
    files: list[str] = []

    for root, dirs, filenames in os.walk(folder_path, followlinks=False):
        # Check file limit
        if len(files) >= MAX_FILES:
            break

        # Filter out skip directories in-place (also skip symlinked dirs)
        dirs[:] = [
            d
            for d in dirs
            if d not in _SKIP_DIRS
            and not d.startswith(".")
            and not (Path(root) / d).is_symlink()
        ]

        rel_root = Path(root).relative_to(folder_path)

        for filename in filenames:
            # Check file limit
            if len(files) >= MAX_FILES:
                break

            # Skip hidden files
            if filename.startswith("."):
                continue

            # Skip by extension
            ext = Path(filename).suffix.lower()
            if ext in _SKIP_EXTENSIONS:
                continue

            full_path = Path(root) / filename
            rel_path = rel_root / filename

            # Skip symlinks
            if full_path.is_symlink():
                continue

            # Check gitignore
            if gitignore and gitignore.match_file(str(rel_path)):
                continue

            # Skip binary files
            if _is_binary(str(full_path)):
                continue

            files.append(str(full_path))

    return sorted(files)


# ── Tree-sitter Parsing ──────────────────────────────────────────────────

# Language detection by extension
_EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyw": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",  # TSX needs special parser
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
}

# Cached parsers (dict[str, Parser] when tree-sitter is available)
_parsers: dict[str, Any] = {}


def _get_parser(language: str) -> Any:
    """Get or create a tree-sitter parser for the language.

    Returns a tree_sitter.Parser or None.
    """
    if not _HAS_TREE_SITTER or Language is None or Parser is None:
        return None

    if language in _parsers:
        return _parsers[language]

    lang_obj = None
    if language == "python" and tree_sitter_python is not None:
        lang_obj = Language(tree_sitter_python.language())
    elif language == "javascript" and tree_sitter_javascript is not None:
        lang_obj = Language(tree_sitter_javascript.language())
    elif language == "typescript" and tree_sitter_typescript is not None:
        lang_obj = Language(tree_sitter_typescript.language_typescript())
    elif language == "tsx" and tree_sitter_typescript is not None:
        lang_obj = Language(tree_sitter_typescript.language_tsx())
    elif language == "go" and tree_sitter_go is not None:
        lang_obj = Language(tree_sitter_go.language())
    elif language == "rust" and tree_sitter_rust is not None:
        lang_obj = Language(tree_sitter_rust.language())

    if lang_obj is None:
        return None

    parser = Parser(lang_obj)
    _parsers[language] = parser
    return parser


def _detect_language(path: str) -> str | None:
    """Detect language from file extension."""
    ext = Path(path).suffix.lower()
    return _EXTENSION_TO_LANGUAGE.get(ext)


def _get_node_text(node: tree_sitter.Node, source: bytes) -> str:
    """Extract text from a tree-sitter node."""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _extract_python_symbols(
    tree: tree_sitter.Tree, source: bytes, file_path: str, file_md5: str
) -> list[Chunk]:
    """Extract Python symbols from AST."""
    chunks: list[Chunk] = []
    root = tree.root_node

    # Collect file-level imports
    file_imports: list[str] = []
    for node in root.children:
        if node.type in ("import_statement", "import_from_statement"):
            file_imports.append(_get_node_text(node, source).strip())

    def process_node(node: tree_sitter.Node, parent_class: str | None = None) -> None:
        if node.type == "function_definition":
            # Extract function/method
            name_node = node.child_by_field_name("name")
            symbol_name = (
                _get_node_text(name_node, source) if name_node else "anonymous"
            )
            symbol_type = "method" if parent_class else "function"

            # Extract decorators
            decorators: list[str] = []
            prev = node.prev_sibling
            while prev and prev.type == "decorator":
                decorators.insert(0, _get_node_text(prev, source).strip())
                prev = prev.prev_sibling

            # Extract calls within function
            calls: list[str] = []
            for child in _iter_tree(node):
                if child.type == "call":
                    func_node = child.child_by_field_name("function")
                    if func_node:
                        calls.append(_get_node_text(func_node, source))

            source_code = _get_node_text(node, source)
            # Include decorators in source
            if decorators:
                source_code = "\n".join(decorators) + "\n" + source_code

            chunk = Chunk(
                file_path=file_path,
                file_md5=file_md5,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                source_code=source_code,
                embedded_text="",  # Built later
                metadata=ChunkMetadata(
                    imports=file_imports.copy(),
                    calls=calls,
                    decorators=decorators,
                    parent_class=parent_class,
                ),
            )
            chunks.append(chunk)

        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            class_name = _get_node_text(name_node, source) if name_node else "anonymous"

            # Extract decorators
            decorators: list[str] = []
            prev = node.prev_sibling
            while prev and prev.type == "decorator":
                decorators.insert(0, _get_node_text(prev, source).strip())
                prev = prev.prev_sibling

            source_code = _get_node_text(node, source)
            if decorators:
                source_code = "\n".join(decorators) + "\n" + source_code

            chunk = Chunk(
                file_path=file_path,
                file_md5=file_md5,
                symbol_name=class_name,
                symbol_type="class",
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                source_code=source_code,
                embedded_text="",
                metadata=ChunkMetadata(
                    imports=file_imports.copy(),
                    decorators=decorators,
                ),
            )
            chunks.append(chunk)

            # Process methods within class
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    process_node(child, parent_class=class_name)

        else:
            # Recurse into other nodes
            for child in node.children:
                process_node(child, parent_class)

    for child in root.children:
        process_node(child)

    return chunks


def _iter_tree(node: tree_sitter.Node):
    """Iterate over all nodes in tree."""
    yield node
    for child in node.children:
        yield from _iter_tree(child)


def _extract_js_symbols(
    tree: tree_sitter.Tree, source: bytes, file_path: str, file_md5: str
) -> list[Chunk]:
    """Extract JavaScript/TypeScript symbols from AST."""
    chunks: list[Chunk] = []
    root = tree.root_node

    # Collect imports
    file_imports: list[str] = []
    for node in root.children:
        if node.type in ("import_statement", "import_declaration"):
            file_imports.append(_get_node_text(node, source).strip())

    def process_node(node: tree_sitter.Node, parent_class: str | None = None) -> None:
        if node.type in ("function_declaration", "method_definition"):
            name_node = node.child_by_field_name("name")
            symbol_name = (
                _get_node_text(name_node, source) if name_node else "anonymous"
            )
            symbol_type = "method" if parent_class else "function"

            chunk = Chunk(
                file_path=file_path,
                file_md5=file_md5,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                source_code=_get_node_text(node, source),
                embedded_text="",
                metadata=ChunkMetadata(
                    imports=file_imports.copy(),
                    parent_class=parent_class,
                ),
            )
            chunks.append(chunk)

        elif node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            class_name = _get_node_text(name_node, source) if name_node else "anonymous"

            chunk = Chunk(
                file_path=file_path,
                file_md5=file_md5,
                symbol_name=class_name,
                symbol_type="class",
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                source_code=_get_node_text(node, source),
                embedded_text="",
                metadata=ChunkMetadata(imports=file_imports.copy()),
            )
            chunks.append(chunk)

            # Process methods within class
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    process_node(child, parent_class=class_name)

        elif node.type in ("arrow_function", "function_expression"):
            # Named arrow functions via variable declarations
            parent = node.parent
            if parent and parent.type == "variable_declarator":
                name_node = parent.child_by_field_name("name")
                symbol_name = _get_node_text(name_node, source) if name_node else None
                if symbol_name:
                    # Get grandparent for const/let declaration, fallback to parent
                    grandparent = parent.parent
                    source_node = grandparent if grandparent else parent
                    chunk = Chunk(
                        file_path=file_path,
                        file_md5=file_md5,
                        symbol_name=symbol_name,
                        symbol_type="function",
                        line_start=parent.start_point[0] + 1,
                        line_end=parent.end_point[0] + 1,
                        source_code=_get_node_text(source_node, source),
                        embedded_text="",
                        metadata=ChunkMetadata(imports=file_imports.copy()),
                    )
                    chunks.append(chunk)
        else:
            for child in node.children:
                process_node(child, parent_class)

    for child in root.children:
        process_node(child)

    return chunks


def _chunk_by_lines(
    content: str, file_path: str, file_md5: str, chunk_size: int = 50
) -> list[Chunk]:
    """Fallback: chunk file by lines for unsupported languages."""
    lines = content.splitlines()
    chunks: list[Chunk] = []

    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i : i + chunk_size]
        chunk = Chunk(
            file_path=file_path,
            file_md5=file_md5,
            symbol_name=None,
            symbol_type="file",
            line_start=i + 1,
            line_end=min(i + chunk_size, len(lines)),
            source_code="\n".join(chunk_lines),
            embedded_text="",
        )
        chunks.append(chunk)

    return chunks


def _parse_file(path: str, file_md5: str) -> list[Chunk]:
    """Parse file and extract chunks using tree-sitter or fallback."""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    source = content.encode("utf-8")
    language = _detect_language(path)

    if language and _HAS_TREE_SITTER:
        parser = _get_parser(language)
        if parser:
            tree = parser.parse(source)
            if language == "python":
                chunks = _extract_python_symbols(tree, source, path, file_md5)
            elif language in ("javascript", "typescript"):
                chunks = _extract_js_symbols(tree, source, path, file_md5)
            elif language == "tsx":
                chunks = _extract_js_symbols(tree, source, path, file_md5)
            else:
                # Other languages: fallback to line chunking
                chunks = _chunk_by_lines(content, path, file_md5)

            # If no symbols found, create a file-level chunk
            if not chunks:
                chunks = [
                    Chunk(
                        file_path=path,
                        file_md5=file_md5,
                        symbol_name=None,
                        symbol_type="file",
                        line_start=1,
                        line_end=len(content.splitlines()),
                        source_code=content[:10000],  # Limit size
                        embedded_text="",
                    )
                ]
            return chunks

    # Fallback for unsupported languages or missing tree-sitter
    return _chunk_by_lines(content, path, file_md5)


# ── Embedding ────────────────────────────────────────────────────────────

# Default model - small, fast, good quality
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Cached embedding model
_embedding_model: Any = None


def _get_embedding_model() -> Any:
    """Get or initialize the embedding model (downloads on first use).

    Returns a sentence_transformers.SentenceTransformer instance.
    """
    global _embedding_model
    if _embedding_model is None:
        import logging as _logging
        import warnings

        from sentence_transformers import SentenceTransformer

        # Suppress harmless transformer library warnings
        warnings.filterwarnings("ignore", message=".*position_ids.*")

        # Suppress BertModel LOAD REPORT printed by transformers (not a warning)
        _logging.getLogger("transformers.modeling_utils").setLevel(_logging.ERROR)

        model_name = config.EMBEDDING_MODEL or DEFAULT_EMBEDDING_MODEL
        try:
            _embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                f"Check internet connection and model availability. Error: {e}"
            ) from e
    return _embedding_model


def _build_enriched_text(chunk: Chunk) -> str:
    """Build enriched text for embedding from chunk."""
    parts = [f"File: {chunk.file_path}"]

    if chunk.metadata.imports:
        # Simplify imports to just module names
        import_names = []
        for imp in chunk.metadata.imports[:5]:  # Limit to first 5
            # Extract module name from import statement
            if "from " in imp:
                parts_split = imp.split("from ")
                if len(parts_split) > 1:
                    module = parts_split[1].split()[0]
                    import_names.append(module)
            elif "import " in imp:
                module = imp.replace("import ", "").split(",")[0].split()[0]
                import_names.append(module)
        if import_names:
            parts.append(f"Imports: {', '.join(import_names)}")

    if chunk.metadata.decorators:
        parts.append(f"Decorators: {', '.join(chunk.metadata.decorators[:3])}")

    if chunk.metadata.parent_class:
        parts.append(f"Class: {chunk.metadata.parent_class}")

    type_label = chunk.symbol_type.capitalize()
    if chunk.symbol_name:
        parts.append(f"{type_label}: {chunk.symbol_name}")
    else:
        parts.append(f"Type: {type_label}")

    parts.append("")  # Empty line before code
    parts.append(chunk.source_code)

    return "\n".join(parts)


def _embed_text(text: str) -> np.ndarray:
    """Generate embedding for text using local sentence-transformers model."""
    model = _get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype(np.float32)


def _embed_texts_batch(texts: list[str]) -> list[np.ndarray]:
    """Embed multiple texts in a batch (more efficient)."""
    if not texts:
        return []
    model = _get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [emb.astype(np.float32) for emb in embeddings]


# ── Database ─────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_path TEXT PRIMARY KEY,
    file_md5 TEXT NOT NULL,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    symbol_name TEXT,
    symbol_type TEXT,
    line_start INTEGER,
    line_end INTEGER,
    source_code TEXT NOT NULL,
    embedded_text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    FOREIGN KEY (file_path) REFERENCES files(file_path) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metadata (
    chunk_id INTEGER PRIMARY KEY,
    imports TEXT,
    calls TEXT,
    decorators TEXT,
    parent_class TEXT,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol_name);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(symbol_type);
CREATE INDEX IF NOT EXISTS idx_files_md5 ON files(file_md5);
CREATE INDEX IF NOT EXISTS idx_metadata_parent ON metadata(parent_class);
"""


async def _init_db() -> aiosqlite.Connection:
    """Initialize database connection and schema."""
    db_path = config.get_embedding_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path))
    await conn.executescript(_SCHEMA)
    await conn.execute("PRAGMA foreign_keys = ON")
    await conn.commit()
    return conn


async def _get_file_md5(conn: aiosqlite.Connection, file_path: str) -> str | None:
    """Get stored MD5 for a file."""
    async with conn.execute(
        "SELECT file_md5 FROM files WHERE file_path = ?", (file_path,)
    ) as cursor:
        row = await cursor.fetchone()
        return row[0] if row else None


async def _delete_file_chunks(conn: aiosqlite.Connection, file_path: str) -> None:
    """Delete all chunks for a file."""
    await conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
    await conn.commit()


async def _save_chunks(conn: aiosqlite.Connection, chunks: list[Chunk]) -> None:
    """Save chunks to database."""
    if not chunks:
        return

    file_path = chunks[0].file_path
    file_md5 = chunks[0].file_md5

    # Delete existing chunks for this file first (cascade will handle metadata)
    await conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))

    # Insert/replace file record
    await conn.execute(
        "INSERT OR REPLACE INTO files (file_path, file_md5) VALUES (?, ?)",
        (file_path, file_md5),
    )

    for chunk in chunks:
        # Insert chunk
        cursor = await conn.execute(
            """
            INSERT INTO chunks
            (file_path, symbol_name, symbol_type, line_start, line_end,
             source_code, embedded_text, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.file_path,
                chunk.symbol_name,
                chunk.symbol_type,
                chunk.line_start,
                chunk.line_end,
                chunk.source_code,
                chunk.embedded_text,
                chunk.embedding.tobytes() if chunk.embedding is not None else b"",
            ),
        )
        chunk_id = cursor.lastrowid

        # Insert metadata
        await conn.execute(
            """
            INSERT INTO metadata (chunk_id, imports, calls, decorators, parent_class)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                json.dumps(chunk.metadata.imports),
                json.dumps(chunk.metadata.calls),
                json.dumps(chunk.metadata.decorators),
                chunk.metadata.parent_class,
            ),
        )

    await conn.commit()


async def _cleanup_stale_files(
    conn: aiosqlite.Connection, indexed_paths: set[str]
) -> int:
    """Remove files from DB that no longer exist. Returns count of removed."""
    async with conn.execute("SELECT file_path FROM files") as cursor:
        db_paths = {row[0] async for row in cursor}

    stale = db_paths - indexed_paths
    for path in stale:
        await conn.execute("DELETE FROM files WHERE file_path = ?", (path,))

    await conn.commit()
    return len(stale)


async def _load_all_embeddings(
    conn: aiosqlite.Connection,
) -> tuple[list[int], np.ndarray | None]:
    """Load all embeddings from database for FAISS index building."""
    async with conn.execute("SELECT id, embedding FROM chunks") as cursor:
        rows = [(row[0], row[1]) async for row in cursor]

    if not rows:
        return [], None

    ids = [r[0] for r in rows]
    # Determine embedding dimension from first non-empty embedding
    dim = 0
    for _, emb_bytes in rows:
        if emb_bytes:
            arr = np.frombuffer(emb_bytes, dtype=np.float32)
            dim = len(arr)
            break

    if dim == 0:
        return [], None

    embeddings = np.zeros((len(rows), dim), dtype=np.float32)
    for i, (_, emb_bytes) in enumerate(rows):
        if emb_bytes:
            embeddings[i] = np.frombuffer(emb_bytes, dtype=np.float32)

    return ids, embeddings


# ── FAISS Index ──────────────────────────────────────────────────────────


def _get_faiss_index_path() -> Path:
    """Get path to FAISS index file."""
    return config.get_embedding_db_path().with_suffix(".faiss")


def _load_faiss_index() -> tuple[faiss_module.Index | None, list[int]]:
    """Load FAISS index from disk if it exists."""
    if not _HAS_FAISS or faiss is None:
        return None, []

    index_path = _get_faiss_index_path()
    ids_path = index_path.with_suffix(".ids.json")

    if not index_path.is_file() or not ids_path.is_file():
        return None, []

    try:
        index = faiss.read_index(str(index_path))
        with open(ids_path, "r") as f:
            ids = json.load(f)
        return index, ids
    except Exception:
        return None, []


def _save_faiss_index(index: faiss_module.Index, ids: list[int]) -> None:
    """Save FAISS index to disk."""
    if not _HAS_FAISS or faiss is None:
        return

    index_path = _get_faiss_index_path()
    ids_path = index_path.with_suffix(".ids.json")
    try:
        faiss.write_index(index, str(index_path))
        with open(ids_path, "w") as f:
            json.dump(ids, f)
    except Exception:
        print("Warning: Failed to save FAISS index")


async def _rebuild_faiss_index(conn: aiosqlite.Connection) -> tuple[Any, list[int]]:
    """Rebuild FAISS index from database embeddings."""
    if not _HAS_FAISS or faiss is None:
        return None, []
    _faiss = faiss  # Local reference for type checker

    ids, embeddings = await _load_all_embeddings(conn)
    if embeddings is None or len(ids) == 0:
        return None, []

    dim = embeddings.shape[1]
    index = _faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)

    # Normalize for cosine similarity
    _faiss.normalize_L2(embeddings)
    index.add(embeddings)  # type: ignore[arg-type]

    _save_faiss_index(index, ids)
    return index, ids


# ── Main Functions ───────────────────────────────────────────────────────


async def embed_folder(path: str) -> AsyncGenerator[tuple[str, Any], None]:
    """Index a folder by parsing files and generating embeddings.

    Yields progress events:
        ("start", {"total_files": int})
        ("file_indexed", {"path": str, "chunks": int})
        ("file_skipped", {"path": str, "reason": str})
        ("file_error", {"path": str, "error": str})
        ("complete", {"files": int, "chunks": int, "duration": float})
    """
    import asyncio
    import time

    start_time = time.time()

    # Collect files (blocking I/O - run in thread)
    files = await asyncio.to_thread(_collect_files, path)
    yield ("start", {"total_files": len(files)})

    conn = await _init_db()
    total_chunks = 0
    indexed_files = 0

    try:
        indexed_paths: set[str] = set()

        for file_path in files:
            indexed_paths.add(file_path)

            # Check if file changed (blocking I/O - run in thread)
            current_md5 = await asyncio.to_thread(_compute_md5, file_path)
            if not current_md5:
                yield (
                    "file_error",
                    {"path": file_path, "error": "Could not read file"},
                )
                continue

            stored_md5 = await _get_file_md5(conn, file_path)
            if stored_md5 == current_md5:
                yield ("file_skipped", {"path": file_path, "reason": "unchanged"})
                continue

            # Delete old chunks if file changed
            if stored_md5:
                await _delete_file_chunks(conn, file_path)

            # Parse file (blocking I/O + CPU - run in thread)
            try:
                chunks = await asyncio.to_thread(_parse_file, file_path, current_md5)
            except Exception as e:
                yield ("file_error", {"path": file_path, "error": str(e)})
                continue

            if not chunks:
                yield ("file_skipped", {"path": file_path, "reason": "no chunks"})
                continue

            # Build enriched text and generate embeddings
            try:
                for chunk in chunks:
                    chunk.embedded_text = _build_enriched_text(chunk)
                # Batch embed all chunks at once (CPU-intensive - run in thread)
                texts = [c.embedded_text for c in chunks]
                embeddings = await asyncio.to_thread(_embed_texts_batch, texts)
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding
            except Exception as e:
                yield (
                    "file_error",
                    {"path": file_path, "error": f"Embedding failed: {e}"},
                )
                continue

            # Save to database
            await _save_chunks(conn, chunks)
            total_chunks += len(chunks)
            indexed_files += 1
            yield ("file_indexed", {"path": file_path, "chunks": len(chunks)})

        # Cleanup stale files
        removed = await _cleanup_stale_files(conn, indexed_paths)
        if removed > 0:
            yield ("cleanup", {"removed_files": removed})

        # Rebuild FAISS index
        await _rebuild_faiss_index(conn)

        # Optimize and compress database
        await conn.execute("PRAGMA optimize")
        await conn.execute("VACUUM")
        await conn.commit()

    finally:
        await conn.close()

    duration = time.time() - start_time
    yield (
        "complete",
        {
            "files": indexed_files,
            "chunks": total_chunks,
            "duration": round(duration, 2),
        },
    )


async def query_similar(query: str, top_k: int = 10) -> list[Chunk]:
    """Query for similar code chunks.

    Args:
        query: Natural language query or code snippet
        top_k: Number of results to return

    Returns:
        List of Chunk objects with source code and metadata
    """
    if not _HAS_FAISS or faiss is None:
        return []
    _faiss = faiss  # Local reference for type checker

    # Load FAISS index
    index, ids = _load_faiss_index()
    if index is None or not ids:
        return []

    # Generate query embedding
    query_embedding = _embed_text(query)
    query_embedding = query_embedding.reshape(1, -1)
    _faiss.normalize_L2(query_embedding)

    # Search
    k = min(top_k, len(ids))
    _, indices = index.search(query_embedding, k)  # type: ignore[misc]

    # Get chunk IDs from results
    result_ids = [ids[i] for i in indices[0] if i < len(ids)]

    if not result_ids:
        return []

    # Fetch chunk details from database
    conn = await _init_db()
    try:
        chunks: list[Chunk] = []
        for chunk_id in result_ids:
            async with conn.execute(
                """
                SELECT c.file_path, c.symbol_name, c.symbol_type,
                       c.line_start, c.line_end, c.source_code,
                       m.imports, m.calls, m.decorators, m.parent_class
                FROM chunks c
                LEFT JOIN metadata m ON c.id = m.chunk_id
                WHERE c.id = ?
                """,
                (chunk_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    chunk = Chunk(
                        file_path=row[0],
                        file_md5="",
                        symbol_name=row[1],
                        symbol_type=row[2],
                        line_start=row[3],
                        line_end=row[4],
                        source_code=row[5],
                        embedded_text="",
                        chunk_id=chunk_id,
                        metadata=ChunkMetadata(
                            imports=json.loads(row[6]) if row[6] else [],
                            calls=json.loads(row[7]) if row[7] else [],
                            decorators=json.loads(row[8]) if row[8] else [],
                            parent_class=row[9],
                        ),
                    )
                    chunks.append(chunk)
        return chunks
    finally:
        await conn.close()


# ── File Watcher ──────────────────────────────────────────────────────────


@dataclass
class FileChangeEvent:
    """Represents a file system change event."""

    event_type: str  # "created", "modified", "deleted", "moved"
    file_path: str
    dest_path: str | None = None  # Only for "moved" events


def _should_watch_file(file_path: str, folder: str, gitignore: Any) -> bool:
    """Check if a file should be watched based on exclusion rules.

    Uses the same rules as _collect_files for consistency.
    """
    path = Path(file_path)

    # Skip hidden files
    if path.name.startswith("."):
        return False

    # Skip by extension
    ext = path.suffix.lower()
    if ext in _SKIP_EXTENSIONS:
        return False

    # Skip symlinks
    if path.is_symlink():
        return False

    # Check if any parent directory is in skip list
    try:
        rel_path = path.relative_to(folder)
    except ValueError:
        return False

    for part in rel_path.parts[:-1]:  # Check parent directories
        if part in _SKIP_DIRS or part.startswith("."):
            return False

    # Check gitignore
    if gitignore and gitignore.match_file(str(rel_path)):
        return False

    # Skip binary files (only for existing files)
    if path.is_file() and _is_binary(str(path)):
        return False

    return True


class _FolderEventHandler:
    """File system event handler that queues events for async processing."""

    def __init__(self, folder: str, gitignore: Any, queue: Any) -> None:
        self.folder = folder
        self.gitignore = gitignore
        self.queue = queue

    def _handle_event(self, event: Any, event_type: str) -> None:
        """Handle a file system event."""
        if event.is_directory:
            return

        src_path = event.src_path

        # Explicitly skip log directory to prevent logging loop
        if "/log/" in src_path or src_path.endswith(".log"):
            return
        dest_path = getattr(event, "dest_path", None)

        if not _should_watch_file(src_path, self.folder, self.gitignore):
            # For moved events, check if destination should be watched
            if event_type == "moved" and dest_path:
                if _should_watch_file(dest_path, self.folder, self.gitignore):
                    # Treat as a new file creation
                    change = FileChangeEvent(
                        event_type="created",
                        file_path=dest_path,
                    )
                    self.queue.put_nowait(change)
            return

        change = FileChangeEvent(
            event_type=event_type,
            file_path=src_path,
            dest_path=dest_path if event_type == "moved" else None,
        )
        self.queue.put_nowait(change)

    def on_created(self, event: Any) -> None:
        self._handle_event(event, "created")

    def on_modified(self, event: Any) -> None:
        self._handle_event(event, "modified")

    def on_deleted(self, event: Any) -> None:
        self._handle_event(event, "deleted")

    def on_moved(self, event: Any) -> None:
        self._handle_event(event, "moved")


async def watch_folder(
    folder: str,
) -> AsyncGenerator[FileChangeEvent, None]:
    """Watch a folder for file changes and yield events.

    This is an async generator that yields FileChangeEvent objects
    when files are created, modified, deleted, or moved.

    Respects the same exclusion rules as embed_folder:
    - Hidden files and directories
    - Binary files
    - .gitignore patterns
    - Default excluded directories (node_modules, .git, etc.)

    Args:
        folder: Path to the folder to watch

    Yields:
        FileChangeEvent objects for each relevant file change

    Raises:
        RuntimeError: If watchdog is not installed

    Example:
        async for event in watch_folder("/path/to/project"):
            if event.event_type in ("created", "modified"):
                # Re-index the changed file
                async for progress in embed_folder("/path/to/project"):
                    pass
    """
    if not _HAS_WATCHDOG or Observer is None or FileSystemEventHandler is None:
        raise RuntimeError(
            "watchdog is not installed. Install it with: pip install watchdog"
        )

    import asyncio

    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    gitignore = _load_gitignore(str(folder_path))
    queue: asyncio.Queue[FileChangeEvent] = asyncio.Queue()

    # Create event handler
    handler = _FolderEventHandler(str(folder_path), gitignore, queue)

    # Create a watchdog-compatible handler by subclassing FileSystemEventHandler
    class WatchdogHandler(FileSystemEventHandler):  # type: ignore[misc]
        def on_created(self, event: Any) -> None:
            handler.on_created(event)

        def on_modified(self, event: Any) -> None:
            handler.on_modified(event)

        def on_deleted(self, event: Any) -> None:
            handler.on_deleted(event)

        def on_moved(self, event: Any) -> None:
            handler.on_moved(event)

    observer = Observer()
    observer.daemon = True
    observer.schedule(WatchdogHandler(), str(folder_path), recursive=True)
    observer.start()

    try:
        while True:
            try:
                # Use wait_for with a timeout to allow for cancellation
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                # No event, continue waiting
                continue
            except asyncio.CancelledError:
                break
    finally:
        observer.stop()
        observer.join(timeout=5.0)


def start_watcher(
    folder: str,
    callback: Any,
) -> Any:
    """Start a file watcher with a callback function.

    This is a synchronous alternative to watch_folder that uses
    a callback instead of an async generator.

    Args:
        folder: Path to the folder to watch
        callback: Function called with FileChangeEvent for each change.
                  Signature: callback(event: FileChangeEvent) -> None

    Returns:
        The watchdog Observer instance (call .stop() to stop watching)

    Raises:
        RuntimeError: If watchdog is not installed
        ValueError: If folder is not a directory

    Example:
        def on_change(event):
            print(f"{event.event_type}: {event.file_path}")

        observer = start_watcher("/path/to/project", on_change)
        # ... later ...
        observer.stop()
        observer.join()
    """
    if not _HAS_WATCHDOG or Observer is None or FileSystemEventHandler is None:
        raise RuntimeError(
            "watchdog is not installed. Install it with: pip install watchdog"
        )

    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    gitignore = _load_gitignore(str(folder_path))

    class CallbackHandler(FileSystemEventHandler):  # type: ignore[misc]
        def _handle(self, event: Any, event_type: str) -> None:
            if event.is_directory:
                return

            src_path = event.src_path

            # Explicitly skip log directory to prevent logging loop
            if "/log/" in src_path or src_path.endswith(".log"):
                return

            dest_path = getattr(event, "dest_path", None)

            if not _should_watch_file(src_path, str(folder_path), gitignore):
                if event_type == "moved" and dest_path:
                    if _should_watch_file(dest_path, str(folder_path), gitignore):
                        change = FileChangeEvent(
                            event_type="created",
                            file_path=dest_path,
                        )
                        callback(change)
                return

            change = FileChangeEvent(
                event_type=event_type,
                file_path=src_path,
                dest_path=dest_path if event_type == "moved" else None,
            )
            callback(change)

        def on_created(self, event: Any) -> None:
            self._handle(event, "created")

        def on_modified(self, event: Any) -> None:
            self._handle(event, "modified")

        def on_deleted(self, event: Any) -> None:
            self._handle(event, "deleted")

        def on_moved(self, event: Any) -> None:
            self._handle(event, "moved")

    observer = Observer()
    observer.daemon = True
    observer.schedule(CallbackHandler(), str(folder_path), recursive=True)
    observer.start()
    return observer
