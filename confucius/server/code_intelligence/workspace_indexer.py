# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Filesystem-based workspace indexer for CCA.

Replaces MCP's Nextcloud-based codebase_indexer with direct filesystem
access.  Walks configurable paths, extracts AST via tree-sitter, embeds
via Qwen3-Embedding-8B, and upserts to the shared Qdrant `codebase_files`
collection + Memgraph knowledge graph.

Change detection: SHA256 content hash (replaces Nextcloud etag).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Language detection from file extension
INDEXED_EXTENSIONS: Dict[str, str] = {
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".sh": "bash",
    ".bash": "bash",
    ".py": "python",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".md": "markdown",
    ".json": "json",
    ".txt": "text",
}

# Directories to skip during walk
DEFAULT_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "build", "dist", ".tox", ".mypy_cache", ".pytest_cache",
    "eggs", "*.egg-info", ".cache", ".idea", ".vscode",
    "worktrees", "artifacts", ".claude",
}

# Limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE = 8000
CHUNK_OVERLAP = 1000
COLLECTION_NAME = "codebase_files"
EMBEDDING_DIMS = 4096


def _detect_project(file_path: str, index_root: str) -> str:
    """Detect project name from file path relative to index root."""
    rel = os.path.relpath(file_path, index_root)
    parts = rel.split(os.sep)
    if len(parts) > 1:
        return parts[0].replace("-", "_")
    return "default"


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _doc_id(file_path: str, name: str) -> str:
    return hashlib.md5(f"{file_path}::{name}".encode()).hexdigest()


class WorkspaceIndexer:
    """Filesystem-based workspace indexer.

    Requires BackendClients for Qdrant + Embedding + Memgraph access.
    """

    def __init__(self, backend_clients: Any) -> None:
        self._clients = backend_clients
        self._parser: Any = None  # TreeSitterParser (lazy)
        self._graph: Any = None   # MemgraphClient (lazy)

    def _get_parser(self) -> Any:
        """Lazy-load tree-sitter parser."""
        if self._parser is None:
            try:
                from .tree_sitter_parser import TreeSitterParser
                self._parser = TreeSitterParser.get_instance()
            except Exception as e:
                logger.warning("TreeSitterParser not available: %s", e)
        return self._parser

    def _get_graph(self) -> Any:
        """Lazy-load Memgraph client."""
        if self._graph is None and self._clients.memgraph is not None:
            try:
                from .memgraph_client import MemgraphClient
                self._graph = MemgraphClient(self._clients.memgraph)
            except Exception as e:
                logger.warning("MemgraphClient not available: %s", e)
        return self._graph

    async def index_paths(
        self,
        paths: List[str],
        skip_dirs: Optional[set] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Index all files under the given paths.

        Args:
            paths: List of directories to walk.
            skip_dirs: Directories to skip (defaults to DEFAULT_SKIP_DIRS).
            force: If True, re-index all files regardless of content hash.

        Returns:
            Stats dict: {files_scanned, indexed, skipped, errors, functions}.
        """
        if not self._clients.available:
            return {"error": "BackendClients not available (Qdrant/Embedding missing)"}

        skip = skip_dirs or DEFAULT_SKIP_DIRS
        stats = {
            "files_scanned": 0,
            "indexed": 0,
            "skipped": 0,
            "errors": 0,
            "functions": 0,
            "paths": paths,
            "started_at": datetime.now().isoformat(),
        }

        # Collect files from all paths
        all_files: List[str] = []
        for root_path in paths:
            if not os.path.isdir(root_path):
                logger.warning("Index path does not exist: %s", root_path)
                continue
            for dirpath, dirnames, filenames in os.walk(root_path):
                # Prune skipped directories in-place
                dirnames[:] = [
                    d for d in dirnames
                    if d not in skip and not d.endswith(".egg-info")
                ]
                for fname in filenames:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in INDEXED_EXTENSIONS:
                        full = os.path.join(dirpath, fname)
                        try:
                            if os.path.getsize(full) <= MAX_FILE_SIZE:
                                all_files.append(full)
                        except OSError:
                            pass

        stats["files_scanned"] = len(all_files)
        logger.info(
            "WorkspaceIndexer: found %d indexable files in %s",
            len(all_files), paths,
        )

        # Load existing content hashes from Qdrant for change detection
        existing_hashes: Dict[str, str] = {}
        if not force:
            existing_hashes = await self._load_existing_hashes()

        # Index files with concurrency control
        sem = asyncio.Semaphore(5)

        async def _index_one(fpath: str) -> None:
            async with sem:
                try:
                    result = await self._index_file(
                        fpath, paths, existing_hashes, force
                    )
                    if result == "indexed":
                        stats["indexed"] += 1
                    elif result == "skipped":
                        stats["skipped"] += 1
                    elif result.startswith("functions:"):
                        stats["indexed"] += 1
                        stats["functions"] += int(result.split(":")[1])
                except Exception as e:
                    logger.error("Error indexing %s: %s", fpath, e)
                    stats["errors"] += 1

        await asyncio.gather(*[_index_one(f) for f in all_files])

        stats["finished_at"] = datetime.now().isoformat()
        logger.info("WorkspaceIndexer: done — %s", stats)
        return stats

    async def _load_existing_hashes(self) -> Dict[str, str]:
        """Load content_hash values from Qdrant for CCA-sourced files."""
        hashes: Dict[str, str] = {}
        qdrant = self._clients.qdrant
        if not qdrant:
            return hashes

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Scroll through all CCA-sourced points to get file_path → content_hash
            offset = None
            while True:
                results = await qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="source",
                                match=MatchValue(value="cca"),
                            )
                        ]
                    ),
                    limit=500,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, next_offset = results
                for pt in points:
                    payload = pt.payload or {}
                    fp = payload.get("file_path", "")
                    ch = payload.get("content_hash", "")
                    if fp and ch:
                        hashes[fp] = ch

                if next_offset is None:
                    break
                offset = next_offset

        except Exception as e:
            logger.warning("Could not load existing hashes: %s", e)

        logger.info("Loaded %d existing content hashes", len(hashes))
        return hashes

    async def _index_file(
        self,
        file_path: str,
        index_roots: List[str],
        existing_hashes: Dict[str, str],
        force: bool,
    ) -> str:
        """Index a single file. Returns status string."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            logger.warning("Cannot read %s: %s", file_path, e)
            return "error"

        content_hash = _sha256(content)

        # Change detection
        if not force and existing_hashes.get(file_path) == content_hash:
            return "skipped"

        ext = os.path.splitext(file_path)[1].lower()
        language = INDEXED_EXTENSIONS.get(ext, "text")

        # Detect project from the closest index root
        project = "default"
        for root in index_roots:
            if file_path.startswith(root):
                project = _detect_project(file_path, root)
                break

        now = datetime.now().isoformat()
        parser = self._get_parser()
        functions: List[Dict] = []

        # AST extraction
        if parser:
            if language in parser.SECTION_LANGUAGES:
                functions = parser.extract_sections(content, language, file_path)
            elif language in parser.SUPPORTED_LANGUAGES:
                functions = parser.extract_functions(content, language, file_path)

        # Build documents to embed
        documents: List[str] = []
        metadatas: List[Dict] = []
        ids: List[str] = []

        if functions:
            for func in functions:
                func_name = func.get("name", "unknown")
                func_type = func.get("type", "function")
                doc_text = (
                    f"File: {file_path}\n"
                    f"Language: {language}\n"
                    f"{'Function' if func_type == 'function' else 'Section'}: "
                    f"{func.get('signature', func_name)}\n"
                )
                if func.get("docstring"):
                    doc_text += f"\n{func['docstring']}\n"
                doc_text += f"\nCode:\n{func.get('body', '')}"

                meta = {
                    "file_path": file_path,
                    "language": language,
                    "type": func_type,
                    "name": func_name,
                    "project": project,
                    "content_hash": content_hash,
                    "source": "cca",
                    "indexed_at": now,
                    "signature": func.get("signature", ""),
                    "line_start": func.get("line_start", 0),
                    "line_end": func.get("line_end", 0),
                    "loc": func.get("loc", 0),
                    "is_async": func.get("is_async", False),
                    "is_generator": func.get("is_generator", False),
                    "return_type": func.get("return_type") or "",
                }
                # JSON-encode list fields
                import json
                for list_field in ("parameters", "decorators", "calls", "imports"):
                    val = func.get(list_field, [])
                    if isinstance(val, list):
                        meta[list_field] = json.dumps(val)
                    else:
                        meta[list_field] = str(val)

                if func.get("class_name"):
                    meta["class_name"] = func["class_name"]

                documents.append(doc_text[:10000])
                metadatas.append(meta)
                ids.append(_doc_id(file_path, func_name))
        else:
            # Chunk file content
            chunks = self._chunk_content(content)
            for i, chunk in enumerate(chunks):
                doc_text = (
                    f"File: {file_path}\n"
                    f"Language: {language}\n"
                    f"Chunk: {i + 1}/{len(chunks)}\n\n"
                    f"{chunk}"
                )
                meta = {
                    "file_path": file_path,
                    "language": language,
                    "type": "chunk",
                    "name": f"chunk_{i + 1}",
                    "project": project,
                    "content_hash": content_hash,
                    "source": "cca",
                    "indexed_at": now,
                }
                documents.append(doc_text[:10000])
                metadatas.append(meta)
                ids.append(_doc_id(file_path, f"chunk_{i}"))

        # Delete old documents for this file before upserting
        await self._delete_file_docs(file_path)

        # Embed and upsert
        try:
            vectors = await self._clients.embed(documents)
            await self._upsert_to_qdrant(ids, vectors, metadatas, documents)
        except Exception as e:
            logger.error("Embedding/upsert error for %s: %s", file_path, e)
            return "error"

        # Build graph
        graph = self._get_graph()
        if graph and functions:
            try:
                await graph.index_file_graph(
                    file_path=file_path,
                    project=project,
                    language=language,
                    functions=functions,
                )
            except Exception as e:
                logger.warning("Graph indexing failed for %s: %s", file_path, e)

        func_count = len(functions) if functions else 0
        return f"functions:{func_count}" if func_count > 0 else "indexed"

    def _chunk_content(self, content: str) -> List[str]:
        """Split content into overlapping chunks."""
        if len(content) <= CHUNK_SIZE:
            return [content]

        chunks = []
        start = 0
        while start < len(content):
            end = start + CHUNK_SIZE

            # Find natural boundary
            if end < len(content):
                for boundary in ["\n\n", "\n", ". ", " "]:
                    pos = content.rfind(boundary, start + CHUNK_SIZE // 2, end)
                    if pos > start:
                        end = pos + len(boundary)
                        break

            chunks.append(content[start:end])
            new_start = end - CHUNK_OVERLAP
            start = new_start if new_start > start else end

        return chunks

    async def _delete_file_docs(self, file_path: str) -> None:
        """Delete all existing Qdrant points for a file (before re-indexing)."""
        qdrant = self._clients.qdrant
        if not qdrant:
            return

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            await qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=file_path),
                        ),
                        FieldCondition(
                            key="source",
                            match=MatchValue(value="cca"),
                        ),
                    ]
                ),
            )
        except Exception as e:
            logger.warning("Could not delete old docs for %s: %s", file_path, e)

    async def _upsert_to_qdrant(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict],
        documents: List[str],
    ) -> None:
        """Upsert pre-embedded documents to Qdrant."""
        qdrant = self._clients.qdrant
        if not qdrant:
            return

        from qdrant_client.models import PointStruct, Distance, VectorParams

        # Ensure collection exists
        try:
            await qdrant.get_collection(COLLECTION_NAME)
        except Exception:
            await qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMS,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", COLLECTION_NAME)

        # Build points
        points = []
        for i, (pid, vec, meta, doc) in enumerate(
            zip(ids, vectors, metadatas, documents)
        ):
            meta["_content"] = doc[:5000]  # Store truncated content for retrieval
            points.append(PointStruct(
                id=pid,
                vector=vec,
                payload=meta,
            ))

        # Batch upsert (100 at a time)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
            )
