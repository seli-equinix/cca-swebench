# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""CodeTraceExtension — 2 LLM-callable tools for code execution tracing.

Tools:
  trace_execution       — Trace from an entry point, collect all needed functions
  assemble_traced_code  — Assemble trace results into a single output file

ToolGroup: TRACE
Routes: CODER, INFRASTRUCTURE

Algorithm:
  1. Build function dictionary from Qdrant (all functions in project)
  2. Parse entry file with tree-sitter (detect top-level calls)
  3. BFS call resolution using func_dict (not graph edges)
  4. Read full function bodies from disk (no truncation)
  5. Assemble into topologically sorted output
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from ...core.analect import AnalectRunContext
from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...orchestrator.extensions.tool_use import ToolUseExtension

logger = logging.getLogger(__name__)

# In-memory trace result cache (per-process, cleared on restart)
_trace_cache: Dict[str, Dict[str, Any]] = {}

COLLECTION_NAME = "codebase_files"


class CodeTraceExtension(ToolUseExtension):
    """CCA extension providing code execution tracing tools."""

    name: str = "CodeTraceExtension"
    enable_tool_use: bool = True
    included_in_system_prompt: bool = False

    _backend_clients: Any

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, backend_clients: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_backend_clients", backend_clients)

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    @property
    async def tools(self) -> list[ant.ToolLike]:
        return [
            ant.Tool(
                name="trace_execution",
                description=(
                    "Trace code execution from an entry point, collecting all "
                    "functions needed to run the code.\n"
                    "- Parses the entry file to find top-level function calls\n"
                    "- Recursively resolves all called functions (BFS)\n"
                    "- Returns function names, source files, line ranges, and call chains\n"
                    "- Works with PowerShell, Bash, and Python\n"
                    "Use this to understand execution flow before code review or migration."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "entry_file": {
                            "type": "string",
                            "description": (
                                "Entry script path or filename to trace from "
                                "(e.g., 'JobStart.ps1' or '/workspace/EVA/code/JobStart.ps1')"
                            ),
                        },
                        "entry_function": {
                            "type": "string",
                            "description": "Specific function name to trace from (instead of entry_file)",
                        },
                        "project": {
                            "type": "string",
                            "description": "Project scope (e.g., 'EVA'). Filters functions to this project.",
                        },
                        "language": {
                            "type": "string",
                            "description": "Language filter (e.g., 'powershell', 'python', 'bash')",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Max BFS traversal depth (1-15, default 15)",
                        },
                    },
                    "required": [],
                },
            ),
            ant.Tool(
                name="assemble_traced_code",
                description=(
                    "Assemble previously traced functions into a single output.\n"
                    "Reads full function bodies from source files (no truncation).\n"
                    "Output includes entry file code, all needed functions, and external deps.\n"
                    "Use after trace_execution to get the complete code assembly."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "trace_id": {
                            "type": "string",
                            "description": "Trace ID from a previous trace_execution result",
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["single_file", "grouped_by_source", "minimal"],
                            "description": (
                                "Output format: single_file (all in one), "
                                "grouped_by_source (organized by source file), "
                                "minimal (signatures only)"
                            ),
                        },
                        "include_entry_file": {
                            "type": "boolean",
                            "description": "Include entry file content (default true)",
                        },
                    },
                    "required": ["trace_id"],
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def on_tool_use(
        self,
        tool_use: ant.MessageContentToolUse,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        name = tool_use.name
        inp = tool_use.input or {}

        try:
            if name == "trace_execution":
                result = await self._handle_trace(inp)
            elif name == "assemble_traced_code":
                result = await self._handle_assemble(inp)
            else:
                result = json.dumps({"error": f"Unknown tool: {name}"})

            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=result if isinstance(result, str) else json.dumps(result),
            )

        except Exception as e:
            logger.error("Trace tool '%s' failed: %s", name, e, exc_info=True)
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=f"Error: {e}",
                is_error=True,
            )

    # ------------------------------------------------------------------
    # Phase 1: Build function dictionary from Qdrant
    # ------------------------------------------------------------------

    async def _build_func_dict(
        self,
        project: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Scroll all indexed functions from Qdrant for a project.

        Returns: {func_name: {file_path, line_start, line_end, calls, language, signature}}
        """
        qdrant = self._backend_clients.qdrant
        if not qdrant:
            return {}

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        must_conditions = [
            FieldCondition(key="type", match=MatchValue(value="function")),
            FieldCondition(key="source", match=MatchValue(value="cca")),
        ]
        if project:
            must_conditions.append(
                FieldCondition(key="project", match=MatchValue(value=project))
            )
        if language:
            must_conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language))
            )

        func_dict: Dict[str, Dict[str, Any]] = {}
        offset = None

        while True:
            points, next_offset = await qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(must=must_conditions),
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for pt in points:
                payload = pt.payload or {}
                name = payload.get("name", "")
                if not name:
                    continue

                # Parse JSON-encoded calls list
                calls_raw = payload.get("calls", "[]")
                try:
                    calls = json.loads(calls_raw) if isinstance(calls_raw, str) else calls_raw
                except (json.JSONDecodeError, TypeError):
                    calls = []

                func_dict[name] = {
                    "file_path": payload.get("file_path", ""),
                    "line_start": payload.get("line_start", 0),
                    "line_end": payload.get("line_end", 0),
                    "calls": calls if isinstance(calls, list) else [],
                    "language": payload.get("language", ""),
                    "signature": payload.get("signature", ""),
                    "project": payload.get("project", ""),
                }

            if next_offset is None:
                break
            offset = next_offset

        logger.info(
            "Built func_dict: %d functions (project=%s, language=%s)",
            len(func_dict), project, language,
        )
        return func_dict

    # ------------------------------------------------------------------
    # Phase 2: Parse entry file for top-level calls
    # ------------------------------------------------------------------

    def _find_entry_file(self, entry_file: str) -> Optional[str]:
        """Resolve entry_file to an absolute path on disk.

        Tries:
        1. Exact path
        2. Under /workspace/*/code/
        3. Recursive search under /workspace/
        """
        if os.path.isfile(entry_file):
            return entry_file

        # Search under /workspace
        workspace = "/workspace"
        if os.path.isdir(workspace):
            for root, dirs, files in os.walk(workspace):
                # Skip hidden dirs and common excludes
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                           ("node_modules", "__pycache__", ".venv", "venv")]
                for f in files:
                    if f == os.path.basename(entry_file):
                        return os.path.join(root, f)

        return None

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        lang_map = {
            ".ps1": "powershell",
            ".psm1": "powershell",
            ".psd1": "powershell",
            ".py": "python",
            ".sh": "bash",
            ".bash": "bash",
        }
        return lang_map.get(ext, "unknown")

    def _extract_toplevel_calls(
        self, content: str, language: str, functions: List[Dict],
    ) -> List[str]:
        """Extract function calls from top-level code (not inside any function)."""
        # Build set of lines that belong to functions
        func_lines: Set[int] = set()
        for f in functions:
            start = f.get("line_start", 0)
            end = f.get("line_end", 0)
            if start and end:
                func_lines.update(range(start, end + 1))

        # Extract top-level code
        lines = content.split("\n")
        toplevel_lines = []
        for i, line in enumerate(lines, 1):
            if i not in func_lines:
                toplevel_lines.append(line)
        toplevel_code = "\n".join(toplevel_lines)

        # Extract calls based on language
        return self._extract_calls_for_language(toplevel_code, language)

    def _extract_calls_for_language(self, code: str, language: str) -> List[str]:
        """Extract function calls from code based on language."""
        calls: Set[str] = set()

        if language == "powershell":
            # Verb-Noun pattern: Get-VM, Build-WindowsVM, etc.
            for m in re.finditer(r'\b([A-Z][a-z]+-[A-Z][a-zA-Z0-9]+)\b', code):
                calls.add(m.group(1))

        elif language == "python":
            # function_name( pattern
            for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code):
                name = m.group(1)
                # Skip builtins and common keywords
                if name not in {
                    "print", "len", "range", "str", "int", "float", "list",
                    "dict", "set", "tuple", "type", "isinstance", "hasattr",
                    "getattr", "setattr", "open", "super", "enumerate",
                    "zip", "map", "filter", "sorted", "reversed", "any",
                    "all", "min", "max", "sum", "abs", "round", "if",
                    "for", "while", "with", "assert", "raise", "return",
                    "import", "from", "class", "def", "async", "await",
                }:
                    calls.add(name)

        elif language == "bash":
            # function calls: word at start of line or after | ; && ||
            for m in re.finditer(r'(?:^|[|;&]\s*)([a-zA-Z_][a-zA-Z0-9_-]*)\b', code, re.MULTILINE):
                name = m.group(1)
                # Skip common bash builtins
                if name not in {
                    "echo", "cd", "ls", "cat", "grep", "awk", "sed",
                    "if", "then", "else", "fi", "for", "do", "done",
                    "while", "until", "case", "esac", "in", "function",
                    "return", "exit", "export", "local", "declare",
                    "read", "source", "true", "false", "test", "set",
                    "unset", "shift", "trap", "eval", "exec",
                }:
                    calls.add(name)

        return sorted(calls)

    # ------------------------------------------------------------------
    # Phase 3: BFS call resolution
    # ------------------------------------------------------------------

    def _bfs_resolve(
        self,
        entry_calls: List[str],
        func_dict: Dict[str, Dict[str, Any]],
        max_depth: int = 15,
    ) -> Tuple[Set[str], Set[str]]:
        """BFS traversal to collect all needed functions.

        Returns: (needed_functions, external_deps)
        """
        needed: Set[str] = set()
        external: Set[str] = set()
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque(
            (name, 0) for name in entry_calls
        )

        while queue:
            name, depth = queue.popleft()
            if name in visited:
                continue
            visited.add(name)

            if name in func_dict:
                needed.add(name)
                if depth < max_depth:
                    for callee in func_dict[name]["calls"]:
                        if callee not in visited:
                            queue.append((callee, depth + 1))
            else:
                external.add(name)

        return needed, external

    # ------------------------------------------------------------------
    # Phase 4: Read full bodies from disk
    # ------------------------------------------------------------------

    def _read_function_body(
        self, func_info: Dict[str, Any],
    ) -> Optional[str]:
        """Read the full function body from the source file on disk."""
        file_path = func_info.get("file_path", "")
        line_start = func_info.get("line_start", 0)
        line_end = func_info.get("line_end", 0)

        if not file_path or not line_start or not line_end:
            return None

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[line_start - 1 : line_end])
        except (FileNotFoundError, IOError) as e:
            logger.warning("Cannot read %s: %s", file_path, e)
            return None

    # ------------------------------------------------------------------
    # trace_execution handler
    # ------------------------------------------------------------------

    async def _handle_trace(self, inp: Dict[str, Any]) -> str:
        """Handle trace_execution tool call."""
        entry_file = inp.get("entry_file", "").strip()
        entry_function = inp.get("entry_function", "").strip()
        project = inp.get("project", "").strip() or None
        language = inp.get("language", "").strip() or None
        max_depth = min(max(inp.get("max_depth", 15), 1), 15)

        if not entry_file and not entry_function:
            return json.dumps({
                "error": "Provide either entry_file or entry_function",
            })

        # Build function dictionary from Qdrant
        func_dict = await self._build_func_dict(project=project, language=language)
        if not func_dict:
            return json.dumps({
                "error": "No functions found in index. Run workspace reindex first.",
                "project": project,
                "language": language,
            })

        entry_calls: List[str] = []
        entry_file_path: Optional[str] = None
        entry_file_content: Optional[str] = None
        detected_language = language

        if entry_file:
            # Resolve entry file path
            entry_file_path = self._find_entry_file(entry_file)
            if not entry_file_path:
                return json.dumps({
                    "error": f"Entry file not found: {entry_file}",
                    "searched": "/workspace/",
                })

            # Read and parse entry file
            try:
                with open(entry_file_path, "r", encoding="utf-8", errors="replace") as f:
                    entry_file_content = f.read()
            except IOError as e:
                return json.dumps({"error": f"Cannot read entry file: {e}"})

            detected_language = language or self._detect_language(entry_file_path)

            # Parse with tree-sitter to find functions in the entry file
            try:
                from .tree_sitter_parser import TreeSitterParser
                parser = TreeSitterParser.get_instance()
                functions = parser.extract_functions(
                    entry_file_content, detected_language, entry_file_path,
                )
            except Exception as e:
                logger.warning("Tree-sitter parse failed, using regex: %s", e)
                functions = []

            # Extract top-level calls (outside any function)
            entry_calls = self._extract_toplevel_calls(
                entry_file_content, detected_language, functions,
            )

            # Also add calls from functions defined in the entry file
            for func in functions:
                func_name = func.get("name", "")
                if func_name:
                    entry_calls.append(func_name)
                    # Add the function's own callees
                    calls = func.get("calls", [])
                    if isinstance(calls, list):
                        entry_calls.extend(calls)

        elif entry_function:
            # Start from a specific function
            if entry_function in func_dict:
                entry_calls = [entry_function]
            else:
                # Try fuzzy match
                matches = [
                    name for name in func_dict
                    if name.lower() == entry_function.lower()
                ]
                if matches:
                    entry_calls = [matches[0]]
                else:
                    return json.dumps({
                        "error": f"Function not found in index: {entry_function}",
                        "available_count": len(func_dict),
                    })

        # BFS resolution
        needed, external = self._bfs_resolve(entry_calls, func_dict, max_depth)

        # Build trace result
        trace_id = str(uuid.uuid4())[:8]
        trace_result = {
            "trace_id": trace_id,
            "entry_file": entry_file_path,
            "entry_function": entry_function or None,
            "project": project,
            "language": detected_language,
            "max_depth": max_depth,
            "needed_functions": sorted(needed),
            "needed_count": len(needed),
            "external_deps": sorted(external),
            "external_count": len(external),
            "func_dict": {
                name: {
                    "file_path": func_dict[name]["file_path"],
                    "line_start": func_dict[name]["line_start"],
                    "line_end": func_dict[name]["line_end"],
                    "signature": func_dict[name]["signature"],
                    "calls": func_dict[name]["calls"],
                }
                for name in needed
                if name in func_dict
            },
            "source_files": sorted(set(
                func_dict[name]["file_path"]
                for name in needed
                if name in func_dict and func_dict[name]["file_path"]
            )),
            "entry_file_content": entry_file_content,
        }

        # Cache for assemble_traced_code
        _trace_cache[trace_id] = trace_result

        # Return summary (without full content to save tokens)
        summary = {
            "trace_id": trace_id,
            "entry_file": entry_file_path,
            "entry_function": entry_function or None,
            "project": project,
            "language": detected_language,
            "needed_functions": sorted(needed),
            "needed_count": len(needed),
            "external_deps": sorted(external),
            "external_count": len(external),
            "source_files": trace_result["source_files"],
            "source_file_count": len(trace_result["source_files"]),
        }

        # Add brief info about each needed function
        func_summary = []
        for name in sorted(needed):
            if name in func_dict:
                info = func_dict[name]
                func_summary.append({
                    "name": name,
                    "file": os.path.basename(info["file_path"]),
                    "lines": f"{info['line_start']}-{info['line_end']}",
                    "calls": info["calls"][:5],  # First 5 callees
                })
        summary["functions"] = func_summary

        return json.dumps(summary, indent=2)

    # ------------------------------------------------------------------
    # assemble_traced_code handler
    # ------------------------------------------------------------------

    async def _handle_assemble(self, inp: Dict[str, Any]) -> str:
        """Handle assemble_traced_code tool call."""
        trace_id = inp.get("trace_id", "").strip()
        output_format = inp.get("output_format", "single_file")
        include_entry = inp.get("include_entry_file", True)

        if not trace_id or trace_id not in _trace_cache:
            return json.dumps({
                "error": f"Trace not found: {trace_id}. Run trace_execution first.",
                "available_traces": list(_trace_cache.keys()),
            })

        trace = _trace_cache[trace_id]
        func_dict = trace["func_dict"]
        needed = trace["needed_functions"]
        external = trace["external_deps"]
        entry_file = trace["entry_file"]
        entry_content = trace["entry_file_content"]
        language = trace.get("language", "unknown")

        # Determine comment prefix
        comment = "#" if language in ("powershell", "python", "bash") else "//"

        # Read all function bodies from disk
        bodies: Dict[str, str] = {}
        missing: List[str] = []
        for name in needed:
            if name not in func_dict:
                missing.append(name)
                continue
            body = self._read_function_body(func_dict[name])
            if body:
                bodies[name] = body
            else:
                missing.append(name)

        if output_format == "minimal":
            # Just signatures and call chains
            lines = [
                f"{comment} ═══════════════════════════════════════════════",
                f"{comment} Code Trace: {os.path.basename(entry_file or 'function')}",
                f"{comment} Functions: {len(bodies)} | External: {len(external)}",
                f"{comment} ═══════════════════════════════════════════════",
                "",
            ]
            for name in sorted(bodies.keys()):
                info = func_dict[name]
                sig = info.get("signature", name)
                calls_str = ", ".join(info.get("calls", [])[:10])
                lines.append(f"{comment} {sig}")
                if calls_str:
                    lines.append(f"{comment}   calls: {calls_str}")
                lines.append("")

            if external:
                lines.append(f"{comment} External dependencies:")
                for dep in sorted(external):
                    lines.append(f"{comment}   - {dep}")

            return "\n".join(lines)

        elif output_format == "grouped_by_source":
            # Group functions by source file
            by_file: Dict[str, List[str]] = {}
            for name in sorted(bodies.keys()):
                fp = func_dict[name]["file_path"]
                by_file.setdefault(fp, []).append(name)

            lines = [
                f"{comment} ═══════════════════════════════════════════════",
                f"{comment} Code Trace: {os.path.basename(entry_file or 'function')}",
                f"{comment} Project: {trace.get('project', 'unknown')}",
                f"{comment} Functions: {len(bodies)} from {len(by_file)} files",
                f"{comment} External deps: {len(external)}",
                f"{comment} ═══════════════════════════════════════════════",
                "",
            ]

            if include_entry and entry_content:
                lines.append(f"{comment} ──── Entry File: {entry_file} ────")
                lines.append(entry_content)
                lines.append("")

            for fp, func_names in sorted(by_file.items()):
                lines.append(
                    f"{comment} ──── Source: {os.path.basename(fp)} ────"
                )
                for name in func_names:
                    lines.append(f"{comment} Function: {name}")
                    lines.append(bodies[name])
                    lines.append("")
                lines.append("")

            if external:
                lines.append(f"{comment} External dependencies (not in project):")
                for dep in sorted(external):
                    lines.append(f"{comment}   - {dep}")

            return "\n".join(lines)

        else:
            # single_file (default)
            lines = [
                f"{comment} ═══════════════════════════════════════════════",
                f"{comment} Code Trace Assembly",
                f"{comment} Entry: {os.path.basename(entry_file or 'function')}",
                f"{comment} Project: {trace.get('project', 'unknown')}",
                f"{comment} Functions: {len(bodies)}",
                f"{comment} External deps: {len(external)}",
                f"{comment} ═══════════════════════════════════════════════",
                "",
            ]

            if external:
                lines.append(f"{comment} External dependencies (not in this project):")
                for dep in sorted(external):
                    lines.append(f"{comment}   - {dep}")
                lines.append("")

            if include_entry and entry_content:
                lines.append(
                    f"{comment} ═══ Entry File: "
                    f"{os.path.basename(entry_file or '')} ═══"
                )
                lines.append(entry_content)
                lines.append("")

            # Sort by call depth: functions called directly by entry first
            entry_calls = set()
            if entry_content:
                entry_calls = set(
                    self._extract_calls_for_language(entry_content, language)
                )

            # Direct callees first, then deeper functions
            direct = [n for n in sorted(bodies.keys()) if n in entry_calls]
            indirect = [n for n in sorted(bodies.keys()) if n not in entry_calls]

            for name in direct + indirect:
                info = func_dict[name]
                src = os.path.basename(info["file_path"])
                lines.append(
                    f"{comment} ═══ {name} (from {src}:"
                    f"{info['line_start']}-{info['line_end']}) ═══"
                )
                lines.append(bodies[name])
                lines.append("")

            if missing:
                lines.append(f"{comment} WARNING: Could not read bodies for:")
                for name in sorted(missing):
                    lines.append(f"{comment}   - {name}")

            return "\n".join(lines)
