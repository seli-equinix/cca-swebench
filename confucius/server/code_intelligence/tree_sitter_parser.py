"""
Tree-sitter AST Parser for Code Intelligence

Provides accurate function/class extraction with rich metadata using
Abstract Syntax Tree parsing instead of regex-based heuristics.

Supports: Python, Bash, PowerShell, JavaScript, TypeScript, Rust, Go

Author: Claude Code (Code Intelligence Upgrade - Phase 1)
Date: 2026-02-09
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from tree_sitter import Language, Parser, Node

logger = logging.getLogger(__name__)

# Path to built language library
LANGUAGE_LIBRARY_PATH = "/usr/local/lib/tree-sitter-languages.so"


class TreeSitterParser:
    """
    Singleton AST parser for multiple programming languages

    Extracts functions, classes, and metadata with high accuracy using
    tree-sitter's language-specific parsers.

    Usage:
        parser = TreeSitterParser.get_instance()
        functions = parser.extract_functions(code, "python", "example.py")
    """

    _instance = None

    # Languages with full tree-sitter support
    # PowerShell: Using airbus-cert grammar (PS 7.3 spec, comprehensive function support)
    SUPPORTED_LANGUAGES = {
        "python": "python",
        "bash": "bash",
        "powershell": "powershell",  # airbus-cert exports tree_sitter_powershell() (lowercase)
        "yaml": "yaml",
        "markdown": "markdown",
    }

    # Languages that extract sections (not functions)
    SECTION_LANGUAGES = {"yaml", "markdown"}

    def __init__(self):
        """Initialize parsers for all supported languages"""
        self.parsers: Dict[str, Parser] = {}
        self.languages: Dict[str, Language] = {}
        self._init_parsers()
        logger.info(f"TreeSitterParser initialized with {len(self.parsers)} languages")

    @classmethod
    def get_instance(cls) -> "TreeSitterParser":
        """Get singleton instance of TreeSitterParser"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_parsers(self):
        """Initialize tree-sitter parsers for all supported languages"""
        for lang_key, lang_name in self.SUPPORTED_LANGUAGES.items():
            try:
                # Load language from built library
                language = Language(LANGUAGE_LIBRARY_PATH, lang_name)
                parser = Parser()
                parser.set_language(language)
                self.languages[lang_key] = language
                self.parsers[lang_key] = parser
                logger.debug(f"Loaded parser for {lang_key}")
            except Exception as e:
                logger.warning(f"Failed to load parser for {lang_key}: {e}")

    def extract_functions(
        self,
        code: str,
        language: str,
        file_path: str
    ) -> List[Dict]:
        """
        Extract function definitions with rich metadata

        Args:
            code: Source code content
            language: Language identifier (python, bash, javascript, etc.)
            file_path: File path for error reporting

        Returns:
            List of function metadata dictionaries:
            {
                "type": "function",
                "name": str,
                "signature": str,
                "parameters": List[{"name": str, "type": Optional[str], "default": Optional[str]}],
                "return_type": Optional[str],
                "decorators": List[str],
                "is_async": bool,
                "is_generator": bool,
                "docstring": Optional[str],
                "body": str,
                "line_start": int,
                "line_end": int,
                "calls": List[str],  # Function calls within this function
                "imports": List[str],
                "loc": int,  # Lines of code
            }
        """
        if language not in self.SUPPORTED_LANGUAGES:
            logger.debug(f"Language {language} not supported by tree-sitter, using fallback")
            return []  # Will fallback to regex in codebase_indexer

        if language not in self.parsers:
            logger.warning(f"Parser for {language} failed to initialize, using fallback")
            return []

        try:
            parser = self.parsers[language]
            tree = parser.parse(bytes(code, "utf8"))

            if language == "python":
                return self._extract_python_functions(tree, code, file_path)
            elif language == "bash":
                return self._extract_bash_functions(tree, code, file_path)
            elif language == "powershell":
                return self._extract_powershell_functions(tree, code, file_path)
            else:
                logger.warning(f"Extraction not implemented for {language}")
                return []

        except Exception as e:
            logger.error(f"Error parsing {file_path} as {language}: {e}")
            return []

    def extract_sections(
        self,
        code: str,
        language: str,
        file_path: str
    ) -> List[Dict]:
        """
        Extract structured sections from non-code files (YAML, Markdown).

        Returns a list of section dicts with the same shape as function dicts
        so the indexer can store them uniformly:
        {
            "type": "section",
            "name": str,           # section/key name
            "body": str,           # section content
            "docstring": str,      # description or summary
            "line_start": int,
            "line_end": int,
            "loc": int,
            "signature": str,      # key path or heading level
            "parameters": [],
            "decorators": [],
            "calls": [],
            "imports": [],
            "is_async": False,
            "is_generator": False,
            "return_type": None,
            "metadata": dict,      # extra type-specific metadata
        }
        """
        if language not in self.SECTION_LANGUAGES:
            return []
        if language not in self.parsers:
            return []

        try:
            parser = self.parsers[language]
            tree = parser.parse(bytes(code, "utf8"))

            if language == "yaml":
                return self._extract_yaml_sections(tree, code, file_path)
            elif language == "markdown":
                return self._extract_markdown_sections(tree, code, file_path)
            else:
                return []
        except Exception as e:
            logger.error(f"Error extracting sections from {file_path} as {language}: {e}")
            return []

    def _make_section(
        self,
        name: str,
        body: str,
        signature: str,
        line_start: int,
        line_end: int,
        docstring: str = "",
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Create a section dict with the same shape as function dicts for uniform indexing."""
        return {
            "type": "section",
            "name": name,
            "body": body[:10000],
            "docstring": docstring[:2000],
            "signature": signature,
            "line_start": line_start,
            "line_end": line_end,
            "loc": line_end - line_start + 1,
            "parameters": [],
            "decorators": [],
            "calls": [],
            "imports": [],
            "is_async": False,
            "is_generator": False,
            "return_type": None,
            "metadata": metadata or {},
        }

    # ── YAML section extractor ──────────────────────────────────────────

    def _extract_yaml_sections(self, tree, code: str, file_path: str) -> List[Dict]:
        """
        Extract structured sections from YAML files using tree-sitter AST.

        Extracts top-level mapping keys as sections. For each key, captures:
        - The key name and full key path
        - The value content (scalar, sequence, or nested mapping)
        - Nested keys for Docker Compose service detection (image, ports, etc.)
        """
        sections = []
        code_lines = code.split("\n")
        root = tree.root_node

        # Walk the AST for top-level block_mapping_pair nodes
        for doc_node in root.children:
            if doc_node.type == "document":
                self._walk_yaml_mapping(doc_node, code_lines, sections, file_path, key_path="")
            elif doc_node.type == "block_mapping_pair":
                self._parse_yaml_pair(doc_node, code_lines, sections, file_path, key_path="")

        # If nothing found via document nodes, try root children directly
        if not sections:
            for child in root.children:
                if child.type == "block_mapping":
                    for pair in child.children:
                        if pair.type == "block_mapping_pair":
                            self._parse_yaml_pair(pair, code_lines, sections, file_path, key_path="")

        return sections

    def _walk_yaml_mapping(
        self, node, code_lines: List[str], sections: List[Dict],
        file_path: str, key_path: str
    ):
        """Recursively walk YAML nodes to find block_mapping_pair entries."""
        for child in node.children:
            if child.type == "block_mapping_pair":
                self._parse_yaml_pair(child, code_lines, sections, file_path, key_path)
            elif child.type == "block_mapping":
                for pair in child.children:
                    if pair.type == "block_mapping_pair":
                        self._parse_yaml_pair(pair, code_lines, sections, file_path, key_path)
            elif child.type in ("block_node", "document"):
                self._walk_yaml_mapping(child, code_lines, sections, file_path, key_path)

    def _parse_yaml_pair(
        self, pair_node, code_lines: List[str], sections: List[Dict],
        file_path: str, key_path: str
    ):
        """Parse a single YAML block_mapping_pair into a section."""
        key_node = pair_node.child_by_field_name("key")
        value_node = pair_node.child_by_field_name("value")

        if not key_node:
            return

        key_text = key_node.text.decode("utf8").strip()
        full_path = f"{key_path}.{key_text}" if key_path else key_text

        line_start = pair_node.start_point[0] + 1
        line_end = pair_node.end_point[0] + 1
        body = "\n".join(code_lines[line_start - 1 : line_end])

        # Collect nested key names for metadata
        nested_keys = []
        if value_node:
            self._collect_yaml_nested_keys(value_node, nested_keys)

        # Build a description from nested keys (useful for services, config blocks)
        docstring = ""
        if nested_keys:
            docstring = f"Keys: {', '.join(nested_keys[:20])}"

        meta = {"key_path": full_path, "nested_keys": nested_keys[:50]}

        sections.append(self._make_section(
            name=key_text,
            body=body,
            signature=full_path,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            metadata=meta,
        ))

    def _collect_yaml_nested_keys(self, node, keys: List[str], depth: int = 0):
        """Collect immediate child key names from a YAML mapping value."""
        if depth > 2:
            return
        for child in node.children:
            if child.type == "block_mapping_pair":
                key_node = child.child_by_field_name("key")
                if key_node:
                    keys.append(key_node.text.decode("utf8").strip())
            elif child.type in ("block_mapping", "block_node"):
                self._collect_yaml_nested_keys(child, keys, depth + 1)

    # ── Markdown section extractor ──────────────────────────────────────

    def _extract_markdown_sections(self, tree, code: str, file_path: str) -> List[Dict]:
        """
        Extract structured sections from Markdown files using tree-sitter AST.

        Extracts:
        - Headings (ATX: # H1, ## H2, etc.) with their following content
        - Fenced code blocks with language info
        """
        sections = []
        code_lines = code.split("\n")
        root = tree.root_node

        # Pass 1: Collect all headings and their positions
        headings = []
        self._find_markdown_headings(root, code_lines, headings)

        # Pass 2: Assign content ranges to headings (content = text until next heading or EOF)
        for i, heading in enumerate(headings):
            content_start = heading["line_end"]
            if i + 1 < len(headings):
                content_end = headings[i + 1]["line_start"] - 1
            else:
                content_end = len(code_lines)

            body_lines = code_lines[heading["line_start"] - 1 : content_end]
            body = "\n".join(body_lines)

            sections.append(self._make_section(
                name=heading["text"],
                body=body,
                signature=f"{'#' * heading['level']} {heading['text']}",
                line_start=heading["line_start"],
                line_end=content_end,
                docstring=heading.get("summary", ""),
                metadata={"heading_level": heading["level"]},
            ))

        # Pass 3: Extract fenced code blocks as separate sections
        self._find_markdown_code_blocks(root, code_lines, sections)

        return sections

    def _find_markdown_headings(self, node, code_lines: List[str], headings: List[Dict]):
        """Find all ATX headings in the Markdown AST."""
        if node.type == "atx_heading":
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            heading_text = node.text.decode("utf8").strip()

            # Determine heading level by counting leading #
            level = 0
            for ch in heading_text:
                if ch == "#":
                    level += 1
                else:
                    break

            # Remove # markers from heading text
            clean_text = heading_text.lstrip("#").strip()

            # Grab first non-empty line after heading as summary
            summary = ""
            for line in code_lines[line_end : min(line_end + 5, len(code_lines))]:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    summary = stripped[:200]
                    break

            headings.append({
                "text": clean_text,
                "level": level,
                "line_start": line_start,
                "line_end": line_end,
                "summary": summary,
            })
            return  # Don't recurse into heading children

        for child in node.children:
            self._find_markdown_headings(child, code_lines, headings)

    def _find_markdown_code_blocks(self, node, code_lines: List[str], sections: List[Dict]):
        """Find fenced code blocks in the Markdown AST."""
        if node.type == "fenced_code_block":
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            body = "\n".join(code_lines[line_start - 1 : line_end])

            # Extract language from info string (```python, ```bash, etc.)
            lang = ""
            for child in node.children:
                if child.type == "info_string":
                    lang = child.text.decode("utf8").strip()
                    break

            name = f"code_block:{lang}" if lang else "code_block"

            sections.append(self._make_section(
                name=name,
                body=body,
                signature=f"```{lang}" if lang else "```",
                line_start=line_start,
                line_end=line_end,
                metadata={"block_language": lang},
            ))
            return

        for child in node.children:
            self._find_markdown_code_blocks(child, code_lines, sections)

    def _extract_python_functions(
        self,
        tree,
        code: str,
        file_path: str
    ) -> List[Dict]:
        """
        Extract Python function and class definitions

        Captures:
        - Function name, parameters with type hints, return type
        - Decorators (@property, @staticmethod, etc.)
        - Async/generator detection
        - Docstrings
        - Function calls within body
        """
        functions = []
        code_lines = code.split("\n")

        # Extract file-level imports once (injected into every function)
        file_imports = self._extract_python_imports(tree)

        # Query for function definitions
        function_query = self.languages["python"].query("""
            (function_definition
                name: (identifier) @func_name
                parameters: (parameters) @params
                return_type: (type)? @return_type
                body: (block) @body
            ) @function
        """)

        # Query for class definitions
        class_query = self.languages["python"].query("""
            (class_definition
                name: (identifier) @class_name
                superclasses: (argument_list)? @superclasses
                body: (block) @body
            ) @class
        """)

        # Extract top-level functions (skip class methods - handled below)
        captures = function_query.captures(tree.root_node)
        for node, capture_name in captures:
            if capture_name == "function":
                # Skip methods inside classes (grandparent is class_definition)
                parent = node.parent
                grandparent = parent.parent if parent else None
                if grandparent and grandparent.type == "class_definition":
                    continue
                func_data = self._parse_python_function(node, code_lines, file_path)
                if func_data:
                    func_data["imports"] = file_imports
                    functions.append(func_data)

        # Extract class methods
        class_captures = class_query.captures(tree.root_node)
        for node, capture_name in class_captures:
            if capture_name == "class":
                # Also extract methods within classes
                class_methods = self._extract_python_class_methods(node, code_lines, file_path)
                for method in class_methods:
                    method["imports"] = file_imports
                functions.extend(class_methods)

        return functions

    def _parse_python_function(
        self,
        node: Node,
        code_lines: List[str],
        file_path: str
    ) -> Optional[Dict]:
        """Parse a single Python function node into metadata dict"""
        try:
            # Get function name
            name_node = node.child_by_field_name("name")
            func_name = name_node.text.decode("utf8") if name_node else "unknown"

            # Get parameters
            params_node = node.child_by_field_name("parameters")
            parameters = self._parse_python_parameters(params_node) if params_node else []

            # Get return type
            return_type_node = node.child_by_field_name("return_type")
            return_type = return_type_node.text.decode("utf8") if return_type_node else None

            # Get body
            body_node = node.child_by_field_name("body")
            body_text = body_node.text.decode("utf8") if body_node else ""

            # Build signature
            params_str = ", ".join([self._param_to_string(p) for p in parameters])
            signature = f"def {func_name}({params_str})"
            if return_type:
                signature += f" -> {return_type}"
            signature += ":"

            # Check for async
            is_async = False
            for child in node.children:
                if child.type == "async":
                    is_async = True
                    signature = "async " + signature
                    break

            # Check for decorators
            decorators = []
            prev_sibling = node.prev_sibling
            while prev_sibling and prev_sibling.type == "decorator":
                decorator_text = prev_sibling.text.decode("utf8")
                decorators.insert(0, decorator_text)  # Insert at beginning to preserve order
                prev_sibling = prev_sibling.prev_sibling

            # Extract docstring
            docstring = self._extract_python_docstring(body_node)

            # Extract function calls
            calls = self._extract_python_calls(body_node)

            # Calculate LOC
            line_start = node.start_point[0]
            line_end = node.end_point[0]
            loc = line_end - line_start + 1

            # Check if generator
            is_generator = "yield" in body_text

            return {
                "type": "function",
                "name": func_name,
                "signature": signature,
                "parameters": parameters,
                "return_type": return_type,
                "decorators": decorators,
                "is_async": is_async,
                "is_generator": is_generator,
                "docstring": docstring,
                "body": body_text,
                "line_start": line_start + 1,  # 1-indexed for humans
                "line_end": line_end + 1,
                "calls": calls,
                "imports": [],  # Will be extracted separately at file level
                "loc": loc,
            }

        except Exception as e:
            logger.error(f"Error parsing Python function in {file_path}: {e}")
            return None

    def _parse_python_parameters(self, params_node: Node) -> List[Dict]:
        """Parse Python function parameters with type hints"""
        parameters = []

        for child in params_node.children:
            if child.type == "identifier":
                # Simple parameter without type hint
                parameters.append({
                    "name": child.text.decode("utf8"),
                    "type": None,
                    "default": None
                })
            elif child.type == "typed_parameter":
                # Parameter with type hint
                name = None
                param_type = None
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = subchild.text.decode("utf8")
                    elif subchild.type == "type":
                        param_type = subchild.text.decode("utf8")

                if name:
                    parameters.append({
                        "name": name,
                        "type": param_type,
                        "default": None
                    })
            elif child.type == "typed_default_parameter":
                # Parameter with type hint AND default value (e.g. y: str = "default")
                name = None
                param_type = None
                default = None
                for subchild in child.children:
                    if subchild.type == "identifier" and name is None:
                        name = subchild.text.decode("utf8")
                    elif subchild.type == "type":
                        param_type = subchild.text.decode("utf8")
                    elif subchild.type in ["integer", "string", "true", "false", "none",
                                           "float", "list", "dictionary", "tuple", "set",
                                           "unary_operator", "call", "attribute"]:
                        default = subchild.text.decode("utf8")

                if name:
                    parameters.append({
                        "name": name,
                        "type": param_type,
                        "default": default
                    })
            elif child.type == "default_parameter":
                # Parameter with default value only (e.g. z = 5)
                name = None
                default = None
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = subchild.text.decode("utf8")
                    elif subchild.type in ["integer", "string", "true", "false", "none",
                                           "float", "list", "dictionary", "tuple", "set",
                                           "unary_operator", "call", "attribute"]:
                        default = subchild.text.decode("utf8")

                if name:
                    parameters.append({
                        "name": name,
                        "type": None,
                        "default": default
                    })

        return parameters

    def _param_to_string(self, param: Dict) -> str:
        """Convert parameter dict to string representation"""
        parts = [param["name"]]
        if param["type"]:
            parts.append(f": {param['type']}")
        if param["default"]:
            parts.append(f" = {param['default']}")
        return "".join(parts)

    def _extract_python_docstring(self, body_node: Node) -> Optional[str]:
        """Extract docstring from function body"""
        if not body_node or not body_node.children:
            return None

        # Find the first expression_statement in the body (docstring)
        for child in body_node.children:
            if child.type == "expression_statement":
                for subchild in child.children:
                    if subchild.type == "string":
                        docstring = subchild.text.decode("utf8")
                        # Remove quotes
                        docstring = docstring.strip('"""').strip("'''").strip('"').strip("'")
                        return docstring.strip()
                break  # Only check first expression_statement
            elif child.type not in ["comment", "newline"]:
                break  # Non-docstring statement found first

        return None

    # Python wrappers that take a callable as the first argument
    _PYTHON_CALLABLE_WRAPPERS = {
        "asyncio.to_thread", "asyncio.run_in_executor",
        "threading.Thread", "multiprocessing.Process",
        "functools.partial", "map", "filter",
        "concurrent.futures.ThreadPoolExecutor.submit",
        "concurrent.futures.ProcessPoolExecutor.submit",
    }

    def _extract_python_calls(self, body_node: Node) -> List[str]:
        """Extract function/method calls from function body"""
        calls = set()

        def traverse(node: Node):
            if node.type == "call":
                # Extract the function being called
                function_node = node.child_by_field_name("function")
                if function_node:
                    call_text = function_node.text.decode("utf8")
                    # Limit to reasonable length (avoid long expressions)
                    if len(call_text) < 100:
                        calls.add(call_text)

                    # Detect callable-wrapper patterns (e.g. asyncio.to_thread(self.func, ...))
                    # The first positional argument is the actual function being called
                    if call_text in self._PYTHON_CALLABLE_WRAPPERS or call_text.endswith((".submit", ".to_thread")):
                        args_node = node.child_by_field_name("arguments")
                        if args_node:
                            for arg_child in args_node.children:
                                if arg_child.type in ("identifier", "attribute"):
                                    indirect_call = arg_child.text.decode("utf8")
                                    if len(indirect_call) < 100:
                                        calls.add(indirect_call)
                                    break  # Only first positional argument

            for child in node.children:
                traverse(child)

        if body_node:
            traverse(body_node)

        return sorted(list(calls))[:50]  # Limit to 50 calls

    def _extract_python_imports(self, tree) -> List[str]:
        """Extract top-level module names from Python import statements.

        Returns unique top-level module names, e.g.:
        - 'import os.path' -> 'os'
        - 'from typing import Dict' -> 'typing'
        - 'import json' -> 'json'
        """
        modules = set()

        def traverse(node):
            if node.type == "import_statement":
                # 'import os.path' or 'import json, sys'
                for child in node.children:
                    if child.type == "dotted_name":
                        # Take the first component: os.path -> os
                        parts = child.text.decode("utf8").split(".")
                        if parts and parts[0]:
                            modules.add(parts[0])
                    elif child.type == "aliased_import":
                        # 'import numpy as np' -> numpy
                        for sub in child.children:
                            if sub.type == "dotted_name":
                                parts = sub.text.decode("utf8").split(".")
                                if parts and parts[0]:
                                    modules.add(parts[0])
                                break
            elif node.type == "import_from_statement":
                # 'from typing import Dict' -> typing
                for child in node.children:
                    if child.type == "dotted_name":
                        parts = child.text.decode("utf8").split(".")
                        if parts and parts[0]:
                            modules.add(parts[0])
                        break  # Only the first dotted_name is the module
                    elif child.type == "relative_import":
                        # 'from . import foo' or 'from ..utils import bar'
                        # Skip relative imports - they're internal
                        break
            else:
                for child in node.children:
                    traverse(child)

        traverse(tree.root_node)
        return sorted(list(modules))

    def _extract_python_class_methods(
        self,
        class_node: Node,
        code_lines: List[str],
        file_path: str
    ) -> List[Dict]:
        """Extract methods from a Python class"""
        methods = []

        # Find the class body
        body_node = class_node.child_by_field_name("body")
        if not body_node:
            return methods

        # Get class name
        class_name_node = class_node.child_by_field_name("name")
        class_name = class_name_node.text.decode("utf8") if class_name_node else "Unknown"

        # Find all function definitions in the class body
        for child in body_node.children:
            if child.type == "function_definition":
                method_data = self._parse_python_function(child, code_lines, file_path)
                if method_data:
                    # Add class_name to metadata
                    method_data["class_name"] = class_name
                    methods.append(method_data)

        return methods

    def _extract_bash_functions(
        self,
        tree,
        code: str,
        file_path: str
    ) -> List[Dict]:
        """
        Extract Bash function definitions

        Captures:
        - Function name
        - Function body
        - Comments preceding function (treated as documentation)
        """
        functions = []
        code_lines = code.split("\n")

        # Extract file-level sourced files once (injected into every function)
        file_imports = self._extract_bash_imports(tree)

        # Query for function definitions
        # Bash has two syntaxes: "function name() { }" and "name() { }"
        function_query = self.languages["bash"].query("""
            (function_definition
                name: (word) @func_name
                body: (compound_statement) @body
            ) @function
        """)

        captures = function_query.captures(tree.root_node)
        for node, capture_name in captures:
            if capture_name == "function":
                func_data = self._parse_bash_function(node, code_lines, file_path)
                if func_data:
                    func_data["imports"] = file_imports
                    functions.append(func_data)

        return functions

    def _parse_bash_function(
        self,
        node: Node,
        code_lines: List[str],
        file_path: str
    ) -> Optional[Dict]:
        """Parse a single Bash function node into metadata dict"""
        try:
            # Get function name
            name_node = node.child_by_field_name("name")
            func_name = name_node.text.decode("utf8") if name_node else "unknown"

            # Get body
            body_node = node.child_by_field_name("body")
            body_text = body_node.text.decode("utf8") if body_node else ""

            # Build signature (Bash doesn't have formal parameters)
            signature = f"{func_name}() {{"

            # Extract preceding comments as documentation
            docstring = self._extract_bash_comments(node, code_lines)

            # Extract command calls using AST
            calls = self._extract_bash_calls(body_node)

            # Calculate LOC
            line_start = node.start_point[0]
            line_end = node.end_point[0]
            loc = line_end - line_start + 1

            return {
                "type": "function",
                "name": func_name,
                "signature": signature,
                "parameters": [],  # Bash uses $1, $2, etc. - not formal params
                "return_type": None,
                "decorators": [],
                "is_async": False,
                "is_generator": False,
                "docstring": docstring,
                "body": body_text,
                "line_start": line_start + 1,
                "line_end": line_end + 1,
                "calls": calls,
                "imports": [],
                "loc": loc,
            }

        except Exception as e:
            logger.error(f"Error parsing Bash function in {file_path}: {e}")
            return None

    def _extract_bash_comments(self, node: Node, code_lines: List[str]) -> Optional[str]:
        """Extract comments preceding a Bash function"""
        line_start = node.start_point[0]

        # Look back up to 5 lines for comments
        comments = []
        for i in range(max(0, line_start - 5), line_start):
            line = code_lines[i].strip()
            if line.startswith("#"):
                comments.append(line[1:].strip())
            elif line and not line.startswith("#"):
                # Non-comment, non-empty line - stop looking
                break

        return "\n".join(comments) if comments else None

    def _extract_bash_calls(self, body_node: Node) -> List[str]:
        """Extract command/function calls from Bash function body using AST"""
        calls = set()

        bash_keywords = {"if", "then", "else", "elif", "fi", "for", "while",
                         "do", "done", "case", "esac", "function", "local",
                         "return", "exit", "export", "declare", "readonly",
                         "echo", "printf", "true", "false", "set", "unset"}

        def traverse(node: Node):
            if node.type == "command_name":
                cmd = node.text.decode("utf8")
                if cmd not in bash_keywords and len(cmd) < 100:
                    calls.add(cmd)
            for child in node.children:
                traverse(child)

        if body_node:
            traverse(body_node)

        return sorted(list(calls))[:50]

    def _extract_bash_imports(self, tree) -> List[str]:
        """Extract sourced files from Bash scripts.

        Detects:
        - 'source file.sh'
        - '. file.sh'
        Returns basenames of sourced files.
        """
        sources = set()

        def traverse(node):
            if node.type == "command":
                # Look for command_name that is 'source' or '.'
                children = list(node.children)
                if len(children) >= 2:
                    cmd_name = children[0]
                    if cmd_name.type == "command_name":
                        cmd_text = cmd_name.text.decode("utf8")
                        if cmd_text in ("source", "."):
                            # Next argument is the file being sourced
                            for arg in children[1:]:
                                if arg.type in ("word", "string", "raw_string", "simple_expansion", "concatenation"):
                                    source_path = arg.text.decode("utf8").strip("'\"")
                                    # Extract basename
                                    basename = source_path.rsplit("/", 1)[-1]
                                    if basename:
                                        sources.add(basename)
                                    break
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return sorted(list(sources))

    def _extract_powershell_functions(
        self,
        tree,
        code: str,
        file_path: str
    ) -> List[Dict]:
        """
        Extract PowerShell function definitions

        Captures:
        - Function name (including hyphenated names like Get-VMInfo)
        - Parameters with types, defaults, and attributes
        - Comment-based help (<# ... #>)
        - Function calls within body
        - Advanced function attributes
        """
        functions = []
        code_lines = code.split("\n")

        # Extract file-level imports once (injected into every function)
        file_imports = self._extract_powershell_imports(code)

        # Query for function definitions
        # PowerShell uses 'function_statement' node (airbus-cert grammar)
        function_query = self.languages["powershell"].query("""
            (function_statement
                (function_name) @func_name
            ) @function
        """)

        captures = function_query.captures(tree.root_node)
        for node, capture_name in captures:
            if capture_name == "function":
                func_data = self._parse_powershell_function(node, code_lines, file_path)
                if func_data:
                    func_data["imports"] = file_imports
                    functions.append(func_data)

        return functions

    def _parse_powershell_function(
        self,
        node: Node,
        code_lines: List[str],
        file_path: str
    ) -> Optional[Dict]:
        """Parse a single PowerShell function node"""
        try:
            # Get function name (airbus-cert grammar uses child node, not named field)
            name_node = None
            for child in node.children:
                if child.type == "function_name":
                    name_node = child
                    break

            if not name_node:
                return None

            func_name = code_lines[name_node.start_point[0]][
                name_node.start_point[1]:name_node.end_point[1]
            ].strip()

            # Get full body text
            line_start = node.start_point[0]
            line_end = node.end_point[0]

            # Extract signature (function definition line)
            signature_end = min(line_start + 5, line_end)  # Look ahead max 5 lines
            signature_lines = []
            for i in range(line_start, signature_end + 1):
                if i < len(code_lines):
                    signature_lines.append(code_lines[i])
                    # Stop at opening brace
                    if '{' in code_lines[i]:
                        break

            signature = "\n".join(signature_lines).strip()

            # Get full body
            body_text = "\n".join(code_lines[line_start:line_end + 1])

            # Extract parameters from param() block
            parameters = self._extract_powershell_parameters(node, code_lines)

            # Extract comment-based help (docstring)
            docstring = self._extract_powershell_help(node, code_lines)

            # Extract function calls
            calls = self._extract_powershell_calls(body_text)

            # Calculate LOC (excluding empty lines and comments)
            loc = sum(
                1 for line in code_lines[line_start:line_end + 1]
                if line.strip() and not line.strip().startswith('#')
            )

            # Check for [CmdletBinding()] attribute (advanced function)
            is_advanced = '[CmdletBinding' in body_text or '[cmdletbinding' in body_text.lower()

            return {
                "type": "function",
                "name": func_name,
                "signature": signature,
                "parameters": parameters,
                "return_type": None,  # PowerShell doesn't have explicit return types
                "decorators": ["CmdletBinding"] if is_advanced else [],
                "is_async": False,  # PowerShell doesn't have async/await like Python
                "is_generator": False,
                "docstring": docstring,
                "body": body_text[:10000],  # Limit to 10KB
                "line_start": line_start + 1,
                "line_end": line_end + 1,
                "calls": calls,
                "imports": [],  # Could extract Import-Module calls if needed
                "loc": loc,
            }

        except Exception as e:
            logger.error(f"Error parsing PowerShell function in {file_path}: {e}")
            return None

    def _extract_powershell_parameters(self, node: Node, code_lines: List[str]) -> List[Dict]:
        """Extract PowerShell parameters from param() block"""
        parameters = []

        try:
            # Look for param() block in the function body
            body_text = "\n".join(
                code_lines[node.start_point[0]:node.end_point[0] + 1]
            )

            # Use regex to extract param block (tree-sitter PowerShell param parsing can be complex)
            param_match = re.search(r'param\s*\((.*?)\)', body_text, re.DOTALL | re.IGNORECASE)
            if not param_match:
                return []

            param_block = param_match.group(1)

            # Extract individual parameters (simplified parsing)
            # Format: [Type]$Name = DefaultValue or [Parameter(...)]$Name
            param_pattern = r'\$([a-zA-Z_][a-zA-Z0-9_]*)'
            param_names = re.findall(param_pattern, param_block)

            for param_name in param_names:
                # Try to find type hint before $param_name
                type_pattern = rf'\[([^\]]+)\]\s*\${param_name}'
                type_match = re.search(type_pattern, param_block)
                param_type = type_match.group(1) if type_match else None

                # Try to find default value
                default_pattern = rf'\${param_name}\s*=\s*([^,\)]+)'
                default_match = re.search(default_pattern, param_block)
                default_value = default_match.group(1).strip() if default_match else None

                parameters.append({
                    "name": param_name,
                    "type": param_type,
                    "default": default_value
                })

        except Exception as e:
            logger.debug(f"Error extracting PowerShell parameters: {e}")

        return parameters[:20]  # Limit to 20 parameters

    def _extract_powershell_help(self, node: Node, code_lines: List[str]) -> Optional[str]:
        """Extract PowerShell comment-based help (<# ... #>)"""
        try:
            # Look for <# ... #> block before or at the start of the function
            line_start = node.start_point[0]

            # Check within the function first (help can be inside)
            body_text = "\n".join(code_lines[line_start:min(line_start + 20, len(code_lines))])
            help_match = re.search(r'<#(.*?)#>', body_text, re.DOTALL)

            if help_match:
                help_text = help_match.group(1).strip()
                # Extract .SYNOPSIS and .DESCRIPTION if present
                synopsis = re.search(r'\.SYNOPSIS\s+(.*?)(?=\n\s*\.[A-Z]|$)', help_text, re.DOTALL | re.IGNORECASE)
                description = re.search(r'\.DESCRIPTION\s+(.*?)(?=\n\s*\.[A-Z]|$)', help_text, re.DOTALL | re.IGNORECASE)

                if synopsis or description:
                    result = []
                    if synopsis:
                        result.append(synopsis.group(1).strip())
                    if description:
                        result.append(description.group(1).strip())
                    return "\n\n".join(result)
                else:
                    return help_text

            # Also check for single-line comments before function
            comments = []
            for i in range(max(0, line_start - 5), line_start):
                line = code_lines[i].strip()
                if line.startswith('#') and not line.startswith('#>'):
                    comments.append(line[1:].strip())
                elif line and not line.startswith('#'):
                    break

            return "\n".join(comments) if comments else None

        except Exception as e:
            logger.debug(f"Error extracting PowerShell help: {e}")
            return None

    def _extract_powershell_calls(self, body_text: str) -> List[str]:
        """Extract cmdlet/function calls from PowerShell function body"""
        calls = set()

        try:
            # PowerShell cmdlets follow Verb-Noun pattern
            cmdlet_pattern = r'\b([A-Z][a-z]+-[A-Z][a-zA-Z0-9]+)\b'
            matches = re.findall(cmdlet_pattern, body_text)
            calls.update(matches)

            # Also look for common PowerShell commands (non-capturing group so findall returns full match)
            common_pattern = r'\b(?:Get-|Set-|New-|Remove-|Test-|Start-|Stop-|Invoke-|Write-|Read-)[A-Z][a-zA-Z0-9]*'
            common_matches = re.findall(common_pattern, body_text)
            calls.update(common_matches)

        except Exception as e:
            logger.debug(f"Error extracting PowerShell calls: {e}")

        return sorted(list(calls))[:50]  # Limit to 50 calls

    def _extract_powershell_imports(self, code: str) -> List[str]:
        """Extract imported modules from PowerShell scripts.

        Detects:
        - Import-Module ModuleName
        - #Requires -Module ModuleName
        - using module ModuleName
        """
        modules = set()
        try:
            # Import-Module ModuleName (with or without -Name)
            for match in re.finditer(r'Import-Module\s+(?:-Name\s+)?["\']?([A-Za-z][A-Za-z0-9._-]*)', code, re.IGNORECASE):
                modules.add(match.group(1))

            # #Requires -Module ModuleName or #Requires -Modules ModuleName
            for match in re.finditer(r'#Requires\s+-Modules?\s+([A-Za-z][A-Za-z0-9._-]*)', code, re.IGNORECASE):
                modules.add(match.group(1))

            # using module ModuleName
            for match in re.finditer(r'using\s+module\s+([A-Za-z][A-Za-z0-9._-]*)', code, re.IGNORECASE):
                modules.add(match.group(1))

            # Dot-sourcing: . .\script.ps1 or . $PSScriptRoot\module.psm1
            for match in re.finditer(
                r'(?:^|\n)\s*\.\s+["\']?\$?(?:PSScriptRoot[\\/])?([^\s"\'#;]+\.ps[md]*1)["\']?',
                code, re.IGNORECASE,
            ):
                source_file = match.group(1)
                basename = source_file.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
                if basename and len(basename) > 4:
                    modules.add(basename)

        except Exception as e:
            logger.debug(f"Error extracting PowerShell imports: {e}")

        return sorted(list(modules))

    # NOTE: JS/TS extraction removed — not in SUPPORTED_LANGUAGES, never compiled.
    # Re-add when JS/TS tree-sitter grammars are built.

# Singleton instance getter for convenience
def get_parser() -> TreeSitterParser:
    """Get the singleton TreeSitterParser instance"""
    return TreeSitterParser.get_instance()
