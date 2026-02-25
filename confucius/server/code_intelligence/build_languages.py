#!/usr/bin/env python3
"""
Build tree-sitter language grammars for Python, Bash, PowerShell, YAML, and Markdown

Uses specific versions compatible with tree-sitter 0.21.3 ABI v14.
PowerShell requires tree-sitter CLI to generate parser.c from grammar.js.

Manual compile+link to avoid setuptools/distutils linker bugs
(missing -shared flag in newer setuptools versions).

Versions:
- Python: v0.20.4 (pre-compiled)
- Bash: v0.20.5 (pre-compiled)
- PowerShell: master branch (requires generation)
- YAML: v0.5.0 (pre-compiled)
- Markdown: v0.7.1 (pre-compiled)
"""

import os
import subprocess

# Language repositories with versions
# Python, Bash, YAML, Markdown have pre-compiled parsers
# PowerShell needs generation via tree-sitter CLI
LANGUAGES = {
    "python": ("https://github.com/tree-sitter/tree-sitter-python.git", "v0.20.4", False),
    "bash": ("https://github.com/tree-sitter/tree-sitter-bash.git", "v0.20.5", False),
    # PowerShell: Using Airbus CERT grammar (comprehensive function support, PS 7.3 spec)
    "powershell": ("https://github.com/airbus-cert/tree-sitter-powershell.git", "main", True),
    "yaml": ("https://github.com/ikatyang/tree-sitter-yaml.git", "v0.5.0", False),
    "markdown": ("https://github.com/ikatyang/tree-sitter-markdown.git", "v0.7.1", False),
}


def main():
    """Build language grammars into a shared library"""
    build_dir = "/tmp/tree-sitter-build"
    os.makedirs(build_dir, exist_ok=True)

    # Clone repositories at specific versions
    repo_paths = []
    for lang_name, (repo_url, version_tag, needs_generate) in LANGUAGES.items():
        repo_path = os.path.join(build_dir, f"tree-sitter-{lang_name}")
        if not os.path.exists(repo_path):
            print(f"Cloning {lang_name} @ {version_tag}...")
            subprocess.run(
                ["git", "clone", "--depth=1", "--branch", version_tag, repo_url, repo_path],
                check=True,
                capture_output=True
            )

        # Generate parser if needed (PowerShell)
        # Force regeneration to ensure ABI v14 compatibility with tree-sitter 0.21.3
        parser_c = os.path.join(repo_path, "src", "parser.c")
        if needs_generate:
            # Remove existing parser.c to force regeneration with our tree-sitter-cli
            if os.path.exists(parser_c):
                os.remove(parser_c)
                print(f"  Removed pre-compiled parser.c (ABI compatibility)")

            print(f"Generating parser for {lang_name}...")
            result = subprocess.run(
                ["tree-sitter", "generate"],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Warning: Failed to generate {lang_name} parser:")
                print(f"  {result.stderr.strip()}")
                print(f"  Skipping {lang_name} — AST parsing will fall back to regex")
                continue
            print(f"  Generated parser for {lang_name}")

        # Verify parser.c exists
        if not os.path.exists(parser_c):
            print(f"Warning: {lang_name}: parser.c not found, skipping")
            continue

        repo_paths.append(repo_path)
        print(f"  {lang_name}: {version_tag}")

    # Build shared library manually (bypasses setuptools/distutils linker issues)
    library_path = "/usr/local/lib/tree-sitter-languages.so"
    print(f"\nCompiling grammars...")

    object_files = []
    for repo_path in repo_paths:
        src_dir = os.path.join(repo_path, "src")

        # Only compile the standard tree-sitter files:
        #   parser.c (required), scanner.c or scanner.cc (optional)
        # Skip other files like schema.generated.cc in YAML grammar
        parser_c = os.path.join(src_dir, "parser.c")
        if os.path.exists(parser_c):
            obj = parser_c.replace(".c", ".o")
            subprocess.run(
                ["cc", "-fPIC", "-c", "-I", src_dir, parser_c, "-o", obj],
                check=True, capture_output=True,
            )
            object_files.append(obj)

        scanner_c = os.path.join(src_dir, "scanner.c")
        scanner_cc = os.path.join(src_dir, "scanner.cc")
        if os.path.exists(scanner_c):
            obj = scanner_c.replace(".c", ".o")
            subprocess.run(
                ["cc", "-fPIC", "-c", "-I", src_dir, scanner_c, "-o", obj],
                check=True, capture_output=True,
            )
            object_files.append(obj)
        elif os.path.exists(scanner_cc):
            obj = scanner_cc.replace(".cc", ".o")
            subprocess.run(
                ["c++", "-fPIC", "-c", "-I", src_dir, scanner_cc, "-o", obj],
                check=True, capture_output=True,
            )
            object_files.append(obj)

    # Link into shared library with explicit -shared flag
    print(f"Linking {len(object_files)} objects into shared library...")
    subprocess.run(
        ["c++", "-shared", "-o", library_path] + object_files,
        check=True,
        capture_output=True,
    )

    print(f"Built {len(repo_paths)} grammars: {', '.join(LANGUAGES.keys())}")
    print(f"Library: {library_path}")


if __name__ == "__main__":
    main()
