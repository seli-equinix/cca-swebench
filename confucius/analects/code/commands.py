# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
from __future__ import annotations

from typing import Dict


def get_allowed_commands() -> Dict[str, str]:
    """Return allowed commands for coding tasks.

    Conservative but practical — covers typical developer workflows
    for Python, Bash, Node.js, and general software development.
    """
    return {
        # ── Filesystem basics ──
        "pwd": "Print current working directory",
        "ls": "List directory contents",
        "cat": "Print file contents",
        "head": "Show first lines of a file",
        "tail": "Show last lines of a file",
        "wc": "Count lines/words/bytes",
        "stat": "Display file or file system status",
        "du": "Estimate file space usage",
        "df": "Report file system disk space usage",
        "chmod": "Change file access permissions",
        "chown": "Change file ownership",
        "cp": "Copy files and directories",
        "mv": "Move (rename) files",
        "rm": "Remove (delete) files or directories",
        "mkdir": "Create directories",
        "rmdir": "Remove empty directories",
        "touch": "Update access and modification times of files",
        "find": "Search for files in a directory hierarchy",
        "ln": "Create hard and symbolic links",
        "tree": "List directory contents in a tree format",
        # ── File & path utilities ──
        "file": "Determine file type",
        "basename": "Strip directory from filename",
        "dirname": "Strip last component from path",
        "realpath": "Print resolved absolute path",
        "readlink": "Print value of a symbolic link",
        "diff": "Compare files line by line",
        "patch": "Apply a diff file to an original",
        "tee": "Read from stdin and write to stdout and files",
        "md5sum": "Compute MD5 message digest",
        "sha256sum": "Compute SHA-256 message digest",
        # ── System information (read-only) ──
        "uname": "Print system information (kernel, architecture)",
        "whoami": "Print current username",
        "hostname": "Print system hostname",
        "id": "Print user identity (uid, gid, groups)",
        "date": "Print or set the system date and time",
        "env": "Print environment variables",
        "printenv": "Print specific or all environment variables",
        # ── Text processing ──
        "grep": "Search for patterns in files",
        "sed": "Stream editor for filtering and transforming text",
        "awk": "Text processing and data extraction",
        "cut": "Remove sections from each line of files",
        "sort": "Sort lines of text files",
        "uniq": "Report or omit repeated lines",
        "tr": "Translate or delete characters",
        "xargs": "Build and execute command lines from standard input",
        "column": "Format input into columns",
        "paste": "Merge lines of files side by side",
        "comm": "Compare two sorted files line by line",
        "join": "Join lines of two files on a common field",
        "expand": "Convert tabs to spaces",
        # ── Data processing ──
        "jq": "Command-line JSON processor",
        "yq": "Command-line YAML/XML/TOML processor",
        "base64": "Encode or decode base64 data",
        "iconv": "Convert text between character encodings",
        "strings": "Print printable strings from binary files",
        # ── Output ──
        "echo": "Print arguments to standard output",
        "printf": "Format and print data",
        # ── Archiving / compression ──
        "tar": "Archive files",
        "gzip": "Compress or decompress named files",
        "gunzip": "Decompress gzip files",
        "zip": "Package and compress files into a ZIP archive",
        "unzip": "Extract files from a ZIP archive",
        "bzip2": "Block-sorting file compressor",
        "xz": "Compress or decompress .xz files",
        "zcat": "View contents of compressed files",
        "zgrep": "Search compressed files for patterns",
        # ── Networking (safe reads only) ──
        "curl": "Transfer data from or to a server",
        "wget": "Non-interactive network downloader",
        # ── Shell & scripting ──
        "bash": "GNU Bourne Again SHell",
        "sh": "POSIX shell",
        "which": "Locate a command on PATH",
        "type": "Describe how a shell name would be interpreted",
        "command": "Run a command or display information about commands",
        "test": "Evaluate conditional expression",
        "true": "Return successful exit status",
        "false": "Return unsuccessful exit status",
        "expr": "Evaluate expressions",
        "bc": "Arbitrary precision calculator",
        "seq": "Print a sequence of numbers",
        "read": "Read a line from standard input",
        "export": "Set or export environment variables",
        "source": "Execute commands from a file in the current shell",
        # ── Process & job control ──
        "sleep": "Delay for a specified amount of time",
        "timeout": "Run a command with a time limit",
        "wait": "Wait for background processes to finish",
        "nohup": "Run a command immune to hangups",
        # ── Git ──
        "git": (
            "Git version control (status, diff, add, commit, branch, "
            "checkout, switch, log, show, grep, rev-parse, etc.)"
        ),
        # ── Python ecosystem ──
        "python3": "Run Python 3 scripts",
        "python": "Python interpreter",
        "pip": "Python package installer",
        "pip3": "Python 3 package installer",
        "pytest": "Python test framework runner",
        "mypy": "Python static type checker",
        "black": "Python code formatter",
        "ruff": "Fast Python linter and formatter",
        "flake8": "Python style guide enforcement",
        "isort": "Sort Python imports",
        "uv": "Fast Python package manager (Rust-based)",
        "coverage": "Python code coverage measurement",
        # ── Node.js / JavaScript ──
        "node": "Node.js JavaScript runtime",
        "npm": "Node.js package manager",
        "npx": "Execute npm package binaries",
        "yarn": "Alternative Node.js package manager",
        "bun": "Fast JavaScript runtime and package manager",
        "pnpm": "Fast, disk-efficient package manager",
        "tsc": "TypeScript compiler",
        # ── Build tools & compilers ──
        "make": "Build automation tool",
        "cmake": "Cross-platform build system generator",
        "gcc": "GNU C compiler",
        "g++": "GNU C++ compiler",
        "go": "Go programming language tool",
        "cargo": "Rust package manager and build tool",
        "rustc": "Rust compiler",
    }
