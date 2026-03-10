# CCA Agent-as-a-Model Endpoint — Session Summary

**Date**: 2026-02-22
**Scope**: Converting CCA from CLI-only agent to OpenAI-compatible HTTP endpoint + MCP tool audit

---

## 1. What Was Built

CCA (Confucius Code Agent) gained a second mode: an **HTTP server** that accepts OpenAI-compatible `/v1/chat/completions` requests, runs the **same orchestrator and full agent loop** as CLI mode, and returns responses formatted as OpenAI chat completions — with full user identification and personalization ported from the MCP server.

### Architecture

```
                    ┌──────────────────────────────────────────────────────┐
                    │              CCA Codebase                            │
                    │                                                      │
  CLI Mode          │  confucius/cli/main.py                               │
  (unchanged)       │    └─ entry_repl.py → invoke_analect(CodeAssistEntry)│
                    │         └─ AnthropicLLMOrchestrator + Extensions     │
                    │                                                      │
  HTTP Mode ──────► │  confucius/server/app.py                     (NEW)  │
  (new)             │    ├─ SessionPool → Confucius instances      (NEW)  │
                    │    ├─ UserSessionManager → identify user     (NEW)  │
                    │    ├─ invoke_analect(HttpCodeAssistEntry)     (NEW)  │
                    │    │    └─ user context injected into task_def       │
                    │    │    └─ UserToolsExtension added to tools         │
                    │    │    └─ AnthropicLLMOrchestrator (SAME)           │
                    │    │         └─ All extensions (SAME)               │
                    │    ├─ HttpIOInterface captures output         (NEW)  │
                    │    └─ Format as OpenAI response               (NEW)  │
                    │                                                      │
  Shared            │  AnthropicLLMOrchestrator (the agent engine)         │
  (untouched)       │  Extensions: FileEdit, CommandLine, Planning, etc.   │
                    │  LLM Managers: Auto, OpenAI (→ vLLM)                │
                    │  Config: config.toml                                 │
                    └──────────────────────────────────────────────────────┘
```

Both CLI and HTTP mode use the same agent path:
```
invoke_analect(Entry, EntryInput(question=...))
  → impl() → build extensions → AnthropicLLMOrchestrator → full agent loop
```

### New Files Created (15 files, ~3,990 lines)

```
confucius/server/
├── __init__.py              # Package init
├── models.py                # OpenAI-compatible Pydantic models
├── io_adapter.py            # HttpIOInterface (buffer + SSE queue)
├── session_pool.py          # Concurrent Confucius session management
├── streaming.py             # SSE formatting with keepalive
├── http_entry.py            # HttpCodeAssistEntry (extends CodeAssistEntry)
├── app.py                   # FastAPI application & routes
└── user/
    ├── __init__.py
    ├── session_manager.py   # Ported from MCP (user identification, Redis + Qdrant)
    ├── critical_facts.py    # CriticalFactsExtractor (regex credential extraction)
    ├── user_context.py      # System prompt personalization builder
    └── tools_extension.py   # UserToolsExtension (5 LLM tools, CCA native extension)
```

### Service Details

| Property | Value |
|----------|-------|
| Container | `cca-http` |
| Host | Spark1 (192.168.4.205) |
| Port | 8500 |
| Compose | `cca-compose.yml` |
| Entry | `confucius serve --port 8500` |
| Model Name | Read from `config.toml` → `/models/Qwen3-Next-80B-A3B-Thinking-FP8` |

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/chat/completions` | OpenAI-compatible completions (full agent loop) |
| GET | `/v1/models` | List served model (real model name from config) |
| GET | `/health` | Health check |
| GET | `/users` | List known user profiles |
| GET | `/sessions` | List active sessions (debugging) |
| GET | `/stats` | Diagnostic statistics |

---

## 2. Key Technical Decisions

### Model Name Resolution
- `/v1/models` returns the **real underlying model** from `config.toml` (e.g., `/models/Qwen3-Next-80B-A3B-Thinking-FP8`)
- Not a custom name like `cca-agent` — external systems (Open WebUI, Continue.dev, LM Studio) need recognizable model names
- Any model name accepted in `/v1/chat/completions` requests (CCA is a single agent, not a model router)
- **Crashes at startup** if config is unavailable or model name is empty — no silent fallback

```python
def _resolve_served_model_name() -> str:
    params = get_llm_params("coder")
    if not params.model:
        raise RuntimeError(
            "CCA config has no model name for 'coder' role. "
            "Check config.toml [providers.*.coder] section."
        )
    return params.model

SERVED_MODEL_NAME: str = _resolve_served_model_name()  # Resolved at import time
```

### Session ID Derivation
Priority chain (ported from MCP server) — ensures memory works with any OpenAI client:
1. `request.session_id` (CCA extension field)
2. `X-Session-Id` header
3. `request.user` → `"user-{user}"` (OpenAI standard)
4. System prompt hash → `"sys-{hash}"` (Continue.dev pattern)
5. Conversation prefix hash → `"conv-{hash}"` (last resort)

### User Awareness System
Ported from MCP server's `session_manager.py`:
- **Smart auto-identification**: 25+ regex patterns for name extraction ("Hi, I'm Sean", "my name is Sean", etc.)
- **Qdrant semantic matching**: Vector similarity search against known user profiles
- **Confidence thresholds**: 0.80+ auto-link, 0.60-0.80 LLM asks via tool, <0.60 anonymous
- **Cross-agent recognition**: Shares Qdrant user_profiles with MCP server — users identified in one are recognized in the other
- **5 LLM Tools** (via `UserToolsExtension`, native CCA extension):
  - `identify_user` — Link session to user
  - `remember_user_fact` — Store persistent user fact
  - `update_user_preference` — Update user preference
  - `infer_user` — Semantic user matching
  - `get_user_context` — Get current user info

### Infrastructure Dependencies
| Service | Location | Purpose |
|---------|----------|---------|
| Redis | Spark1:6379 | Session storage, critical facts |
| Qdrant | Spark1:6333 | User profiles (vector search) |
| Embedding | Spark1:8200 | Semantic user matching |
| vLLM | Spark2:8000 | LLM inference (Qwen3-80B) |

---

## 3. Fixes Applied

### Port Conflict (8100 → 8500)
MCP server already uses port 8100 on Spark1. Changed `cca-http` to port 8500.

### Code Was Never Committed
All server code from the initial planning/coding session was uncommitted. Committed 3,990 lines across 15 files, pushed submodule, updated parent pointer.

### Submodule Update Fix
`git submodule update --init --recursive` failed due to orphaned `asus-dgx-spark/flashinfer` reference. Fixed with targeted: `git submodule update --init nvidia-dgx-spark/cca`.

### Model Name `cca-agent` Not Recognizable
External systems won't recognize custom names. Fixed to read real model from config.toml.

### Silent Fallback on Missing Config
User feedback: "should go to the error system so that the UI would know it is not working." Removed try/except fallback — now raises `RuntimeError` at startup.

---

## 4. Deployment Workflow

```bash
# 1. Commit in CCA submodule
cd /home/seli/docker-swarm-stacks/nvidia-dgx-spark/cca
git add <files> && git commit -m "msg" && git push origin main

# 2. Update parent submodule pointer
cd /home/seli/docker-swarm-stacks
git add nvidia-dgx-spark/cca && git commit -m "update CCA submodule" && git push

# 3. Pull on Spark1
sshpass -p "$CCA_DEPLOY_PASS" ssh user@CCA_HOST "cd docker-swarm-stacks && git pull && git submodule update --init nvidia-dgx-spark/cca"

# 4. Build and restart
sshpass -p "$CCA_DEPLOY_PASS" ssh user@CCA_HOST "cd docker-swarm-stacks/nvidia-dgx-spark/cca && docker compose -f cca-compose.yml build cca-http && docker compose -f cca-compose.yml up -d cca-http"

# 5. Verify
curl http://192.168.4.205:8500/health
curl http://192.168.4.205:8500/v1/models
```

---

## 5. MCP Server Tool Audit (Complete Inventory)

### Overview
| Category | Count | Registration |
|----------|-------|-------------|
| LLM-callable tools (`AVAILABLE_TOOLS`) | 33 | OpenAI function-calling schema in `mcp_server.py` |
| MCP-protocol-only tools (`MCPToolsManager`) | 5 | MCP protocol in `mcp_tools.py` |
| **Total** | **38** | |

### Category 1: Memory & Search (5 tools)

| Tool | Description | Backend |
|------|-------------|---------|
| `search_memory` | Search Redis conversation history + Qdrant long-term memory | Redis + Qdrant |
| `web_search` | Search the web via SearXNG | SearXNG (Spark1:8888) |
| `search_knowledge` | Unified search across ephemeral docs, user knowledge, project knowledge | Qdrant |
| `search_codebase` | Search indexed codebase from Nextcloud WebDAV | Qdrant |
| `fetch_url` | Fetch and extract content from a URL | Direct HTTP |

### Category 2: Document Management (4 tools)

| Tool | Description | Backend |
|------|-------------|---------|
| `upload_document` | Upload and chunk a document for search | Qdrant |
| `search_documents` | Search within uploaded documents | Qdrant |
| `get_document` | Retrieve a specific document by ID | Qdrant |
| `delete_document` | Delete an uploaded document | Qdrant |

### Category 3: Project Context (3 tools)

| Tool | Description | Backend |
|------|-------------|---------|
| `get_project_context` | Get current project context and metadata | Redis |
| `switch_project` | Switch active project context | Redis |
| `list_projects` | List all known projects | Nextcloud WebDAV |

### Category 4: File Operations — LLM-callable (8 tools)

| Tool | Description | Backend |
|------|-------------|---------|
| `read_file` | Read file contents from workspace | MCPToolsManager |
| `write_file` | Write/overwrite file in workspace | MCPToolsManager |
| `edit_file` | Apply targeted edits to a file | MCPToolsManager |
| `create_file` | Create a new file with content | MCPToolsManager |
| `move_file` | Move/rename a file | MCPToolsManager |
| `list_directory` | List directory contents | MCPToolsManager |
| `glob_search` | Find files matching glob patterns | MCPToolsManager |
| `search_files_content` | Search file contents with regex | MCPToolsManager |

### Category 5: File Operations — MCP-protocol-only (5 tools)

| Tool | Description | Notes |
|------|-------------|-------|
| `move_file` (MCP) | Move file via MCP protocol | Duplicate of LLM version |
| `copy_file` | Copy file in workspace | MCP-only, not exposed to LLM |
| `replace_lines` | Replace specific line range | MCP-only |
| `insert_lines` | Insert lines at position | MCP-only |
| `search_replace` | Find and replace in file | MCP-only |

### Category 6: Code Intelligence / Graph (3 tools)

| Tool | Description | Backend |
|------|-------------|---------|
| `query_call_graph` | Query function call graph (callers/callees) | Memgraph (node3:7687) |
| `find_orphan_functions` | Find functions never called by other functions | Memgraph |
| `analyze_dependencies` | Analyze module/file dependency relationships | Memgraph |

### Category 7: User Management (6 tools)

| Tool | Description | Backend |
|------|-------------|---------|
| `manage_user_profile` | CRUD operations on user profile | Qdrant |
| `search_user_knowledge` | Search user's personal knowledge base | Qdrant |
| `store_user_knowledge` | Store knowledge to user's personal KB | Qdrant |
| `get_user_rules` | Get user's custom rules/preferences | Redis |
| `set_user_rules` | Set user's custom rules | Redis |
| `delete_user_rules` | Delete user's custom rules | Redis |

### Category 8: Code Quality (2 tools)

| Tool | Description | Backend |
|------|-------------|---------|
| `view_diff` | View git diff or file comparison | MCPToolsManager |
| `run_shell_command` | Execute shell commands with security validation | MCPToolsManager |

### Category 9: Vision (1 tool)

| Tool | Description | Backend |
|------|-------------|---------|
| `analyze_image` | Analyze images using Qwen3-VL-30B | Vision server (Spark1:8300) |

### Category 10: Miscellaneous (1 tool)

| Tool | Description | Backend |
|------|-------------|---------|
| `get_current_time` | Get current date/time with timezone | Local |

### External Service Dependencies

| Service | URL | Tools That Use It |
|---------|-----|-------------------|
| Redis | Spark1:6379 | search_memory, get/set/delete_user_rules, get/switch_project, critical facts |
| Qdrant | Spark1:6333 | search_memory, search_knowledge, search_codebase, documents, user profiles |
| Embedding | Spark1:8200 | All semantic search tools (vector embeddings) |
| Memgraph | node3:7687 | query_call_graph, find_orphan_functions, analyze_dependencies |
| SearXNG | Spark1:8888 | web_search |
| Vision | Spark1:8300 | analyze_image |
| vLLM | Spark2:8000 | LLM inference (the agent loop itself) |
| Nextcloud WebDAV | node4 | search_codebase, list_projects, codebase indexing |

### Tool Dispatch Architecture

```
Client Request → /v1/chat/completions
  → LLM generates tool_calls (OpenAI format)
  → execute_tool() — giant if/elif dispatch (~1,700 lines)
    ├─ memory tools → MemoryManager
    ├─ knowledge tools → KnowledgeSearch
    ├─ file tools → MCPToolsManager
    ├─ graph tools → MemgraphAdapter
    ├─ user tools → SessionManager
    ├─ document tools → DocumentProcessor
    ├─ project tools → ProjectContext
    ├─ web tools → SearXNG / direct HTTP
    └─ vision tools → Vision server
  → Tool results fed back as messages
  → LLM iterates (up to 20 iterations, 1 hour timeout)
  → Final response returned
```

---

## 6. Phoenix Observability Integration

### Phoenix Service
- **URL**: http://192.168.4.204:6006
- **OTel gRPC**: port 4317
- **OTel HTTP**: port 4318
- **Stack**: `phoenix/phoenix-stack.yml` (PostgreSQL 16 + Phoenix 13.1.1 on node5)

### Instrumentation Pattern (for future CCA integration)
```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://192.168.4.204:4317", insecure=True)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
```

### Environment Variables for CCA HTTP Container
```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://192.168.4.204:4317
  - OTEL_SERVICE_NAME=cca-http
  - OTEL_TRACES_EXPORTER=otlp
```

---

## 7. CCA Agent Engine — Why It Works

The agent engine in CCA is cleanly decoupled from the CLI through the `IOInterface` abstraction:

| Interface | CLI Mode | HTTP Mode |
|-----------|----------|-----------|
| `IOInterface.ai()` | Rich console output | Buffer to response |
| `IOInterface.system()` | Console status | Tag as thinking/progress |
| `IOInterface.get_input()` | `aioconsole.ainput()` | Return empty (non-blocking) |
| `IOInterface.confirm()` | Console prompt | Auto-confirm (autonomous) |

Only 2 files from the original CCA codebase needed to be **extended** (not modified):
- `CodeAssistEntry` → extended by `HttpCodeAssistEntry` (adds user context + tools)
- `IOInterface` → implemented by `HttpIOInterface` (buffers output)

**Zero modifications** to:
- `AnthropicLLMOrchestrator` (the recursive tool-calling loop)
- Any extension (FileEdit, CommandLine, Planning, Memory, etc.)
- Any LLM manager (Auto, OpenAI, Azure, Bedrock)
- Config system
- Memory system

---

## 8. Current Status

| Item | Status |
|------|--------|
| Server code written | Done (15 files, ~3,990 lines) |
| Committed & pushed | Done (submodule + parent) |
| Port changed to 8500 | Done |
| Model name from config | Done |
| Error handling (no silent fallback) | Done |
| CLAUDE.md updated | Done |
| Container deployed & running | Needs verification on Spark1 |
| Phoenix integration | Future work |
| Continue.dev testing | Future work |
| Cross-agent user recognition testing | Future work |

### Verification Commands
```bash
# Health check
curl http://192.168.4.205:8500/health

# List models
curl http://192.168.4.205:8500/v1/models

# Basic chat test
curl -X POST http://192.168.4.205:8500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"cca","messages":[{"role":"user","content":"What files are in the current directory?"}]}'

# User identification test
curl -X POST http://192.168.4.205:8500/v1/chat/completions \
  -H "X-Session-Id: test-session" \
  -H "Content-Type: application/json" \
  -d '{"model":"cca","messages":[{"role":"user","content":"Hi, I'\''m Sean"}]}'

# Streaming test
curl -N -X POST http://192.168.4.205:8500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"cca","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```
