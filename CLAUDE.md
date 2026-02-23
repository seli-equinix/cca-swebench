# Confucius Code Agent (CCA) - Development Guide

**Base Directory**: `nvidia-dgx-spark/cca/`
**Fork**: https://github.com/seli-equinix/cca-swebench
**Upstream**: https://github.com/facebookresearch/cca-swebench
**Origin**: Meta + Harvard research project (arXiv:2512.10398)

---

## CRITICAL: Two-Repo Git Workflow

CCA is a **git submodule** of `docker-swarm-stacks`. It has its own repo.

```
docker-swarm-stacks/          <- parent repo (seli-equinix/docker-swarm-stacks)
  nvidia-dgx-spark/
    cca/                      <- submodule (seli-equinix/cca-swebench)
      confucius/              <- the agent framework
```

### Committing CCA Changes

CCA changes require **two commits** - one in the submodule, one in the parent:

```bash
# 1. Commit inside the submodule
cd nvidia-dgx-spark/cca
git add <files> && git commit -m "description" && git push origin main

# 2. Update the parent repo's submodule pointer
cd /home/seli/docker-swarm-stacks
git add nvidia-dgx-spark/cca && git commit -m "update CCA submodule" && git push
```

### Pulling Upstream Updates

```bash
cd nvidia-dgx-spark/cca
git fetch upstream
git merge upstream/main
git push origin main
```

### Cloning Fresh (on Spark1 or new machine)

```bash
cd docker-swarm-stacks
git submodule update --init --recursive nvidia-dgx-spark/cca
```

---

## Architecture Overview

CCA has two modes: **CLI** (interactive REPL) and **HTTP** (Agent-as-a-Model endpoint). Both use the same agent engine. HTTP mode includes an **Expert Router** that classifies requests and selects the right entry.

```
CLI Mode (confucius code)               HTTP Mode (confucius serve --port 8500)
  -> Confucius (session manager)           -> FastAPI app (app.py)
    -> CodeAssistEntry (Analect)             -> SessionPool -> Confucius instances
      -> AnthropicLLMOrchestrator              -> UserSessionManager (Redis + Qdrant)
        -> Extensions (tools)                  -> Expert Router (Functionary on node5:8001)
        -> AutoLLMManager -> LLM                 -> classify_request() → RouteDecision
                                                 -> HttpCodeAssistEntry (coding)
                                                 -> HttpInfraEntry (infrastructure)
                                                 -> Direct answer / Clarification (no agent loop)
                                                   -> AnthropicLLMOrchestrator (SAME engine)
                                                   -> All extensions (SAME) + UserToolsExtension
                                                   -> AutoLLMManager -> LLM (SAME)
```

**Key principle**: The HTTP layer is a thin shell around the existing CLI agent. The `AnthropicLLMOrchestrator` IS the agent — it handles recursive tool calling, extension lifecycle, memory, thinking/reasoning, error recovery, and multi-step task execution. Both modes invoke it identically via `invoke_analect(Entry, EntryInput(question=...))`.

### Core Components

| Directory | Purpose | Key Classes |
|-----------|---------|-------------|
| `confucius/cli/` | CLI entry + `serve` command | `main.py` (Click: `confucius code`, `confucius serve`) |
| `confucius/lib/` | Runtime bootstrap | `Confucius`, `run_entry_repl` |
| `confucius/analects/code/` | Code agent config | `CodeAssistEntry`, LLM params, allowed commands |
| `confucius/analects/note_taker/` | Note-taking agent | Observes traces, persists long-term memory |
| `confucius/core/` | Foundation layer | `Analect`, `AutoLLMManager`, memory, storage, config |
| `confucius/core/config.py` | TOML config loader | `get_llm_params("role")` → `LLMParams` |
| `confucius/orchestrator/` | Agent execution loop | `AnthropicLLMOrchestrator`, Extension pipeline |
| `confucius/orchestrator/extensions/` | Tool implementations | File edit, bash, planning, memory, caching |
| `confucius/server/` | **HTTP server (Agent-as-a-Model)** | FastAPI app, session pool, streaming, user system |
| `confucius/utils/` | Helpers | JSON, async, string, pydantic utilities |
| `scripts/` | SWE-bench harness | `run_swebench.py`, `run_sbp.sh` |

---

## Config System (TOML)

All model selection is managed via `config.toml` (mounted at `/etc/cca/config.toml` in Docker).

**Config file**: `nvidia-dgx-spark/cca/config.toml`
**Loader**: `confucius/core/config.py` — pydantic-validated, lazy singleton, Python 3.12 `tomllib`
**Access**: `get_llm_params("role")` returns `LLMParams` (synchronous, safe at module level)

```toml
# Top-level: model prefixes that route through OpenAILLMManager
openai_model_prefixes = ["qwen", "/models/"]

[active]
coder = "local"           # "cloud" = Azure GPT-5.2, "local" = Spark2 Qwen3-80B
note_taker = "local"      # "cloud" = Bedrock Claude, "local" = Spark1 Qwen3-8B
planner = "local"
reviewer = "local"
tester = "local"

[providers.local.coder]
model = "/models/Qwen3-Next-80B-A3B-Thinking-FP8"
provider = "openai"
base_url = "http://192.168.4.208:8000/v1"
api_key_env = "OPENAI_API_KEY"
initial_max_tokens = 8192
max_tokens = 16384
temperature = 0.3

[providers.local.note_taker]
model = "/models/Qwen3-8B-FP8"
provider = "openai"
base_url = "http://192.168.4.205:8400/v1"
temperature = 0.3
```

**Switching providers**: Edit `[active]` section — no code changes needed. `"local"` routes to Spark infrastructure, `"cloud"` routes to Azure/Bedrock.

---

## Agent-as-a-Model HTTP Endpoint

CCA runs as a persistent HTTP server on **Spark1 port 8500**, accepting OpenAI-compatible `/v1/chat/completions` requests and running the full agent loop internally.

### Service Info

| Property | Value |
|----------|-------|
| **URL** | `http://192.168.4.205:8500` |
| **Health** | `http://192.168.4.205:8500/health` |
| **Container** | `cca-http` on Spark1 |
| **Restart policy** | `unless-stopped` (survives reboots) |
| **Underlying LLM** | Qwen3-Next-80B-A3B-Thinking-FP8 on Spark2:8000 |
| **Model name in /v1/models** | Read from `config.toml` coder role (currently `/models/Qwen3-Next-80B-A3B-Thinking-FP8`) |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming) |
| `GET` | `/v1/models` | Lists served model (real name from config.toml) |
| `GET` | `/health` | Health check (status, version, active sessions) |
| `POST` | `/route/test` | Test expert router classification (no agent loop) |
| `GET` | `/users` | List all known user profiles |
| `GET` | `/sessions` | List active agent sessions |
| `GET` | `/stats` | Diagnostic statistics |

### Quick Test Commands

```bash
# Health check
curl http://192.168.4.205:8500/health

# List models
curl http://192.168.4.205:8500/v1/models

# Non-streaming chat
curl -X POST http://192.168.4.205:8500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"any-name-works","messages":[{"role":"user","content":"What files are in /workspace?"}]}'

# Streaming chat
curl -X POST http://192.168.4.205:8500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"cca","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# User identification test
curl -X POST http://192.168.4.205:8500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: test-session" \
  -d '{"messages":[{"role":"user","content":"Hi, I'\''m Sean. What can you help me with?"}]}'
```

### Model Name Behavior

- `/v1/models` advertises the real model from `config.toml` (e.g., `/models/Qwen3-Next-80B-A3B-Thinking-FP8`)
- Completions handler accepts **any** model name (CCA is a single agent, not a router)
- Response echoes back whatever model name the client sent
- If config is missing or broken, the server **crashes at startup** with a clear error (no silent fallback)

### Session ID Derivation

Clients are identified via session ID, derived in priority order:
1. `session_id` field in request body (CCA extension)
2. `X-Session-Id` header
3. `user` field → `user-{user}` (OpenAI standard)
4. System prompt hash → `sys-{hash}` (Continue.dev pattern)
5. Conversation prefix hash → `conv-{hash}` (last resort)

### Server Module Structure

```
confucius/server/
├── __init__.py              # Package init
├── app.py                   # FastAPI app, routes, expert routing, session ID derivation
├── models.py                # OpenAI-compatible Pydantic models (request/response/streaming)
├── io_adapter.py            # HttpIOInterface — captures orchestrator output for HTTP
├── session_pool.py          # Concurrent Confucius instance management (per-session locks)
├── streaming.py             # SSE formatting with 8s keepalive
├── expert_router.py         # Functionary-based request classifier → RouteDecision
├── http_entry.py            # HttpCodeAssistEntry — coding agent with user context
├── http_infra_entry.py      # HttpInfraEntry — infrastructure agent (Docker, SSH, etc.)
├── utility_tools.py         # UtilityToolsExtension (web_search, fetch_url_content)
└── user/
    ├── __init__.py
    ├── session_manager.py   # UserSessionManager (Redis + Qdrant), profiles, identification
    ├── critical_facts.py    # CriticalFactsExtractor (auto-extract IPs, passwords, keys)
    ├── user_context.py      # Build personalization text for system prompt injection
    └── tools_extension.py   # UserToolsExtension (5 LLM-callable tools, CCA native)
```

### Infrastructure Analect

```
confucius/analects/infrastructure/
├── __init__.py              # Package init
├── commands.py              # Expanded CLI allowlist (docker, ssh, systemctl, etc.)
└── tasks.py                 # Infrastructure system prompt with cluster topology
```

### Key Integration Points

**HttpCodeAssistEntry** (`http_entry.py`): Extends `CodeAssistEntry` pattern to:
1. Prepend user personalization context to the system prompt (`task_def`)
2. Append `UserToolsExtension` to the extensions list
3. Everything else (orchestrator, all other extensions, LLM) stays **identical** to CLI mode

**HttpIOInterface** (`io_adapter.py`): Implements CCA's `IOInterface` ABC:
- `ai()` → tagged as "assistant" output
- `system()` → tagged as "thinking" or "progress"
- `_get_input()` → returns empty string (non-blocking, so orchestrator never blocks on interactive prompts)
- Stream queue for SSE forwarding

**SessionPool** (`session_pool.py`): Manages concurrent Confucius instances:
- Per-session `asyncio.Lock` (serializes requests within a session)
- TTL-based cleanup (default 1 hour)
- Max 50 concurrent sessions

---

## Expert Router (Functionary Classification)

Incoming HTTP requests are classified by a small Functionary model (8B, Q4_0) running on node5 via llama.cpp. This happens **before** the main agent loop, with ~50-100ms overhead.

### Router Infrastructure

| Property | Value |
|----------|-------|
| **Model** | `functionary-small-v3.2.Q4_0.gguf` (4.34GB) |
| **Server** | llama.cpp `server-cuda` b8124 on node5 (192.168.4.204:8001) |
| **GPU** | RTX 5070 SM120, ~115 tok/s generation |
| **Template** | Custom Jinja from HuggingFace `meetkai/functionary-small-v3.2` |
| **Config** | `config.toml` `[router]` section |

### Expert Types

| Expert | Entry Class | Description |
|--------|-------------|-------------|
| `coder` | `HttpCodeAssistEntry` | Code editing, debugging, git, file operations |
| `infrastructure` | `HttpInfraEntry` | Docker, SSH, Swarm, monitoring, deployments |
| `search` | `HttpCodeAssistEntry` | (Future) Dedicated search expert |
| `planner` | `HttpCodeAssistEntry` | (Future) Multi-step planning expert |
| `direct` | *(no agent loop)* | Simple Q&A answered by Functionary directly |
| `clarify` | *(no agent loop)* | Ambiguous request — asks user for more info |

### How Routing Works

1. User message arrives at `/v1/chat/completions`
2. `classify_request()` sends message to Functionary with 6 routing tools
3. Functionary calls one tool (e.g., `route_to_infrastructure`) with parameters
4. `RouteDecision` is built from the tool call response
5. If `DIRECT` or `CLARIFY` → return immediately (no agent loop)
6. Otherwise → select entry class based on expert type, inject `route.to_context_header()` into system prompt

### Config (config.toml)

```toml
[router]
enabled = true
url = "http://192.168.4.204:8001"
timeout_ms = 10000
fallback_entry = "coder"
temperature = 0.1

[tool_router]      # Phase 2 — in-loop tool selection (not yet enabled)
enabled = false
```

### Test Classification

```bash
# Quick classification test (no agent loop)
curl -X POST http://192.168.4.205:8500/route/test \
  -H "Content-Type: application/json" \
  -d '{"message": "check docker swarm node status"}'

# Expected: {"expert": "infrastructure", "task_summary": "...", ...}

curl -X POST http://192.168.4.205:8500/route/test \
  -H "Content-Type: application/json" \
  -d '{"message": "refactor the login handler in auth.py"}'

# Expected: {"expert": "coder", "task_summary": "...", ...}
```

### HttpInfraEntry vs HttpCodeAssistEntry

| Feature | HttpCodeAssistEntry | HttpInfraEntry |
|---------|--------------------|--------------------|
| System prompt | Code-focused task definition | Cluster topology, SSH access, services |
| Allowed commands | Conservative (git, python, grep) | Expanded (docker, ssh, systemctl, sshpass) |
| Extensions | + CodeReviewer, TestGenerator | No code review/test gen (not relevant) |
| Max iterations | 20 | 30 (infra tasks need more steps) |
| Max output lines | 300 | 500 (infra output is verbose) |

---

## User Awareness System

Ported from the MCP server's user identification system. Shares infrastructure (Redis, Qdrant, Embedding server) so users identified in MCP are recognized in CCA.

### Infrastructure (shared with MCP server)

| Service | URL | Purpose |
|---------|-----|---------|
| Redis | `redis://192.168.4.205:6379` | Session storage, critical facts |
| Qdrant | `http://192.168.4.205:6333` | User profiles (vector search) |
| Embedding | `http://192.168.4.205:8200` | Semantic user matching (Qwen3-Embedding-8B) |

### Smart Auto-Identification

On first message of new sessions, the system runs a 3-step pipeline:
1. **Alias consolidation** — detects "I am both seli and sean" (11 regex patterns)
2. **Name extraction** — detects "Hi, I'm Sean" or "My name is Sean" (27 regex patterns)
3. **Semantic search** — matches message content against known user profiles in Qdrant

**Confidence thresholds**:
- 0.80+ → auto-link session to user
- 0.60–0.80 → LLM can ask user to confirm via `identify_user` tool
- <0.60 → stay anonymous

### 5 LLM-Callable User Tools

Implemented as a CCA native `ToolUseExtension` (same pattern as `FileEditExtension`):

| Tool | Description |
|------|-------------|
| `identify_user(name)` | Link session to user by name |
| `remember_user_fact(key, value)` | Store persistent fact (employer, project, etc.) |
| `update_user_preference(key, value)` | Update response preference (verbosity, code style) |
| `infer_user(message)` | Semantic user matching via Qdrant |
| `get_user_context()` | Get current session/user info |

### CriticalFactsExtractor

Auto-extracts from conversations and stores in Redis:
- Passwords, IP addresses, URLs, API keys, hostnames, usernames, ports
- Storage: `facts:{session_id}:{fact_type}` with auto-expiry

---

## LLM Provider Routing

`AutoLLMManager` routes by model name prefix (overridable via `openai_model_prefixes` in config.toml):

| Model Pattern | Provider | Backend | Env Vars |
|---------------|----------|---------|----------|
| Contains `claude` | `BedrockLLMManager` | AWS Bedrock | `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Contains `gemini` | `GoogleLLMManager` | Google AI | Google credentials |
| Matches `gpt`, `o1`, `o3`, `o4` | `AzureLLMManager` | Azure OpenAI | Azure config |
| Matches config `openai_model_prefixes` | `OpenAILLMManager` | OpenAI API / vLLM | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |

**Current config**: `openai_model_prefixes = ["qwen", "/models/"]` routes Qwen models to `OpenAILLMManager` → Spark2 vLLM.

---

## Extensions (Agent Tools)

The orchestrator runs an extension pipeline. Each extension registers tag handlers and tool schemas.

| Extension | File | Capabilities |
|-----------|------|-------------|
| `LLMCodingArchitectExtension` | `extensions/plan/llm.py` | Planning phase before execution |
| `FileEditExtension` | `extensions/file/edit.py` | View, create, edit files (line-range diffs) |
| `CommandLineExtension` | `extensions/command_line/base.py` | Bash execution (allowlisted commands) |
| `FunctionExtension` | `extensions/function/` | Python function calling (placeholder) |
| `CodeReviewerExtension` | `extensions/expert/reviewer.py` | Code review expert |
| `TestGeneratorExtension` | `extensions/expert/test_gen.py` | Test generation expert |
| `PlainTextExtension` | `extensions/plain_text.py` | Raw text pass-through |
| `HierarchicalMemoryExtension` | `extensions/memory/hierarchical.py` | Long-term memory with summarization |
| `AnthropicPromptCaching` | `extensions/caching/anthropic.py` | Anthropic token cache reuse |
| `SoloModeExtension` | `extensions/solo.py` | Single-shot execution mode |
| `UserToolsExtension` | `server/user/tools_extension.py` | User identification + memory (**HTTP mode only**) |

### Allowed CLI Commands

Defined in `analects/code/commands.py`. Conservative allowlist:
- Filesystem: `pwd`, `ls`, `cat`, `head`, `tail`, `find`, `cp`, `mv`, `rm`, `mkdir`, `touch`, `chmod`
- Text: `grep`, `sed`, `awk`, `cut`, `sort`, `uniq`, `tr`, `xargs`
- Archive: `tar`, `gzip`
- Network: `curl`, `wget`
- Git: `git` (all subcommands)
- Python: `python3`

---

## Orchestrator Loop

```
while iterations < max_iterations:
  1. on_input_messages()     -> preprocess (extensions inject context)
  2. get_llm_params()        -> select model/temperature for this turn
  3. invoke_llm()            -> call LLM via AutoLLMManager
  4. parse response          -> extract XML tags, plain text, tool_use blocks
  5. on_plain_text()         -> extension handlers for text output
  6. on_tag()                -> match tag to extension, execute tool
  7. on_process_messages_complete() -> post-processing
```

Exceptions: `OrchestratorInterruption` (pause), `OrchestratorTermination` (stop), `MaxIterationsReachedError`.

---

## Phoenix Observability Integration

Phoenix (Arize Phoenix v13.1.1) provides trace visualization for CCA agent debugging. Tracing is **built-in** — the CCA HTTP server automatically exports spans to Phoenix on startup.

| Property | Value |
|----------|-------|
| **Web UI** | `http://192.168.4.204:6006` |
| **OTel gRPC** | `192.168.4.204:4317` (for Python instrumentors) |
| **OTel HTTP** | `192.168.4.204:4318` (alternative) |
| **Stack file** | `phoenix/phoenix-stack.yml` |
| **Deploy** | `docker stack deploy -c phoenix/phoenix-stack.yml phoenix` |
| **Database** | PostgreSQL 16 on GlusterFS (`/mnt/glusterfs/phoenix/postgres`) |
| **Auth** | Disabled (`PHOENIX_ENABLE_AUTH=false`) |

### Phoenix Projects

Traces are routed to named projects via the `openinference.project.name` resource attribute (NOT `phoenix.project.name` — that doesn't work).

| Project | Source | Set By |
|---------|--------|--------|
| `cca-http` | Production HTTP server | `PHOENIX_PROJECT_NAME` env var |
| `cca-aaam-tests` | Test suite (`tests/conftest.py`) | Hardcoded in test config |

### What Gets Traced

- Every `/v1/chat/completions` request (session_id, user info, timing, response length)
- All OpenAI/vLLM HTTP calls (auto-instrumented via `openinference-instrumentation-openai`)
- Trace hierarchy: parent `cca.agent` span → child vLLM LLM call spans

### Tracing Module

**File**: `confucius/core/tracing.py` — centralized OTel initialization

- `init_tracing()` — called in server lifespan startup
- `shutdown_tracing()` — called at shutdown, flushes remaining spans
- `get_tracer()` — get a tracer instance for custom spans
- Graceful no-op if Phoenix is unreachable (never crashes the server)

### Phoenix Environment Variables

Set in `cca-compose.yml` for `cca-http`:
```yaml
environment:
  - PHOENIX_COLLECTOR_ENDPOINT=http://192.168.4.204:4317  # OTel gRPC endpoint
  - PHOENIX_PROJECT_NAME=cca-http                          # Phoenix project name
  - PHOENIX_TRACING_DISABLED=false                         # Set to "true" to disable
```

---

## Session & Storage

| Item | Location |
|------|----------|
| Session data | `~/.confucius/sessions/{session_id}/` |
| Memory | `~/.confucius/sessions/{session_id}/memory/` |
| Storage | `~/.confucius/sessions/{session_id}/storage/` |
| Artifacts | `~/.confucius/sessions/{session_id}/artifacts/` |
| Trajectory dumps | `/tmp/confucius/traj_{session_id}.json` |
| User sessions (HTTP) | Redis `cca:session:{session_id}` |
| User profiles (HTTP) | Qdrant `user_profiles` collection |
| Critical facts (HTTP) | Redis `facts:{session_id}:{fact_type}` |

Session IDs are UUID-based, auto-generated per run (CLI) or derived from request context (HTTP).

---

## Build & Deploy (Spark1)

### Docker Commands

```bash
cd nvidia-dgx-spark/cca

# Build all images
docker compose -f cca-compose.yml build

# Run interactive REPL
docker compose -f cca-compose.yml run --rm cca

# Run with source mounted (no rebuild for code changes)
docker compose -f cca-compose.yml --profile dev run --rm cca-dev

# Open shell for debugging
docker compose -f cca-compose.yml run --rm cca bash

# Start HTTP server (runs as daemon)
docker compose -f cca-compose.yml up -d cca-http

# View HTTP server logs
docker logs --tail 100 -f cca-http

# Restart HTTP server after code changes
docker compose -f cca-compose.yml down cca-http
docker compose -f cca-compose.yml build --no-cache cca-http
docker compose -f cca-compose.yml up -d cca-http
```

### Deploy Script (from node5)

```bash
./nvidia-dgx-spark/cca/deploy-cca.sh build   # pull + build on Spark1
./nvidia-dgx-spark/cca/deploy-cca.sh run     # interactive REPL on Spark1
./nvidia-dgx-spark/cca/deploy-cca.sh dev     # source-mounted dev mode
./nvidia-dgx-spark/cca/deploy-cca.sh shell   # bash inside container
```

### Docker Services

| Service | Purpose | Profile | Port | Persistent |
|---------|---------|---------|------|------------|
| `cca` | Interactive REPL (production) | default | — | No (run --rm) |
| `cca-dev` | Source-mounted for live editing | `dev` | — | No (run --rm) |
| `cca-swebench` | SWE-bench runner | `swebench` | — | No (run --rm) |
| `cca-http` | **Agent-as-a-Model HTTP server** | default | **8500** | **Yes** (restart: unless-stopped) |

### CCA HTTP Deployment (from node5)

```bash
# Full redeploy cycle
cd nvidia-dgx-spark/cca && git add <files> && git commit -m "msg" && git push origin main
cd /home/seli/docker-swarm-stacks && git add nvidia-dgx-spark/cca && git commit -m "update CCA submodule" && git push
sshpass -p 'Loveme-sex64' ssh seli@192.168.4.205 "cd docker-swarm-stacks && git pull && git submodule update --init nvidia-dgx-spark/cca"
sshpass -p 'Loveme-sex64' ssh seli@192.168.4.205 "cd docker-swarm-stacks/nvidia-dgx-spark/cca && docker compose -f cca-compose.yml build --no-cache cca-http && docker compose -f cca-compose.yml down cca-http && docker compose -f cca-compose.yml up -d cca-http"
curl http://192.168.4.205:8500/health
```

---

## Environment Variables

### Required (all services)

```env
OPENAI_API_KEY=dummy                              # or real key for cloud providers
OPENAI_BASE_URL=http://192.168.4.208:8000/v1     # local vLLM on Spark2
CCA_CONFIG_PATH=/etc/cca/config.toml              # TOML config (mounted in Docker)
```

### HTTP Server Additional

```env
REDIS_URL=redis://:Loveme-sex64@192.168.4.205:6379/0   # User sessions + facts
QDRANT_URL=http://192.168.4.205:6333                     # User profiles
EMBEDDING_URL=http://192.168.4.205:8200                  # Semantic matching
```

### Optional (provider-specific)

```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_BEARER_TOKEN_BEDROCK=...
CCA_WORKSPACE=/home/seli/code
```

---

## Key Files to Modify

| File | What to Change |
|------|---------------|
| `config.toml` | Model selection, provider switching, model prefixes, router config |
| `confucius/server/app.py` | HTTP routes, session handling, expert routing, user identification flow |
| `confucius/server/expert_router.py` | Router classification logic, tool definitions, system prompt |
| `confucius/server/http_entry.py` | User context injection, extension list for coding tasks |
| `confucius/server/http_infra_entry.py` | Infrastructure entry, expanded commands, cluster-aware prompt |
| `confucius/server/user/session_manager.py` | User identification logic, Redis/Qdrant operations |
| `confucius/server/user/tools_extension.py` | LLM-callable user tools (identify, remember, etc.) |
| `confucius/analects/code/commands.py` | Add/remove allowed CLI commands (coding) |
| `confucius/analects/code/tasks.py` | Modify coding system prompt / task definition |
| `confucius/analects/infrastructure/commands.py` | Add/remove allowed CLI commands (infra) |
| `confucius/analects/infrastructure/tasks.py` | Modify infrastructure system prompt / cluster topology |
| `cca-compose.yml` | Docker service configuration, ports, env vars |

---

## Testing

### SWE-bench (upstream validation)

```bash
docker compose -f cca-compose.yml --profile swebench run --rm cca-swebench
```

### HTTP Server Testing

```bash
# Health
curl http://192.168.4.205:8500/health

# Models
curl http://192.168.4.205:8500/v1/models

# Chat (non-streaming)
curl -X POST http://192.168.4.205:8500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"List files in /workspace"}]}'

# Chat (streaming)
curl -X POST http://192.168.4.205:8500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'

# User identification
curl -X POST http://192.168.4.205:8500/v1/chat/completions \
  -H "X-Session-Id: test-session" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi, I'\''m Sean"}]}'

# Active sessions
curl http://192.168.4.205:8500/sessions

# Known users
curl http://192.168.4.205:8500/users
```

---

## Upstream Sync

```bash
cd nvidia-dgx-spark/cca
git fetch upstream
git log --oneline upstream/main..HEAD   # see what we've added
git merge upstream/main                 # pull in upstream changes
git push origin main                    # push merged result to fork
```
