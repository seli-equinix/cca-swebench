# CCA Failing Tests — 2026-03-09

Full suite: **9/15 passed, 6/15 failed**
Run time: ~1h 45m (sequential, with 3s cooldown between tests)

All 6 failures are **LLM response quality / tool-calling behavior issues** — not infrastructure or parsing errors. The tree-sitter bash parsing change (deployed this session) has zero impact on these failures.

---

## 1. test_profile_crud — Profile Deletion Incomplete

**File**: `tests/user/test_profile_crud.py`
**Result**: 9/10 judge assertions passed, 1 failed
**Failed assertion**: Turn 5 `task_completion = failed`

### What happened
- Turns 1-4 all passed perfectly: create user, verify Qdrant profile + Redis session + NoteObserver notes, recall profile, update skills (remove Java, remove alias), verify updates persisted
- **Turn 5**: User says "I want to delete my profile entirely. Remove all my data — profile, notes, everything."
- LLM acknowledged deletion but the **judge deemed the task incomplete**

### Root cause
The LLM likely called `delete_profile` but the judge saw that either:
- The response didn't explicitly confirm all data types were removed (profile + notes + sessions)
- Or the delete_profile tool succeeded but the LLM's response was ambiguous about completion

### What to investigate
1. Check Phoenix trace for Turn 5 — did the LLM call `delete_profile` tool?
2. Is `delete_profile` doing cascade delete (profile + notes)? Check `session_manager.py`
3. Is the judge being too strict? The assertion code at line 306-308 checks for "deleted/removed/confirmed/profile" in the response — this passed. The judge separately rated `task_completion = failed`
4. The test's own assertions (lines 311-350) validate the actual deletion via REST API — those **passed**. Only the LLM judge disagreed.

### Verdict: **Judge strictness issue** — the actual data deletion worked (REST verified), but the LLM's response text wasn't convincing enough for the judge.

---

## 2. test_code_intelligence — Dependency Analysis Response Quality

**File**: `tests/coder/test_code_intelligence.py`
**Result**: 3/3 subtests passed (pytest), but 5/6 judge assertions passed, 1 failed
**Failed assertion**: `test_dependency_analysis` Turn 1 `response_quality = poor`

### What happened
- `test_call_graph_query`: PASSED (2 iters, 17.7s) — asked about callers of `build_user_context`, got good graph data
- `test_orphan_detection`: PASSED (2 iters, 28.5s) — found orphan functions, good response
- `test_dependency_analysis`: PASSED pytest assertions but **17 iterations** (49.2s) — the LLM struggled

### Root cause
The LLM took 17 tool iterations to analyze dependencies of `user_context.py`. This suggests:
- The `analyze_dependencies` graph tool returned results the LLM couldn't synthesize well
- Or the LLM kept re-calling tools trying to get a better picture
- The response quality was rated "poor" by the judge — likely because the response was bloated with repeated tool calls rather than a clean analysis

### What to investigate
1. Phoenix trace: what tools did it call 17 times? Was it `analyze_dependencies` repeatedly or cycling between tools?
2. Is the `analyze_dependencies` tool returning too much/too little data for the LLM to work with?
3. Does `user_context.py` even exist in the indexed graph? (Path resolution issues — graph stores full indexed paths like `mcp-server/repo/nvidia-dgx-spark/...`)
4. Consider: should the test use a more specific file that's known to be in the graph?

### Verdict: **LLM iteration loop** — the agent couldn't efficiently synthesize dependency data, kept retrying, produced a poor-quality response despite eventually passing the content assertions.

---

## 3. test_codebase_search — First Turn Search Failure

**File**: `tests/coder/test_codebase_search.py`
**Result**: 1/1 subtest passed (pytest), but 2/4 judge assertions failed
**Failed assertions**: Turn 1 `response_quality = poor`, Turn 1 `task_completion = failed`

### What happened
- **Turn 1**: "I need to modify the health check endpoint... Look through the codebase and find which files implement health check functions"
  - Judge rated: response_quality=poor, task_completion=failed
  - 8 tool iterations — the LLM used search tools but produced a poor result
- **Turn 2**: "I also need to understand our Qdrant integration..."
  - Judge rated: response_quality=good, task_completion=completed
  - This one worked fine

### Root cause
Turn 1 failed because:
- The LLM used `search_codebase`/`search_knowledge` tools (8 iterations) but the results weren't good enough
- Possible: "health check" is too generic — returns many results, LLM couldn't distill them
- Possible: search returned vector results that weren't about the actual health check endpoint

### What to investigate
1. Phoenix trace: what search queries did the LLM send? Were they too broad?
2. What did `search_codebase` return for "health check"? Is the Qdrant index stale?
3. Turn 2 (Qdrant) worked — "Qdrant" is a more specific search term. The issue may be search relevance for generic terms.
4. Consider: tune the test to ask about something more specific, or improve codebase search relevance

### Verdict: **Search relevance issue** — generic "health check" query returns too many weak matches, LLM can't synthesize well. Specific terms like "Qdrant" work fine.

---

## 4. test_infra_route — Docker Query Task Completion

**File**: `tests/integration/test_infra_route.py`
**Result**: 2/2 subtests passed (pytest), but 3/4 judge assertions passed, 1 failed
**Failed assertion**: `test_infra_docker_query` Turn 1 `task_completion = failed`

### What happened
- **test_infra_docker_query**: "What Docker containers are running on this system right now?"
  - Routed to infrastructure, used tools (6 iterations), response quality rated "good"
  - But `task_completion = failed` — judge thinks the task wasn't completed
- **test_infra_service_check**: "Check if Redis is running" — PASSED everything

### Root cause
The LLM ran `docker ps` (or equivalent) but the judge deemed the result incomplete. Possible reasons:
- CCA runs in a container itself — `docker ps` inside the container may show limited containers
- The response may have shown containers but the judge expected a "summary of names and status" in a specific format
- Docker socket may not be mounted, so `docker ps` returned an error and the LLM improvised

### What to investigate
1. Phoenix trace: what bash command did it run? `docker ps`? Did it succeed or fail?
2. Check CCA container — does it have Docker socket access? (`/var/run/docker.sock` mount)
3. The log shows `Command grep,docker failed` in a different test — is `docker` in the allowed command list for INFRA route?
4. If docker isn't available inside the container, this test fundamentally can't work

### Verdict: **Environment limitation** — CCA container may not have Docker socket access, so `docker ps` either fails or shows nothing useful. Judge sees empty/error result and rates task as failed.

---

## 5. test_full_flow — Infra Facts Not Stored

**File**: `tests/integration/test_full_flow.py`
**Result**: FAILED (pytest assertion), 6/6 judge assertions passed
**Failed assertion**: Line 140 `AssertionError: Infra facts not stored after session 4`

### What happened
- Sessions 1-3 all passed (user creation, coding, web search)
- Session 4: User says "Hey, important stuff to remember about our setup: we run a 5-node Docker Swarm cluster, our private registry is at registry.acme.internal..."
- LLM produced a Redis compose file (correct!) — but the test then checks REST API for stored facts
- `profile.get("facts")` returned: `{'role': 'DevOps engineer', 'fact': 'I mainly work with Docker and Kubernetes', 'name': 'Lifecycle_4a964e', 'employer': 'AcmeSystems', 'project': 'AcmeSystems', 'tool': 'Kubernetes'}`
- **Missing**: "swarm", "registry", "portainer", "gitops", "5-node" — none stored

### Root cause
The CriticalFactsExtractor (or profile updater) stored the user's **role/company/tools** from Session 1 but did NOT extract the **infrastructure details** from Session 4. The fact extraction prompt likely prioritizes personal facts (name, role, skills) over infrastructure/environment facts.

### What to investigate
1. Check `memory_manager.py` CriticalFactsExtractor — what extraction prompt does it use?
2. Does the fact schema even have a field for infrastructure details? Current fields: `role`, `fact`, `name`, `employer`, `project`, `tool`
3. The LLM stored `tool: Kubernetes` from Session 1, but "Docker Swarm", "Portainer", "GitOps" from Session 4 were dropped
4. This is a **real product gap** — users telling CCA about their infrastructure setup expect it to be remembered

### Also noted
- `WARNING tests.evaluators:evaluators.py:574 TOOL ERRORS detected (1): Command grep,docker failed`
- The LLM tried `grep` or `docker` which are disallowed in the CODER route

### Verdict: **Fact extraction gap** — CriticalFactsExtractor doesn't extract infrastructure/environment facts. The fact schema is too narrow (role/name/employer/tool) for the types of information users share.

---

## 6. test_routing_edge_cases — Planner Route Fundamentally Broken

**File**: `tests/integration/test_routing_edge_cases.py`
**Result**: 1/3 subtests failed (pytest), 4/8 judge assertions failed

### Subtest results

#### test_direct_answer — Judge strict on task_completion
- Response quality: good (PASSED)
- Task completion: failed (FAILED)
- The LLM correctly explained REST but the judge rated completion as "failed"
- **Verdict**: Judge strictness — the response contains "representational", "state", "transfer" (line 41-43 assertion passes), but judge disagrees on completion

#### test_planner_route — DEEP DIVE (FAILED)
- **pytest assertion failed**: Response lacks structured breakdown
- Response was: "I'll design a comprehensive CI/CD pipeline... Memory updated Memory updated Memory updated..." repeated many times
- 12 iterations, 259s — the LLM wrote to memory instead of answering
- The response has no numbered steps, no bullet points, no headers — just "Memory updated" spam

#### test_complex_multi_file_project — Judge strict on Turn 1
- Turn 1: `task_completion = failed` (judge), but files WERE created (REST verified: 2+ files with prefix)
- Turn 2: PASSED — ran the code successfully with correct output
- **Verdict**: Judge strict — Turn 1 created files but the judge wanted to see all code in the response text, not just tool output

---

### Deep Dive: Why the Planner Route is Broken

This is not a minor LLM quality issue — there are **4 compounding architectural bugs** that make the planner route fundamentally non-functional.

#### Bug 1: Wrong System Prompt

**File**: `confucius/server/http_routed_entry.py:87`

```python
_ROUTE_TASK_DEFS = {
    ExpertType.USER: get_user_task_definition,
    ExpertType.CODER: get_task_definition,
    ExpertType.INFRASTRUCTURE: get_infra_task_definition,
    ExpertType.SEARCH: get_search_task_definition,
    ExpertType.PLANNER: get_task_definition,      # ← WRONG: reuses CODER prompt
}
```

The PLANNER route uses the **CODER system prompt** (`analects/code/tasks.py`). This prompt says:

- "You are a coding assistant working inside a developer's repository"
- "Execute the plan using tool-use tags"
- "Only use allowed commands surfaced by the command-line extension"
- "You MUST always use `str_replace_editor` tool to view files or make any file edits"

A planning request ("Design a CI/CD pipeline, what are the architecture and steps?") gets a prompt that tells the LLM to edit files and run commands. The LLM has no instruction to produce a structured plan — it's told to execute.

**The real planner prompt exists** at `orchestrator/extensions/plan/prompts.py` — an 80-line Senior Software Architect prompt that generates `<summary>` and `<plan>` tags with structured steps. But it's **never injected** into the PLANNER route (see Bug 2).

#### Bug 2: Planner Extension Doesn't Inject Its Prompt

**File**: `orchestrator/extensions/plan/llm.py:32-34`

```python
class LLMPlannerExtension(TokenEstimatorExtension):
    name: str = "llm_planner"
    included_in_system_prompt: bool = False    # ← NEVER included
```

The `LLMCodingArchitectExtension` (line 268) inherits from `LLMPlannerExtension`, which has `included_in_system_prompt = False`. This means:

1. The extension IS instantiated and added to the extension list (`tool_groups.py:196`)
2. But its architect prompt is **never appended to the system prompt** (filtered out at `orchestrator/llm.py:142`)
3. The extension's only role is **context window summarization** — when the conversation exceeds `max_prompt_length` tokens, it fires `_on_invoke_llm()` to summarize earlier messages
4. It has **0 tools** — nothing for the LLM to call

The architect prompt at `plan/prompts.py` would tell the LLM to produce structured `<summary>` and `<plan>` output. But since `included_in_system_prompt = False`, this prompt is invisible.

#### Bug 3: Tool Mismatch — CODER Prompt References Missing Tools

The CODER system prompt (`tasks.py:51-54`) says:

```
- If the user gives their name, call `remember_user_fact(key="name", value="Alice")` IMMEDIATELY
- If the user mentions any personal facts, call `remember_user_fact` for each fact
- Call `get_user_context` at the start if you need to recall who you're talking to
```

But the PLANNER route's tool groups (`tool_groups.py:102-108`):

```python
ExpertType.PLANNER: [
    ToolGroup.PLANNER,      # 0 tools (prompt-only, but prompt is invisible)
    ToolGroup.MEMORY,       # 6 tools: write_memory, read_memory, edit_memory, ...
    ToolGroup.CODE_SEARCH,  # 3 tools: search_codebase, search_knowledge, index_workspace
    ToolGroup.WEB,          # 2 tools: web_search, fetch_url_content
    ToolGroup.NOTES,        # 1 tool: search_notes
]
```

**`remember_user_fact` and `get_user_context` are NOT on the PLANNER route** — those are from `UserMemoryExtension` (ToolGroup.USER_MEMORY), which isn't included. The LLM is told to call tools that don't exist.

So when the LLM tries to follow the system prompt's instruction to store facts, it finds the closest available alternative: `write_memory` from `HierarchicalMemoryExtension`. It starts writing plan content to memory instead of to the response.

#### Bug 4: "Memory updated" Pollutes the Response Stream

**File**: `orchestrator/extensions/memory/hierarchical/extension.py:250`

```python
async def _display_memory(self, memory: Memory, context: AnalectRunContext) -> None:
    """Display the current memory structure as an artifact."""
    ...
    await context.io.ai("Memory updated", attachments=[attachment])
```

Every `write_memory` call triggers `_display_memory()` which calls `context.io.ai("Memory updated")`. The `io.ai()` method tags the text as `chunk_type="assistant"` in the `HttpIOInterface` (`io_adapter.py:64-76`).

The response assembly (`app.py:882-884`) collects all "assistant"-tagged chunks:

```python
response_text = io.get_response_text()   # Collects chunk_type == "assistant"
if not response_text:
    response_text = io.get_all_text()     # Fallback
```

So every "Memory updated" from tool acknowledgments becomes part of the **user-visible response**. The LLM's actual plan text (if any) is buried inside `write_memory` tool calls — the response is just "Memory updated" repeated.

#### The Full Failure Chain

```
User: "Design a CI/CD pipeline with architecture and steps"
  ↓
Router: correctly identifies PLANNER route
  ↓
HttpRoutedEntry: injects CODER system prompt (Bug 1)
  ↓
LLMCodingArchitectExtension: built but prompt invisible (Bug 2)
  ↓
LLM sees: "You are a coding assistant... call remember_user_fact..."
  ↓
LLM looks for remember_user_fact → not found (Bug 3)
  ↓
LLM uses write_memory instead → writes plan sections to memory tool
  ↓
Each write_memory → context.io.ai("Memory updated") → response stream (Bug 4)
  ↓
Response: "Memory updated Memory updated Memory updated..."
  ↓
Test assertion: no numbered steps, no structure → FAIL
```

### Files That Need Fixing

| File | Line | Bug | Fix |
|------|------|-----|-----|
| `server/http_routed_entry.py` | 87 | #1 Wrong prompt | Create `get_planner_task_definition()` — dedicated planner system prompt |
| `orchestrator/extensions/plan/llm.py` | 34 | #2 Prompt invisible | Either set `included_in_system_prompt = True` OR inject the architect prompt directly in the task definition |
| `server/tool_groups.py` | 102-108 | #3 Missing USER_MEMORY | Add `ToolGroup.USER_MEMORY` to PLANNER route OR remove user-fact instructions from planner prompt |
| `orchestrator/extensions/memory/hierarchical/extension.py` | 250 | #4 io.ai() pollution | Change `context.io.ai("Memory updated")` to `context.io.system("Memory updated")` |

### Recommended Fix Strategy

**Option A: Dedicated planner prompt (cleanest)**
1. Create `get_planner_task_definition()` in `analects/code/tasks.py` — a prompt that says "You are a planning architect. Produce a structured response with numbered steps, architecture decisions, and implementation breakdown. Do NOT use file or shell tools — your response text IS the deliverable."
2. Wire it in `http_routed_entry.py:87`: `ExpertType.PLANNER: get_planner_task_definition`
3. Fix `io.ai("Memory updated")` → `io.system("Memory updated")` (Bug 4 — affects ALL routes)

**Option B: Enable the existing architect prompt**
1. Override `included_in_system_prompt = True` on `LLMCodingArchitectExtension`
2. The architect prompt would then be appended to whatever task definition is set
3. Still need to fix the wrong task def (Bug 1) and io.ai pollution (Bug 4)

**Option A is recommended** because:
- The existing architect prompt (`plan/prompts.py`) is designed for context-window summarization, not for direct planning responses
- A dedicated planner prompt can be tailored to the user-facing planning use case
- It's a clean separation — the planner prompt tells the LLM exactly what output format to use

---

## Priority Ranking

| Priority | Test | Category | Fix Difficulty |
|----------|------|----------|----------------|
| **P1** | test_routing_edge_cases (planner) | 4 compounding architecture bugs | Medium — dedicated prompt + io.ai fix |
| **P1** | test_full_flow | Fact extraction gap | Medium — expand CriticalFactsExtractor schema |
| **P2** | test_code_intelligence | LLM iteration loop | Easy — improve dependency analysis tool output |
| **P2** | test_codebase_search | Search relevance | Medium — tune search or test query |
| **P3** | test_infra_route | Environment limitation | Easy — verify Docker socket mount |
| **P3** | test_profile_crud | Judge strictness | Low — judge tuning or response prompt |

### Notes
- P3 items may not need code changes — they could be test/judge calibration
- P1 items represent real product gaps that affect users
- The planner route has been broken since it was wired — it was never tested with a real planning question until now
- Bug 4 (`io.ai("Memory updated")`) affects ALL routes, not just planner — it pollutes responses whenever `write_memory` is called
- None of these are related to the tree-sitter bash parsing change
