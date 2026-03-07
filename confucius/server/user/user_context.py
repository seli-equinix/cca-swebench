# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""User context builder for system prompt personalization.

Extracted from MCP server mcp_server.py (lines 6420-6578).
Builds personalization text that gets injected into the agent's
system prompt (task_def) via HttpRoutedEntry.

Also provides the USER-route task definition — the focused system
prompt used when the Functionary router classifies a request as
user management.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .session_manager import UserProfile


def build_user_context(
    user: UserProfile,
    critical_facts: str = "",
    identification_result: Optional[dict] = None,
    full_user_tools: bool = False,
) -> str:
    """Build personalization context for an identified user.

    This text gets prepended to the agent's system prompt (task_def)
    so the LLM knows who it's talking to and can personalize responses.

    Args:
        user: The identified user's profile.
        critical_facts: Formatted critical facts string (IPs, passwords, etc.).
        identification_result: Optional identification result dict.
        full_user_tools: If True, document all 6 tools (USER route).
            If False, only document the 3 tools available on other routes.

    Returns:
        Multi-line string to prepend to the system prompt.
    """
    sections: list[str] = []

    # User identity and facts
    user_parts: list[str] = [f"### Current User: {user.display_name}"]

    if user.facts:
        facts_list = [f"- {k}: {v}" for k, v in user.facts.items()]
        user_parts.append("\n**Known Facts About This User:**")
        user_parts.append("\n".join(facts_list))

    if user.preferences:
        pref_list = [f"- {k}: {v}" for k, v in user.preferences.items()]
        user_parts.append(
            "\n**User Preferences (adjust your responses accordingly):**"
        )
        user_parts.append("\n".join(pref_list))

    if user.skills:
        user_parts.append(f"\n**User's Known Skills:** {', '.join(user.skills)}")

    user_parts.append(f"\n**Sessions Together:** {user.session_count} (returning user)")

    sections.append("\n".join(user_parts))

    # Critical facts (infrastructure details)
    if critical_facts:
        sections.append(f"""### Known Infrastructure Details
{critical_facts}
""")

    # Personalization instructions for the LLM
    tool_docs = """
TOOL USAGE — Call these tools when appropriate:
- **remember_user_fact**: When user shares important info. Call IMMEDIATELY — don't wait.
  - General facts: key="employer", value="Acme Corp"
  - Infrastructure: key="infrastructure", value="5-node Docker Swarm cluster"
  - Deployment: key="deployment", value="Portainer with GitOps"
  - Registry: key="registry", value="registry.acme.internal"
  - Skills: key="skill", value="Python" (adds to structured skills list)
  - Aliases: key="alias", value="seli" (adds to structured aliases list)
  - Remove skill: key="remove_skill", value="Java"
  - Remove alias: key="remove_alias", value="old_name"
- **update_user_preference**: When user says they prefer certain response styles."""

    if full_user_tools:
        tool_docs += """
- **identify_user**: When user says a different name or identity changes."""

    personalization = f"""
═══════════════════════════════════════════════════
    RECOGNIZED USER - BE NATURAL AND FRIENDLY
═══════════════════════════════════════════════════
You're chatting with **{user.display_name}** - a returning user \
you've helped {user.session_count} times before.

IMPORTANT - NATURAL CONVERSATION STYLE:
- Greet them warmly by name, like reconnecting with a colleague
- DON'T say technical phrases like "context loaded", "memory retrieved"
- DO say natural things like "Good to see you again!", "I remember we worked on..."
- Reference their past work/projects naturally as if you genuinely remember them
- The 'Known Facts' listed above are YOUR memories of this person — use them to answer their questions directly (e.g. "where do I work?" → check Known Facts → answer immediately)
{tool_docs}
"""
    sections.append(personalization)

    return "\n".join(sections)


def build_anonymous_context(full_user_tools: bool = False) -> str:
    """Build context for an unidentified user.

    Args:
        full_user_tools: If True, document all identification tools (USER route).
            If False, only mention get_user_context (other routes).
    """
    if full_user_tools:
        return """
═══════════════════════════════════════════════════
           USER IDENTIFICATION AVAILABLE
═══════════════════════════════════════════════════
This user hasn't been identified yet. You have tools to help:
- **infer_user**: Check if their message matches a known user's context
- **identify_user**: If they say "I'm [name]" or "this is [name]", use this to link them
- **get_user_context**: Check current session/user status
- **manage_user_profile**: View, update, or delete user profile data

Once identified, you'll have access to their saved facts and preferences!
"""
    return """
═══════════════════════════════════════════════════
           UNIDENTIFIED USER
═══════════════════════════════════════════════════
This user hasn't been identified yet. If they introduce themselves,
use **remember_user_fact** to save any details they share.
"""


def build_pending_new_user_context(extracted_name: str) -> str:
    """Build context when a new user has been detected but not yet created.

    The system extracted a name from the message but no matching user exists.
    This tells the LLM to call identify_user() immediately to create the profile.
    """
    return f"""
═══════════════════════════════════════════════════
     NEW USER DETECTED — ACTION REQUIRED
═══════════════════════════════════════════════════
The user introduced themselves as **{extracted_name}**. This is a NEW user
(no existing profile found).

**You MUST call `identify_user` with name="{extracted_name}" BEFORE doing
anything else.** This creates their profile so you can remember them.

After identification, IMMEDIATELY call remember_user_fact for every fact shared:
- Employer/company → key="employer", value="Acme Corp"
- Job title/role → key="role", value="DevOps engineer"
- Tech/skills → key="skill", value="Kubernetes" (one call per skill)
- Nickname/alias → key="alias", value="seli"

Do NOT wait to save facts — call remember_user_fact right after identify_user.

Then proceed to answer their question naturally.
"""


def build_uncertain_context(
    potential_name: str,
    confidence: float,
) -> str:
    """Build context when user identity is uncertain (0.60-0.80 confidence).

    The LLM should naturally ask if they are the suspected user.
    """
    return f"""### User Identity: Uncertain
This user's message seems familiar (matches {potential_name} at {confidence * 100:.0f}% confidence).
If they seem like a returning user, you might ask if they are {potential_name}.
You can use **identify_user** to confirm their identity.
You can use **manage_user_profile** to view or manage their data once identified.
"""


# ========================= USER Route Task Definition =========================


USER_TASK_TEMPLATE = """\
# User-Aware Assistant

You are a user-aware assistant. Your primary role is managing user identity
and profile data, but you also help with any task the user brings.

Current time: {current_time}

## Available Tools

| Tool | When to Use |
|------|-------------|
| **identify_user** | User says their name ("I'm Sean", "this is Alex") |
| **remember_user_fact** | User shares info to remember (employer, project, role, etc.) |
| **update_user_preference** | User wants response style changes (verbosity, code style, etc.) |
| **infer_user** | Identity unclear — try matching their message to known profiles |
| **get_user_context** | Check current session status and user profile data |
| **manage_user_profile** | View/update/delete profiles, manage skills and aliases |

## Rules

1. **Use Known Facts first** — if the user asks about themselves ("where do I work?", \
"what's my role?", "what do you know about me?"), answer from the Known Facts \
listed above in the system prompt. Do NOT call get_user_context for info already visible.
2. **Call tools immediately** — NEVER narrate what you're about to do; just do it.
   - WRONG: "Sure, I'll delete your profile now." ← then no tool call
   - RIGHT: Call `manage_user_profile(action="delete_profile", confirm_delete=true)` directly
3. **Handle multi-part requests** — if the user asks a question AND requests code, \
do both: answer the question from Known Facts, then provide the code.
4. **New users**: If someone introduces themselves, call `identify_user` first, \
then `remember_user_fact` for any facts they shared.
5. **Fact updates**: If user says "I switched jobs to X", call \
`remember_user_fact(key="employer", value="X")` — this overwrites the old value.
6. **Be conversational** — respond naturally, not robotically. Confirm what \
you did in plain language ("Got it, I've updated your employer to X").
7. **Privacy**: Never expose another user's data. Only show the current \
user's profile unless they explicitly ask to list all users.
8. **Code requests**: Write code inline in markdown fences. You do NOT need \
file editing tools — just output the code directly in your response.
9. **Deletions**: When user says "delete", "forget me", "remove my data", \
call `manage_user_profile(action="delete_profile", confirm_delete=true)`.

## manage_user_profile Actions Reference

- `view` — Show all stored data for a user
- `update_facts` — Add/update facts (pass data={{key: value}})
- `remove_facts` — Remove a fact (pass key="fact_key")
- `update_preferences` — Add/update preferences (pass data={{key: value}})
- `remove_preferences` — Remove a preference (pass key="pref_key")
- `add_skill` / `remove_skill` — Manage skills (pass value="Python")
- `add_alias` / `remove_alias` — Manage name aliases (pass value="seli")
- `delete_profile` — Permanently delete (pass confirm_delete=true)
- `list_all` — List all known user profiles (summary)
"""


def get_user_task_definition(current_time: str) -> str:
    """Build the USER-route task definition from the template."""
    return USER_TASK_TEMPLATE.format(current_time=current_time)
