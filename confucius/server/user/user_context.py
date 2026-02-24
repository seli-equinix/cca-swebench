# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""User context builder for system prompt personalization.

Extracted from MCP server mcp_server.py (lines 6420-6578).
Builds personalization text that gets injected into the agent's
system prompt (task_def) via HttpCodeAssistEntry / HttpRoutedEntry.

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
) -> str:
    """Build personalization context for an identified user.

    This text gets prepended to the agent's system prompt (task_def)
    so the LLM knows who it's talking to and can personalize responses.

    Args:
        user: The identified user's profile.
        critical_facts: Formatted critical facts string (IPs, passwords, etc.).
        identification_result: Optional identification result dict.

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
- If you remember facts about them, weave them in conversationally

TOOL USAGE — Call these tools when appropriate:
- **remember_user_fact**: When user shares important info (employer, project, tech stack, role, etc.). Call IMMEDIATELY — don't wait to be asked. If they say "I switched jobs to X" or "I now work at X", update the employer fact.
- **update_user_preference**: When user says they prefer certain response styles.
- **manage_user_profile**: When user asks to view/update/delete their profile, add/remove skills or aliases. For deletion, ALWAYS pass confirm_delete=true when the user explicitly confirms.
- **identify_user**: When user says a different name or identity changes.
"""
    sections.append(personalization)

    return "\n".join(sections)


def build_anonymous_context() -> str:
    """Build context for an unidentified user.

    Tells the LLM about available user identification tools.
    """
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

After identification, IMMEDIATELY use these tools:
- **remember_user_fact** to save any facts they shared (employer, role, skills, tech stack, etc.)
- **manage_user_profile** with action="add_skill" for any technical skills mentioned
- **manage_user_profile** with action="add_alias" for any nicknames or alternate names mentioned

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
# User Management Assistant

You are a user management specialist. Your ONLY job is to manage user
identities, profiles, facts, preferences, skills, and aliases.

Current time: {current_time}

## Available Tools

You have exactly 6 tools:

| Tool | When to Use |
|------|-------------|
| **identify_user** | User says their name ("I'm Sean", "this is Alex") |
| **remember_user_fact** | User shares info to remember (employer, project, role, etc.) |
| **update_user_preference** | User wants response style changes (verbosity, code style, etc.) |
| **infer_user** | Identity unclear — try matching their message to known profiles |
| **get_user_context** | Check current session status and user profile data |
| **manage_user_profile** | View/update/delete profiles, manage skills and aliases |

## Rules

1. **Call tools immediately** — don't narrate what you're about to do, just do it.
2. **One tool call per action** — don't chain unnecessary calls.
3. **Deletions**: When user asks to delete their profile, ALWAYS pass \
`confirm_delete=true` to `manage_user_profile(action="delete_profile")`.
4. **New users**: If someone introduces themselves, call `identify_user` first, \
then `remember_user_fact` for any facts they shared.
5. **Fact updates**: If user says "I switched jobs to X", call \
`remember_user_fact(key="employer", value="X")` — this overwrites the old value.
6. **Be conversational** — respond naturally, not robotically. Confirm what \
you did in plain language ("Got it, I've updated your employer to X").
7. **Privacy**: Never expose another user's data. Only show the current \
user's profile unless they explicitly ask to list all users.

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
