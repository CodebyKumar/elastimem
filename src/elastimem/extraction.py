"""LLM job bodies: fact extraction, rolling summaries, session summaries.

Every function here takes ``complete_fn`` — the host's completion callable —
and is defensive to the bone: a local 2B model will return malformed JSON,
prefix chatter, or nothing at all, and none of that may ever surface as an
exception. All prompts are sized for tiny models (short instructions, hard
token caps) because the worker may share the host's only model instance.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from typing import Callable

from . import semantic
from .config import ElastimemConfig
from .db import utcnow

log = logging.getLogger("elastimem")

CompleteFn = Callable[..., str]

_EXTRACT_PROMPT = (
    "You extract information ABOUT THE USER ONLY from one chat exchange — "
    "never facts about the assistant itself, and never the exchange itself. "
    "Output ONLY a JSON object with three keys:\n"
    '"facts": a flat object of snake_case keys to short string values, e.g. '
    '{"favorite_color": "blue"}. Valid: the user\'s name, preferences, '
    "location, occupation, interests, plans, corrections about THEMSELVES. "
    "Invalid: keys like 'user_message' or 'assistant_reply' — copying what "
    "either side said is not a fact.\n"
    '"entities": a list of {"name": str, "type": str} for notable people, '
    'places, organizations, or things mentioned. type is one of "person", '
    '"place", "org", "thing". Omit the user themself.\n'
    '"relationships": a list of {"source": str, "relation": str, "target": '
    'str} short snake_case relations between two entities or between the '
    'user and an entity (use "user" for the user), e.g. {"source": "user", '
    '"relation": "works_at", "target": "Acme Corp"}.\n'
    "Use an empty object/list for any key with nothing to report. If there "
    "is nothing at all to extract, output exactly NONE."
)

_ROLLING_PROMPT = (
    "Condense the following conversation notes into 2-3 short plain sentences "
    "capturing what the user wanted and any decisions made. Past tense. "
    "Output only the summary."
)

_SESSION_PROMPT = (
    "Summarize what the user talked about and wanted in this chat session in "
    "1-2 short plain sentences, past tense. Output only the summary."
)

_MERGE_PROMPT = (
    "A memory key changed value recently. Decide if the NEW value fully "
    "replaces the OLD one, or if they should be merged into one short value. "
    "Reply with ONLY the final value to keep (no explanation)."
)


def _call(complete_fn: CompleteFn, system: str, user: str, max_tokens: int) -> str:
    """One guarded completion. Returns '' on any failure."""
    try:
        out = complete_fn(
            f"{system}\n\n{user}", max_tokens=max_tokens, temperature=0.0
        )
        return (out or "").strip()
    except Exception:
        log.exception("elastimem: background completion failed")
        return ""


def _store_facts_from_payload(
    facts_payload: object, store_fn: Callable[[str, str, str], tuple[bool, str]]
) -> dict[str, str]:
    if not isinstance(facts_payload, dict):
        return {}
    stored: dict[str, str] = {}
    for key, value in facts_payload.items():
        if not isinstance(value, (str, int, float)):
            continue
        changed, _ = store_fn(str(key), str(value), "auto")
        if changed:
            stored[str(key)] = str(value)
    return stored


def extract_facts(
    conn: sqlite3.Connection,
    config: ElastimemConfig,
    complete_fn: CompleteFn,
    user_text: str,
    assistant_text: str,
    store_fn: Callable[[str, str, str], tuple[bool, str]],
    graph_hops: int = 0,
    source_chunk_id: int | None = None,
) -> dict[str, str]:
    """Reflection pass over one exchange; validated stores via ``store_fn``.

    One completion call produces both facts and (when ``graph_hops`` > 0,
    i.e. the governor's tier permits graph retrieval) entities/relationships
    for the knowledge graph — never a second model call per turn. Graph
    storage failures are caught and logged, never allowed to affect fact
    extraction's own result.
    """
    text = _call(
        complete_fn,
        _EXTRACT_PROMPT,
        f"User said: {user_text}\nAssistant replied: {assistant_text[:400]}",
        max_tokens=config.worker_max_tokens,
    )
    if not text or text.upper().startswith("NONE"):
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}

    # Backward compat: a model that ignores the new prompt shape and
    # returns the old flat {key: value} JSON (no "facts" wrapper) is
    # treated as if the whole payload were the facts dict.
    stored = _store_facts_from_payload(payload.get("facts", payload), store_fn)

    if graph_hops > 0:
        try:
            from . import graph as graph_mod

            graph_mod.store_extraction(
                conn, payload.get("entities"), payload.get("relationships"),
                node_cap=config.graph_node_cap, edge_cap=config.graph_edge_cap,
                source_chunk_id=source_chunk_id,
            )
        except Exception:
            log.exception("elastimem: graph extraction storage failed")

    return stored


def rolling_summary(
    complete_fn: CompleteFn,
    config: ElastimemConfig,
    previous_summary: str | None,
    evicted_turns: list[tuple[str, str]],
) -> str:
    """Fold evicted turns into the running conversation summary."""
    notes = []
    if previous_summary:
        notes.append(f"Earlier summary: {previous_summary}")
    for user_text, assistant_text in evicted_turns:
        notes.append(f"User: {user_text[:200]}\nAssistant: {assistant_text[:200]}")
    return _call(
        complete_fn, _ROLLING_PROMPT, "\n\n".join(notes),
        max_tokens=config.worker_max_tokens,
    )


def session_summary(
    complete_fn: CompleteFn, config: ElastimemConfig, user_turns: list[str]
) -> str:
    """1-2 sentence episodic summary; '' when the session was trivial."""
    if len(user_turns) < 2:
        return ""
    transcript = "\n".join(f"- {t[:160]}" for t in user_turns[-12:])
    return _call(complete_fn, _SESSION_PROMPT, transcript,
                 max_tokens=config.worker_max_tokens)


# --------------------------------------------------------------------------- #
# consolidation
# --------------------------------------------------------------------------- #
def consolidate(
    conn: sqlite3.Connection,
    config: ElastimemConfig,
    complete_fn: CompleteFn | None,
    *,
    llm_merge: bool,
) -> dict[str, int]:
    """Maintenance sweep: decay/archival, contradiction merges, caps.

    ``llm_merge`` (FULL tier + complete_fn present) reviews keys whose value
    changed recently and lets the model produce a merged value when the new
    one doesn't cleanly supersede the old ("moving to Austin in May" vs
    "lives in Austin").
    """
    stats = {"archived": semantic.apply_decay(conn, config), "merged": 0}

    if llm_merge and complete_fn is not None:
        recent_changes = conn.execute(
            "SELECT old.key, old.value AS old_value, new.value AS new_value,"
            " new.id AS new_id FROM facts old JOIN facts new"
            " ON new.id = old.invalidated_by"
            " WHERE old.invalidated_at >= datetime('now', ?)"
            " AND new.invalidated_at IS NULL LIMIT 5",
            (f"-{config.fact_merge_review_window_days} days",),
        ).fetchall()
        for row in recent_changes:
            merged = _call(
                complete_fn, _MERGE_PROMPT,
                f"Key: {row['key']}\nOLD: {row['old_value']}\nNEW: {row['new_value']}",
                max_tokens=48,
            )
            if merged and merged != row["new_value"] and len(merged) < 200:
                with conn:
                    conn.execute(
                        "UPDATE facts SET value=?, last_accessed_at=? WHERE id=?",
                        (merged, utcnow(), row["new_id"]),
                    )
                stats["merged"] += 1

    with conn:
        conn.execute(
            "DELETE FROM quarantine WHERE id NOT IN"
            " (SELECT id FROM quarantine ORDER BY id DESC LIMIT ?)",
            (config.quarantine_cap,),
        )
    return stats
