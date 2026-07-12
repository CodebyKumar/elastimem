"""Fact validation: the gate between extractors and the store.

A small local model running the extraction pass will, with some regularity,
produce junk: placeholder values ("unknown"), echoes of the transcript filed
as "facts" ({"user_message": ...}), or facts about *itself* filed under the
user's profile. This module rejects those before they ever reach the facts
table, and quarantines automatic rejections so misbehavior stays inspectable
instead of silently vanishing.

Hosts extend the defaults via ``ElastimemConfig.reserved_keys`` (keys the host
owns — e.g. an agent's identity fields) and may pass extra self-referential
value markers when constructing the store.
"""

from __future__ import annotations

import re
import sqlite3

from .db import utcnow

# Values that carry no information — refusing them stops "goals: unknown"
# style noise from being stored.
_PLACEHOLDER_VALUES = {
    "unknown", "n/a", "na", "none", "null", "not specified", "not provided",
    "not sure", "unclear", "tbd", "?", "-", "",
}

# Keys shaped like a conversation transcript rather than a durable fact. An
# extractor sometimes echoes the exchange back ({"user_message": "...",
# "assistant_reply": "..."}); these describe an EXCHANGE, not the user.
_TRANSCRIPT_KEY_MARKERS = frozenset({
    "user", "assistant", "message", "reply", "replied", "response",
    "responded", "said", "says", "conversation", "exchange", "dialogue",
    "utterance",
})

# Value substrings marking a fact as describing the AGENT rather than the
# user, regardless of key ("I'm a local AI agent" filed under plain "role").
_SELF_REFERENTIAL_MARKERS = (
    "ai agent", "language model", "i run on", "running locally",
    "large language model", "ai assistant",
)


def normalize_key(key: str) -> str:
    """Lowercase snake_case, stripped of anything but [a-z0-9_]."""
    return re.sub(r"[^a-z0-9_]", "", key.strip().lower().replace(" ", "_"))


def is_placeholder(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDER_VALUES


def is_transcript_key(normalized_key: str) -> bool:
    return any(part in _TRANSCRIPT_KEY_MARKERS for part in normalized_key.split("_"))


def is_reserved_key(normalized_key: str, reserved: frozenset[str]) -> bool:
    """True when the key is host-reserved, exactly or as a ``*_<reserved>`` suffix."""
    return normalized_key in reserved or any(
        normalized_key.endswith(f"_{r}") for r in reserved
    )


def is_self_referential_value(value: str, extra_markers: tuple[str, ...] = ()) -> bool:
    lowered = value.lower()
    return any(m in lowered for m in (*_SELF_REFERENTIAL_MARKERS, *extra_markers))


def validate_fact(
    key: str,
    value: str,
    source: str,
    *,
    reserved_keys: frozenset[str] = frozenset(),
    self_ref_markers: tuple[str, ...] = (),
) -> tuple[str | None, str]:
    """Validate one candidate fact.

    Returns ``(normalized_key, "ok")`` when storable, ``(None, reason)`` when
    rejected. Transcript-shaped keys are only rejected for automatic sources —
    a user explicitly asking to remember "last_message" gets to keep it.
    """
    normalized = normalize_key(key)
    value = str(value).strip()

    if not normalized or not value:
        return None, "empty key or value"
    if is_reserved_key(normalized, reserved_keys):
        return None, "reserved key (owned by the host application)"
    if source in ("auto", "rule") and is_transcript_key(normalized):
        return None, "transcript-shaped key, not a fact"
    if is_self_referential_value(value, self_ref_markers):
        return None, "self-referential value (describes the agent, not the user)"
    if is_placeholder(value):
        return None, "placeholder value, nothing to store"
    return normalized, "ok"


def quarantine(
    conn: sqlite3.Connection,
    key: str,
    value: str,
    reason: str,
    source: str,
    cap: int = 200,
) -> None:
    """Record a rejected automatic extraction for inspection (never prompted)."""
    with conn:
        conn.execute(
            "INSERT INTO quarantine(ts, key, value, reason, source) VALUES (?,?,?,?,?)",
            (utcnow(), key[:200], str(value)[:500], reason, source),
        )
        conn.execute(
            "DELETE FROM quarantine WHERE id NOT IN "
            "(SELECT id FROM quarantine ORDER BY id DESC LIMIT ?)",
            (cap,),
        )
