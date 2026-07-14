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

# Short, common all-consonant (or otherwise vowel-less) English words/
# abbreviations that must NOT be flagged as gibberish fragments below:
# ordinary compact keys ("dob", "mfg_date", "hr_contact") use these
# constantly, and none of them are LLM-hallucination artifacts.
_VOWELLESS_ALLOWLIST = frozenset({
    "by", "my", "dry", "fly", "sky", "shy", "try", "why", "cry", "spy",
    "dob", "ssn", "id", "mfg", "hr", "hq", "vp", "ceo", "cfo", "cto",
    "st", "dr", "rd", "ave", "blvd", "nth", "mph", "gym", "gps", "tv",
})

# A key fragment with no vowel at all and long enough that it's very
# unlikely to be a real word/abbreviation (short ones are allowlisted
# above) is the strongest single signal of hallucinated LLM-extraction
# output rather than a human-chosen key: real snake_case keys are written
# by either the user's own wording or an extraction prompt explicitly
# asked for "short snake_case labels" from real vocabulary, and virtually
# never land on a 4+ letter, vowel-free fragment by chance. Observed in
# practice: a Q4 quantized 2B model produced the key "ol_taht_you_have"
# ("taht" - scrambled "that", vowel-free is incidental here but paired
# with the anagram check below it's a strong combined signal).
_VOWELLESS_MIN_LEN = 4

_VOWELS = frozenset("aeiou")


def normalize_key(key: str) -> str:
    """Lowercase snake_case, stripped of anything but [a-z0-9_]."""
    return re.sub(r"[^a-z0-9_]", "", key.strip().lower().replace(" ", "_"))


def is_placeholder(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDER_VALUES


def is_transcript_key(normalized_key: str) -> bool:
    return any(part in _TRANSCRIPT_KEY_MARKERS for part in normalized_key.split("_"))


def is_gibberish_key(normalized_key: str) -> bool:
    """True when a key fragment looks like scrambled/hallucinated output
    rather than a real word a human or a well-behaved extraction prompt
    would produce - a long, vowel-free, non-allowlisted fragment. Narrow
    and conservative on purpose: false negatives (a genuinely garbled key
    that happens to contain a vowel) are fine, since this only needs to
    catch the pattern actually observed in practice, not every possible
    hallucination shape. Numeric-only fragments (ids, years, counts) are
    never flagged - digits carry no "is this a real word" signal at all."""
    for part in normalized_key.split("_"):
        if not part or part in _VOWELLESS_ALLOWLIST or part.isdigit():
            continue
        if len(part) >= _VOWELLESS_MIN_LEN and not any(c in _VOWELS for c in part):
            return True
    return False


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
    if source == "auto" and is_gibberish_key(normalized):
        return None, "gibberish key, likely hallucinated extraction output"
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
