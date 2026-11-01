"""Background LLM calls: prompts and parsing.

Every function takes the host's completion function ``(prompt, *,
max_tokens, temperature) -> str`` and never raises: a small local model
returns malformed JSON, chatter or nothing, and none of that may reach the
host. Prompts are short and every call is capped at ``tiers.LLM_MAX_TOKENS``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable

from . import tiers

log = logging.getLogger("elastimem")

CompleteFn = Callable[..., str]

_EXTRACT = (
    "Extract information about the user from one chat exchange. Never facts "
    "about the assistant, never a copy of the exchange. Output only JSON with "
    "three keys:\n"
    '"facts": snake_case keys to short values the user stated, e.g. '
    '{"favorite_color": "blue"};\n'
    '"entities": [{"name": str, "type": "person"|"place"|"org"|"thing"}] '
    "mentioned, excluding the user;\n"
    '"relations": [{"source": str, "relation": str, "target": str}] with '
    'snake_case relations; use "user" for the user.\n'
    "Use empty values when there is nothing to report."
)
_FOLD = ("Condense these conversation notes into 2-3 short plain sentences: what "
         "the user wanted and what was decided. Past tense. Output only the summary.")
_SESSION = ("Summarize what the user talked about and wanted in this session in 1-2 "
            "short plain sentences, past tense. Output only the summary.")
_MERGE = ("A memory changed value. Reply with only the value to keep: NEW if it "
          "replaces OLD, or one short value combining both.")
_SAME = ("Are these two labels the same real-world thing written differently? "
         'Reply only "yes" or "no".')
_TOPIC = "Name the topic these related things share in 1-4 words. Output only the name."


def _call(complete: CompleteFn, instruction: str, text: str, max_tokens: int = tiers.LLM_MAX_TOKENS) -> str:
    try:
        return (complete(f"{instruction}\n\n{text}", max_tokens=max_tokens, temperature=0.0) or "").strip()
    except Exception:
        log.exception("elastimem: background LLM call failed")
        return ""


def extract(complete: CompleteFn, user: str, reply: str) -> dict:
    """``{"facts": {...}, "entities": [...], "relations": [...]}``; empty on failure."""
    text = _call(complete, _EXTRACT, f"User said: {user}\nAssistant replied: {reply[:400]}")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        payload = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        return {}
    facts = payload.get("facts")
    return {
        "facts": {str(k): str(v) for k, v in facts.items() if isinstance(v, (str, int, float))}
                 if isinstance(facts, dict) else {},
        "entities": payload.get("entities"),
        "relations": payload.get("relations"),
    }


def fold(complete: CompleteFn, previous: str | None, turns: list[tuple[str, str]]) -> str:
    notes = [f"Earlier summary: {previous}"] if previous else []
    notes += [f"User: {u[:200]}\nAssistant: {a[:200]}" for u, a in turns]
    return _call(complete, _FOLD, "\n\n".join(notes))


def session_summary(complete: CompleteFn, user_turns: list[str]) -> str:
    if len(user_turns) < 2:
        return ""
    return _call(complete, _SESSION, "\n".join(f"- {t[:160]}" for t in user_turns[-12:]))


def merge_value(complete: CompleteFn, key: str, old: str, new: str) -> str:
    out = _call(complete, _MERGE, f"Key: {key}\nOLD: {old}\nNEW: {new}", max_tokens=48)
    return out if out and len(out) < 200 else ""


def same_thing(complete: CompleteFn, a: str, b: str) -> bool:
    return _call(complete, _SAME, f"A: {a}\nB: {b}", max_tokens=4).casefold().startswith("y")


def topic_name(complete: CompleteFn, names: list[str]) -> str:
    out = _call(complete, _TOPIC, ", ".join(names[:12]), max_tokens=12).strip("\"'. ")
    return out if 0 < len(out) <= 60 else ""
