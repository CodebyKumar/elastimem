"""Rule-based fact capture: the zero-cost extraction floor.

A handful of compiled regexes over the user's message catch the highest-value
personal facts inline (microseconds, works in every tier, no LLM). This is
deliberately conservative — high precision over recall; the LLM extraction
pass (when the tier affords it) handles the long tail.
"""

from __future__ import annotations

import re

# (compiled pattern, fact key or None to use group('key'))
_RULES: list[tuple[re.Pattern, str]] = [
    # Name/location values must be capitalized words — the patterns stay
    # case-sensitive on the captured group ("Priya and I live…" must yield
    # "Priya", not "Priya and") while tolerating lowercased prefixes.
    (re.compile(r"\b[Mm]y name is ([A-Z][\w'-]+(?: [A-Z][\w'-]+)?)"), "name"),
    (re.compile(r"\b[Cc]all me ([A-Z][\w'-]+)\b"), "name"),
    (re.compile(r"\b[Ii] live in ([A-Z][\w'-]+(?:[ ,][A-Z][\w'-]+){0,2})"), "location"),
    (re.compile(r"\b[Ii](?:'m| am) from ([A-Z][\w'-]+(?:[ ,][A-Z][\w'-]+){0,2})"), "location"),
    (re.compile(r"\bi work (?:as an? |at )([\w &'-]{2,40})", re.I), "occupation"),
    (re.compile(r"\bi(?:'m| am) an? ([\w-]{2,25}) by (?:profession|trade)\b", re.I), "occupation"),
    (re.compile(r"\bi(?:'m| am) allergic to ([\w ,'-]{2,40})", re.I), "allergies"),
    (re.compile(r"\bi(?:'m| am) (vegetarian|vegan|pescatarian)\b", re.I), "diet"),
]

_PREFERENCE = re.compile(
    r"\bmy (?:favorite|favourite|preferred) ([\w ]{2,20}) is ([\w ,'-]{2,40})", re.I
)

# Conversational filler that trails a captured value ("teal by the way",
# "ramen though") — cut the value at the first filler token.
_TRAILING_FILLER = frozenset(
    {"by", "though", "actually", "anyway", "btw", "honestly", "right", "now",
     "these", "at", "for", "in", "so", "but"}
)


def _clean_value(raw: str) -> str:
    words = raw.strip().rstrip(".,!?").split()
    for i, word in enumerate(words):
        if word.lower() in _TRAILING_FILLER:
            words = words[:i]
            break
    return " ".join(words)


def capture(user_text: str) -> list[tuple[str, str]]:
    """Extract (key, value) facts from one user message. Pure, no I/O."""
    found: list[tuple[str, str]] = []
    for pattern, key in _RULES:
        m = pattern.search(user_text)
        if m:
            value = _clean_value(m.group(1))
            if value:
                found.append((key, value))
    for m in _PREFERENCE.finditer(user_text):
        subject = m.group(1).strip().lower().replace(" ", "_")
        value = _clean_value(m.group(2))
        if value:
            found.append((f"favorite_{subject}", value))
    return found
