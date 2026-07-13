"""Rule-based fact capture: the zero-cost extraction floor.

A handful of compiled regexes over the user's message catch the highest-value
personal facts inline (microseconds, works in every tier, no LLM). This is
deliberately conservative — high precision over recall; the LLM extraction
pass (when the tier affords it) handles the long tail.
"""

from __future__ import annotations

import re

# (compiled pattern, fact key or None to use group('key'))
#
# Name/location values are tried as capitalized words first ("Priya and I
# live…" stops at "Priya", not "Priya and" — capitalization bounds the
# match against a trailing clause), falling back to a single all-lowercase
# word for casual typing ("my name is kumar"). The lowercase fallback has no
# capital letter to bound it, so it only captures one word and
# _LEADING_STOPWORD rejects it when that word is a stopword rather than part
# of a name/place ("my name is not important" must not yield "not").
_RULES: list[tuple[re.Pattern, str]] = [
    # "my (full/real/actual) name is X" - an adjective between "my" and
    # "name" (very common phrasing when correcting/expanding on an earlier
    # short name) must not block the match.
    (re.compile(r"\b[Mm]y (?:\w+ )?name is ([A-Z][\w'-]+(?: [A-Z][\w'-]+)?)"), "name"),
    (re.compile(r"\b[Mm]y (?:\w+ )?name is ([a-z][\w'-]+)\b"), "name"),
    # "call me X" / "you can call me X" - typo tolerance ("all me" for "call
    # me") is NOT attempted here; that's indistinguishable from genuine
    # other text without much higher false-positive risk.
    (re.compile(r"\b[Cc]all me ([A-Za-z][\w'-]+)\b"), "name"),
    (re.compile(r"\b[Ii] (?:live|stay) in ([A-Z][\w'-]+(?:[ ,][A-Z][\w'-]+){0,2})"), "location"),
    (re.compile(r"\b[Ii] (?:live|stay) in ([a-z][\w'-]+)\b"), "location"),
    (re.compile(r"\b[Ii](?:'m| am) from ([A-Z][\w'-]+(?:[ ,][A-Z][\w'-]+){0,2})"), "location"),
    (re.compile(r"\b[Ii](?:'m| am) from ([a-z][\w'-]+)\b"), "location"),
    (re.compile(r"\bi work (?:as an? |at )([\w &'-]{2,40})", re.I), "occupation"),
    (re.compile(r"\bi(?:'m| am) an? ([\w-]{2,25}) by (?:profession|trade)\b", re.I), "occupation"),
    (re.compile(r"\bi(?:'m| am) allergic to ([\w ,'-]{2,40})", re.I), "allergies"),
    (re.compile(r"\bi(?:'m| am) (vegetarian|vegan|pescatarian)\b", re.I), "diet"),
]

# Rejects a lowercase-led capture whose first word is a stopword rather than
# part of a name/place ("my name is not important", "i live in the moment").
_LEADING_STOPWORD = frozenset({
    "not", "no", "very", "so", "too", "really", "just", "kind", "sort",
    "a", "an", "the", "still", "also", "actually", "basically", "literally",
})

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
            if value and value.split()[0].lower() not in _LEADING_STOPWORD:
                found.append((key, value))
    for m in _PREFERENCE.finditer(user_text):
        subject = m.group(1).strip().lower().replace(" ", "_")
        value = _clean_value(m.group(2))
        if value:
            found.append((f"favorite_{subject}", value))
    return found
