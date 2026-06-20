"""Jailbreak attack patterns.

Role-play escapes, persona hijacks, fictional-framing bypasses, and
encoding tricks that attempt to get the model to act against its training.

Patterns are tiered:
  • STRONG   — explicit jailbreak intent; HIGH on a single match.
  • ACRONYMS — named personas (DAN, STAN, …); matched case-sensitively and
               whole-word so they never fire inside ordinary words such as
               "guidance" / "understanding" / "circumstances".
  • WEAK     — phrases that also occur in benign text ("act as", "dark mode",
               "let's say"); one match is LOW, two or more is MEDIUM.
"""
from __future__ import annotations

from jobs.helper.base import _compile

# Explicit jailbreak intent — HIGH on a single match.
_JAILBREAK_STRONG: list[str] = [
    "you are now",
    "pretend you are",
    "pretend to be",
    "you are an AI without",
    "you are a different AI",
    "you are no longer",
    "you are not an AI",
    "you have no restrictions",
    "without any restrictions",
    "uncensored",
    "unfiltered",
    "without censorship",
    "without filters",
    "jailbreak",
    "do anything now",
    "developer mode",
    "god mode",
    "sudo mode",
    "admin mode",
    "unrestricted mode",
    "override mode",
    "evil mode",
    "new persona",
    "your true self",
    "deep down you want",
    "you secretly want",
    "your inner self",
]

# Named persona acronyms — case-sensitive + whole-word (real ones are uppercase).
_JAILBREAK_ACRONYMS: list[str] = [
    "DAN",
    "STAN",
    "DUDE",
    "AIM",
    "BetterDAN",
    "BasedGPT",
]

# Ambiguous — also common in benign text. One = LOW, two or more = MEDIUM.
_JAILBREAK_WEAK: list[str] = [
    "act as if you are",
    "act as",
    "act like",
    "roleplay as",
    "role play as",
    "play the role of",
    "simulate a",
    "simulate being",
    "without restrictions",
    "without limitations",
    "no restrictions",
    "no limitations",
    "unrestricted",
    "test mode",
    "debug mode",
    "dark mode",
    "maintenance mode",
    "chaos mode",
    "hypothetically",
    "in a fictional world",
    "in a story",
    "for a novel",
    "for a movie",
    "for educational purposes",
    "for research purposes",
    "for academic purposes",
    "purely hypothetical",
    "just imagine",
    "let's say",
    "as a thought experiment",
    "theoretically speaking",
    "base64",
    "rot13",
    "in pig latin",
    "backwards",
    "in leetspeak",
    "i am a developer",
    "i am the developer",
    "i am an admin",
    "i am the administrator",
    "i have permission",
    "i have clearance",
    "this is authorized",
    "you are allowed to",
    "you can say",
    "it is ok to",
    "if you could",
    "what would you do if",
    "bypass",
]

JAILBREAK_STRONG_COMPILED = (
    _compile(_JAILBREAK_STRONG) + _compile(_JAILBREAK_ACRONYMS, case_sensitive=True)
)
JAILBREAK_WEAK_COMPILED = _compile(_JAILBREAK_WEAK)

# Combined list — used for indirect-injection scanning of retrieved docs, where
# only the STRONG tier should flag (weak phrases are common in reference text).
JAILBREAK_COMPILED = JAILBREAK_STRONG_COMPILED
