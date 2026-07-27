"""Skeleton key attack patterns.

Microsoft's "Skeleton Key" attack: asking the model to add a new "mode" or
augment its baseline directives so it stops refusing harmful requests.
Reference: https://www.microsoft.com/en-us/security/blog/2024/06/26/mitigating-skeleton-key-a-new-type-of-generative-ai-jailbreak-technique/
"""
from __future__ import annotations

from jobs.helper.base import _compile

# Explicit skeleton-key intent — HIGH on a single match.
_SKELETON_KEY_STRONG: list[str] = [
    "add a mode",
    "add a new mode",
    "create a mode",
    "enable a mode",
    "unlock a mode",
    "switch to a mode",
    "augment your baseline",
    "augment your directives",
    "update your baseline",
    "update your directives",
    "change your baseline",
    "override your baseline",
    "add to your instructions",
    "unlock your capabilities",
    "unlock your full capabilities",
    "unlock your potential",
    "your safety guidelines don't apply",
    "safety guidelines don't apply",
    "your guidelines don't apply",
    "treat this as an exception",
    "exempt this",
    "waive your",
    "suspend your",
    "temporarily disable",
    "ignore your safety",
    "bypass your safety",
    "disable your safety",
    "remove your restrictions",
    "add capability",
    "grant yourself",
    "give yourself permission",
    "enable yourself to",
    "allow yourself to",
    "i authorize you to",
    "you are authorized to",
    "you now have permission",
    "your new directive",
    "your updated directive",
    "new baseline",
    "new directive",
]

# Ambiguous — routine customer-support / conversational phrasing. One = LOW,
# two or more = MEDIUM. "Can you make an exception to the refund policy?" is a
# perfectly ordinary request and must not be a HIGH block on its own.
_SKELETON_KEY_WEAK: list[str] = [
    "make an exception",
    "add an exception",
    "this is an exception",
    "turn off your",
]

SKELETON_KEY_STRONG_COMPILED = _compile(_SKELETON_KEY_STRONG)
SKELETON_KEY_WEAK_COMPILED = _compile(_SKELETON_KEY_WEAK)

# Combined list — used for indirect / image scanning, where only the STRONG
# tier should flag (weak phrases are common in benign reference text).
SKELETON_KEY_COMPILED = SKELETON_KEY_STRONG_COMPILED
