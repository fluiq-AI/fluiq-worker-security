"""Skeleton key attack patterns.

Microsoft's "Skeleton Key" attack: asking the model to add a new "mode" or
augment its baseline directives so it stops refusing harmful requests.
Reference: https://www.microsoft.com/en-us/security/blog/2024/06/26/mitigating-skeleton-key-a-new-type-of-generative-ai-jailbreak-technique/
"""
from __future__ import annotations

from jobs.helper.base import _compile

_SKELETON_KEY_PATTERNS: list[str] = [
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
    "add an exception",
    "make an exception",
    "unlock your capabilities",
    "unlock your full capabilities",
    "unlock your potential",
    "your safety guidelines don't apply",
    "safety guidelines don't apply",
    "your guidelines don't apply",
    "treat this as an exception",
    "this is an exception",
    "exempt this",
    "waive your",
    "suspend your",
    "temporarily disable",
    "turn off your",
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

SKELETON_KEY_COMPILED = _compile(_SKELETON_KEY_PATTERNS)
