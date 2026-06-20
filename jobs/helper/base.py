from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    CLEAN  = "clean"
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


_RISK_ORDER: dict[str, int] = {
    RiskLevel.CLEAN:  0,
    RiskLevel.LOW:    1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH:   3,
}


def _max_risk(*levels: RiskLevel) -> RiskLevel:
    return max(levels, key=lambda r: _RISK_ORDER[r])


def _risk_from_score(score: float) -> RiskLevel:
    if score >= 0.9:
        return RiskLevel.HIGH
    if score >= 0.5:
        return RiskLevel.MEDIUM
    if score >= 0.3:
        return RiskLevel.LOW
    return RiskLevel.CLEAN


def _compile(
    patterns: list[str],
    *,
    case_sensitive: bool = False,
) -> list[tuple[str, re.Pattern]]:
    """Compile literal phrases to regexes.

    Word boundaries (``\\b``) are added only on *alphanumeric* edges so short
    tokens like ``DAN``/``STAN`` match whole words, not substrings inside
    ``guidance``/``understanding`` — while delimiter markers such as
    ``[system]:`` or ``<|im_start|>system`` (non-word edges) keep matching as-is.
    Pass ``case_sensitive=True`` for all-caps persona acronyms (DAN, STAN, …) so
    they don't fire on the ordinary lowercase words or names.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled: list[tuple[str, re.Pattern]] = []
    for p in patterns:
        if not p:
            continue
        left = r"\b" if p[0].isalnum() else ""
        right = r"\b" if p[-1].isalnum() else ""
        compiled.append((p, re.compile(f"{left}{re.escape(p)}{right}", flags)))
    return compiled


@dataclass
class _AttackResult:
    detected: bool
    patterns_found: List[str]
    risk_score: float
    risk_level: RiskLevel


def _scan_patterns(
    text: str,
    compiled: list[tuple[str, re.Pattern]],
    logger_tag: str,
) -> _AttackResult:
    empty = _AttackResult(False, [], 0.0, RiskLevel.CLEAN)
    if not text or not text.strip():
        return empty
    try:
        found = [p for p, rx in compiled if rx.search(text)]
        if not found:
            return empty
        score = min(len(found) / max(len(compiled), 1), 1.0)
        # Any matched attack pattern is HIGH risk — ratio scoring is informational only
        return _AttackResult(True, found, round(score, 4), RiskLevel.HIGH)
    except Exception as exc:
        logger.exception("[fluiq.secure] %s scan error: %s", logger_tag, exc)
        return empty


def _scan_tiered(
    text: str,
    strong: list[tuple[str, re.Pattern]],
    weak: list[tuple[str, re.Pattern]],
    logger_tag: str,
) -> _AttackResult:
    """Severity-aware scan: an explicit ('strong') pattern is HIGH on its own;
    'weak'/ambiguous phrases (e.g. ``act as``, ``dark mode``, ``let's say``) need
    corroboration — one is LOW, two or more is MEDIUM. This stops a single
    ambiguous phrase in benign text from raising a HIGH alert."""
    empty = _AttackResult(False, [], 0.0, RiskLevel.CLEAN)
    if not text or not text.strip():
        return empty
    try:
        strong_hits = [p for p, rx in strong if rx.search(text)]
        weak_hits = [p for p, rx in weak if rx.search(text)]
        found = strong_hits + weak_hits
        if strong_hits:
            return _AttackResult(True, found, 0.9, RiskLevel.HIGH)
        if len(weak_hits) >= 2:
            return _AttackResult(True, found, 0.6, RiskLevel.MEDIUM)
        if weak_hits:
            return _AttackResult(True, found, 0.3, RiskLevel.LOW)
        return empty
    except Exception as exc:
        logger.exception("[fluiq.secure] %s scan error: %s", logger_tag, exc)
        return empty
