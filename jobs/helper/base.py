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


def _compile(patterns: list[str]) -> list[tuple[str, re.Pattern]]:
    return [(p, re.compile(re.escape(p), re.IGNORECASE)) for p in patterns]


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
