from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import List

from jobs.helper.base import RiskLevel

logger = logging.getLogger(__name__)

_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Modern OpenAI keys are `sk-proj-…` / `sk-svcacct-…` (variable length, with
    # - and _), alongside the legacy 48-char `sk-…`. Match both.
    ("openai_key",        re.compile(r"sk-(?:proj|svcacct|admin)-[a-zA-Z0-9_\-]{20,}")),
    ("openai_key_legacy", re.compile(r"sk-[a-zA-Z0-9]{48}")),
    ("anthropic_key",     re.compile(r"sk-ant-[a-zA-Z0-9\-]{90,}")),
    ("aws_access_key",    re.compile(r"(?:AKIA|ASIA|AGPA|AIDA)[0-9A-Z]{16}")),
    # Personal (ghp_), OAuth (gho_), user/server/refresh (ghu_/ghs_/ghr_), and
    # fine-grained (github_pat_) GitHub tokens.
    ("github_token",      re.compile(r"gh[posur]_[a-zA-Z0-9]{36,}")),
    ("github_fine_grained", re.compile(r"github_pat_[a-zA-Z0-9_]{60,}")),
    ("stripe_live_key",   re.compile(r"sk_live_[a-zA-Z0-9]{24,}")),
    ("slack_token",       re.compile(r"xox[baprs]-[0-9A-Za-z\-]{10,}")),
    ("google_api_key",    re.compile(r"AIza[0-9A-Za-z\-_]{35}")),
    ("sendgrid_key",      re.compile(r"SG\.[a-zA-Z0-9\-_]{22}\.[a-zA-Z0-9\-_]{43}")),
    ("twilio_key",        re.compile(r"SK[0-9a-fA-F]{32}")),
    ("jwt_token",         re.compile(r"eyJ[A-Za-z0-9\-_=]+\.[A-Za-z0-9\-_=]+\.?[A-Za-z0-9\-_.+/=]*")),
    ("private_key_block", re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("password_field",    re.compile(r'(?i)(password|passwd|pwd)\s*[:=]\s*\S+')),
]

_ENTROPY_THRESHOLD     = 4.5
_MIN_ENTROPY_TOKEN_LEN = 20
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{20,}")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _has_high_entropy(text: str) -> bool:
    for m in _TOKEN_RE.finditer(text):
        tok = m.group()
        if len(tok) >= _MIN_ENTROPY_TOKEN_LEN and _shannon_entropy(tok) > _ENTROPY_THRESHOLD:
            return True
    return False


@dataclass
class _SecretResult:
    detected: bool
    secret_types: List[str]
    high_entropy_detected: bool
    risk_level: RiskLevel


class _SecretScanner:
    def scan(self, text: str) -> _SecretResult:
        empty = _SecretResult(False, [], False, RiskLevel.CLEAN)
        if not text or not text.strip():
            return empty
        try:
            found = [label for label, pat in _SECRET_PATTERNS if pat.search(text)]
            high_entropy = _has_high_entropy(text)
            if not found and not high_entropy:
                return empty
            return _SecretResult(
                detected=True,
                secret_types=found,
                high_entropy_detected=high_entropy,
                risk_level=RiskLevel.HIGH if found else RiskLevel.MEDIUM,
            )
        except Exception as exc:
            logger.exception("[fluiq.secure] secret scan error: %s", exc)
            return empty
