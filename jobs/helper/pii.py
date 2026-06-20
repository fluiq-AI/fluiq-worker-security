from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from jobs.helper.base import RiskLevel, _risk_from_score

logger = logging.getLogger(__name__)

# Harmonized Tariff Schedule (HTS) codes — a 4-digit heading followed by one to
# four dot-separated groups of 2–4 digits. This covers both canonical groupings:
#   USITC     6109.10.00.04   (dotted 2-digit groups)
#   CBP/CROSS  6109.10.0040    (8-digit subheading + 4-digit statistical suffix)
# Each is 10 digits that Presidio's phone-number recognizer reads as a
# PHONE_NUMBER. Customs/trade traces are full of them, so we strip phone matches
# that fall inside an HTS code. The 4-digit heading anchor keeps real dotted
# phone formats (415.555.1234, 3-digit lead group) from matching.
_HTS_CODE_RE = re.compile(r"\b\d{4}(?:\.\d{2,4}){1,4}\b")

# Schedule B / HTSUS codes also travel as a *bare* 10-digit string — most often
# as the value of a `schedule_b` field in classification traces (e.g.
# 'schedule_b': '9013809100'). A bare 10-digit run is indistinguishable from a
# phone number on its own, so we only treat it as a tariff code when there is
# positive evidence and never drop a generic 10-digit number: (1) it is the
# value of a schedule_b key, or (2) its digits equal a dotted HTS code elsewhere
# in the same text. This keeps real bare-number PII intact while clearing the
# tariff-code false positives.
_SCHEDULE_B_RE = re.compile(r"schedule[_\s]?b['\"]?\s*[:=]\s*['\"]?(\d{10})\b", re.IGNORECASE)
_NON_DIGIT_RE = re.compile(r"\D")

# US ZIP+4 postal codes (NNNNN-NNNN) are byte-identical to Presidio's very-weak
# US_SSN pattern (SSN1: 5 digits, hyphen, 4 digits). A real SSN is always 3-2-4,
# never 5-4, so a US_SSN match in 5-4 form is always a postal-code false positive
# and is dropped. Customs/trade traces carry importer/consignee addresses full of
# these (e.g. 'Spring Valley, NY 10977-2006').
_ZIP4_RE = re.compile(r"\b\d{5}-\d{4}\b")

try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine
    _PRESIDIO_OK = True
except ImportError:
    _PRESIDIO_OK = False
    logger.warning("[fluiq.secure] presidio not installed — PII scanning disabled")


_ENTITY_WEIGHTS: dict[str, float] = {
    "US_SSN":             1.0,
    "CREDIT_CARD":        1.0,
    "IBAN_CODE":          0.9,
    "CRYPTO":             0.8,
    "US_PASSPORT":        1.0,
    "EMAIL_ADDRESS":      0.5,
    "PHONE_NUMBER":       0.5,
    "PERSON":             0.4,
    "LOCATION":           0.3,
    "IP_ADDRESS":         0.3,
    "OPENAI_API_KEY":     1.0,
    "ANTHROPIC_API_KEY":  1.0,
    "AWS_ACCESS_KEY":     1.0,
    "GITHUB_TOKEN":       1.0,
    "STRIPE_LIVE_KEY":    1.0,
}
_SUPPORTED_ENTITIES = list(_ENTITY_WEIGHTS.keys())


def _build_custom_recognizers() -> list:
    specs = [
        ("OPENAI_API_KEY",    [Pattern("OpenAI key",      r"sk-[a-zA-Z0-9]{48}",        0.95)]),
        ("ANTHROPIC_API_KEY", [Pattern("Anthropic key",   r"sk-ant-[a-zA-Z0-9\-]{90,}", 0.95)]),
        ("AWS_ACCESS_KEY",    [Pattern("AWS key",         r"AKIA[0-9A-Z]{16}",           0.95)]),
        ("GITHUB_TOKEN",      [Pattern("GitHub token",    r"ghp_[a-zA-Z0-9]{36}",        0.95)]),
        ("STRIPE_LIVE_KEY",   [Pattern("Stripe live key", r"sk_live_[a-zA-Z0-9]{24}",    0.95)]),
        # Passport patterns — merged with Presidio's built-in to avoid duplicate entities.
        # US_DRIVER_LICENSE excluded: formats overlap with passport numbers causing false positives.
        ("US_PASSPORT", [
            Pattern("Passport 1L+8D",    r"\b[A-Z][0-9]{8}\b",     0.85),
            Pattern("Passport 2L+7D",    r"\b[A-Z]{2}[0-9]{7}\b",  0.80),
        ]),
    ]
    return [
        PatternRecognizer(supported_entity=e, patterns=p, supported_language="en")
        for e, p in specs
    ]


def _drop_hts_phone_false_positives(results: list, text: str) -> list:
    """Remove PHONE_NUMBER matches that are really Harmonized Tariff Schedule
    codes. Presidio's phone recognizer misreads tariff codes (10 digits, dotted
    NNNN.NN.NN.NN or bare NNNNNNNNNN) as phone numbers. A match is dropped when
    it (a) overlaps a dotted HTS code, (b) sits in the value of a `schedule_b`
    field, or (c) its digits equal a dotted HTS code elsewhere in the text.
    Bare 10-digit numbers with no such corroboration are left untouched so real
    phone-number PII still surfaces. Runs before the entity list, risk score,
    and redaction are computed.
    """
    dotted_spans = [(m.start(), m.end()) for m in _HTS_CODE_RE.finditer(text)]
    sched_spans = [m.span(1) for m in _SCHEDULE_B_RE.finditer(text)]
    suppress_spans = dotted_spans + sched_spans
    # Digit-only forms of dotted codes corroborate bare 10-digit tariff codes
    # that appear elsewhere (e.g. schedule_b mirroring a dotted `code`).
    hts_digits = {
        d for d in (_NON_DIGIT_RE.sub("", text[s:e]) for s, e in dotted_spans)
        if len(d) >= 8
    }
    if not suppress_spans and not hts_digits:
        return results
    kept = []
    for r in results:
        if r.entity_type == "PHONE_NUMBER":
            overlaps = any(r.start < end and start < r.end for start, end in suppress_spans)
            digits = _NON_DIGIT_RE.sub("", text[r.start:r.end])
            if overlaps or (digits in hts_digits):
                continue
        kept.append(r)
    return kept


def _drop_zip_ssn_false_positives(results: list, text: str) -> list:
    """Remove US_SSN matches that are really US ZIP+4 postal codes. Presidio's
    very-weak SSN1 pattern (NNNNN-NNNN) is identical to a ZIP+4; a genuine SSN is
    always 3-2-4, so any US_SSN overlapping a ZIP+4 span is a false positive.
    """
    zip_spans = [(m.start(), m.end()) for m in _ZIP4_RE.finditer(text)]
    if not zip_spans:
        return results
    kept = []
    for r in results:
        if r.entity_type == "US_SSN" and any(
            r.start < end and start < r.end for start, end in zip_spans
        ):
            continue
        kept.append(r)
    return kept


@dataclass
class _PIIResult:
    detected: bool
    entities: List[str]
    risk_level: RiskLevel
    redacted_text: str
    score: float


class _PIIScanner:
    def __init__(self) -> None:
        if not _PRESIDIO_OK:
            self._ready = False
            return
        self._analyzer   = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        for r in _build_custom_recognizers():
            self._analyzer.registry.add_recognizer(r)
        self._ready = True
        logger.info("[fluiq.secure] PII scanner ready")

    def scan(self, text, ignore: set[str] | None = None) -> _PIIResult:
        """Detect PII in ``text``. ``ignore`` is a set of entity types (e.g.
        ``{"PERSON", "LOCATION"}``) that are dropped before scoring, redaction
        and reporting — the per-org Guardrail PII policy. Ignored entities are
        treated as if never found, so they neither inflate the risk score nor
        get redacted (the org has opted to keep seeing them in the clear).
        """
        if not isinstance(text, str):
            text = str(text) if text else ""
        empty = _PIIResult(False, [], RiskLevel.CLEAN, text or "", 0.0)
        if not text or not text.strip() or not self._ready:
            return empty
        try:
            entities = _SUPPORTED_ENTITIES
            if ignore:
                entities = [e for e in entities if e not in ignore]
                if not entities:
                    return empty
            results = self._analyzer.analyze(text=text, entities=entities, language="en")
            results = _drop_hts_phone_false_positives(results, text)
            results = _drop_zip_ssn_false_positives(results, text)
            if not results:
                return empty
            entity_types = list({r.entity_type for r in results})
            score = max((_ENTITY_WEIGHTS.get(e, 0.3) for e in entity_types), default=0.0)
            redacted = self._anonymizer.anonymize(text=text, analyzer_results=results).text
            return _PIIResult(True, entity_types, _risk_from_score(score), redacted, score)
        except Exception as exc:
            logger.exception("[fluiq.secure] PII scan error: %s", exc)
            return empty
