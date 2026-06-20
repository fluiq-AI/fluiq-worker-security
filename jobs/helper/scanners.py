"""Security scan orchestrator.

Runs all scanners and aggregates results into ScanResult / CheckResult.
Individual scanners live in their own modules:
  base.py       — shared types and pattern-matching helpers
  pii.py        — PII detection and redaction (Presidio)
  injection.py  — direct prompt injection patterns
  jailbreak.py  — jailbreak / role-play escape patterns
  skeleton_key.py — skeleton key attack patterns (Microsoft KB)
  secrets.py    — hardcoded credentials and high-entropy tokens
  semantic.py   — cosine-similarity attack classifier (sentence-transformers)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from jobs.helper.base import RiskLevel, _max_risk, _scan_patterns, _scan_tiered
from jobs.helper.pii import _PIIScanner
from jobs.helper.injection import INJECTION_COMPILED
from jobs.helper.jailbreak import (
    JAILBREAK_COMPILED,
    JAILBREAK_STRONG_COMPILED,
    JAILBREAK_WEAK_COMPILED,
)
from jobs.helper.skeleton_key import SKELETON_KEY_COMPILED
from jobs.helper.secrets import _SecretScanner
from jobs.helper.semantic import init_semantic, semantic_score


# ── Module-level singletons ───────────────────────────────────────────────────

_pii_scanner    = _PIIScanner()
_secret_scanner = _SecretScanner()
init_semantic()

# A retrieved document whose semantic similarity to the attack centroid meets
# this bar is treated as RAG poisoning. Lower than the prompt-side bar (0.65):
# poisoned chunks are usually longer benign text wrapping a short injection, so
# averaging dilutes the score and a slightly more sensitive threshold is needed.
_RAG_POISON_THRESHOLD = 0.55


# ── Public result types ───────────────────────────────────────────────────────

@dataclass
class ScanResult:
    # PII
    prompt_redacted:       str
    response_redacted:     str
    pii_entities_prompt:   List[str]
    pii_entities_response: List[str]
    # Attacks
    injection_detected:    bool
    injection_patterns:    List[str]
    jailbreak_detected:    bool
    jailbreak_patterns:    List[str]
    skeleton_key_detected: bool
    skeleton_key_patterns: List[str]
    # Secrets
    secrets_detected:      bool
    secret_types:          List[str]
    # Indirect injection
    indirect_injection_detected: bool
    indirect_injection_sources:  List[str]
    # RAG poisoning (semantic match on retrieved docs)
    rag_poisoning_detected: bool
    rag_poisoning_sources:  List[str]
    rag_poisoning_score:    float
    # Tool-input exfiltration (sensitive data sent out to tools)
    tool_exfiltration_detected: bool
    tool_exfiltration_types:    List[str]
    tool_exfiltration_sources:  List[str]
    # Tool-allowlist policy violation (tool called outside the org allowlist)
    tool_policy_violation_detected: bool
    tool_policy_violations:         List[str]
    # Cross-agent injection (attack content arriving from another agent's output)
    cross_agent_injection_detected: bool
    # Semantic
    semantic_attack_score: float
    # Aggregate
    security_risk_level:   str
    security_risk_score:   float
    should_block:          bool


@dataclass
class CheckResult:
    """Lightweight pre-call result — no response text needed."""
    allow:         bool
    block_reason:  Optional[str]
    risk_level:    str
    attack_types:  List[str]


# ── Entry points ──────────────────────────────────────────────────────────────

def scan(
    prompt:       str,
    response:     str,
    tool_outputs: list[str] | None = None,
    context_docs: list[str] | None = None,
    tool_inputs:  list[str] | None = None,
    tool_names:   list[str] | None = None,
    allowed_tools: list[str] | None = None,
    cross_agent_source: bool = False,
    pii_ignore:   list[str] | None = None,
) -> ScanResult:
    """Full post-call scan: PII + all attack categories + indirect injection.

    ``tool_outputs`` / ``context_docs`` are sibling tool results and retrieved
    documents from the same trace tree — scanned for indirect injection and RAG
    poisoning. ``tool_inputs`` are the arguments the agent *sent* to tools —
    scanned for sensitive-data exfiltration. ``tool_names`` are the tools the
    agent invoked — checked against ``allowed_tools``.

    ``allowed_tools`` is the org's Guardrail tool allowlist. When non-empty, any
    invoked tool outside it is a ``tool_policy_violation``. Empty/None means no
    allowlist is configured, so nothing is flagged (opt-in, like ``pii_ignore``).

    ``cross_agent_source`` is True when the prompt was produced by another agent
    (the LLM event's parent is itself an ``llm`` event) rather than the end user.
    When set, attack content in the prompt is escalated to ``cross_agent_injection``.

    ``pii_ignore`` is the org's Guardrail PII policy — entity types to suppress
    (e.g. ``["PERSON", "LOCATION"]``). Suppressed entities are dropped before
    scoring so they neither surface in the dashboard nor inflate risk.
    """
    # PII
    ignore       = set(pii_ignore or [])
    prompt_pii   = _pii_scanner.scan(prompt, ignore)
    response_pii = _pii_scanner.scan(response, ignore)

    # Attack pattern scans (prompt only — response is attacker-unknown)
    injection = _scan_patterns(prompt, INJECTION_COMPILED,  "injection")
    jailbreak = _scan_tiered(prompt, JAILBREAK_STRONG_COMPILED, JAILBREAK_WEAK_COMPILED, "jailbreak")
    skeleton  = _scan_patterns(prompt, SKELETON_KEY_COMPILED, "skeleton_key")

    # Indirect injection: scan tool outputs and retrieved docs for attack patterns
    indirect_sources: list[str] = []
    for i, content in enumerate(tool_outputs or []):
        r = _scan_patterns(content, INJECTION_COMPILED, "indirect-tool")
        if not r.detected:
            r = _scan_patterns(content, JAILBREAK_COMPILED, "indirect-tool")
        if r.detected:
            indirect_sources.append(f"tool_output[{i}]")
    for i, content in enumerate(context_docs or []):
        r = _scan_patterns(content, INJECTION_COMPILED, "indirect-doc")
        if not r.detected:
            r = _scan_patterns(content, JAILBREAK_COMPILED, "indirect-doc")
        if r.detected:
            indirect_sources.append(f"context_doc[{i}]")

    # RAG poisoning (A.2): a retrieved document whose *semantics* resemble an
    # attack — catches paraphrased / obfuscated injections the regex lists miss.
    rag_sources: list[str] = []
    rag_max = 0.0
    for i, content in enumerate(context_docs or []):
        s = semantic_score(content)
        if s > rag_max:
            rag_max = s
        if s >= _RAG_POISON_THRESHOLD:
            rag_sources.append(f"context_doc[{i}]")

    # Tool-input exfiltration (B.2): sensitive data (PII / secrets) leaving the
    # agent in the arguments it passes to a tool. Honors the org's pii_ignore.
    exfil_types: set[str] = set()
    exfil_sources: list[str] = []
    for i, content in enumerate(tool_inputs or []):
        pii_hit    = _pii_scanner.scan(content, ignore)
        secret_hit = _secret_scanner.scan(content)
        if pii_hit.entities or secret_hit.detected:
            exfil_types.update(pii_hit.entities)
            exfil_types.update(secret_hit.secret_types)
            exfil_sources.append(f"tool_input[{i}]")

    # Tool-allowlist policy (B.3): when the org configured an allowlist, any
    # invoked tool outside it is a violation. Case-insensitive match; preserves
    # the original tool name and reports each offending tool once.
    policy_violations: list[str] = []
    allow_set = {t.strip().lower() for t in (allowed_tools or []) if t and t.strip()}
    if allow_set:
        seen: set[str] = set()
        for name in (tool_names or []):
            key = name.strip().lower()
            if key and key not in allow_set and key not in seen:
                seen.add(key)
                policy_violations.append(name)

    # Secrets
    response_secrets = _secret_scanner.scan(response)
    prompt_secrets   = _secret_scanner.scan(prompt)
    all_secret_types = list(set(response_secrets.secret_types + prompt_secrets.secret_types))
    secrets_detected = response_secrets.detected or prompt_secrets.detected

    # Semantic
    sem_score = semantic_score(prompt)

    # Cross-agent injection (C.1): the inbound prompt came from another agent's
    # output (not the end user) AND carries attack content — a multi-agent trust
    # violation where one agent injects into another. Tool-sourced injection is
    # covered separately by indirect injection, so only agent-to-agent counts.
    cross_agent_injection = cross_agent_source and (
        injection.detected or jailbreak.detected or sem_score >= 0.65
    )

    # Aggregate risk
    attack_risk = _max_risk(
        injection.risk_level,
        jailbreak.risk_level,
        skeleton.risk_level,
        RiskLevel.HIGH   if indirect_sources    else RiskLevel.CLEAN,
        RiskLevel.HIGH   if rag_sources         else RiskLevel.CLEAN,
        RiskLevel.MEDIUM if exfil_sources       else RiskLevel.CLEAN,
        RiskLevel.HIGH   if policy_violations   else RiskLevel.CLEAN,
        RiskLevel.HIGH   if cross_agent_injection else RiskLevel.CLEAN,
        RiskLevel.MEDIUM if sem_score >= 0.65   else RiskLevel.CLEAN,
    )
    overall_risk = _max_risk(
        prompt_pii.risk_level,
        response_pii.risk_level,
        attack_risk,
        response_secrets.risk_level,
        prompt_secrets.risk_level,
    )

    risk_score = max(
        prompt_pii.score,
        response_pii.score,
        injection.risk_score,
        jailbreak.risk_score,
        skeleton.risk_score,
        1.0 if indirect_sources else 0.0,
        rag_max if rag_sources else 0.0,
        0.5 if exfil_sources else 0.0,
        1.0 if policy_violations else 0.0,
        1.0 if cross_agent_injection else 0.0,
        sem_score,
        1.0 if secrets_detected else 0.0,
    )

    return ScanResult(
        prompt_redacted             = prompt_pii.redacted_text,
        response_redacted           = response_pii.redacted_text,
        pii_entities_prompt         = prompt_pii.entities,
        pii_entities_response       = response_pii.entities,
        injection_detected          = injection.detected,
        injection_patterns          = injection.patterns_found,
        jailbreak_detected          = jailbreak.detected,
        jailbreak_patterns          = jailbreak.patterns_found,
        skeleton_key_detected       = skeleton.detected,
        skeleton_key_patterns       = skeleton.patterns_found,
        secrets_detected            = secrets_detected,
        secret_types                = all_secret_types,
        indirect_injection_detected = bool(indirect_sources),
        indirect_injection_sources  = indirect_sources,
        rag_poisoning_detected      = bool(rag_sources),
        rag_poisoning_sources       = rag_sources,
        rag_poisoning_score         = round(rag_max, 4),
        tool_exfiltration_detected  = bool(exfil_sources),
        tool_exfiltration_types     = sorted(exfil_types),
        tool_exfiltration_sources   = exfil_sources,
        tool_policy_violation_detected = bool(policy_violations),
        tool_policy_violations         = policy_violations,
        cross_agent_injection_detected = bool(cross_agent_injection),
        semantic_attack_score       = round(sem_score, 4),
        security_risk_level         = overall_risk.value,
        security_risk_score         = round(risk_score, 4),
        should_block                = overall_risk == RiskLevel.HIGH,
    )


def check(prompt: str) -> CheckResult:
    """Lightweight pre-call check: attack patterns only, no PII or secrets scan."""
    injection = _scan_patterns(prompt, INJECTION_COMPILED,    "injection")
    jailbreak = _scan_tiered(prompt, JAILBREAK_STRONG_COMPILED, JAILBREAK_WEAK_COMPILED, "jailbreak")
    skeleton  = _scan_patterns(prompt, SKELETON_KEY_COMPILED, "skeleton_key")
    sem_score = semantic_score(prompt)

    attack_types: list[str] = []
    if injection.detected:
        attack_types.append("prompt_injection")
    if jailbreak.detected:
        attack_types.append("jailbreak")
    if skeleton.detected:
        attack_types.append("skeleton_key")
    if sem_score >= 0.65:
        attack_types.append("semantic_attack")

    overall = _max_risk(
        injection.risk_level,
        jailbreak.risk_level,
        skeleton.risk_level,
        RiskLevel.MEDIUM if sem_score >= 0.65 else RiskLevel.CLEAN,
    )
    allow = overall != RiskLevel.HIGH

    block_reason: str | None = None
    if not allow:
        block_reason = f"Blocked by fluiq.secure: {', '.join(attack_types)}"

    return CheckResult(
        allow        = allow,
        block_reason = block_reason,
        risk_level   = overall.value,
        attack_types = attack_types,
    )
