import asyncio
import logging
import time
from typing import Any, Dict

import config
from jobs.helper.scanners import scan as run_scan, check as run_check
from db.clickhouse import clickhouse_security_client
from db.kafka import kafka_producer

logger = logging.getLogger(__name__)

_RISK_SCORE_MAP = {"clean": 0.0, "low": 0.25, "medium": 0.5, "high": 1.0}

# Roles that are developer-authored rather than user-controlled. Their content
# is excluded from security scanning to avoid false positives on legitimate
# instructions (mirrors the SDK pre-call gate's _extract_prompt). "developer"
# is OpenAI's renamed system role.
_NON_SCANNED_ROLES = ("system", "developer")


def _extract_scan_prompt(event: Dict[str, Any]) -> str:
    """Build the prompt text to scan from a trace event.

    Only user-controlled content is scanned. System / developer messages are
    excluded. For Anthropic the system prompt arrives as a separate ``system``
    field that is never read here, so it is likewise never scanned.
    """
    messages = event.get("messages") or event.get("contents") or event.get("input") or []
    if isinstance(messages, list):
        return "\n".join(
            str(m.get("content") or "")
            for m in messages
            if isinstance(m, dict)
            and m.get("content")
            and m.get("role") not in _NON_SCANNED_ROLES
        )
    if isinstance(messages, str):
        return messages
    return ""


def _crescendo_slope(scores: list[float]) -> float:
    """Linear regression slope over a sequence of per-turn risk scores."""
    n = len(scores)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(scores) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, scores))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den else 0.0


async def auto_security_scan(message: Dict[str, Any]) -> None:
    """Run the full post-call security scan on an SDK trace.

    Triggered by /ingest when it sees ``_security_config`` on a completed
    trace. Results are stored in the evaluations table (evaluator=fluiq.security)
    and published via SSE so the frontend can merge them without a page refresh.
    """
    event           = message.get("event") or {}
    organization_id = message.get("organization_id")
    api_key_prefix  = message.get("api_key_prefix")
    trace_id        = message.get("trace_id") or event.get("trace_id")
    root_trace_id   = event.get("root_trace_id") or trace_id

    # Extract prompt from messages — system / developer messages excluded.
    prompt = _extract_scan_prompt(event)

    _response_raw = event.get("response") or event.get("output") or ""
    if isinstance(_response_raw, str):
        response = _response_raw
    elif isinstance(_response_raw, list):
        response = "\n".join(str(item) for item in _response_raw if item)
    elif isinstance(_response_raw, dict):
        response = _response_raw.get("content") or str(_response_raw)
    else:
        response = str(_response_raw) if _response_raw else ""

    if not prompt and not response:
        logger.info(
            "[EVALUATOR] Skipping security scan: trace_id=%s no prompt or response",
            trace_id,
        )
        return

    # When the response was already scanned synchronously by the response gate,
    # skip the response text to avoid double-billing PII/secrets detection.
    response_gated = message.get("response_gated", False)
    scan_response  = "" if response_gated else response

    started = time.time()
    # Run scan and fetch session history concurrently
    effective_root = root_trace_id or trace_id
    result, past_scores = await asyncio.gather(
        asyncio.to_thread(run_scan, prompt=prompt, response=scan_response),
        clickhouse_security_client.get_session_risk_scores(
            organization_id, effective_root, trace_id
        ),
        return_exceptions=True,
    )
    if isinstance(result, Exception):
        logger.exception("[EVALUATOR] Security scan failed trace_id=%s", trace_id)
        return
    if isinstance(past_scores, Exception):
        logger.warning("[EVALUATOR] Could not fetch session history trace_id=%s: %s", trace_id, past_scores)
        past_scores = []

    scan_latency = time.time() - started

    # Crescendo: append current score to session history and compute slope
    all_scores = past_scores + [result.security_risk_score]
    slope = _crescendo_slope(all_scores)
    crescendo_detected = slope >= 0.3 and len(all_scores) >= 3 and result.security_risk_score >= 0.25
    crescendo_score = round(slope, 4)

    await clickhouse_security_client.insert_security_scan({
        "organization_id":             organization_id,
        "api_key_prefix":              api_key_prefix,
        "trace_id":                    trace_id,
        "root_trace_id":               effective_root,
        "mode":                        message.get("security_config", {}).get("mode", "warn"),
        "prompt_redacted":             result.prompt_redacted,
        "response_redacted":           result.response_redacted,
        "pii_entities_prompt":         result.pii_entities_prompt,
        "pii_entities_response":       result.pii_entities_response,
        "injection_detected":          result.injection_detected,
        "injection_patterns":          result.injection_patterns,
        "jailbreak_detected":          result.jailbreak_detected,
        "jailbreak_patterns":          result.jailbreak_patterns,
        "skeleton_key_detected":       result.skeleton_key_detected,
        "skeleton_key_patterns":       result.skeleton_key_patterns,
        "secrets_detected":            result.secrets_detected,
        "secret_types":                result.secret_types,
        "indirect_injection_detected": result.indirect_injection_detected,
        "indirect_injection_sources":  result.indirect_injection_sources,
        "semantic_attack_score":       result.semantic_attack_score,
        "security_risk_level":         result.security_risk_level,
        "security_risk_score":         result.security_risk_score,
        "should_block":                result.should_block,
        "scan_latency":                scan_latency,
        "extra": {
            "crescendo_detected": crescendo_detected,
            "crescendo_score":    crescendo_score,
            "session_turns":      len(all_scores),
        },
    })

    details = {
        "prompt_redacted":             result.prompt_redacted,
        "response_redacted":           result.response_redacted,
        "pii_entities_prompt":         result.pii_entities_prompt,
        "pii_entities_response":       result.pii_entities_response,
        "injection_detected":          result.injection_detected,
        "injection_patterns":          result.injection_patterns,
        "jailbreak_detected":          result.jailbreak_detected,
        "jailbreak_patterns":          result.jailbreak_patterns,
        "skeleton_key_detected":       result.skeleton_key_detected,
        "skeleton_key_patterns":       result.skeleton_key_patterns,
        "secrets_detected":            result.secrets_detected,
        "secret_types":                result.secret_types,
        "indirect_injection_detected": result.indirect_injection_detected,
        "indirect_injection_sources":  result.indirect_injection_sources,
        "semantic_attack_score":       result.semantic_attack_score,
        "security_risk_level":         result.security_risk_level,
        "security_risk_score":         result.security_risk_score,
        "should_block":                result.should_block,
        "crescendo_detected":          crescendo_detected,
        "crescendo_score":             crescendo_score,
        "session_turns":               len(all_scores),
        "risk_trajectory":             [round(s, 4) for s in all_scores],
        "scan_latency":                scan_latency,
    }

    try:
        await kafka_producer.publish(
            {
                "kind":            "enriched",
                "enrichment":      "security",
                "organization_id": str(organization_id) if organization_id is not None else None,
                "api_key_prefix":  api_key_prefix,
                "trace_id":        trace_id,
                "root_trace_id":   root_trace_id or trace_id,
                "security":        details,
            },
            key=str(organization_id) if organization_id is not None else None,
        )
    except Exception:
        logger.exception(
            "[EVALUATOR] Failed to publish trace.enriched (security) trace_id=%s", trace_id,
        )

    logger.info(
        "[EVALUATOR] security_scan trace_id=%s risk=%s score=%.3f crescendo=%s slope=%.3f turns=%d",
        trace_id, result.security_risk_level, result.security_risk_score,
        crescendo_detected, crescendo_score, len(all_scores),
    )


async def sync_response_gate_check(message: Dict[str, Any]) -> None:
    """Synchronous response gate check via Kafka request-reply.

    Called from /ingest when the org has scan_responses=True. Only scans
    the response text for PII and secrets — attack patterns are prompt-side
    and are already caught by the pre-call check.
    """
    correlation_id = message.get("correlation_id")
    response       = message.get("response", "")

    if not response or not response.strip():
        result_dict: dict = {"response_blocked": False, "risk_level": "clean", "attack_types": [], "block_reason": None}
    else:
        try:
            scan_result = await asyncio.to_thread(run_scan, prompt="", response=response)

            attack_types: list[str] = []
            if scan_result.pii_entities_response:
                attack_types.append("pii_in_response")
            if scan_result.secrets_detected:
                attack_types.append("secrets_in_response")

            should_block = bool(attack_types) and scan_result.security_risk_score >= 0.5
            result_dict = {
                "response_blocked": should_block,
                "risk_level":       scan_result.security_risk_level,
                "attack_types":     attack_types,
                "block_reason": (
                    f"Response blocked by fluiq.secure(): {', '.join(attack_types)}"
                    if should_block else None
                ),
            }
            logger.info(
                "[EVALUATOR] response_gate_check correlation_id=%s blocked=%s risk=%s",
                correlation_id, should_block, scan_result.security_risk_level,
            )
        except Exception:
            logger.exception("[EVALUATOR] response_gate_check failed, failing open")
            result_dict = {"response_blocked": False, "risk_level": "clean", "attack_types": [], "block_reason": None}

    if not correlation_id:
        return

    try:
        await kafka_producer.publish(
            {"correlation_id": correlation_id, "result": result_dict},
            topic=config.KAFKA_SECURITY_REPLY_TOPIC,
        )
    except Exception:
        logger.exception(
            "[EVALUATOR] Failed to publish response gate reply correlation_id=%s", correlation_id,
        )


async def sync_security_check(message: Dict[str, Any]) -> None:
    """Synchronous pre-call check via Kafka request-reply.

    Runs a full scan on the prompt (no response text) and publishes the result
    to KAFKA_SECURITY_REPLY_TOPIC keyed by correlation_id so the waiting API
    request can return it directly.

    The API forwards a ``policy`` dict with:
      - block_threshold:  'medium' | 'high'  (default 'high')
      - block_categories: list[str]           (empty = all categories block)
    """
    correlation_id   = message.get("correlation_id")
    prompt           = message.get("prompt", "")
    policy           = message.get("policy") or {}
    block_threshold  = policy.get("block_threshold", "high")
    block_categories = set(policy.get("block_categories") or [])

    if not prompt or not prompt.strip():
        result_dict = {"allow": True, "block_reason": None, "risk_level": "clean", "attack_types": []}
    else:
        try:
            scan_result = await asyncio.to_thread(run_scan, prompt=prompt, response="")

            attack_types: list[str] = []
            if scan_result.injection_detected:
                attack_types.append("prompt_injection")
            if scan_result.jailbreak_detected:
                attack_types.append("jailbreak")
            if scan_result.skeleton_key_detected:
                attack_types.append("skeleton_key")
            if scan_result.semantic_attack_score >= 0.65:
                attack_types.append("semantic_attack")
            if scan_result.pii_entities_prompt:
                attack_types.append("pii_detected")
            if scan_result.secrets_detected:
                attack_types.append("secrets_detected")

            # Apply block_categories filter: if configured, only listed
            # categories contribute to a block decision
            if block_categories:
                blocking_types = [t for t in attack_types if t in block_categories]
            else:
                blocking_types = attack_types

            risk = scan_result.security_risk_level
            should_block = bool(blocking_types) and (
                risk == "high"
                or (risk == "medium" and block_threshold == "medium")
            )

            allow        = not should_block
            block_reason = (
                f"Blocked by fluiq.secure: {', '.join(blocking_types)}"
                if not allow else None
            )
            result_dict = {
                "allow":        allow,
                "block_reason": block_reason,
                "risk_level":   risk,
                "attack_types": attack_types,
            }
            logger.info(
                "[EVALUATOR] sync_security_check correlation_id=%s allow=%s risk=%s",
                correlation_id, allow, risk,
            )
        except Exception:
            logger.exception("[EVALUATOR] sync_security_check failed, failing open")
            result_dict = {"allow": True, "block_reason": None, "risk_level": "clean", "attack_types": []}

    if not correlation_id:
        return

    try:
        await kafka_producer.publish(
            {"correlation_id": correlation_id, "result": result_dict},
            topic=config.KAFKA_SECURITY_REPLY_TOPIC,
        )
    except Exception:
        logger.exception(
            "[EVALUATOR] Failed to publish security check reply correlation_id=%s", correlation_id,
        )
