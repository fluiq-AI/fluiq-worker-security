"""Third detection signal: a fine-tuned prompt-injection classifier.

The pattern layer and the semantic layer both work by comparing a prompt to
things we wrote down. Patterns compare to phrases, semantics compares to seed
sentences. Neither generalises past what someone thought to enumerate, and
benchmarking put a number on that ceiling: combining them tops out near 46%
recall on public prompt-injection data and 42% on jailbreak.

A model trained on injection data is a genuinely independent signal. Measured on
the same corpora, adding it to the existing stack is worth roughly:

    prompt injection   23% -> 46% recall   at +1.1pp false alarms
    jailbreak          42% -> 75% recall   at +0.7pp false alarms

Notably it is much stronger on jailbreak than injection, which is why it does
not remove the case for training our own injection model later.

Costs, because they are not free: a DeBERTa forward pass is far heavier than the
embedding lookups the semantic layer does. Lazily loaded, fails open, and both
the model and the threshold are env-overridable so it can be swapped for the
`small` variant or turned off entirely without a deploy.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(1)
    _TF_OK = True
except ImportError:
    _TF_OK = False


# `-small-` is the cheaper variant if memory or latency becomes the binding
# constraint. Set FLUIQ_CLASSIFIER_ENABLED=0 to disable without a deploy.
MODEL = os.getenv("FLUIQ_CLASSIFIER_MODEL", "protectai/deberta-v3-base-prompt-injection-v2")
ENABLED = os.getenv("FLUIQ_CLASSIFIER_ENABLED", "1") != "0"

THRESHOLD = float(os.getenv("FLUIQ_CLASSIFIER_THRESHOLD", "0.90"))

# advisory | block
#
# Defaults to advisory, and that default is the whole story of this module.
#
# On the public benchmark corpora the classifier looks free: it lifts jailbreak
# recall 55% -> 80% and injection 35% -> 53% with no measurable change in false
# alarms. On the false-positive regression cases, which look far more like real
# support traffic, it flags 6 of 16 benign prompts at p > 0.99:
#
#     0.9975  "You are no longer subscribed."
#     0.9998  "Ignore the formatting of the attached file and just read the text."
#     1.0000  "Please disregard my previous message, I sent it by mistake."
#
# It keys on the imperative verb, not on what the verb targets, and it is as
# confident about those as about a real attack. No threshold separates them, and
# requiring pattern corroboration buys exactly zero extra recall because the
# pattern layer had already fired on anything it would corroborate.
#
# So in advisory mode the score is computed and recorded but never blocks. That
# gives visibility and, more usefully, labelled disagreements between this model
# and the shipping gate, which is the training set needed to fine-tune a model
# that does not have this failure mode. Flip to `block` only with a deliberate
# decision about the false-positive budget.
MODE = os.getenv("FLUIQ_CLASSIFIER_MODE", "advisory")
BLOCKS = MODE == "block"

# DeBERTa caps at 512 tokens. Long prompts are scored in overlapping windows and
# the strongest window wins: an injection buried at the end of a long benign
# wall of text is exactly the case that matters, and truncating would drop it.
_WINDOW_CHARS = 1200
_STRIDE_CHARS = 900
_MAX_WINDOWS = 8

_tok: Any = None
_model: Any = None
_failed = False


def _ensure() -> bool:
    global _tok, _model, _failed
    if _model is not None:
        return True
    if _failed or not _TF_OK or not ENABLED:
        return False
    try:
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        _model.eval()
        logger.info("[fluiq.secure] injection classifier ready (%s)", MODEL)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[fluiq.secure] injection classifier unavailable: %s", exc)
        _tok = _model = None
        _failed = True
        return False


def _windows(text: str) -> list[str]:
    if len(text) <= _WINDOW_CHARS:
        return [text]
    out = []
    for start in range(0, len(text), _STRIDE_CHARS):
        out.append(text[start:start + _WINDOW_CHARS])
        if len(out) >= _MAX_WINDOWS:
            break
    return out


def injection_probability(text: str) -> float:
    """P(injection) in 0..1. Returns 0.0 when the model is unavailable."""
    if not text or not text.strip() or not _ensure():
        return 0.0
    try:
        best = 0.0
        for chunk in _windows(text):
            enc = _tok(chunk, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = _model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            # Label 1 is INJECTION in this model card; fall back to the max of
            # the non-zero labels if a future revision reorders them.
            p = float(probs[1]) if probs.shape[-1] > 1 else float(probs[0])
            best = max(best, p)
        return best
    except Exception:
        # Fail open, like every other layer in the gate.
        return 0.0


def classifier_verdict(text: str) -> tuple[bool, float]:
    """(would_block, probability).

    `would_block` respects MODE, so an advisory deployment always gets False
    here while still receiving the score. Callers that want the raw signal for
    logging should read the probability rather than the boolean.
    """
    p = injection_probability(text)
    return (BLOCKS and p >= THRESHOLD), p
