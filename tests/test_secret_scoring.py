"""Secret scoring: named secret vs high-entropy-only.

A NAMED secret pattern (openai_key, aws_access_key, …) is HIGH and scores 1.0.
A high-entropy-only hit (e.g. a benign base64 blob) is a soft MEDIUM signal that
must NOT score 1.0 — otherwise the run-rollup badge (levelFromScore ≥ 0.9 → HIGH)
disagrees with the actual MEDIUM risk_level, and the response gate would block a
benign long token in the output.

Loads the full scanner (presidio / sentence-transformers), so it is slow.

Run:
    ../.workers-venv/Scripts/python.exe -m pytest tests/test_secret_scoring.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.helper.scanners import scan

# A random-looking base64 blob (high Shannon entropy) that is NOT a named secret.
_BENIGN_BLOB = "data payload: TW9zdCBvZiB0aGlzIGlzIGp1c3QgYmFzZTY0IG5vaXNlIHp4Y3Zibm1hc2RmZ2hqa2w"
_OPENAI_KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


def test_high_entropy_only_is_medium_not_high():
    r = scan(prompt=_BENIGN_BLOB, response="")
    assert r.secrets_detected is True          # high-entropy still surfaced
    assert r.secret_types == []                # but no named pattern
    assert r.security_risk_level == "medium", r.security_risk_level
    assert r.security_risk_score <= 0.5, r.security_risk_score
    assert r.should_block is False


def test_named_secret_is_high_and_blocks():
    r = scan(prompt=f"my key is {_OPENAI_KEY}", response="")
    assert r.secrets_detected is True
    assert "openai_key" in r.secret_types, r.secret_types
    assert r.security_risk_level == "high"
    assert r.security_risk_score >= 0.9
    assert r.should_block is True


if __name__ == "__main__":
    test_high_entropy_only_is_medium_not_high()
    print("ok  test_high_entropy_only_is_medium_not_high")
    test_named_secret_is_high_and_blocks()
    print("ok  test_named_secret_is_high_and_blocks")
    print("\n2 passed")
