"""Normalize OpenInference-shaped events for security scanning.

External observability traces (Arize Phoenix, Langfuse, LangSmith, Braintrust)
are normally converted to Fluiq events at the API's ``/ingest/otel`` endpoint, so
the security worker sees native fields. This is a defensive fallback: if a raw
OpenInference span (``attributes`` dict, no ``messages`` / ``response``) reaches
the worker on any path, populate the ``messages`` / ``response`` / ``tools`` /
``tool_calls`` fields the scanner reads from the OpenInference attributes.

Best-effort and non-destructive: returns the event unchanged when it isn't
OpenInference-shaped or already has the standard fields.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()
        if s and s[0] in "[{":
            try:
                return json.loads(s)
            except (json.JSONDecodeError, ValueError):
                return value
    return value


def is_openinference_event(event: Dict[str, Any]) -> bool:
    return (
        isinstance(event, dict)
        and isinstance(event.get("attributes"), dict)
        and not event.get("messages")
        and not event.get("response")
    )


def _collect_messages(attrs: Dict[str, Any], base: str) -> List[Dict[str, Any]]:
    by_index: Dict[int, Dict[str, Any]] = {}
    prefix = base + "."
    for key, val in attrs.items():
        if not key.startswith(prefix):
            continue
        idx_str, _, tail = key[len(prefix):].partition(".")
        if not idx_str.isdigit():
            continue
        entry = by_index.setdefault(int(idx_str), {})
        if tail == "message.role":
            entry["role"] = val
        elif tail == "message.content":
            entry["content"] = val
    return [by_index[i] for i in sorted(by_index)]


def normalize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Return an event with scanner fields populated from OpenInference attributes."""
    if not is_openinference_event(event):
        return event
    attrs = event["attributes"]
    out = dict(event)

    messages = _collect_messages(attrs, "llm.input_messages")
    if messages:
        out["messages"] = messages
    elif attrs.get("input.value") is not None:
        parsed = _maybe_json(attrs["input.value"])
        out["messages"] = parsed if isinstance(parsed, list) else [{"role": "user", "content": str(parsed)}]

    resp = attrs.get("output.value")
    if resp is None:
        out_msgs = _collect_messages(attrs, "llm.output_messages")
        resp = out_msgs[-1].get("content") if out_msgs else None
    if resp is not None:
        out["response"] = resp if isinstance(resp, str) else json.dumps(resp, default=str)

    tools = attrs.get("llm.tools")
    if isinstance(tools, list) and tools:
        out["tools"] = [_maybe_json(t) for t in tools]

    # Tool spans expose the invoked tool — surface it for tool-policy checks.
    tool_name = attrs.get("tool.name") or attrs.get("tool_call.function.name")
    if tool_name and not out.get("tool_calls"):
        raw_args = attrs.get("tool.parameters") or attrs.get("tool_call.function.arguments") or attrs.get("input.value")
        args = _maybe_json(raw_args)
        out["tool_calls"] = [{
            "id": str(event.get("trace_id") or ""), "type": "function",
            "function": {"name": tool_name,
                         "arguments": args if isinstance(args, str) else json.dumps(args or {}, default=str)},
        }]

    return out
