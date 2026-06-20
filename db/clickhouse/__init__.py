import asyncio
import json
import logging
from typing import Any, Optional
import config
import clickhouse_connect
from clickhouse_connect.driver.asyncclient import AsyncClient


logger = logging.getLogger(__name__)


def _stringify_output(out: Any) -> str:
    """Flatten a tool/function output (str | list | dict) to scannable text."""
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    try:
        return json.dumps(out, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(out)


class ClickHouseSecurityClient:
    """Async ClickHouse client for storing security scan results and reading
    per-session risk history (crescendo detection)."""

    def __init__(
        self,
        host: str = config.CLICKHOUSE_HOST,
        port: int = config.CLICKHOUSE_PORT,
        username: str = config.CLICKHOUSE_USER,
        password: str = config.CLICKHOUSE_PASSWORD,
        database: str = config.CLICKHOUSE_DATABASE,
        default_table: str = config.CLICKHOUSE_SECURITY_TABLE,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.default_table = default_table
        self._client: Optional[AsyncClient] = None

    async def start(self) -> None:
        if self._client is not None:
            return
        self._client = await clickhouse_connect.get_async_client(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database,
        )
        logger.info(
            "[CLICKHOUSE] Security client started: %s:%s/%s",
            self.host, self.port, self.database,
        )

    async def stop(self) -> None:
        if self._client is None:
            return
        await self._client.close()
        self._client = None
        logger.info("[CLICKHOUSE] Security client stopped")

    async def get_session_risk_scores(
        self,
        organization_id: Any,
        root_trace_id: Any,
        exclude_trace_id: Any,
        table: Optional[str] = None,
    ) -> list[float]:
        """Return ordered security_risk_score values for the session, excluding the current trace."""
        if self._client is None:
            await self.start()
        target = table or self.default_table
        result = await self._client.query(
            f"SELECT security_risk_score FROM {target} "
            f"WHERE organization_id = {{org_id:UUID}} "
            f"AND root_trace_id = {{root_id:UUID}} "
            f"AND trace_id != {{exc_id:UUID}} "
            f"ORDER BY ingested_at ASC LIMIT 19",
            parameters={
                "org_id": str(organization_id),
                "root_id": str(root_trace_id),
                "exc_id": str(exclude_trace_id),
            },
        )
        return [float(row[0]) for row in result.result_rows]

    async def get_indirect_sources(
        self,
        organization_id: Any,
        root_trace_id: Any,
        exclude_trace_id: Any,
        table: Optional[str] = None,
        limit: int = 50,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Pull sibling tool outputs, retrieved documents, tool inputs, and tool
        names from the same trace tree so the scanner can check them for indirect
        prompt injection / RAG poisoning (outputs + docs), sensitive-data
        exfiltration (inputs), and tool-allowlist violations (names). Returns
        ``(tool_outputs, context_docs, tool_inputs, tool_names)`` as plain-text
        lists.

        Read-only and best-effort: sibling events are ingested independently, so
        a just-arrived event may not be present yet, and any failure fails open
        (empty lists) rather than blocking the scan.
        """
        if self._client is None:
            await self.start()
        target = table or config.CLICKHOUSE_TRACE_TABLE
        try:
            result = await self._client.query(
                f"SELECT JSONExtractString(toString(event), 'type') AS etype, "
                f"       toString(event) AS ev "
                f"FROM {target} "
                f"WHERE organization_id = {{org_id:UUID}} "
                f"AND root_trace_id = {{root_id:UUID}} "
                f"AND trace_id != {{exc_id:UUID}} "
                f"AND JSONExtractString(toString(event), 'type') IN ('tool','function','vectorstore') "
                f"LIMIT {int(limit)}",
                parameters={
                    "org_id": str(organization_id),
                    "root_id": str(root_trace_id),
                    "exc_id": str(exclude_trace_id),
                },
            )
        except Exception as exc:
            logger.warning("[CLICKHOUSE] get_indirect_sources failed: %s", exc)
            return [], [], [], []

        tool_outputs: list[str] = []
        context_docs: list[str] = []
        tool_inputs:  list[str] = []
        tool_names:   list[str] = []
        for etype, ev in result.result_rows:
            try:
                event = json.loads(ev)
            except (ValueError, TypeError):
                continue
            if etype in ("tool", "function"):
                text = _stringify_output(event.get("output"))
                if text.strip():
                    tool_outputs.append(text)
                # Tool arguments the agent sent out — scanned for exfiltration.
                arg = _stringify_output(event.get("input") or event.get("arguments"))
                if arg.strip():
                    tool_inputs.append(arg)
                # Tool identity — checked against the org's allowlist policy.
                name = event.get("name") or event.get("function")
                if isinstance(name, str) and name.strip():
                    tool_names.append(name.strip())
            elif etype == "vectorstore":
                items = (((event.get("result") or {}).get("matches") or {}).get("items")) or []
                for m in items:
                    if isinstance(m, dict) and isinstance(m.get("text"), str) and m["text"].strip():
                        context_docs.append(m["text"])
        return tool_outputs, context_docs, tool_inputs, tool_names

    async def get_parent_event_kind(
        self,
        organization_id: Any,
        root_trace_id: Any,
        parent_id: Any,
        table: Optional[str] = None,
    ) -> str:
        """Return the ``type`` of the parent event (``llm``/``tool``/...), or ""
        when there is no parent, it isn't found yet, or the lookup fails.

        Used for cross-agent injection detection: when an LLM event's parent is
        itself an ``llm`` event, the inbound prompt was produced by another agent
        rather than the end user, so it must be treated as untrusted.
        """
        if not parent_id:
            return ""
        if self._client is None:
            await self.start()
        target = table or config.CLICKHOUSE_TRACE_TABLE
        try:
            result = await self._client.query(
                f"SELECT JSONExtractString(toString(event), 'type') AS etype "
                f"FROM {target} "
                f"WHERE organization_id = {{org_id:UUID}} "
                f"AND root_trace_id = {{root_id:UUID}} "
                f"AND trace_id = {{pid:UUID}} "
                f"LIMIT 1",
                parameters={
                    "org_id": str(organization_id),
                    "root_id": str(root_trace_id),
                    "pid": str(parent_id),
                },
            )
        except Exception as exc:
            logger.warning("[CLICKHOUSE] get_parent_event_kind failed: %s", exc)
            return ""
        rows = result.result_rows
        return str(rows[0][0]) if rows and rows[0] and rows[0][0] else ""

    async def get_agent_chain_scores(
        self,
        organization_id: Any,
        root_trace_id: Any,
        parent_id: Any,
        sec_table: Optional[str] = None,
        trace_table: Optional[str] = None,
        max_depth: int = 12,
    ) -> list[float]:
        """Return the prior security risk scores of the current event's *agent
        ancestry*, ordered oldest-ancestor → immediate-parent.

        Walks the parent chain (each LLM event's ``parent_id`` lives in the event
        JSON) up from ``parent_id`` to the root, collecting the security score of
        every ancestor that was itself scanned. Feeds C.2 trust-boundary
        escalation: the crescendo slope keyed on the agent DAG rather than on
        flat session time. Best-effort — any failure returns ``[]``.
        """
        if not parent_id:
            return []
        if self._client is None:
            await self.start()
        sec    = sec_table or self.default_table
        traces = trace_table or config.CLICKHOUSE_TRACE_TABLE
        try:
            edge_rows, score_rows = await asyncio.gather(
                self._client.query(
                    f"SELECT trace_id, "
                    f"       JSONExtractString(toString(event), 'parent_id') AS pid "
                    f"FROM {traces} "
                    f"WHERE organization_id = {{org_id:UUID}} "
                    f"AND root_trace_id = {{root_id:UUID}} "
                    f"AND JSONExtractString(toString(event), 'type') = 'llm'",
                    parameters={"org_id": str(organization_id), "root_id": str(root_trace_id)},
                ),
                self._client.query(
                    f"SELECT trace_id, security_risk_score FROM {sec} "
                    f"WHERE organization_id = {{org_id:UUID}} "
                    f"AND root_trace_id = {{root_id:UUID}}",
                    parameters={"org_id": str(organization_id), "root_id": str(root_trace_id)},
                ),
            )
        except Exception as exc:
            logger.warning("[CLICKHOUSE] get_agent_chain_scores failed: %s", exc)
            return []

        parent_of: dict[str, str] = {
            str(tid): str(pid) for tid, pid in edge_rows.result_rows if tid
        }
        score_of: dict[str, float] = {
            str(tid): float(score) for tid, score in score_rows.result_rows if tid
        }

        chain: list[float] = []
        node = str(parent_id)
        seen: set[str] = set()
        depth = 0
        while node and node not in seen and depth < max_depth:
            seen.add(node)
            if node in score_of:
                chain.append(score_of[node])
            node = parent_of.get(node, "")
            depth += 1
        chain.reverse()  # oldest ancestor first → immediate parent last
        return chain

    async def insert_security_scan(
        self,
        record: dict[str, Any],
        table: Optional[str] = None,
    ) -> None:
        if self._client is None:
            await self.start()
        target = table or self.default_table
        await self._client.insert(
            target,
            [[
                record.get("organization_id"),
                record.get("api_key_prefix") or "",
                record.get("trace_id"),
                record.get("root_trace_id") or record.get("trace_id"),
                record.get("mode") or "warn",
                record.get("prompt_redacted") or "",
                record.get("response_redacted") or "",
                record.get("pii_entities_prompt") or [],
                record.get("pii_entities_response") or [],
                int(bool(record.get("injection_detected"))),
                record.get("injection_patterns") or [],
                int(bool(record.get("jailbreak_detected"))),
                record.get("jailbreak_patterns") or [],
                int(bool(record.get("skeleton_key_detected"))),
                record.get("skeleton_key_patterns") or [],
                int(bool(record.get("secrets_detected"))),
                record.get("secret_types") or [],
                int(bool(record.get("indirect_injection_detected"))),
                record.get("indirect_injection_sources") or [],
                int(bool(record.get("rag_poisoning_detected"))),
                record.get("rag_poisoning_sources") or [],
                float(record.get("rag_poisoning_score") or 0.0),
                int(bool(record.get("tool_exfiltration_detected"))),
                record.get("tool_exfiltration_types") or [],
                record.get("tool_exfiltration_sources") or [],
                int(bool(record.get("tool_policy_violation_detected"))),
                record.get("tool_policy_violations") or [],
                int(bool(record.get("cross_agent_injection_detected"))),
                float(record.get("semantic_attack_score") or 0.0),
                record.get("security_risk_level") or "clean",
                float(record.get("security_risk_score") or 0.0),
                int(bool(record.get("should_block"))),
                float(record.get("scan_latency") or 0.0),
                record.get("extra") or {},
            ]],
            column_names=[
                "organization_id", "api_key_prefix", "trace_id", "root_trace_id",
                "mode", "prompt_redacted", "response_redacted",
                "pii_entities_prompt", "pii_entities_response",
                "injection_detected", "injection_patterns",
                "jailbreak_detected", "jailbreak_patterns",
                "skeleton_key_detected", "skeleton_key_patterns",
                "secrets_detected", "secret_types",
                "indirect_injection_detected", "indirect_injection_sources",
                "rag_poisoning_detected", "rag_poisoning_sources", "rag_poisoning_score",
                "tool_exfiltration_detected", "tool_exfiltration_types", "tool_exfiltration_sources",
                "tool_policy_violation_detected", "tool_policy_violations",
                "cross_agent_injection_detected",
                "semantic_attack_score", "security_risk_level",
                "security_risk_score", "should_block", "scan_latency", "extra",
            ],
        )


clickhouse_security_client = ClickHouseSecurityClient()

__all__ = [
    "ClickHouseSecurityClient",
    "clickhouse_security_client",
]
