# fluiq-worker-security

Scans agent traces for prompt injection, jailbreaks, PII leakage, secrets and
indirect attacks that arrive through tool output or retrieved documents.

> **Status: archived.** Part of [Fluiq](https://github.com/fluiq-AI), which ran
> from 10 April to September 2026 and never found customers. The hosted service
> is shut down. MIT, unmaintained, fork freely.

| | |
|---|---|
| **Consumes** | `KAFKA_SECURITY_TOPIC` as group `KAFKA_SECURITY_GROUP_ID` |
| **Replies on** | `KAFKA_SECURITY_REPLY_TOPIC` — correlated, for the blocking gate |
| **Writes** | ClickHouse `security_scans` |

## Two entry points

[`jobs/helper/scanners.py`](jobs/helper/scanners.py) exposes two functions with
deliberately different costs.

**`check(prompt)`** — pre-call. Attack patterns only: injection, jailbreak,
skeleton key, plus a semantic verdict. No PII or secret scanning. This runs
while a user is waiting, so it is regex-tiered and fast.

**`scan(prompt, response, ...)`** — post-call, the full pass. PII, secrets, every
attack category, and the indirect paths that only exist once you have the whole
trace tree:

- `tool_outputs` and `context_docs` — sibling tool results and retrieved
  documents, scanned for **indirect injection and RAG poisoning**
- `tool_inputs` — the arguments the agent *sent* to a tool, scanned for
  **sensitive-data exfiltration**
- `tool_names` against `allowed_tools` — **tool allow-listing**
- `cross_agent_source` — content that arrived from another agent, which is a
  **trust-boundary crossing** and is treated as untrusted input

Detectors live in [`jobs/helper/`](jobs/helper/): `injection`, `jailbreak`,
`skeleton_key`, `pii`, `secrets`, `semantic` / `semantic_v2`, `classifier`,
`image_scan`, and `openinference` for traces imported from other vendors.

## Strong and weak patterns

Every pattern-based detector is tiered. A STRONG pattern is one that is almost
never benign; a WEAK pattern only fires in combination or raises a lower-confidence
finding.

This exists because the untiered version had an ugly false-positive class.
Substring matching meant "guidance" contained "dan" and tripped the jailbreak
detector. Ordinary phrases — "act as", "you are now", "from now on" — flagged
benign prompts, and `{{` in a templating example read as injection. The fix was
word boundaries, case-sensitive acronyms, tiering, and demoting bare "you are"
openers to WEAK.

If you take one thing from this repo, take the false-positive work rather than
the detectors. A guardrail that flags a third of normal traffic gets turned off
in week two, and then it protects nothing. The
[benchmark](https://github.com/SaurabhKumbhar24/guardrail-bench) makes the same
point with numbers: one commercial product catches 97.3% of jailbreaks while
flagging 83.3% of benign text as an attack.

## Configuration

Read in [`config.py`](config.py):

```
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_SECURITY_PROTOCOL=PLAINTEXT
KAFKA_SECURITY_TOPIC=fluiq.security
KAFKA_SECURITY_GROUP_ID=fluiq-security
KAFKA_SECURITY_REPLY_TOPIC=fluiq.security.reply
KAFKA_TRACE_PERSISTED_TOPIC=fluiq.traces.persisted
KAFKA_MAX_FETCH_BYTES=10485760

CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
CLICKHOUSE_DATABASE=fluiq
CLICKHOUSE_SECURITY_TABLE=fluiq.security_scans
CLICKHOUSE_TRACE_TABLE=fluiq.traces

ANTHROPIC_API_KEY=sk-ant-...     # semantic classifier only
```

## Running

Bring up the data stores from
[fluiq-api](https://github.com/fluiq-AI/fluiq-api)'s compose file, then:

```bash
docker build -t fluiq-security .
docker run --network fluiq-ai --env-file .env.development fluiq-security
```

Or directly:

```bash
pip install -r requirements.txt
python app.py
```

## Tests

```bash
python -m pytest tests/
```

The suite includes the false-positive corpus. If you change a pattern, that is
the thing that will tell you what you broke.

## Notable decisions

**The gate is opt-in and fails open.** `fluiq.secure()` is a no-op unless a
caller asks for it, and if this worker is down, slow, or errors, the request
proceeds unblocked. A security tool positioned inline on someone's production
traffic is a liability the moment it becomes a single point of failure. That is a
deliberate posture, not an oversight — if you fork this for an environment that
needs fail-closed, that is a real change to make, and you should make it
knowingly.

**Synchronous requests reply on a correlated topic.** The blocking pre-call check
has a live requester waiting on `KAFKA_SECURITY_REPLY_TOPIC` with a timeout
(`KAFKA_SECURITY_CHECK_TIMEOUT`, default 8s in the API). Async scans have no
requester and simply write to ClickHouse. Both arrive on the same consumer, which
is why `dispatch` branches on whether a reply is expected.

**SSRF and allow-list hardening.** Fetching remote content for scanning is an
obvious SSRF vector; the audit that closed it is in the repo's history.

## Licence

MIT. See [LICENSE](LICENSE).
