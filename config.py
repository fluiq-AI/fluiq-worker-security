import os
import ssl
from dotenv import load_dotenv

load_dotenv()

# ── Kafka ──────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS     = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
# PLAINTEXT (local docker) | SASL_SSL (AWS MSK SASL/SCRAM)
KAFKA_SECURITY_PROTOCOL     = os.getenv("KAFKA_SECURITY_PROTOCOL")
KAFKA_SASL_MECHANISM        = os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512")
KAFKA_SASL_USERNAME         = os.getenv("KAFKA_SASL_USERNAME")
KAFKA_SASL_PASSWORD         = os.getenv("KAFKA_SASL_PASSWORD")

# Security jobs carry the full trace event (can be multi-MB); raise the consumer
# fetch ceiling above the ~1MB default to match the API/broker sizing.
KAFKA_MAX_FETCH_BYTES = int(os.getenv("KAFKA_MAX_FETCH_BYTES", str(10 * 1024 * 1024)))


def kafka_auth_kwargs() -> dict:
    """aiokafka security kwargs derived from env, shared by consumer + producer.

    PLAINTEXT (default, local docker-compose) → no auth.
    SASL_SSL → SCRAM-SHA-512 username/password over TLS (AWS MSK). MSK's broker
    certs chain to Amazon Trust Services, which is in the default CA bundle, so
    no CA file is needed.
    """
    protocol = (KAFKA_SECURITY_PROTOCOL or "PLAINTEXT").upper()
    if protocol == "SASL_SSL":
        return {
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": KAFKA_SASL_MECHANISM,
            "sasl_plain_username": KAFKA_SASL_USERNAME,
            "sasl_plain_password": KAFKA_SASL_PASSWORD,
            "ssl_context": ssl.create_default_context(),
        }
    return {"security_protocol": "PLAINTEXT"}

# Topic this worker consumes (dedicated security topic — NOT the evaluations topic).
KAFKA_SECURITY_TOPIC        = os.getenv("KAFKA_SECURITY_TOPIC")
KAFKA_SECURITY_GROUP_ID     = os.getenv("KAFKA_SECURITY_GROUP_ID")

# Topic for sync request-reply (pre-call check / response gate). The API blocks
# on this keyed by correlation_id.
KAFKA_SECURITY_REPLY_TOPIC  = os.getenv("KAFKA_SECURITY_REPLY_TOPIC")

# Default publish topic for async "enriched" security notifications consumed by
# the API's SSE route (shared with the tracer; routed by the `kind` field).
KAFKA_TRACE_PERSISTED_TOPIC = os.getenv("KAFKA_TRACE_PERSISTED_TOPIC")

# ── ClickHouse ─────────────────────────────────────────────────────────────────
CLICKHOUSE_HOST          = os.getenv("CLICKHOUSE_HOST")
CLICKHOUSE_PORT          = int(os.getenv("CLICKHOUSE_PORT"))
CLICKHOUSE_USER          = os.getenv("CLICKHOUSE_USER")
CLICKHOUSE_PASSWORD      = os.getenv("CLICKHOUSE_PASSWORD")
CLICKHOUSE_DATABASE      = os.getenv("CLICKHOUSE_DATABASE")
CLICKHOUSE_SECURITY_TABLE = os.getenv("CLICKHOUSE_SECURITY_TABLE")
