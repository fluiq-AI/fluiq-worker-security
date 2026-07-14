import asyncio
import json
import logging
import config
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable

from aiokafka import AIOKafkaConsumer, TopicPartition

from db.clickhouse import clickhouse_security_client
from db.kafka import kafka_producer
from jobs.run import (
    auto_security_scan,
    sync_security_check,
    sync_response_gate_check,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OPERATIONS: dict[str, Callable[[dict[str, Any]], Awaitable[None]]] = {
    "sdk_security":        auto_security_scan,
    "security_check_sync": sync_security_check,
    "response_gate_check": sync_response_gate_check,
}

# Sync request-reply ops have a live requester blocking on the reply topic (the
# API's /secure/check or the /ingest response gate, which fail open on timeout).
# Retrying one minutes later is useless — never seek-back-block the partition
# for these; skip on any failure.
_SYNC_OPERATIONS = {"security_check_sync", "response_gate_check"}

# Seek-back retry pacing for transient (connection-class) failures.
_RETRY_INITIAL_S = 1.0
_RETRY_MAX_S = 30.0


def _is_transient(exc: BaseException) -> bool:
    """True for connection-class failures to a backing store (ClickHouse,
    Postgres, Kafka, network). The message itself is fine — the caller seeks
    back to its offset and retries, instead of letting a later ``commit()``
    advance the position past it (which silently loses the message).

    Walks the cause/context chain because drivers wrap the underlying network
    error (e.g. clickhouse_connect raises OperationalError *from* an aiohttp
    ClientConnectorError).
    """
    seen: set[int] = set()
    e: BaseException | None = exc
    depth = 0
    while e is not None and id(e) not in seen and depth < 10:
        seen.add(id(e))
        depth += 1
        if isinstance(e, (OSError, TimeoutError, asyncio.TimeoutError)):
            return True
        mod = type(e).__module__ or ""
        name = type(e).__name__
        if mod.startswith("clickhouse_connect") and name in ("OperationalError", "NetworkError"):
            return True
        if mod.startswith("aiohttp"):
            return True
        if mod.startswith("asyncpg") and (
            "Connect" in name or "TooManyConnections" in name or name == "InterfaceError"
        ):
            return True
        if mod.startswith(("aiokafka", "kafka")) and (
            "Connection" in name or "Timeout" in name or "NodeNotReady" in name
        ):
            return True
        e = e.__cause__ or e.__context__
    return False


async def _start_with_retry(attempt: Callable[[], Awaitable[Any]], what: str) -> Any:
    """Bring up one startup dependency, retrying with backoff until reachable.

    Replaces the boot crash-loop (raise → process exit → docker restart →
    repeat until the store is up) with an in-process wait: a worker's job is to
    outlast its stores, not die with them.
    """
    backoff = _RETRY_INITIAL_S
    while True:
        try:
            return await attempt()
        except Exception as exc:
            logger.warning(
                "[SECURITY] %s not ready at startup (%s) — retrying in %.1fs",
                what, type(exc).__name__, backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RETRY_MAX_S)


async def dispatch(message: dict[str, Any]) -> None:
    # This worker consumes a dedicated security topic, so every message is a
    # security job. The sync request-reply checks always carry an explicit
    # `operation`; raw trace envelopes (with or without `security_config`) are
    # full async post-call scans.
    operation = message.get("operation")
    if operation is None:
        operation = "sdk_security"
    handler = OPERATIONS.get(operation)
    if handler is None:
        logger.warning("[SECURITY] Unknown operation: %s", operation)
        return
    await handler(message)


async def consume() -> None:

    async def _consumer_attempt() -> AIOKafkaConsumer:
        # Recreate the consumer per attempt: a failed AIOKafkaConsumer.start()
        # can leave partial client state, so retrying a fresh instance is the
        # only clean path.
        c = AIOKafkaConsumer(
            config.KAFKA_SECURITY_TOPIC,
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            group_id=config.KAFKA_SECURITY_GROUP_ID,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            max_partition_fetch_bytes=config.KAFKA_MAX_FETCH_BYTES,
            **config.kafka_auth_kwargs(),
        )
        try:
            await c.start()
            return c
        except Exception:
            try:
                await c.stop()
            except Exception:
                pass
            raise

    # Pin all blocking work (asyncio.to_thread → run_scan) to a single executor
    # thread. Messages are processed one at a time, so no parallelism is lost —
    # and keeping torch/spaCy on one thread means one glibc malloc arena instead
    # of one per default-pool thread, the main driver of RSS growth.
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="security-worker")
    )

    consumer = await _start_with_retry(_consumer_attempt, "kafka consumer")
    await _start_with_retry(clickhouse_security_client.start, "clickhouse")
    await _start_with_retry(kafka_producer.start, "kafka producer")
    logger.info(
        "[SECURITY] Consuming topic=%s group=%s servers=%s",
        config.KAFKA_SECURITY_TOPIC, config.KAFKA_SECURITY_GROUP_ID, config.KAFKA_BOOTSTRAP_SERVERS,
    )
    try:
        backoff = _RETRY_INITIAL_S
        retries = 0
        async for msg in consumer:
            try:
                await dispatch(msg.value)
                await consumer.commit()
                backoff, retries = _RETRY_INITIAL_S, 0
            except Exception as exc:
                op = msg.value.get("operation") if isinstance(msg.value, dict) else None
                if _is_transient(exc) and op not in _SYNC_OPERATIONS:
                    # Backing store unreachable — the async scan is fine. Seek
                    # back to it and retry with backoff so a later commit() can
                    # never advance past it (that was silent data loss). Blocks
                    # the partition until the dependency recovers; Kafka
                    # retention holds the backlog.
                    retries += 1
                    consumer.seek(TopicPartition(msg.topic, msg.partition), msg.offset)
                    log = logger.error if retries % 20 == 0 else logger.warning
                    log(
                        "[SECURITY] Transient %s at offset=%s partition=%s — seeking back, "
                        "retry #%d in %.1fs",
                        type(exc).__name__, msg.offset, msg.partition, retries, backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, _RETRY_MAX_S)
                else:
                    # Poison message, or a sync request-reply whose requester has
                    # already failed open — skip DELIBERATELY by committing past
                    # it (previously this skip happened as an accident of the
                    # next successful commit).
                    logger.exception(
                        "[SECURITY] Failed to process message op=%s offset=%s partition=%s — skipping",
                        op, msg.offset, msg.partition,
                    )
                    try:
                        await consumer.commit()
                    except Exception:
                        logger.warning(
                            "[SECURITY] Commit after poison skip failed; message may re-deliver on restart",
                        )
                    backoff, retries = _RETRY_INITIAL_S, 0
    finally:
        await consumer.stop()
        await kafka_producer.stop()
        await clickhouse_security_client.stop()


def main() -> None:
    try:
        asyncio.run(consume())
    except KeyboardInterrupt:
        logger.info("[SECURITY] Shutdown requested")


if __name__ == "__main__":
    main()
