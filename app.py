import asyncio
import json
import logging
import config
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable

from aiokafka import AIOKafkaConsumer

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

    consumer = AIOKafkaConsumer(
        config.KAFKA_SECURITY_TOPIC,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        group_id=config.KAFKA_SECURITY_GROUP_ID,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_partition_fetch_bytes=config.KAFKA_MAX_FETCH_BYTES,
        **config.kafka_auth_kwargs(),
    )

    # Pin all blocking work (asyncio.to_thread → run_scan) to a single executor
    # thread. Messages are processed one at a time, so no parallelism is lost —
    # and keeping torch/spaCy on one thread means one glibc malloc arena instead
    # of one per default-pool thread, the main driver of RSS growth.
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="security-worker")
    )

    await consumer.start()
    await clickhouse_security_client.start()
    await kafka_producer.start()
    logger.info(
        "[SECURITY] Consuming topic=%s group=%s servers=%s",
        config.KAFKA_SECURITY_TOPIC, config.KAFKA_SECURITY_GROUP_ID, config.KAFKA_BOOTSTRAP_SERVERS,
    )
    try:
        async for msg in consumer:
            try:
                await dispatch(msg.value)
                await consumer.commit()
            except Exception:
                logger.exception(
                    "[SECURITY] Failed to process message offset=%s partition=%s",
                    msg.offset, msg.partition,
                )
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
