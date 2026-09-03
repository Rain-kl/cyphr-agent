# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import signal

from .config import load_config
from .job_runner import JobRunner
from .models.registry import default_registry
from .monitor import SystemMonitor
from .reporter import Reporter
from .ws_client import AgentWebSocketClient

logger = logging.getLogger("transcribe.agent")


async def main() -> None:
    """Agent process main async entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config()
    logger.info("Loaded agent config for node '%s'", config.node_name)
    logger.info("Controller URL: %s", config.controller_url)
    logger.info("Max concurrent jobs: %d", config.max_concurrent_jobs)

    monitor = SystemMonitor()
    registry = default_registry
    reporter = Reporter(
        base_url=config.http_base_url,
        agent_token=config.agent_token,
    )
    job_runner = JobRunner(
        reporter=reporter,
        registry=registry,
        media_dir=config.media_dir,
        max_concurrent_jobs=config.max_concurrent_jobs,
    )
    ws_client = AgentWebSocketClient(
        config=config,
        monitor=monitor,
        registry=registry,
        job_runner=job_runner,
    )

    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received, stopping agent...")
        asyncio.create_task(ws_client.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except (NotImplementedError, RuntimeError):
            # Windows or non-main thread fallback
            pass

    try:
        await ws_client.start()
    finally:
        await reporter.close()
        logger.info("Agent process exited cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
