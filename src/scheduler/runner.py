# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from ..core.exceptions import InsufficientVRAMError
from ..models.registry import ModelRegistry
from ..reporter import Reporter
from .handler import BaseTaskHandler, TaskHandlerRegistry
from .handlers.asr import ASRTaskHandler
from .semaphore import DynamicSemaphore

logger = logging.getLogger(__name__)


class JobRunner:
    """Manages concurrent execution of model inference tasks with lifecycle acquisition,
    concurrency control, dynamic exception shielding, and pluggable task handlers.
    """

    def __init__(
        self,
        reporter: Reporter,
        registry: ModelRegistry,
        media_dir: str = "/tmp/transcribe/media",
        max_concurrent_jobs: int = 2,
        on_job_rejected: Callable[[int, str], Any] | None = None,
        on_job_finished: Callable[[], Any] | None = None,
        task_registry: TaskHandlerRegistry | None = None,
    ) -> None:
        self.reporter = reporter
        self.registry = registry
        self.media_dir = media_dir
        self.max_concurrent_jobs = max_concurrent_jobs
        self.on_job_rejected = on_job_rejected
        self.on_job_finished = on_job_finished

        # Task Handler Registry
        if task_registry is None:
            self.task_registry = TaskHandlerRegistry()
            # Register default ASR handler
            self.task_registry.register(ASRTaskHandler(media_dir=media_dir))
        else:
            self.task_registry = task_registry

        self._semaphore = DynamicSemaphore(max_concurrent_jobs)
        self._inference_lock = asyncio.Lock()
        self._active_tasks: dict[int, asyncio.Task[None]] = {}

    def register_task_handler(self, handler: BaseTaskHandler) -> None:
        """Register a new domain task handler (e.g. LLM, TTS)."""
        self.task_registry.register(handler)

    def get_running_jobs_count(self) -> int:
        """Return number of currently active jobs."""
        return len(self._active_tasks)

    def set_max_concurrent_jobs(self, limit: int) -> None:
        """Update maximum concurrency limit dynamically."""
        if limit > 0 and limit != self.max_concurrent_jobs:
            logger.info(
                "Updating max concurrent jobs from %d to %d", self.max_concurrent_jobs, limit
            )
            self.max_concurrent_jobs = limit
            self._semaphore.set_capacity(limit)

    def run_job(self, payload: dict[str, Any]) -> asyncio.Task[None]:
        """Dispatch a job asynchronously in the background.

        Does not block; returns the spawned asyncio.Task.
        """
        job_id = int(payload["job_id"])
        task = asyncio.create_task(self._execute_job(payload))
        self._active_tasks[job_id] = task

        def _cleanup(_: asyncio.Task[None]) -> None:
            self._active_tasks.pop(job_id, None)
            if self.on_job_finished is not None:
                try:
                    cb = self.on_job_finished()
                    if asyncio.iscoroutine(cb):
                        asyncio.create_task(cb)
                except Exception as ex:
                    logger.warning("Error in on_job_finished callback: %s", ex)

        task.add_done_callback(_cleanup)
        return task

    async def _execute_job(self, payload: dict[str, Any]) -> None:
        """Internal execution pipeline wrapped in full exception shielding."""
        job_id = int(payload["job_id"])
        model_name = payload.get("model_name", "mock-whisper-base")
        task_type = payload.get("task_type", "transcribe")
        start_time = time.time()

        try:
            async with self._semaphore:
                # Resolve handler before acquiring engine
                handler = self.task_registry.resolve(task_type, payload)
                if handler is None:
                    raise RuntimeError(f"No task handler found for task_type '{task_type}'")

                # Acquire model engine with lifecycle guard (ref counted, prevents hot-unload crash)
                async with self.registry.acquire_engine(model_name) as engine:
                    await handler.execute(
                        payload=payload,
                        engine=engine,
                        reporter=self.reporter,
                        inference_lock=self._inference_lock,
                    )

        except Exception as exc:
            duration = time.time() - start_time
            is_oom = (
                isinstance(exc, InsufficientVRAMError)
                or "out of memory" in str(exc).lower()
                or "cuda error: out of memory" in str(exc).lower()
                or type(exc).__name__ == "OutOfMemoryError"
            )

            if is_oom and self.on_job_rejected is not None:
                logger.warning("Job %d rejected due to VRAM / OOM condition: %s", job_id, exc)
                try:
                    cb_res = self.on_job_rejected(job_id, str(exc))
                    if asyncio.iscoroutine(cb_res):
                        await cb_res
                except Exception as reject_err:
                    logger.error(
                        "Error during on_job_rejected callback for job %d: %s", job_id, reject_err
                    )
                return

            # Shielding: catch any exception, report failure, never crash agent
            logger.exception("Job %d execution encountered error: %s", job_id, exc)
            try:
                await self.reporter.report_completion(
                    job_id=job_id,
                    status="failed",
                    duration_seconds=round(duration, 2),
                    result_text="",
                    openai_response=None,
                    error_msg=str(exc),
                )
            except Exception as report_err:
                logger.error("Failed to report failure status for job %d: %s", job_id, report_err)
