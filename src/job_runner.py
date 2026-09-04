# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models.registry import ModelRegistry
from .reporter import Reporter

logger = logging.getLogger(__name__)


class DynamicSemaphore:
    """An asyncio semaphore that supports dynamic capacity changes without leaking permits."""

    def __init__(self, initial_capacity: int) -> None:
        self._capacity = max(1, initial_capacity)
        self._acquired = 0
        self._cond = asyncio.Condition()
        self._bg_tasks: set[asyncio.Task[None]] = set()

    @property
    def capacity(self) -> int:
        return self._capacity

    def set_capacity(self, new_capacity: int) -> None:
        if new_capacity <= 0:
            return
        self._capacity = new_capacity

        async def _notify() -> None:
            async with self._cond:
                self._cond.notify_all()

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(_notify())
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        except RuntimeError:
            pass

    async def acquire(self) -> None:
        async with self._cond:
            while self._acquired >= self._capacity:
                await self._cond.wait()
            self._acquired += 1

    async def release(self) -> None:
        async with self._cond:
            self._acquired = max(0, self._acquired - 1)
            self._cond.notify_all()

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.release()


class JobRunner:
    """Manages concurrent execution of transcription tasks with global exception shielding."""

    def __init__(
        self,
        reporter: Reporter,
        registry: ModelRegistry,
        media_dir: str = "/tmp/transcribe/media",
        max_concurrent_jobs: int = 2,
    ) -> None:
        self.reporter = reporter
        self.registry = registry
        self.media_dir = media_dir
        self.max_concurrent_jobs = max_concurrent_jobs
        self._semaphore = DynamicSemaphore(max_concurrent_jobs)
        self._inference_lock = asyncio.Lock()
        self._active_tasks: dict[int, asyncio.Task[None]] = {}

    def get_running_jobs_count(self) -> int:
        """Return number of currently active jobs."""
        return len(self._active_tasks)

    def set_max_concurrent_jobs(self, limit: int) -> None:
        """Update maximum concurrency limit dynamically."""
        if limit > 0 and limit != self.max_concurrent_jobs:
            logger.info("Updating max concurrent jobs from %d to %d", self.max_concurrent_jobs, limit)
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

        task.add_done_callback(_cleanup)
        return task

    async def _execute_job(self, payload: dict[str, Any]) -> None:
        """Internal execution pipeline wrapped in full exception shielding."""
        job_id = int(payload["job_id"])
        model_name = payload.get("model_name", "mock-whisper-base")
        language = payload.get("language")
        task_type = payload.get("task_type", "transcribe")
        media_path = payload.get("media_path", "")

        os.makedirs(self.media_dir, exist_ok=True)
        ext = Path(media_path).suffix or ".mp3"
        local_file_path = os.path.join(
            self.media_dir,
            f"job_{job_id}_{uuid.uuid4().hex[:8]}{ext}",
        )
        start_time = time.time()

        try:
            async with self._semaphore:
                # 1. Announce start
                now_iso = datetime.now(UTC).isoformat()
                await self.reporter.report_logs(
                    job_id=job_id,
                    progress=5,
                    logs=[{
                        "timestamp": now_iso,
                        "level": "info",
                        "message": f"Job {job_id} scheduled on agent node (model: {model_name})",
                    }],
                )

                # 2. Download media
                now_iso = datetime.now(UTC).isoformat()
                await self.reporter.report_logs(
                    job_id=job_id,
                    progress=10,
                    logs=[{
                        "timestamp": now_iso,
                        "level": "info",
                        "message": "Downloading media file...",
                    }],
                )
                await self.reporter.download_media(job_id, local_file_path)

                # 3. Resolve and acquire model with lifecycle guard (ref counted, prevents hot-unload crash)
                async with self.registry.acquire_engine(model_name) as engine:
                    # 4. Engine log callback
                    async def engine_log_cb(progress: int, message: str) -> None:
                        ts = datetime.now(UTC).isoformat()
                        try:
                            await self.reporter.report_logs(
                                job_id=job_id,
                                progress=progress,
                                logs=[{"timestamp": ts, "level": "info", "message": message}],
                            )
                        except Exception as log_err:
                            logger.warning("Failed to report progress log: %s", log_err)

                    # 5. Perform inference with GIL protection for CPU-heavy / blocking tasks
                    loop = asyncio.get_running_loop()
                    if self._inference_lock.locked():
                        await self.reporter.report_logs(
                            job_id=job_id,
                            progress=15,
                            logs=[{
                                "timestamp": datetime.now(UTC).isoformat(),
                                "level": "info",
                                "message": "Waiting for inference engine to become available...",
                            }],
                        )

                    async with self._inference_lock:
                        if asyncio.iscoroutinefunction(engine.transcribe):
                            result = await engine.transcribe(
                                audio_path=local_file_path,
                                language=language,
                                task_type=task_type,
                                log_callback=engine_log_cb,
                            )
                        else:
                            pending_log_futures: list[concurrent.futures.Future[Any]] = []

                            def sync_log_cb(p: int, msg: str) -> None:
                                # Fire-and-track asynchronously; do not block GPU inference worker thread
                                fut = asyncio.run_coroutine_threadsafe(
                                    engine_log_cb(p, msg),
                                    loop,
                                )
                                pending_log_futures.append(fut)

                            result = await loop.run_in_executor(
                                None,
                                engine.transcribe,
                                local_file_path,
                                language,
                                task_type,
                                sync_log_cb,
                            )

                            # Drain any in-flight log callbacks with concurrent wait (bounded to 0.5s)
                            if pending_log_futures:
                                concurrent.futures.wait(pending_log_futures, timeout=0.5)

                # 6. Settle completion
                duration = time.time() - start_time
                result_text = result.get("text", "")
                await self.reporter.report_completion(
                    job_id=job_id,
                    status="completed",
                    duration_seconds=round(duration, 2),
                    result_text=result_text,
                    openai_response=result,
                )
                logger.info("Job %d completed successfully in %.2fs", job_id, duration)

        except Exception as exc:
            # Shielding: catch any exception, report failure, never crash agent
            duration = time.time() - start_time
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

        finally:
            # Clean up local audio file
            if os.path.exists(local_file_path):
                try:
                    os.remove(local_file_path)
                except OSError as os_err:
                    logger.warning("Failed to remove temp file %s: %s", local_file_path, os_err)
