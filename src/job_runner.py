# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
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
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
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
            self._semaphore = asyncio.Semaphore(limit)

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

                # 3. Resolve and load model if needed
                engine = self.registry.get_engine(model_name)
                if engine is None or not engine.loaded:
                    engine = await self.registry.load_model(model_name)

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
                        def sync_log_cb(p: int, msg: str) -> None:
                            fut = asyncio.run_coroutine_threadsafe(
                                engine_log_cb(p, msg),
                                loop,
                            )
                            try:
                                fut.result(timeout=5.0)
                            except Exception as cb_err:
                                logger.warning("Error in sync_log_cb: %s", cb_err)

                        result = await loop.run_in_executor(
                            None,
                            engine.transcribe,
                            local_file_path,
                            language,
                            task_type,
                            sync_log_cb,
                        )

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
