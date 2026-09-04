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
        self._notify_scheduled = False

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
        except RuntimeError:
            return
        # 合并连续扩容/缩容的唤醒：若已有待执行 notify，直接复用，避免后台任务堆积泄漏。
        self._bg_tasks = {t for t in self._bg_tasks if not t.done()}
        if self._bg_tasks or self._notify_scheduled:
            return
        self._notify_scheduled = True
        try:
            # 线程安全地调度唤醒（set_capacity 可能被非事件循环线程调用）
            loop.call_soon_threadsafe(lambda: self._schedule_notify(_notify))
        except RuntimeError:
            # 事件循环已关闭时静默跳过；等待者会在 acquire 轮询新 capacity 时自然通过
            self._notify_scheduled = False
            return

    def _schedule_notify(self, coro_factory: Any) -> None:
        self._notify_scheduled = False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        # 二次合并：若回调排队期间已有 notify 任务在飞，则跳过新建
        self._bg_tasks = {t for t in self._bg_tasks if not t.done()}
        if self._bg_tasks:
            return
        task = loop.create_task(coro_factory())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

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
        # 按 model_name 细粒度串行：同模型推理排队，不同模型可并行。
        # key 仅为已注册模型名（数量有界），不会随 job 增长而泄漏。
        self._inference_locks: dict[str, asyncio.Lock] = {}
        self._active_tasks: dict[int, asyncio.Task[None]] = {}

    def _get_inference_lock(self, model_name: str) -> asyncio.Lock:
        lock = self._inference_locks.get(model_name)
        if lock is None:
            lock = asyncio.Lock()
            self._inference_locks[model_name] = lock
        return lock

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
                inference_lock = self._get_inference_lock(model_name)
                async with self.registry.acquire_engine(model_name) as engine:
                    # 4. Engine log callback (throttled: 同job间隔>=2s或progress增量>=10才上报)
                    _last_log_time = 0.0
                    _last_progress = -100

                    async def engine_log_cb(progress: int, message: str) -> None:
                        nonlocal _last_log_time, _last_progress
                        now = time.monotonic()
                        is_first = _last_progress <= -100
                        if not is_first and progress < 100:
                            if (now - _last_log_time) < 2.0 and (progress - _last_progress) < 10:
                                return
                        ts = datetime.now(UTC).isoformat()
                        try:
                            await self.reporter.report_logs(
                                job_id=job_id,
                                progress=progress,
                                logs=[{"timestamp": ts, "level": "info", "message": message}],
                            )
                        except Exception as log_err:
                            logger.warning("Failed to report progress log: %s", log_err)
                        else:
                            _last_log_time = now
                            _last_progress = progress

                    # 5. Perform inference with GIL protection for CPU-heavy / blocking tasks
                    loop = asyncio.get_running_loop()
                    if inference_lock.locked():
                        await self.reporter.report_logs(
                            job_id=job_id,
                            progress=15,
                            logs=[{
                                "timestamp": datetime.now(UTC).isoformat(),
                                "level": "info",
                                "message": "Waiting for inference engine to become available...",
                            }],
                        )

                    async with inference_lock:
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
