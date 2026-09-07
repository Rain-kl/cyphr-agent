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

from ...reporter import Reporter
from ..handler import BaseTaskHandler

logger = logging.getLogger(__name__)


class ASRTaskHandler(BaseTaskHandler):
    """Task executor specialized for speech-to-text (ASR) audio inference."""

    def __init__(self, media_dir: str = "/tmp/transcribe/media") -> None:
        self.media_dir = media_dir

    def can_handle(self, task_type: str, payload: dict[str, Any]) -> bool:
        """Match audio transcription and translation tasks."""
        return task_type in ("transcribe", "translate") or "media_path" in payload or not task_type

    async def execute(
        self,
        payload: dict[str, Any],
        engine: Any,
        reporter: Reporter,
        inference_lock: asyncio.Lock,
    ) -> Any:
        """Execute ASR transcription pipeline: download media -> run inference -> report completion."""
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
            # 1. Announce start
            now_iso = datetime.now(UTC).isoformat()
            await reporter.report_logs(
                job_id=job_id,
                progress=5,
                logs=[
                    {
                        "timestamp": now_iso,
                        "level": "info",
                        "message": f"Job {job_id} scheduled on agent node (model: {model_name})",
                    }
                ],
            )

            # 2. Download media
            now_iso = datetime.now(UTC).isoformat()
            await reporter.report_logs(
                job_id=job_id,
                progress=10,
                logs=[
                    {
                        "timestamp": now_iso,
                        "level": "info",
                        "message": "Downloading media file...",
                    }
                ],
            )
            await reporter.download_media(job_id, local_file_path)

            # 3. Engine log callback
            async def engine_log_cb(progress: int, message: str) -> None:
                ts = datetime.now(UTC).isoformat()
                try:
                    await reporter.report_logs(
                        job_id=job_id,
                        progress=progress,
                        logs=[{"timestamp": ts, "level": "info", "message": message}],
                    )
                except Exception as log_err:
                    logger.warning("Failed to report progress log: %s", log_err)

            # 4. Perform inference with GIL protection for CPU-heavy / blocking tasks
            loop = asyncio.get_running_loop()
            supports_concurrent = getattr(engine, "supports_concurrent_inference", False)

            async def _run_inference() -> dict[str, Any]:
                if asyncio.iscoroutinefunction(getattr(engine, "transcribe", None)):
                    return await engine.transcribe(
                        audio_path=local_file_path,
                        language=language,
                        task_type=task_type,
                        log_callback=engine_log_cb,
                    )
                pending_log_futures: list[concurrent.futures.Future[Any]] = []

                def sync_log_cb(p: int, msg: str) -> None:
                    fut = asyncio.run_coroutine_threadsafe(
                        engine_log_cb(p, msg),
                        loop,
                    )
                    pending_log_futures.append(fut)

                res = await loop.run_in_executor(
                    None,
                    engine.transcribe,
                    local_file_path,
                    language,
                    task_type,
                    sync_log_cb,
                )

                if pending_log_futures:
                    concurrent.futures.wait(pending_log_futures, timeout=0.5)
                return res

            if supports_concurrent:
                result = await _run_inference()
            else:
                if inference_lock.locked():
                    await reporter.report_logs(
                        job_id=job_id,
                        progress=15,
                        logs=[
                            {
                                "timestamp": datetime.now(UTC).isoformat(),
                                "level": "info",
                                "message": "Waiting for inference engine to become available...",
                            }
                        ],
                    )

                async with inference_lock:
                    result = await _run_inference()

            # 5. Settle completion
            duration = time.time() - start_time
            result_text = result.get("text", "")
            await reporter.report_completion(
                job_id=job_id,
                status="completed",
                duration_seconds=round(duration, 2),
                result_text=result_text,
                openai_response=result,
            )
            logger.info("Job %d completed successfully in %.2fs", job_id, duration)
            return result

        finally:
            if os.path.exists(local_file_path):
                try:
                    os.remove(local_file_path)
                except OSError as os_err:
                    logger.warning("Failed to remove temp file %s: %s", local_file_path, os_err)
