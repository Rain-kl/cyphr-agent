# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import os
from collections.abc import Callable
from typing import Any

from .base import BaseEngine


class MockASREngine(BaseEngine):
    """Mock ASR engine that simulates multi-stage transcription and yields OpenAI verbose_json."""

    def __init__(
        self,
        model_name: str = "mock-whisper-base",
        stage_delay: float = 0.05,
    ) -> None:
        super().__init__(model_name)
        self.stage_delay = stage_delay

    async def load(self, work_mode: str = "gpu") -> None:
        """Simulate loading model weights."""
        if self.stage_delay > 0:
            await asyncio.sleep(self.stage_delay)
        self.loaded = True

    async def unload(self) -> None:
        """Simulate unloading model resources."""
        if self.stage_delay > 0:
            await asyncio.sleep(self.stage_delay)
        self.loaded = False

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        task_type: str = "transcribe",
        log_callback: Callable[[int, str], Any] | None = None,
    ) -> dict[str, Any]:
        """Simulate multi-stage speech-to-text inference.

        Stages:
        - 20%: Loading audio file
        - 30%: Preprocessing audio chunks
        - 80%: Running acoustic model inference
        - 100%: Aligning timestamps and generating verbose_json transcript
        """
        if not self.loaded:
            await self.load()

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        stages = [
            (20, "Loading and decoding audio file..."),
            (30, "Preprocessing audio chunks and extracting features..."),
            (80, "Running acoustic model inference..."),
            (100, "Aligning timestamps and finalizing transcript..."),
        ]

        for progress, message in stages:
            if log_callback:
                res = log_callback(progress, message)
                if inspect.isawaitable(res):
                    await res
            if self.stage_delay > 0:
                await asyncio.sleep(self.stage_delay)

        target_lang = language if language else "english"
        full_text = "This is a mock transcription result for testing purposes."

        return {
            "task": task_type or "transcribe",
            "language": target_lang,
            "duration": 5.0,
            "text": full_text,
            "segments": [
                {
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": 2.5,
                    "text": "This is a mock transcription result",
                    "tokens": [50364, 1212, 318, 257, 12450, 41724, 1146, 50489],
                    "temperature": 0.0,
                    "avg_logprob": -0.15,
                    "compression_ratio": 1.2,
                    "no_speech_prob": 0.01,
                },
                {
                    "id": 1,
                    "seek": 250,
                    "start": 2.5,
                    "end": 5.0,
                    "text": " for testing purposes.",
                    "tokens": [50489, 329, 3960, 9367, 13, 50614],
                    "temperature": 0.0,
                    "avg_logprob": -0.12,
                    "compression_ratio": 1.1,
                    "no_speech_prob": 0.02,
                },
            ],
        }
