# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

from ..core.engine import BaseModelEngine


class BaseASREngine(BaseModelEngine):
    """Abstract base class for all ASR model inference engines."""

    @abstractmethod
    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        task_type: str = "transcribe",
        log_callback: Callable[[int, str], Any] | None = None,
    ) -> dict[str, Any]:
        """Perform speech-to-text inference on audio file.

        Args:
            audio_path: Local filesystem path to the audio file.
            language: Optional target language code (e.g. 'en', 'zh').
            task_type: Task type ('transcribe' or 'translate').
            log_callback: Callback function receiving (progress: int, message: str).

        Returns:
            OpenAI verbose_json compliant dictionary.
        """


# Backwards compatibility alias
BaseEngine = BaseASREngine
