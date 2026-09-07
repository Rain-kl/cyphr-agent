# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod


class BaseModelEngine(ABC):
    """Abstract base class for all model inference engines (ASR, LLM, TTS, VLM, etc.).

    Encapsulates core lifecycle methods: loading, unloading, sleep mode (KV cache offload),
    and resource availability checks.
    """

    supports_concurrent_inference: bool = False

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.loaded = False

    @abstractmethod
    async def load(self, work_mode: str = "gpu") -> None:
        """Load model weights and initialize inference resources."""

    @abstractmethod
    async def unload(self) -> None:
        """Unload model and free GPU/CPU memory."""

    def enter_sleep(self) -> None:  # noqa: B027
        """Offload memory / KV cache during idle periods. Default no-op for models without KV cache."""
        pass

    def is_active(self) -> bool:
        """Check if engine is currently performing inference or awake with memory allocated."""
        return self.loaded and not getattr(self, "is_sleeping", False)

    def check_resources_available(self) -> bool:
        """Check whether system resources allow this engine to load or wake up."""
        return True
