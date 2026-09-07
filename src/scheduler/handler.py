# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from ..reporter import Reporter


class BaseTaskHandler(ABC):
    """Abstract base class for domain-specific task executors (ASR, LLM, TTS, etc.)."""

    @abstractmethod
    def can_handle(self, task_type: str, payload: dict[str, Any]) -> bool:
        """Return True if this handler can process the given task."""

    @abstractmethod
    async def execute(
        self,
        payload: dict[str, Any],
        engine: Any,
        reporter: Reporter,
        inference_lock: asyncio.Lock,
    ) -> Any:
        """Execute task using the acquired engine and reporter."""


class TaskHandlerRegistry:
    """Registry managing domain-specific task handlers."""

    def __init__(self) -> None:
        self._handlers: list[BaseTaskHandler] = []

    def register(self, handler: BaseTaskHandler) -> None:
        """Register a new task handler (checked in reverse order of registration)."""
        self._handlers.append(handler)

    def resolve(self, task_type: str, payload: dict[str, Any]) -> BaseTaskHandler | None:
        """Find the matching handler for a given task type and payload."""
        for handler in reversed(self._handlers):
            if handler.can_handle(task_type, payload):
                return handler
        return None
