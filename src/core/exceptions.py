# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0


class AgentBaseError(Exception):
    """Base exception for all agent domain errors."""

    pass


class InsufficientVRAMError(AgentBaseError, RuntimeError):
    """Raised when GPU VRAM is insufficient to load the model or wake up to restore KV cache."""

    pass


class ModelLoadError(AgentBaseError, RuntimeError):
    """Raised when model loading or initialization fails."""

    pass


class TaskExecutionError(AgentBaseError, RuntimeError):
    """Raised when task execution fails."""

    pass
