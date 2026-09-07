# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatibility module forwarding to src.scheduler."""

from .scheduler.handler import BaseTaskHandler, TaskHandlerRegistry
from .scheduler.handlers.asr import ASRTaskHandler
from .scheduler.runner import JobRunner
from .scheduler.semaphore import DynamicSemaphore

__all__ = [
    "ASRTaskHandler",
    "BaseTaskHandler",
    "DynamicSemaphore",
    "JobRunner",
    "TaskHandlerRegistry",
]
