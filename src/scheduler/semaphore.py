# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any


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
