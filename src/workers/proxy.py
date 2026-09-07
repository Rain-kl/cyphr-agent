# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing as mp
import queue
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class BaseWorkerProxy:
    """Generic base class managing isolated child worker process lifecycle and IPC communication.

    Provides cross-model process spawning, ready signaling, sleep mode offload, and graceful shutdown.
    """

    def __init__(
        self,
        device_id: int | str,
        target_fn: Callable[..., Any],
        args: tuple[Any, ...],
    ) -> None:
        self.device_id = device_id
        self.is_sleeping = False
        ctx = mp.get_context("spawn")
        self.cmd_queue = ctx.Queue()
        self.resp_queue = ctx.Queue()
        full_args = (*args, self.cmd_queue, self.resp_queue)
        self.process = ctx.Process(target=target_fn, args=full_args, daemon=True)
        self.process.start()

    def wait_ready(self, timeout: float = 60.0) -> None:
        """Wait for worker process to signal ready."""
        start = time.time()
        while time.time() - start < timeout:
            if not self.process.is_alive():
                raise RuntimeError(
                    f"Worker process for device {self.device_id} terminated unexpectedly during initialization"
                )
            try:
                msg = self.resp_queue.get(timeout=1.0)
                if msg[0] == "ready":
                    return
                if msg[0] == "error":
                    raise RuntimeError(f"Worker init error on device {self.device_id}: {msg[2]}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.debug("Transient error waiting for worker ready: %s", e)
                continue
        raise TimeoutError(
            f"Worker for device {self.device_id} timed out after {timeout}s waiting for ready"
        )

    def enter_sleep(self) -> None:
        """Signal child worker process to enter sleep mode and offload KV cache."""
        self.is_sleeping = True
        if self.process.is_alive():
            try:
                self.cmd_queue.put(("sleep",))
            except Exception:
                pass

    def stop(self, timeout: float = 3.0) -> None:
        """Gracefully stop worker process with fallback kill."""
        self.is_sleeping = True
        if self.process.is_alive():
            try:
                self.cmd_queue.put(("stop",))
            except Exception:
                pass
            self.process.join(timeout=timeout)
            if self.process.is_alive():
                self.process.kill()
