# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Collects host machine utilization metrics with safe GPU fallback."""

    def __init__(self) -> None:
        # Prime psutil cpu measurement
        psutil.cpu_percent(interval=None)

    def collect(self) -> dict[str, Any]:
        """Collect current CPU, RAM, and GPU stats."""
        # 1. CPU percent
        try:
            cpu_percent = float(psutil.cpu_percent(interval=None))
        except Exception as e:
            logger.warning("Failed to collect CPU percent: %s", e)
            cpu_percent = 0.0

        # 2. RAM stats
        try:
            mem = psutil.virtual_memory()
            ram_percent = float(mem.percent)
            ram_used_mb = int(mem.used // (1024 * 1024))
            ram_total_mb = int(mem.total // (1024 * 1024))
        except Exception as e:
            logger.warning("Failed to collect RAM stats: %s", e)
            ram_percent = 0.0
            ram_used_mb = 0
            ram_total_mb = 0

        # 3. GPU stats (safe fallback to 0.0)
        gpu_percent, gpu_used_mb, gpu_total_mb = self._collect_gpu()

        return {
            "cpu_percent": round(cpu_percent, 1),
            "ram_percent": round(ram_percent, 1),
            "ram_used_mb": ram_used_mb,
            "ram_total_mb": ram_total_mb,
            "gpu_percent": round(gpu_percent, 1),
            "gpu_memory_used_mb": gpu_used_mb,
            "gpu_memory_total_mb": gpu_total_mb,
        }

    def _collect_gpu(self) -> tuple[float, int, int]:
        """Safely probe GPU metrics without failing if GPU or drivers are missing."""
        try:
            # First try pynvml if available
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return (
                    float(util.gpu),
                    int(mem_info.used // (1024 * 1024)),
                    int(mem_info.total // (1024 * 1024)),
                )
        except Exception:
            pass

        try:
            # Fallback check torch.cuda if available
            import torch  # type: ignore

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                mem_allocated = torch.cuda.memory_allocated(0) // (1024 * 1024)
                total_mem = (
                    torch.cuda.get_device_properties(0).total_memory
                    // (1024 * 1024)
                )
                return (0.0, int(mem_allocated), int(total_mem))
        except Exception:
            pass

        # Safe fallback
        return (0.0, 0, 0)
