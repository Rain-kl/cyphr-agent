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

        # 3. GPU stats: single probe, then derive aggregate (keeps old fields compat)
        try:
            gpu_devices = self._collect_gpu_devices()
        except Exception as e:
            logger.warning("Failed to collect per-GPU stats: %s", e)
            gpu_devices = []
        if gpu_devices:
            try:
                gpu_percent = max(float(d.get("percent", 0.0)) for d in gpu_devices)
                gpu_used_mb = sum(int(d.get("used_mb", 0)) for d in gpu_devices)
                gpu_total_mb = sum(int(d.get("total_mb", 0)) for d in gpu_devices)
            except Exception:
                gpu_percent, gpu_used_mb, gpu_total_mb = self._collect_gpu()
        else:
            gpu_percent, gpu_used_mb, gpu_total_mb = 0.0, 0, 0

        return {
            "cpu_percent": round(cpu_percent, 1),
            "ram_percent": round(ram_percent, 1),
            "ram_used_mb": ram_used_mb,
            "ram_total_mb": ram_total_mb,
            "gpu_percent": round(gpu_percent, 1),
            "gpu_memory_used_mb": gpu_used_mb,
            "gpu_memory_total_mb": gpu_total_mb,
            "gpu_devices": gpu_devices,
        }

    def _collect_gpu_devices(self) -> list[dict[str, Any]]:
        """Probe every visible GPU; returns [{index, percent, used_mb, total_mb}]."""
        devices: list[dict[str, Any]] = []
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for idx in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    devices.append({
                        "index": idx,
                        "percent": float(util.gpu),
                        "used_mb": int(mem_info.used // (1024 * 1024)),
                        "total_mb": int(mem_info.total // (1024 * 1024)),
                    })
                except Exception:
                    continue
            if devices:
                return devices
        except Exception:
            pass

        try:
            import torch  # type: ignore

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                for idx in range(torch.cuda.device_count()):
                    try:
                        mem_allocated = torch.cuda.memory_allocated(idx) // (1024 * 1024)
                        total_mem = torch.cuda.get_device_properties(idx).total_memory // (1024 * 1024)
                    except Exception:
                        continue
                    devices.append({
                        "index": idx,
                        "percent": 0.0,
                        "used_mb": int(mem_allocated),
                        "total_mb": int(total_mem),
                    })
                return devices
        except Exception:
            pass

        return []

    def _collect_gpu(self) -> tuple[float, int, int]:
        """Safely probe aggregate GPU metrics (max util, summed memory)."""
        try:
            devices = self._collect_gpu_devices()
            if devices:
                peak = max(float(d.get("percent", 0.0)) for d in devices)
                used = sum(int(d.get("used_mb", 0)) for d in devices)
                total = sum(int(d.get("total_mb", 0)) for d in devices)
                return (peak, used, total)
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
