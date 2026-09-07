# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import logging

logger = logging.getLogger(__name__)


def get_gpu_free_memory_mb(device_id: int | str) -> int:
    """Get remaining free GPU VRAM in megabytes for a given device index."""
    if str(device_id).lower() == "cpu":
        return 1000000
    try:
        import torch

        if not torch.cuda.is_available():
            return 1000000
        dev_idx = int(device_id) if isinstance(device_id, int) or str(device_id).isdigit() else 0
        free_bytes, _ = torch.cuda.mem_get_info(dev_idx)
        return free_bytes // (1024 * 1024)
    except Exception:
        return 1000000


def calculate_dynamic_gpu_utilization(
    device_id: int | str,
    gpu_memory_utilization: float,
    free_ratio: float = 0.70,
    mock_free_bytes: int | None = None,
    mock_total_bytes: int | None = None,
) -> float:
    """Calculate effective vLLM gpu_memory_utilization based on remaining free VRAM.

    Prevents OOM when other processes (e.g. training jobs) occupy parts of the GPU memory.
    """
    if str(device_id).lower() == "cpu":
        return gpu_memory_utilization

    try:
        if mock_free_bytes is not None and mock_total_bytes is not None:
            free_bytes, total_bytes = mock_free_bytes, mock_total_bytes
        else:
            import torch

            if not torch.cuda.is_available():
                return gpu_memory_utilization
            dev_idx = (
                int(device_id) if isinstance(device_id, int) or str(device_id).isdigit() else 0
            )
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)

        if total_bytes > 0:
            target_util = (free_bytes * free_ratio) / total_bytes
            return min(gpu_memory_utilization, max(0.15, target_util))
    except Exception as e:
        logger.warning(
            "Failed to compute dynamic GPU memory utilization for device %s: %s", device_id, e
        )

    return gpu_memory_utilization
