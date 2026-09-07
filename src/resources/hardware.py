# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any

from .vram import get_gpu_free_memory_mb

logger = logging.getLogger(__name__)

DEFAULT_MIN_LOAD_VRAM_MB = 2048


def detect_supported_modes() -> tuple[list[str], str]:
    """Detect available acceleration hardware and multi-GPU devices."""
    modes = ["cpu"]
    default_mode = "cpu"
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            modes.append("gpu")
            count = torch.cuda.device_count()
            for idx in range(count):
                modes.append(f"cuda:{idx}")
            default_mode = "gpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            modes.append("gpu")
            modes.append("mps")
            default_mode = "gpu"
    except Exception:
        pass

    return modes, default_mode


def resolve_devices(work_mode: str = "gpu", min_load_vram_mb: int | None = None) -> list[int | str]:
    """Resolve list of target inference device identifiers (GPU indices or 'cpu'),
    filtering out any GPU devices with insufficient free VRAM for model loading.
    """
    import torch

    env_device = (
        os.getenv("DEVICE", os.getenv("MODEL_DEVICE", os.getenv("QWEN3_ASR_DEVICE", "")))
        .strip()
        .lower()
    )
    explicit_idx = os.getenv("CUDA_DEVICE_INDEX", "").strip()

    candidate_devices: list[int | str]
    if env_device and env_device not in ("all", "multi", "gpu", "cuda"):
        if env_device == "cpu":
            return ["cpu"]
        if env_device.startswith("cuda:"):
            try:
                candidate_devices = [int(env_device.split(":")[1])]
            except ValueError:
                candidate_devices = [0]
        else:
            candidate_devices = [env_device]
    elif explicit_idx:
        try:
            candidate_devices = [int(explicit_idx)]
        except ValueError:
            candidate_devices = [0]
    else:
        req_device = work_mode.strip().lower()
        if req_device == "cpu":
            return ["cpu"]
        if req_device.startswith("cuda:"):
            try:
                candidate_devices = [int(req_device.split(":")[1])]
            except ValueError:
                candidate_devices = [0]
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            if env_vis := os.getenv("CUDA_VISIBLE_DEVICES"):
                parts = [p.strip() for p in env_vis.split(",") if p.strip()]
                valid: list[int | str] = []
                for p in parts:
                    try:
                        valid.append(int(p))
                    except ValueError:
                        pass
                candidate_devices = valid if valid else list(range(torch.cuda.device_count()))
            else:
                candidate_devices = list(range(torch.cuda.device_count()))
        else:
            return ["cpu"]

    # Filter candidate GPUs by free memory against min_load_vram_mb
    if min_load_vram_mb is None:
        min_load_vram_mb = int(os.getenv("MIN_LOAD_VRAM_MB", str(DEFAULT_MIN_LOAD_VRAM_MB)))

    available_devices: list[int | str] = []
    for dev in candidate_devices:
        if str(dev).lower() == "cpu":
            available_devices.append(dev)
        else:
            free_mb = get_gpu_free_memory_mb(dev)
            if free_mb >= min_load_vram_mb:
                available_devices.append(dev)
            else:
                logger.warning(
                    "[resolve_devices] Skipping GPU %s: insufficient free VRAM (%d MB < %d MB required)",
                    dev,
                    free_mb,
                    min_load_vram_mb,
                )

    return available_devices


def resolve_device_and_dtype(
    work_mode: str = "gpu", default_batch_size: int = 16
) -> tuple[str, Any, int]:
    """Resolve target inference device, dtype, and default batch size."""
    import torch

    env_device = (
        os.getenv("DEVICE", os.getenv("MODEL_DEVICE", os.getenv("QWEN3_ASR_DEVICE", "")))
        .strip()
        .lower()
    )
    req_device = env_device or work_mode.strip().lower()
    batch_size = int(
        os.getenv(
            "INFERENCE_BATCH_SIZE", os.getenv("QWEN3_ASR_BATCH_SIZE", str(default_batch_size))
        )
    )

    if req_device == "cpu":
        return "cpu", torch.float32, 1

    if req_device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA requested (%s) but torch.cuda is not available; falling back to CPU",
                req_device,
            )
            return "cpu", torch.float32, 1
        device = req_device if ":" in req_device else "cuda:0"
        use_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        return device, dtype, batch_size

    if req_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float16, min(batch_size, 4)
        return "cpu", torch.float32, 1

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device_idx = int(os.getenv("CUDA_DEVICE_INDEX", "0"))
        device_idx = min(max(0, device_idx), torch.cuda.device_count() - 1)
        use_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        return f"cuda:{device_idx}", dtype, batch_size

    return "cpu", torch.float32, 1
