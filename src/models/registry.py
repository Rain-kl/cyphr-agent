# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from .base import BaseEngine
from .mock_asr import MockASREngine
from .qwen3_asr import (
    MODEL_NAME_0_6B,
    MODEL_NAME_1_7B,
    Qwen3ASREngine,
    resolve_model_dir,
)

logger = logging.getLogger(__name__)


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


def _normalize_device(device: str) -> str:
    """Normalize a device string (lowercase/strip; bare 'cuda' -> 'cuda:0')."""
    d = device.strip().lower()
    if d == "cuda":
        return "cuda:0"
    return d


def _parse_device_list(text: str) -> list[str]:
    """Parse comma-separated device strings into normalized tokens."""
    parts: list[str] = []
    for tok in text.split(","):
        tok = tok.strip().lower()
        if tok:
            parts.append(_normalize_device(tok))
    return parts


def _parse_visible_indices() -> list[int] | None:
    """Parse $CUDA_VISIBLE_DEVICES into index list; None means no filtering."""
    raw = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    indices: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok.startswith("cuda:"):
            tok = tok.split(":", 1)[1].strip()
        if re.fullmatch(r"\d+", tok):
            indices.append(int(tok))
    return indices


class GpuScheduler:
    """Least-loaded GPU selector with env overrides and visibility filtering."""

    def list_devices(self) -> list[str]:
        """List visible CUDA devices (e.g. ['cuda:0', 'cuda:1']), honoring $CUDA_VISIBLE_DEVICES."""
        probed = self._probe_devices()
        visible = _parse_visible_indices()
        if visible is not None:
            probed = [d for d in probed if d["index"] in visible]
        return [f"cuda:{d['index']}" for d in probed]

    def select_device(self, candidates: list[str] | None = None) -> str:
        """Select the least-loaded device.

        - $QWEN3_ASR_DEVICE (except empty/'gpu') wins immediately.
        - Otherwise probe via NVML/torch, filter by $CUDA_VISIBLE_DEVICES
          and optional candidates, pick lowest (util, used_ratio).
        - Falls back to 'cpu' when no GPU is visible.
        """
        env_device = os.getenv("QWEN3_ASR_DEVICE", "").strip().lower()
        if env_device and env_device != "gpu":
            return _normalize_device(env_device)

        probed = self._probe_devices()
        visible = _parse_visible_indices()
        if visible is not None:
            probed = [d for d in probed if d["index"] in visible]
        if candidates:
            wanted: set[int] = set()
            for c in candidates:
                nc = _normalize_device(c)
                if nc.startswith("cuda:"):
                    suffix = nc.split(":", 1)[1]
                    if re.fullmatch(r"\d+", suffix):
                        wanted.add(int(suffix))
                elif nc in ("gpu",):
                    pass  # generic 'gpu' means any probed device
            if wanted:
                probed = [d for d in probed if d["index"] in wanted]
        if not probed:
            return "cpu"
        probed.sort(key=lambda d: (d["util"], d["used_ratio"], d["index"]))
        return f"cuda:{probed[0]['index']}"

    def _probe_devices(self) -> list[dict]:
        """Probe per-GPU util/memory via NVML, falling back to torch.cuda."""
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            out: list[dict] = []
            for idx in range(count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total = int(mem.total)
                    used = int(mem.used)
                    free = max(0, total - used)
                    ratio = (used / total) if total > 0 else 1.0
                    out.append({
                        "index": idx,
                        "util": util,
                        "used": used,
                        "free": free,
                        "total": total,
                        "used_ratio": ratio,
                    })
                except Exception:
                    continue
            if out:
                return out
        except Exception:
            pass

        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                return []
            count = torch.cuda.device_count()
            out = []
            for idx in range(count):
                try:
                    total = int(torch.cuda.get_device_properties(idx).total_memory)
                except Exception:
                    continue
                try:
                    used = int(torch.cuda.memory_allocated(idx))
                except Exception:
                    used = 0
                free = max(0, total - used)
                ratio = (used / total) if total > 0 else 1.0
                # No SM util via torch; use memory ratio as load proxy (scaled to 0-100).
                out.append({
                    "index": idx,
                    "util": ratio * 100.0,
                    "used": used,
                    "free": free,
                    "total": total,
                    "used_ratio": ratio,
                })
            return out
        except Exception:
            return []


class ModelRegistry:
    """Registry managing available ASR engine factories and loaded engine instances."""

    def __init__(self, preload_default: bool = True) -> None:
        self._factories: dict[str, Callable[[], BaseEngine]] = {}
        self._loaded_engines: dict[tuple[str, str], BaseEngine] = {}
        self._load_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()
        self._inference_counts: dict[tuple[str, str], int] = {}
        self._drain_events: dict[tuple[str, str], asyncio.Event] = {}
        self.scheduler = GpuScheduler()

        supported, default_mode = detect_supported_modes()
        self._supported_modes: list[str] = supported
        self._current_mode: str = default_mode

        # Pre-register mock-whisper-base
        self.register("mock-whisper-base", lambda: MockASREngine(model_name="mock-whisper-base"))
        # Real model: local Qwen3-ASR-0.6B package under backend/agent/models/
        self.register(MODEL_NAME_0_6B, lambda: Qwen3ASREngine(model_name=MODEL_NAME_0_6B))
        # Real model: local Qwen3-ASR-1.7B package under backend/agent/models/
        self.register(MODEL_NAME_1_7B, lambda: Qwen3ASREngine(model_name=MODEL_NAME_1_7B))
        # Hugging Face aliases for seamless interoperability
        self.register("Qwen/Qwen3-ASR-0.6B", lambda: Qwen3ASREngine(model_name=MODEL_NAME_0_6B))
        self.register("Qwen/Qwen3-ASR-1.7B", lambda: Qwen3ASREngine(model_name=MODEL_NAME_1_7B))

        if preload_default:
            engine = self._factories["mock-whisper-base"]()
            engine.loaded = True
            self._loaded_engines[("mock-whisper-base", "cpu")] = engine

    def get_supported_modes(self) -> list[str]:
        """List supported inference modes on this machine (e.g. ['cpu', 'gpu'])."""
        return list(self._supported_modes)

    def get_current_mode(self) -> str:
        """Get current inference work mode ('cpu', 'gpu', or 'cuda:0,cuda:1')."""
        return self._current_mode

    async def set_work_mode(self, mode: str) -> None:
        """Set target work mode; supports comma-separated multi-GPU strings.

        按需迁移：仅更新目标模式，不再 unload 全部已加载模型。
        """
        tokens = _parse_device_list(mode)
        if not tokens:
            raise ValueError(f"Mode '{mode}' is not supported on this agent (supported: {self._supported_modes})")
        for tok in tokens:
            if tok not in self._supported_modes and not re.fullmatch(r"cuda:\d+", tok) and tok not in ("cpu", "gpu", "mps"):
                raise ValueError(f"Mode '{mode}' is not supported on this agent (supported: {self._supported_modes})")
        normalized = ",".join(tokens)
        if normalized != self._current_mode:
            logger.info("Switching work mode from %s to %s", self._current_mode, normalized)
            self._current_mode = normalized

    def register(self, model_name: str, factory: Callable[[], BaseEngine]) -> None:
        """Register a model factory."""
        self._factories[model_name] = factory

    def _resolve_device(self, device: str | None) -> str:
        """Resolve effective device: explicit param wins, else derive from current mode."""
        if device is not None:
            return _normalize_device(device)
        cur = self._current_mode.strip().lower()
        if cur in ("cpu", "mps"):
            return cur
        if cur == "gpu":
            return self.scheduler.select_device()
        if "," in cur:
            return self.scheduler.select_device(candidates=_parse_device_list(cur))
        if cur.startswith("cuda:") or cur in ("cuda",):
            return _normalize_device(cur)
        return self.scheduler.select_device()

    def _get_load_lock(self, model_name: str, device: str | None = None) -> asyncio.Lock:
        """Backward-compat lock accessor; resolves to the per-(model, device) lock."""
        return self._lock_for((model_name, self._resolve_device(device)))

    def _lock_for(self, key: tuple[str, str]) -> asyncio.Lock:
        if key not in self._load_locks:
            self._load_locks[key] = asyncio.Lock()
        return self._load_locks[key]

    async def _load_model_unlocked(self, model_name: str, device: str) -> BaseEngine:
        key = (model_name, device)
        if key in self._loaded_engines:
            engine = self._loaded_engines[key]
            if not engine.loaded:
                try:
                    await engine.load(work_mode=device)
                except TypeError:
                    await engine.load()
            return engine

        if model_name not in self._factories:
            raise ValueError(f"Unknown or unregistered model: {model_name}")

        engine = self._factories[model_name]()
        try:
            await engine.load(work_mode=device)
        except TypeError:
            await engine.load()
        self._loaded_engines[key] = engine
        logger.info("Loaded model '%s' on %s (mode=%s)", model_name, device, self._current_mode)
        return engine

    @asynccontextmanager
    async def acquire_engine(self, model_name: str, device: str | None = None) -> AsyncIterator[BaseEngine]:
        """Safely acquire an engine instance for inference with active reference counting.

        Guarantees the model cannot be concurrently unloaded while the context block is executing.
        """
        effective = self._resolve_device(device)
        key = (model_name, effective)
        lock = self._lock_for(key)
        async with lock:
            engine = await self._load_model_unlocked(model_name, effective)
            async with self._registry_lock:
                self._inference_counts[key] = self._inference_counts.get(key, 0) + 1
        try:
            yield engine
        finally:
            async with self._registry_lock:
                count = self._inference_counts.get(key, 1) - 1
                if count <= 0:
                    self._inference_counts.pop(key, None)
                    event = self._drain_events.get(key)
                    if event is not None and not event.is_set():
                        event.set()
                else:
                    self._inference_counts[key] = count

    async def load_model(self, model_name: str, device: str | None = None) -> BaseEngine:
        """Load and cache an engine instance by model name with concurrency lock."""
        effective = self._resolve_device(device)
        lock = self._lock_for((model_name, effective))
        async with lock:
            return await self._load_model_unlocked(model_name, effective)

    async def unload_model(self, model_name: str, device: str | None = None, timeout: float = 30.0) -> bool:
        """Unload engine instance(s) safely after active inferences have drained.

        device=None unloads all devices for the model (backward compat);
        explicit device unloads only that (model, device) copy.
        """
        if device is not None:
            return await self._unload_single(model_name, _normalize_device(device), timeout)
        targets = [k for k in list(self._loaded_engines.keys()) if k[0] == model_name]
        # Also cover legacy string keys if any external code injected them (defensive).
        if not targets and model_name in self._loaded_engines:  # type: ignore[comparison-overlap]
            targets = [model_name]  # type: ignore[list-item]
        ok = False
        for key in targets:
            if isinstance(key, tuple):
                if await self._unload_single(key[0], key[1], timeout):
                    ok = True
            else:
                # Legacy string-keyed entry
                lock = self._get_load_lock(model_name)
                async with lock:
                    engine = self._loaded_engines.pop(key, None)  # type: ignore[call-overload]
                    if engine is not None:
                        await engine.unload()
                        ok = True
        return ok

    async def _unload_single(self, model_name: str, device: str, timeout: float) -> bool:
        key = (model_name, device)
        lock = self._lock_for(key)
        async with lock:
            drain_event: asyncio.Event | None = None
            async with self._registry_lock:
                if self._inference_counts.get(key, 0) > 0:
                    drain_event = asyncio.Event()
                    self._drain_events[key] = drain_event

            if drain_event is not None:
                logger.info(
                    "Waiting for active inferences on model '%s@%s' to drain before unloading (timeout=%.1fs)...",
                    model_name,
                    device,
                    timeout,
                )
                try:
                    await asyncio.wait_for(drain_event.wait(), timeout=timeout)
                except TimeoutError:
                    logger.warning(
                        "Timed out waiting for model '%s@%s' inferences to drain; forcing unload",
                        model_name,
                        device,
                    )
                finally:
                    async with self._registry_lock:
                        self._drain_events.pop(key, None)

            if key in self._loaded_engines:
                engine = self._loaded_engines[key]
                await engine.unload()
                del self._loaded_engines[key]
                logger.info("Unloaded model '%s@%s'", model_name, device)
                return True
            return False

    async def unload_all_models(self, timeout: float = 30.0) -> list[str]:
        """Unload and remove all currently loaded engine instances safely."""
        unloaded = []
        for key in list(self._loaded_engines.keys()):
            try:
                if isinstance(key, tuple):
                    name, dev = key
                    if await self._unload_single(name, dev, timeout=timeout):
                        unloaded.append(name)
                else:
                    # Legacy string key fallback
                    if await self.unload_model(str(key), timeout=timeout):
                        unloaded.append(str(key))
            except Exception as e:
                logger.error("Error unloading model %s: %s", key, e)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Unloaded all models: %s", unloaded)
        return unloaded

    def get_engine(self, model_name: str, device: str | None = None) -> BaseEngine | None:
        """Retrieve a currently loaded engine instance, if any."""
        if device is not None:
            return self._loaded_engines.get((model_name, _normalize_device(device)))
        for key, engine in self._loaded_engines.items():
            if isinstance(key, tuple):
                if key[0] == model_name:
                    return engine
            elif key == model_name:
                return engine  # type: ignore[return-value]
        return None

    def list_loaded_models(self) -> list[str]:
        """List names of all currently loaded models (deduplicated for heartbeat compat)."""
        seen: list[str] = []
        for key, engine in self._loaded_engines.items():
            name = key[0] if isinstance(key, tuple) else str(key)
            if engine.loaded and name not in seen:
                seen.append(name)
        return seen

    def list_loaded_models_detailed(self) -> list[str]:
        """List loaded models as 'model@device' entries."""
        out: list[str] = []
        for key, engine in self._loaded_engines.items():
            if not engine.loaded:
                continue
            if isinstance(key, tuple):
                out.append(f"{key[0]}@{key[1]}")
            else:
                out.append(str(key))
        return out

    def list_available_models(self) -> list[str]:
        """List names of all models known to the registry."""
        return list(self._factories.keys())

    def list_downloaded_models(self) -> list[str]:
        """List names of models whose weights/config are downloaded and ready on local disk."""
        downloaded = ["mock-whisper-base"]

        # Check Qwen models
        for name in [MODEL_NAME_0_6B, MODEL_NAME_1_7B]:
            model_dir = resolve_model_dir(name)
            if model_dir.is_dir():
                # Check for config.json and presence of safetensors/bin/pt weights
                config_file = model_dir / "config.json"
                if config_file.is_file():
                    has_weights = any(
                        p.suffix in {".safetensors", ".bin", ".pt"}
                        for p in model_dir.iterdir()
                        if p.is_file()
                    )
                    if has_weights:
                        downloaded.append(name)
                        if name == MODEL_NAME_0_6B:
                            downloaded.append("Qwen/Qwen3-ASR-0.6B")
                        elif name == MODEL_NAME_1_7B:
                            downloaded.append("Qwen/Qwen3-ASR-1.7B")

        return downloaded


default_registry = ModelRegistry()
