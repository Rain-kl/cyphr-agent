# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

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


class ModelRegistry:
    """Registry managing available ASR engine factories and loaded engine instances."""

    def __init__(self, preload_default: bool = False, debug: bool | None = None) -> None:
        self._factories: dict[str, Callable[[], BaseEngine]] = {}
        self._loaded_engines: dict[str, BaseEngine] = {}
        self._load_locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()
        self._inference_counts: dict[str, int] = {}
        self._drain_events: dict[str, asyncio.Event] = {}
        self._auto_unload_minutes: int = int(os.getenv("AUTO_UNLOAD_MINUTES", os.getenv("MODEL_IDLE_UNLOAD_MINUTES", "0")))
        self._idle_unload_delay: float | None = None
        self._auto_unload_callback: Callable[[], Any] | None = None
        self._idle_unload_task: asyncio.Task[None] | None = None

        if debug is None:
            env_debug = os.getenv("DEBUG", os.getenv("AGENT_DEBUG", "")).lower()
            self._debug = env_debug in ("true", "1", "yes", "on")
        else:
            self._debug = debug

        supported, default_mode = detect_supported_modes()
        self._supported_modes: list[str] = supported
        self._current_mode: str = default_mode

        # Only register mock-whisper-base in debug mode
        if self._debug:
            self.register("mock-whisper-base", lambda: MockASREngine(model_name="mock-whisper-base"))

        # Real model: local Qwen3-ASR-0.6B package under backend/agent/models/
        self.register(MODEL_NAME_0_6B, lambda: Qwen3ASREngine(model_name=MODEL_NAME_0_6B))
        # Real model: local Qwen3-ASR-1.7B package under backend/agent/models/
        self.register(MODEL_NAME_1_7B, lambda: Qwen3ASREngine(model_name=MODEL_NAME_1_7B))
        # Hugging Face aliases for seamless interoperability
        self.register("Qwen/Qwen3-ASR-0.6B", lambda: Qwen3ASREngine(model_name=MODEL_NAME_0_6B))
        self.register("Qwen/Qwen3-ASR-1.7B", lambda: Qwen3ASREngine(model_name=MODEL_NAME_1_7B))

        if preload_default and self._debug and "mock-whisper-base" in self._factories:
            engine = self._factories["mock-whisper-base"]()
            engine.loaded = True
            self._loaded_engines["mock-whisper-base"] = engine

    def get_supported_modes(self) -> list[str]:
        """List supported inference modes on this machine (e.g. ['cpu', 'gpu'])."""
        return list(self._supported_modes)

    def get_current_mode(self) -> str:
        """Get current inference work mode ('cpu' or 'gpu')."""
        return self._current_mode

    async def set_work_mode(self, mode: str) -> None:
        """Set target work mode ('cpu' or 'gpu'). Unloads existing models if mode changed."""
        mode = mode.lower().strip()
        if mode not in self._supported_modes:
            raise ValueError(f"Mode '{mode}' is not supported on this agent (supported: {self._supported_modes})")

        if mode != self._current_mode:
            logger.info("Switching work mode from %s to %s", self._current_mode, mode)
            self._current_mode = mode
            # Unload all models so future loads use the new device mode
            await self.unload_all_models()

    def register(self, model_name: str, factory: Callable[[], BaseEngine]) -> None:
        """Register a model factory."""
        self._factories[model_name] = factory

    def _get_load_lock(self, model_name: str) -> asyncio.Lock:
        if model_name not in self._load_locks:
            self._load_locks[model_name] = asyncio.Lock()
        return self._load_locks[model_name]

    async def _load_model_unlocked(self, model_name: str) -> BaseEngine:
        if model_name in self._loaded_engines:
            engine = self._loaded_engines[model_name]
            if not getattr(engine, "loaded", False):
                if hasattr(engine, "load") and callable(engine.load):
                    try:
                        res = engine.load(work_mode=self._current_mode)
                        if asyncio.iscoroutine(res):
                            await res
                    except TypeError:
                        res = engine.load()
                        if asyncio.iscoroutine(res):
                            await res
            return engine

        if model_name not in self._factories:
            if model_name == "mock-whisper-base" and not self._debug:
                raise ValueError("Model 'mock-whisper-base' is only available in debug mode")
            raise ValueError(f"Unknown or unregistered model: {model_name}")

        engine = self._factories[model_name]()
        if not getattr(engine, "loaded", False):
            if hasattr(engine, "load") and callable(engine.load):
                try:
                    res = engine.load(work_mode=self._current_mode)
                    if asyncio.iscoroutine(res):
                        await res
                except TypeError:
                    res = engine.load()
                    if asyncio.iscoroutine(res):
                        await res
        self._loaded_engines[model_name] = engine
        logger.info("Loaded model '%s' (mode=%s)", model_name, self._current_mode)
        return engine

    def set_auto_unload_minutes(self, minutes: int) -> None:
        """Set idle duration in minutes before automatically unloading all models (0 disables)."""
        self._auto_unload_minutes = max(0, minutes)
        logger.info("Updated model registry auto_unload_minutes to %d", self._auto_unload_minutes)
        if self._auto_unload_minutes == 0 and self._idle_unload_delay is None:
            self._cancel_idle_unload_timer()
        else:
            self._check_and_schedule_idle_unload()

    def get_auto_unload_minutes(self) -> int:
        return self._auto_unload_minutes

    def set_auto_unload_callback(self, cb: Callable[[], Any] | None) -> None:
        """Register a callback (sync or async) invoked after models are auto-unloaded."""
        self._auto_unload_callback = cb

    def _cancel_idle_unload_timer(self) -> None:
        if self._idle_unload_task is not None and not self._idle_unload_task.done():
            self._idle_unload_task.cancel()
            self._idle_unload_task = None

    def _check_and_schedule_idle_unload(self) -> None:
        if self._auto_unload_minutes <= 0 and self._idle_unload_delay is None:
            self._cancel_idle_unload_timer()
            return
        if sum(self._inference_counts.values()) > 0:
            self._cancel_idle_unload_timer()
            return
        if not any(getattr(eng, "loaded", False) for eng in self._loaded_engines.values()):
            self._cancel_idle_unload_timer()
            return

        delay = self._idle_unload_delay if self._idle_unload_delay is not None else float(self._auto_unload_minutes * 60)
        self._cancel_idle_unload_timer()

        async def _idle_worker() -> None:
            try:
                await asyncio.sleep(delay)
                logger.info(
                    "ModelRegistry idle for %.1fs (threshold: %d min); auto-unloading all models to free GPU...",
                    delay,
                    self._auto_unload_minutes,
                )
                unloaded = await self.unload_all_models()
                if unloaded and self._auto_unload_callback is not None:
                    cb_res = self._auto_unload_callback()
                    if asyncio.iscoroutine(cb_res):
                        await cb_res
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("Error during model auto-unload: %s", e)

        try:
            loop = asyncio.get_running_loop()
            self._idle_unload_task = loop.create_task(_idle_worker())
        except RuntimeError:
            pass

    @asynccontextmanager
    async def acquire_engine(self, model_name: str) -> AsyncIterator[BaseEngine]:
        """Safely acquire an engine instance for inference with active reference counting.

        Guarantees the model cannot be concurrently unloaded while the context block is executing.
        """
        self._cancel_idle_unload_timer()
        lock = self._get_load_lock(model_name)
        async with lock:
            # Preemptively offload KV cache of any other loaded models that are currently idle
            async with self._registry_lock:
                for name, eng in self._loaded_engines.items():
                    if name != model_name and self._inference_counts.get(name, 0) == 0:
                        if hasattr(eng, "enter_sleep") and callable(eng.enter_sleep):
                            try:
                                eng.enter_sleep()
                            except Exception as e:
                                logger.warning("Failed to enter sleep for idle engine %s: %s", name, e)

            engine = await self._load_model_unlocked(model_name)
            async with self._registry_lock:
                self._inference_counts[model_name] = self._inference_counts.get(model_name, 0) + 1
        try:
            yield engine
        finally:
            async with self._registry_lock:
                count = self._inference_counts.get(model_name, 1) - 1
                if count <= 0:
                    self._inference_counts.pop(model_name, None)
                    event = self._drain_events.get(model_name)
                    if event is not None and not event.is_set():
                        event.set()
                else:
                    self._inference_counts[model_name] = count
            self._check_and_schedule_idle_unload()

    async def load_model(self, model_name: str) -> BaseEngine:
        """Load and cache an engine instance by model name using current work mode with concurrency lock."""
        lock = self._get_load_lock(model_name)
        async with lock:
            eng = await self._load_model_unlocked(model_name)
            self._check_and_schedule_idle_unload()
            return eng

    async def unload_model(self, model_name: str, timeout: float = 30.0) -> bool:
        """Unload and remove an engine instance safely after active inferences have drained."""
        lock = self._get_load_lock(model_name)
        async with lock:
            drain_event: asyncio.Event | None = None
            async with self._registry_lock:
                if self._inference_counts.get(model_name, 0) > 0:
                    drain_event = asyncio.Event()
                    self._drain_events[model_name] = drain_event

            if drain_event is not None:
                logger.info(
                    "Waiting for active inferences on model '%s' to drain before unloading (timeout=%.1fs)...",
                    model_name,
                    timeout,
                )
                try:
                    await asyncio.wait_for(drain_event.wait(), timeout=timeout)
                except TimeoutError:
                    logger.warning(
                        "Timed out waiting for model '%s' inferences to drain; forcing unload",
                        model_name,
                    )
                finally:
                    async with self._registry_lock:
                        self._drain_events.pop(model_name, None)

            if model_name in self._loaded_engines:
                engine = self._loaded_engines[model_name]
                await engine.unload()
                del self._loaded_engines[model_name]
                logger.info("Unloaded model '%s'", model_name)
                self._check_and_schedule_idle_unload()
                return True
            self._check_and_schedule_idle_unload()
            return False

    async def unload_all_models(self, timeout: float = 30.0) -> list[str]:
        """Unload and remove all currently loaded engine instances safely."""
        unloaded = []
        for name in list(self._loaded_engines.keys()):
            try:
                if await self.unload_model(name, timeout=timeout):
                    unloaded.append(name)
            except Exception as e:
                logger.error("Error unloading model %s: %s", name, e)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._check_and_schedule_idle_unload()
        logger.info("Unloaded all models: %s", unloaded)
        return unloaded

    def get_engine(self, model_name: str) -> BaseEngine | None:
        """Retrieve a currently loaded engine instance, if any."""
        return self._loaded_engines.get(model_name)

    def list_loaded_models(self) -> list[str]:
        """List names of all currently loaded models."""
        return [name for name, engine in self._loaded_engines.items() if engine.loaded]

    def list_available_models(self) -> list[str]:
        """List names of all models known to the registry."""
        return list(self._factories.keys())

    def list_downloaded_models(self) -> list[str]:
        """List names of models whose weights/config are downloaded and ready on local disk."""
        downloaded: list[str] = []
        if self._debug:
            downloaded.append("mock-whisper-base")

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

    def is_model_active(self) -> bool:
        """Check if any loaded model is active (doing inference or awake with allocated KV cache)."""
        for engine in self._loaded_engines.values():
            if getattr(engine, "loaded", False):
                if hasattr(engine, "is_active") and callable(engine.is_active):
                    if engine.is_active():
                        return True
                elif not getattr(engine, "is_sleeping", True):
                    return True
        return False

    def check_resources_available(self, model_name: str | None = None) -> bool:
        """Check if resources are available to load or run the model."""
        if self._current_mode == "cpu":
            return True
        if model_name and model_name in self._loaded_engines:
            eng = self._loaded_engines[model_name]
            if hasattr(eng, "check_resources_available") and callable(eng.check_resources_available):
                return eng.check_resources_available()
        # If any loaded engine is active, resources are available
        if self.is_model_active():
            return True
        # Check loaded engines first
        for eng in self._loaded_engines.values():
            if hasattr(eng, "check_resources_available") and callable(eng.check_resources_available):
                if eng.check_resources_available():
                    return True
        # If no model loaded, check if any GPU has MIN_LOAD_VRAM_MB
        from .qwen3_asr import resolve_devices

        try:
            devs = resolve_devices("gpu")
            return len(devs) > 0
        except Exception:
            return True


default_registry = ModelRegistry()

