# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable

from .base import BaseEngine
from .mock_asr import MockASREngine
from .qwen3_asr import (
    MODEL_NAME_0_6B,
    MODEL_NAME_1_7B,
    Qwen3ASREngine,
)

logger = logging.getLogger(__name__)


def detect_supported_modes() -> tuple[list[str], str]:
    """Detect available acceleration hardware."""
    has_gpu = False
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            has_gpu = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            has_gpu = True
    except Exception:
        pass

    if has_gpu:
        return ["cpu", "gpu"], "gpu"
    return ["cpu"], "cpu"


class ModelRegistry:
    """Registry managing available ASR engine factories and loaded engine instances."""

    def __init__(self, preload_default: bool = True) -> None:
        self._factories: dict[str, Callable[[], BaseEngine]] = {}
        self._loaded_engines: dict[str, BaseEngine] = {}

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

    async def load_model(self, model_name: str) -> BaseEngine:
        """Load and cache an engine instance by model name using current work mode."""
        if model_name in self._loaded_engines:
            engine = self._loaded_engines[model_name]
            if not engine.loaded:
                try:
                    await engine.load(work_mode=self._current_mode)
                except TypeError:
                    await engine.load()
            return engine

        if model_name not in self._factories:
            raise ValueError(f"Unknown or unregistered model: {model_name}")

        engine = self._factories[model_name]()
        try:
            await engine.load(work_mode=self._current_mode)
        except TypeError:
            await engine.load()
        self._loaded_engines[model_name] = engine
        logger.info("Loaded model '%s' (mode=%s)", model_name, self._current_mode)
        return engine

    async def unload_model(self, model_name: str) -> bool:
        """Unload and remove an engine instance."""
        if model_name in self._loaded_engines:
            engine = self._loaded_engines[model_name]
            await engine.unload()
            del self._loaded_engines[model_name]
            logger.info("Unloaded model '%s'", model_name)
            return True
        return False

    async def unload_all_models(self) -> list[str]:
        """Unload and remove all currently loaded engine instances."""
        unloaded = []
        for name in list(self._loaded_engines.keys()):
            engine = self._loaded_engines[name]
            try:
                await engine.unload()
                unloaded.append(name)
            except Exception as e:
                logger.error("Error unloading model %s: %s", name, e)
            finally:
                self._loaded_engines.pop(name, None)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
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


default_registry = ModelRegistry()
