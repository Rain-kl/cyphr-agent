# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable

from .base import BaseEngine
from .mock_asr import MockASREngine
from .qwen3_asr import MODEL_NAME as QWEN3_ASR_MODEL_NAME
from .qwen3_asr import Qwen3ASREngine

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry managing available ASR engine factories and loaded engine instances."""

    def __init__(self, preload_default: bool = True) -> None:
        self._factories: dict[str, Callable[[], BaseEngine]] = {}
        self._loaded_engines: dict[str, BaseEngine] = {}

        # Pre-register mock-whisper-base
        self.register("mock-whisper-base", lambda: MockASREngine(model_name="mock-whisper-base"))
        # Real model: local Qwen3-ASR-0.6B package under backend/agent/models/
        self.register(QWEN3_ASR_MODEL_NAME, lambda: Qwen3ASREngine())

        if preload_default:
            engine = self._factories["mock-whisper-base"]()
            engine.loaded = True
            self._loaded_engines["mock-whisper-base"] = engine

    def register(self, model_name: str, factory: Callable[[], BaseEngine]) -> None:
        """Register a model factory."""
        self._factories[model_name] = factory

    async def load_model(self, model_name: str) -> BaseEngine:
        """Load and cache an engine instance by model name."""
        if model_name in self._loaded_engines:
            engine = self._loaded_engines[model_name]
            if not engine.loaded:
                await engine.load()
            return engine

        if model_name not in self._factories:
            raise ValueError(f"Unknown or unregistered model: {model_name}")

        engine = self._factories[model_name]()
        await engine.load()
        self._loaded_engines[model_name] = engine
        logger.info("Loaded model '%s'", model_name)
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
