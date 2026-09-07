# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ..core.engine import BaseModelEngine
from ..resources.hardware import resolve_devices
from ..resources.vram import calculate_dynamic_gpu_utilization

logger = logging.getLogger(__name__)

MODEL_NAME_HY_MT2 = "tencent/Hy-MT2-1.8B"
ALIAS_HY_MT2_SHORT = "hy-mt2-1.8b"
ALIAS_HY_MT2_CAMEL = "Hy-MT2-1.8B"
HF_REPO_ID = "tencent/Hy-MT2-1.8B"

LANG_MAP = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
}


def resolve_model_dir(model_name: str = MODEL_NAME_HY_MT2) -> Path:
    """Resolve local model directory for Hy-MT2-1.8B."""
    if env := os.getenv("HY_MT2_1_8B_MODEL_DIR"):
        return Path(env)
    if env := os.getenv("HY_MT2_MODEL_DIR"):
        return Path(env)
    base_dir = Path(__file__).resolve().parent.parent.parent / "models"
    for candidate in ["hy-mt2-1.8b", "Hy-MT2-1.8B", "tencent/Hy-MT2-1.8B"]:
        p = base_dir / candidate
        if p.is_dir():
            return p
    return base_dir / "hy-mt2-1.8b"


def format_translation_prompt(
    text: str,
    target_lang: str = "zh",
    source_lang: str | None = None,
) -> str:
    """Format input text with translation instructions for Tencent Hy-MT2-1.8B.

    If the text already contains explicit translation instructions, it is returned as-is.
    """
    clean_text = text.strip()
    if not clean_text:
        return ""

    # Check if translation instruction is already provided
    lower = clean_text.lower()
    if lower.startswith("translate ") or clean_text.startswith("将以下") or clean_text.startswith("翻译"):
        return clean_text

    norm_target = target_lang.lower().strip()
    target_lang_name = LANG_MAP.get(norm_target, target_lang)

    # Use Chinese instruction for Chinese target, English instruction for others
    if norm_target in ("zh", "chinese", "中文"):
        return f"将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{clean_text}"
    return f"Translate the following text into {target_lang_name}. Note that you should only output the translated result without any additional explanation:\n\n{clean_text}"


class HyMT2Engine(BaseModelEngine):
    """Tencent Hy-MT2-1.8B translation engine with real-time streaming inference support."""

    supports_concurrent_inference = True

    def __init__(
        self,
        model_name: str = MODEL_NAME_HY_MT2,
        model_dir: str | Path | None = None,
    ) -> None:
        super().__init__(model_name)
        self.model_dir = Path(model_dir) if model_dir else resolve_model_dir(model_name)
        self.gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.60"))
        self.max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
        self._engine: Any = None
        self._mock_mode = False
        self._active_requests: set[str] = set()
        self._aborted_requests: set[str] = set()
        self._lock = asyncio.Lock()

    def _resolve_model_source(self) -> str:
        if self.model_dir.joinpath("config.json").is_file():
            return str(self.model_dir.resolve())
        return HF_REPO_ID

    async def load(self, work_mode: str = "gpu") -> None:
        """Load weights using vLLM AsyncLLMEngine or initialize mock engine."""
        env_mock = os.getenv("HY_MT2_MOCK", "").lower() in ("1", "true", "yes")
        env_debug = os.getenv("DEBUG", os.getenv("AGENT_DEBUG", "")).lower() in ("1", "true", "yes")

        devices = resolve_devices(work_mode)
        if env_mock or (env_debug and not self.model_dir.joinpath("config.json").is_file()):
            logger.info("Initializing HyMT2Engine in mock mode for '%s'", self.model_name)
            self._mock_mode = True
            self.loaded = True
            return

        model_source = self._resolve_model_source()
        logger.info(
            "Initializing HyMT2Engine for %s from %s (devices=%s)...",
            self.model_name,
            model_source,
            devices,
        )

        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            free_ratio = float(os.getenv("VLLM_FREE_MEMORY_RATIO", "0.70"))
            effective_gpu_util = calculate_dynamic_gpu_utilization(
                device_id=devices[0] if devices else 0,
                gpu_memory_utilization=self.gpu_memory_utilization,
                free_ratio=free_ratio,
            )

            engine_args = AsyncEngineArgs(
                model=model_source,
                gpu_memory_utilization=effective_gpu_util,
                max_model_len=self.max_model_len,
                enforce_eager=True,
                disable_log_stats=True,
                trust_remote_code=True,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.loaded = True
            logger.info("Successfully initialized vLLM AsyncLLMEngine for Hy-MT2")
        except Exception as e:
            if env_debug or "cpu" in devices or not devices:
                logger.warning("vLLM init failed (%s); falling back to mock mode", e)
                self._mock_mode = True
                self.loaded = True
            else:
                logger.exception("Failed to initialize vLLM AsyncLLMEngine for Hy-MT2: %s", e)
                raise

    async def unload(self) -> None:
        """Unload engine and release resources."""
        self._engine = None
        self.loaded = False
        self._active_requests.clear()
        self._aborted_requests.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Unloaded HyMT2Engine '%s'", self.model_name)

    def is_active(self) -> bool:
        return len(self._active_requests) > 0

    async def abort(self, request_id: str) -> None:
        """Abort an active streaming generation request."""
        self._aborted_requests.add(request_id)
        if self._engine is not None and hasattr(self._engine, "abort"):
            try:
                await self._engine.abort(request_id)
            except Exception as e:
                logger.debug("Error aborting vLLM request %s: %s", request_id, e)

    async def generate_stream(
        self,
        request_id: str,
        prompt: str,
        target_lang: str = "zh",
        source_lang: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        messages: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream translation generation chunks (delta, finish_reason)."""
        if not self.loaded:
            raise RuntimeError(f"Model '{self.model_name}' is not loaded")

        self._active_requests.add(request_id)
        self._aborted_requests.discard(request_id)

        # Extract text from messages if provided
        input_text = prompt
        if not input_text and messages:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    input_text = str(msg.get("content", ""))
                    break

        formatted_prompt = format_translation_prompt(
            input_text, target_lang=target_lang, source_lang=source_lang
        )

        try:
            if self._mock_mode or self._engine is None:
                # Mock streaming generator
                mock_translation = self._generate_mock_translation(input_text, target_lang)
                words = mock_translation.split(" ")
                for idx, word in enumerate(words):
                    if request_id in self._aborted_requests:
                        logger.info("Request %s was aborted by client", request_id)
                        return
                    chunk = word + (" " if idx < len(words) - 1 else "")
                    await asyncio.sleep(0.03)
                    yield chunk, None
                yield "", "stop"
                return

            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.6,
                top_k=20,
                repetition_penalty=1.05,
            )

            results_generator = self._engine.generate(
                prompt=formatted_prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            prev_len = 0
            async for request_output in results_generator:
                if request_id in self._aborted_requests:
                    break
                text = request_output.outputs[0].text
                delta = text[prev_len:]
                prev_len = len(text)
                finish_reason = request_output.outputs[0].finish_reason
                yield delta, finish_reason

        finally:
            self._active_requests.discard(request_id)
            self._aborted_requests.discard(request_id)

    @staticmethod
    def _generate_mock_translation(text: str, target_lang: str) -> str:
        """Generate high-quality mock translation for automated testing."""
        clean = text.strip()
        target = target_lang.lower().strip()
        if target in ("zh", "chinese", "中文"):
            if "hello" in clean.lower():
                return "你好，世界！"
            return f"【译文】{clean}"
        elif target in ("en", "english", "英语"):
            if "你好" in clean:
                return "Hello, world!"
            return f"[Translated: {clean}]"
        return f"[{target}: {clean}]"
