# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .base import BaseEngine

logger = logging.getLogger(__name__)

MODEL_NAME = "qwen3-asr-0.6b"
CHUNK_SEC = 30.0
TARGET_SR = 16000

# ISO code -> Qwen3-ASR language name. Unknown values pass through untouched.
LANG_MAP = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}


def resolve_model_dir(model_name: str = MODEL_NAME) -> Path:
    """Local model package dir: $QWEN3_ASR_MODEL_DIR or backend/agent/models/<name>/."""
    if env := os.getenv("QWEN3_ASR_MODEL_DIR"):
        return Path(env)
    return Path(__file__).resolve().parent.parent.parent / "models" / model_name


class Qwen3ASREngine(BaseEngine):
    """Qwen3-ASR local inference engine yielding OpenAI verbose_json."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_dir: str | Path | None = None,
        max_new_tokens: int = 1024,
    ) -> None:
        super().__init__(model_name)
        self.model_dir = Path(model_dir) if model_dir else resolve_model_dir(model_name)
        self.max_new_tokens = max_new_tokens
        self._model = None

    async def load(self) -> None:
        """Load weights (blocking torch work runs in executor via caller)."""
        import asyncio

        await asyncio.to_thread(self._load_blocking)
        self.loaded = True

    def _load_blocking(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        if not self.model_dir.joinpath("config.json").is_file():
            raise FileNotFoundError(f"Model package missing in {self.model_dir}")
        if torch.cuda.is_available():
            device, dtype = "cuda:0", torch.bfloat16
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            batch_size = int(os.getenv("QWEN3_ASR_BATCH_SIZE", "16"))
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
            batch_size = int(os.getenv("QWEN3_ASR_BATCH_SIZE", "4"))
        else:
            device, dtype = "cpu", torch.float32
            batch_size = int(os.getenv("QWEN3_ASR_BATCH_SIZE", "1"))

        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": device,
            "max_inference_batch_size": batch_size,
            "max_new_tokens": self.max_new_tokens,
        }
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            load_kwargs["attn_implementation"] = "sdpa"

        logger.info(
            "Loading %s from %s on %s (%s, batch_size=%d, attn=%s)",
            self.model_name,
            self.model_dir,
            device,
            dtype,
            batch_size,
            load_kwargs.get("attn_implementation", "sdpa"),
        )
        self._model = Qwen3ASRModel.from_pretrained(
            str(self.model_dir),
            **load_kwargs,
        )

    async def unload(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        self.loaded = False

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        task_type: str = "transcribe",
        log_callback: Callable[[int, str], Any] | None = None,
    ) -> dict[str, Any]:
        """Sync (runs in job_runner's executor): ffmpeg -> chunk -> transcribe -> verbose_json."""
        from qwen_asr.inference.utils import (
            SAMPLE_RATE,
            normalize_audio_input,
            split_audio_into_chunks,
        )

        if self._model is None or not self.loaded:
            raise RuntimeError(f"Model '{self.model_name}' is not loaded")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        lang = LANG_MAP.get(language, language) if language else None

        def _log(progress: int, message: str) -> None:
            if log_callback is None:
                return
            try:
                log_callback(progress, message)
            except Exception as e:
                logger.warning("log_callback failed: %s", e)

        _log(20, "Loading and decoding audio file...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            r = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-i",
                    audio_path,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(TARGET_SR),
                    "-c:a",
                    "pcm_s16le",
                    wav_path,
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg failed for {audio_path}: {r.stderr.strip()}")
            _log(50, "Preprocessing audio chunks and extracting features...")
            wav = normalize_audio_input(wav_path)
            chunks = split_audio_into_chunks(wav, SAMPLE_RATE, max_chunk_sec=CHUNK_SEC)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        import torch

        duration = len(wav) / SAMPLE_RATE
        segments, texts, langs = [], [], []
        total = len(chunks)
        batch_size = int(os.getenv("QWEN3_ASR_BATCH_SIZE", "16" if torch.cuda.is_available() else "1"))
        batch_size = max(1, batch_size)

        with torch.inference_mode():
            for i in range(0, total, batch_size):
                batch = chunks[i : i + batch_size]
                cur_end = min(i + batch_size, total)
                _log(50 + int(45 * cur_end / max(total, 1)), f"Running ASR batch inference ({cur_end}/{total})...")

                audio_inputs = [(cwav, SAMPLE_RATE) for cwav, _ in batch]
                try:
                    outs = self._model.transcribe(audio=audio_inputs, language=lang)
                except Exception:
                    outs = [
                        self._model.transcribe(audio=(cwav, SAMPLE_RATE), language=lang)[0]
                        for cwav, _ in batch
                    ]

                for (cwav, offset), out in zip(batch, outs):
                    text = (out.text or "").strip()
                    if out.language and out.language not in langs:
                        langs.append(out.language)
                    if text:
                        texts.append(text)
                        segments.append(
                            {
                                "id": len(segments),
                                "seek": int(offset * 100),
                                "start": round(offset, 2),
                                "end": round(offset + len(cwav) / SAMPLE_RATE, 2),
                                "text": text,
                            }
                        )
        _log(100, "Aligning timestamps and finalizing transcript...")
        return {
            "task": task_type or "transcribe",
            "language": langs[0] if langs else (lang or ""),
            "duration": round(duration, 2),
            "text": " ".join(texts),
            "segments": segments,
        }
