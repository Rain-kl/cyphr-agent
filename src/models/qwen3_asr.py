# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseEngine

logger = logging.getLogger(__name__)

MODEL_NAME_0_6B = "qwen3-asr-0.6b"
MODEL_NAME_1_7B = "qwen3-asr-1.7b"
MODEL_NAME = MODEL_NAME_0_6B
CHUNK_SEC = 30.0
TARGET_SR = 16000

HF_REPO_MAP = {
    MODEL_NAME_0_6B: "Qwen/Qwen3-ASR-0.6B",
    MODEL_NAME_1_7B: "Qwen/Qwen3-ASR-1.7B",
}

DEFAULT_MAX_NEW_TOKENS = 512
MIN_MAX_NEW_TOKENS = 128
MAX_MAX_NEW_TOKENS = 1024

DEFAULT_MAX_BATCH_SECONDS = 180.0

SILENCE_THRESHOLD_DB = -40.0

FFMPEG_BIN = "ffmpeg"
FFMPEG_BASE_ARGS: tuple[str, ...] = ("-y", "-hide_banner", "-nostdin", "-loglevel", "error")
FFMPEG_CONVERT_ARGS: tuple[str, ...] = (
    "-vn", "-ac", "1", "-ar", str(TARGET_SR), "-f", "s16le", "pipe:1"
)


def resolve_max_new_tokens(explicit: int | None = None) -> int:
    """Resolve max_new_tokens: env wins, else explicit/default, clamped 128~1024."""
    raw = os.getenv("QWEN3_ASR_MAX_NEW_TOKENS")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(MIN_MAX_NEW_TOKENS, min(MAX_MAX_NEW_TOKENS, int(float(str(raw).strip()))))
        except (TypeError, ValueError):
            pass
    if explicit is None:
        return DEFAULT_MAX_NEW_TOKENS
    try:
        return max(MIN_MAX_NEW_TOKENS, min(MAX_MAX_NEW_TOKENS, int(explicit)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_NEW_TOKENS


def resolve_max_batch_seconds(explicit: float | None = None) -> float:
    """Resolve per-batch audio budget in seconds: env $QWEN3_ASR_MAX_BATCH_SECONDS wins."""
    raw = os.getenv("QWEN3_ASR_MAX_BATCH_SECONDS")
    if raw is not None and str(raw).strip() != "":
        try:
            v = float(str(raw).strip())
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    if explicit is not None:
        try:
            v = float(explicit)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return DEFAULT_MAX_BATCH_SECONDS


def should_skip_silence() -> bool:
    return os.getenv("QWEN3_ASR_SKIP_SILENCE", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def is_silent_chunk(audio: Any, threshold_db: float = SILENCE_THRESHOLD_DB) -> bool:
    """Lightweight energy VAD: True when RMS < threshold_db (default -40dB)."""
    try:
        import math

        arr = np.asarray(audio, dtype=np.float64)
        if arr.size == 0:
            return True
        rms = float(np.sqrt(np.mean(arr * arr)))
        if rms <= 1e-9:
            return True
        return (20.0 * math.log10(rms)) < threshold_db
    except Exception:
        return False


def is_oom_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower().replace("_", "")
    if "outofmemory" in name or name == "oomerror":
        return True
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or " oom" in msg
        or msg.startswith("oom")
        or "cudnn out of memory" in msg
    )


def split_chunks_by_seconds(
    chunks: list, max_batch_seconds: float, max_count: int | None = None
) -> list:
    """Pack chunks into batches capped by max_batch_seconds (and optional max_count)."""
    if not max_batch_seconds or max_batch_seconds <= 0:
        max_batch_seconds = DEFAULT_MAX_BATCH_SECONDS
    batches: list = []
    cur: list = []
    cur_sec = 0.0
    for ch in chunks:
        cwav, _ = ch
        try:
            dur = len(cwav) / TARGET_SR
        except Exception:
            dur = CHUNK_SEC
        if cur and (
            cur_sec + dur > max_batch_seconds
            or (max_count is not None and len(cur) >= max_count)
        ):
            batches.append(cur)
            cur = []
            cur_sec = 0.0
        cur.append(ch)
        cur_sec += dur
    if cur:
        batches.append(cur)
    return batches


def should_enable_cudnn_benchmark() -> bool:
    return os.getenv("QWEN3_ASR_CUDNN_BENCHMARK", "0").strip().lower() in ("1", "true", "yes", "on")


def should_enable_torch_compile() -> bool:
    return os.getenv("QWEN3_ASR_TORCH_COMPILE", "0").strip().lower() in ("1", "true", "yes", "on")


def resolve_device_and_dtype(work_mode: str = "gpu") -> tuple[str, Any, int]:
    """Resolve target inference device, dtype, and default batch size.

    Supports:
    1. Explicit env override via $QWEN3_ASR_DEVICE (e.g. 'cuda:1', 'cuda', 'mps', 'cpu')
    2. Multi-GPU dynamic index via $CUDA_DEVICE_INDEX (e.g. '0', '1', ...)
    3. work_mode parameter ('cpu', 'gpu', or specific device like 'cuda:0')
    4. Auto-detect CUDA with bf16/fp16 support, MPS on Apple Silicon, or CPU fallback.
    """
    import torch

    env_device = os.getenv("QWEN3_ASR_DEVICE", "").strip().lower()
    req_device = env_device or work_mode.strip().lower()

    if req_device == "cpu":
        batch_size = int(os.getenv("QWEN3_ASR_BATCH_SIZE", "1"))
        return "cpu", torch.float32, batch_size

    if req_device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested (%s) but torch.cuda is not available; falling back to CPU", req_device)
            return "cpu", torch.float32, int(os.getenv("QWEN3_ASR_BATCH_SIZE", "1"))
        device = req_device if ":" in req_device else "cuda:0"
        use_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        batch_size = int(os.getenv("QWEN3_ASR_BATCH_SIZE", "16"))
        return device, dtype, batch_size

    if req_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            batch_size = int(os.getenv("QWEN3_ASR_BATCH_SIZE", "4"))
            return "mps", torch.float16, batch_size
        logger.warning("MPS requested but torch.backends.mps is not available; falling back to CPU")
        return "cpu", torch.float32, int(os.getenv("QWEN3_ASR_BATCH_SIZE", "1"))

    # Generic "gpu" request
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device_idx = int(os.getenv("CUDA_DEVICE_INDEX", "0"))
        device_idx = min(max(0, device_idx), torch.cuda.device_count() - 1)
        device = f"cuda:{device_idx}"
        use_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        batch_size = int(os.getenv("QWEN3_ASR_BATCH_SIZE", "16"))
        return device, dtype, batch_size

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        batch_size = int(os.getenv("QWEN3_ASR_BATCH_SIZE", "4"))
        return "mps", torch.float16, batch_size

    return "cpu", torch.float32, int(os.getenv("QWEN3_ASR_BATCH_SIZE", "1"))

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
    """Local model package dir:
    1. Dedicated env: $QWEN3_ASR_1_7B_MODEL_DIR or $QWEN3_ASR_0_6B_MODEL_DIR
    2. Generic env: $QWEN3_ASR_MODEL_DIR
    3. Default path: backend/agent/models/<model_name>/
    """
    clean_name = model_name.lower().replace("-", "_").replace(".", "_")
    if env := os.getenv(f"{clean_name.upper()}_MODEL_DIR"):
        return Path(env)
    if env := os.getenv("QWEN3_ASR_MODEL_DIR"):
        return Path(env)
    return Path(__file__).resolve().parent.parent.parent / "models" / model_name.lower()


class Qwen3ASREngine(BaseEngine):
    """Qwen3-ASR local inference engine yielding OpenAI verbose_json."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_dir: str | Path | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        super().__init__(model_name)
        self.model_dir = Path(model_dir) if model_dir else resolve_model_dir(model_name)
        self.max_new_tokens = resolve_max_new_tokens(max_new_tokens)
        self._model = None
        # Per-instance lock: replicas on different GPUs infer concurrently;
        # same-replica jobs still serialize (transformers backend is not thread-safe).
        self._inference_lock = threading.Lock()

    async def load(self, work_mode: str = "gpu") -> None:
        """Load weights (blocking torch work runs in executor via caller)."""
        import asyncio

        await asyncio.to_thread(self._load_blocking, work_mode)
        self.loaded = True

    def _load_blocking(self, work_mode: str = "gpu") -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        model_source: str
        if self.model_dir.joinpath("config.json").is_file():
            model_source = str(self.model_dir)
        elif self.model_name.lower() in HF_REPO_MAP:
            # Fallback to Hugging Face repo ID if offline weights directory is missing
            model_source = HF_REPO_MAP[self.model_name.lower()]
            logger.info(
                "Local weights not found at %s; loading directly via Hugging Face ID: %s",
                self.model_dir,
                model_source,
            )
        else:
            raise FileNotFoundError(
                f"Model package missing in {self.model_dir}. "
                f"Please download model '{self.model_name}' first using cyphr-installer."
            )

        device, dtype, batch_size = resolve_device_and_dtype(work_mode)

        if "cuda" in device:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # cudnn.benchmark only helps fixed-shape batches; default off.
            torch.backends.cudnn.benchmark = should_enable_cudnn_benchmark()

        self.max_new_tokens = resolve_max_new_tokens(self.max_new_tokens)
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
            model_source,
            device,
            dtype,
            batch_size,
            load_kwargs.get("attn_implementation", "sdpa"),
        )
        self._model = Qwen3ASRModel.from_pretrained(
            model_source,
            **load_kwargs,
        )
        if should_enable_torch_compile():
            try:
                self._model = torch.compile(self._model, backend="inductor", mode="reduce-overhead")
            except Exception as e:
                logger.warning("torch.compile failed, falling back to eager: %s", e)

    def _transcribe_batch_with_oom_retry(
        self,
        batch: list,
        lang: str | None,
        batch_seconds: float,
        batch_size_cap: int,
    ) -> list:
        """Transcribe one duration-batch; on OOM halve budget and retry, then single-chunk."""
        from qwen_asr.inference.utils import SAMPLE_RATE as _SR

        audio_inputs = [(cwav, _SR) for cwav, _ in batch]
        try:
            return self._model.transcribe(audio=audio_inputs, language=lang)
        except Exception as batch_err:
            if not is_oom_error(batch_err):
                logger.warning("Batch failed, falling back to single-chunk: %s", batch_err)
                return [
                    self._model.transcribe(audio=(cwav, _SR), language=lang)[0]
                    for cwav, _ in batch
                ]
            logger.warning(
                "Batch OOM (size=%d), halving batch_seconds and retrying: %s", len(batch), batch_err
            )

        budget = batch_seconds
        sub_batches: list = [batch]
        for _ in range(3):
            budget = budget / 2.0
            sub_batches = split_chunks_by_seconds(batch, budget, batch_size_cap)
            if len(sub_batches) <= 1 and len(batch) > 1:
                mid = len(batch) // 2
                sub_batches = [batch[:mid], batch[mid:]]
            try:
                outs: list = []
                for sub in sub_batches:
                    sub_inputs = [(cwav, _SR) for cwav, _ in sub]
                    outs.extend(self._model.transcribe(audio=sub_inputs, language=lang))
                return outs
            except Exception as retry_err:
                if not is_oom_error(retry_err):
                    logger.warning("Retry failed (non-OOM), single-chunk fallback: %s", retry_err)
                    break
                logger.warning(
                    "Retry OOM (budget=%.1fs, parts=%d): %s", budget, len(sub_batches), retry_err
                )
                continue

        logger.warning("OOM retries exhausted, falling back to single-chunk")
        return [
            self._model.transcribe(audio=(cwav, _SR), language=lang)[0]
            for cwav, _ in batch
        ]

    async def unload(self) -> None:
        if self._model is not None:
            self._model = None

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                logger.info(
                    "CUDA memory after model unload: allocated=%.2fMB, reserved=%.2fMB",
                    allocated,
                    reserved,
                )
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            logger.warning("Failed during CUDA cache cleanup: %s", e)

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
        # Fast path: any 16kHz mono audio readable by soundfile skips ffmpeg
        # (WAV/MP3/FLAC/OGG/M4A all supported via libsndfile; only sr+channels checked).
        is_standard_audio = False
        try:
            import soundfile as sf
            with sf.SoundFile(audio_path) as info:
                if info.samplerate == TARGET_SR and info.channels == 1:
                    is_standard_audio = True
        except Exception:
            is_standard_audio = False

        if is_standard_audio:
            _log(50, "Direct audio feed detected (standard 16kHz mono audio), skipping ffmpeg conversion...")
            wav = normalize_audio_input(audio_path)
            chunks = split_audio_into_chunks(wav, SAMPLE_RATE, max_chunk_sec=CHUNK_SEC)
        else:
            try:
                r = subprocess.run(
                    [
                        FFMPEG_BIN,
                        *FFMPEG_BASE_ARGS,
                        "-i",
                        audio_path,
                        *FFMPEG_CONVERT_ARGS,
                    ],
                    capture_output=True,
                )
                if r.returncode != 0:
                    err_msg = r.stderr.decode("utf-8", errors="replace").strip() if r.stderr else "unknown error"
                    raise RuntimeError(f"ffmpeg failed for {audio_path}: {err_msg}")
                _log(50, "Preprocessing audio chunks and extracting features...")
                pcm_data = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
                wav = normalize_audio_input((pcm_data, TARGET_SR))
                chunks = split_audio_into_chunks(wav, SAMPLE_RATE, max_chunk_sec=CHUNK_SEC)
            except FileNotFoundError as fnf_err:
                raise RuntimeError(
                    "未在当前系统中检测到 ffmpeg 可执行程序。请安装 ffmpeg 并加入系统 PATH，"
                    "或者使用 cyphr 命令行客户端 (CLI) 在上传前自动完成音频格式转换。"
                ) from fnf_err

        import torch

        duration = len(wav) / SAMPLE_RATE
        segments, texts, langs = [], [], []
        try:
            batch_size_cap = int(os.getenv("QWEN3_ASR_BATCH_SIZE", "16" if torch.cuda.is_available() else "1"))
        except (TypeError, ValueError):
            batch_size_cap = 1
        batch_size_cap = max(1, batch_size_cap)
        max_batch_seconds = resolve_max_batch_seconds()

        if should_skip_silence():
            kept = [(cwav, offset) for cwav, offset in chunks if not is_silent_chunk(cwav)]
            skipped = len(chunks) - len(kept)
            if skipped:
                _log(50, f"Skipped {skipped} silent chunk(s) via energy VAD...")
            chunks = kept

        total = len(chunks)
        if total == 0:
            _log(100, "Aligning timestamps and finalizing transcript...")
            return {
                "task": task_type or "transcribe",
                "language": langs[0] if langs else (lang or ""),
                "duration": round(duration, 2),
                "text": "",
                "segments": segments,
            }
        batches = split_chunks_by_seconds(chunks, max_batch_seconds, batch_size_cap)

        _log(48, "Waiting for model inference slot...")
        with self._inference_lock:
            with torch.inference_mode():
                done = 0
                for batch in batches:
                    done += len(batch)
                    _log(
                        50 + int(45 * done / max(total, 1)),
                        f"Running ASR batch inference ({done}/{total})...",
                    )

                    outs = self._transcribe_batch_with_oom_retry(
                        batch, lang, max_batch_seconds, batch_size_cap
                    )

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
