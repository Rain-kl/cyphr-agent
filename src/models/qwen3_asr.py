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


def resolve_devices_and_configs(
    work_mode: str = "gpu",
    model_name: str = MODEL_NAME,
) -> list[tuple[str, Any, int]]:
    """Resolve target inference devices, dtypes, and optimal batch sizes.

    Features:
    1. Multi-GPU allocation across all healthy CUDA devices.
    2. Dynamic VRAM estimation and batch size calculation to maximize throughput while avoiding OOM.
    3. Respects explicit user overrides: $QWEN3_ASR_DEVICE, $CUDA_DEVICE_INDEX, $QWEN3_ASR_BATCH_SIZE.
    4. Safe fallback to single GPU, MPS, or CPU if VRAM or hardware is insufficient.
    """
    import torch

    env_device = os.getenv("QWEN3_ASR_DEVICE", "").strip().lower()
    explicit_idx = os.getenv("CUDA_DEVICE_INDEX", "").strip()

    # 1. If explicit single device is requested via env or work_mode contains ":"
    if env_device and env_device not in ("all", "multi", "gpu", "cuda"):
        dev, dt, bs = resolve_device_and_dtype(env_device)
        return [(dev, dt, bs)]

    if explicit_idx:
        dev, dt, bs = resolve_device_and_dtype(work_mode)
        return [(dev, dt, bs)]

    req_device = work_mode.strip().lower()
    if req_device == "cpu":
        return [resolve_device_and_dtype("cpu")]

    if req_device.startswith("cuda") and ":" in req_device:
        return [resolve_device_and_dtype(req_device)]

    if req_device == "mps":
        return [resolve_device_and_dtype("mps")]

    # 2. Multi-GPU auto-discovery for generic "gpu" / "cuda"
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        use_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16

        # Estimate minimum VRAM required to load model weights (bytes)
        name_lower = model_name.lower()
        if "1.7b" in name_lower:
            min_model_vram = int(3.8 * 1024 * 1024 * 1024)
        elif "0.6b" in name_lower:
            min_model_vram = int(1.8 * 1024 * 1024 * 1024)
        else:
            min_model_vram = int(2.0 * 1024 * 1024 * 1024)

        devices: list[tuple[str, Any, int]] = []
        user_bs = os.getenv("QWEN3_ASR_BATCH_SIZE")

        for idx in range(torch.cuda.device_count()):
            try:
                free_bytes, _ = torch.cuda.mem_get_info(idx)
            except Exception:
                free_bytes = 0

            if free_bytes >= min_model_vram:
                if user_bs:
                    bs = max(1, int(user_bs))
                else:
                    # Dynamically calculate batch size based on free VRAM headroom
                    usable_vram = max(0, free_bytes - min_model_vram) * 0.70
                    # ~180MB activation memory per 30s chunk in batch
                    bs = max(2, min(36, 16 + int(usable_vram / (180 * 1024 * 1024))))
                devices.append((f"cuda:{idx}", dtype, bs))
            else:
                logger.warning(
                    "Skipping GPU %d for %s due to low free VRAM (%.1fMB < %.1fMB)",
                    idx,
                    model_name,
                    free_bytes / (1024 * 1024),
                    min_model_vram / (1024 * 1024),
                )

        if devices:
            return devices

        # If no GPU met full min_model_vram, try device with largest free memory
        max_idx = -1
        max_free = 0
        for idx in range(torch.cuda.device_count()):
            try:
                f, _ = torch.cuda.mem_get_info(idx)
                if f > max_free:
                    max_free = f
                    max_idx = idx
            except Exception:
                pass

        if max_idx >= 0 and max_free > 1024 * 1024 * 1024:
            logger.warning("No GPU met optimal VRAM threshold; falling back to cuda:%d with minimal batch size", max_idx)
            return [(f"cuda:{max_idx}", dtype, 2)]

        logger.warning("All GPUs insufficient for %s; falling back to CPU", model_name)
        return [resolve_device_and_dtype("cpu")]

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return [resolve_device_and_dtype("mps")]

    return [resolve_device_and_dtype("cpu")]


class WorkerInstance:
    """An inference worker instance bound to a specific hardware device."""

    def __init__(
        self,
        device: str,
        dtype: Any,
        batch_size: int,
        model: Any,
        device_idx: int = -1,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.batch_size = max(1, batch_size)
        self.model = model
        self.device_idx = device_idx

    def transcribe_chunks(
        self,
        indexed_chunks: list[tuple[int, tuple[np.ndarray, float]]],
        lang: str | None = None,
        progress_cb: Callable[[int], None] | None = None,
    ) -> list[tuple[int, np.ndarray, float, str, str | None]]:
        """Transcribe assigned chunks with dynamic batching and OOM self-healing backoff.

        Returns:
            list of (chunk_idx, cwav, offset, text, language)
        """
        import torch

        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        results: list[tuple[int, np.ndarray, float, str, str | None]] = []
        i = 0
        cur_batch_size = self.batch_size

        while i < len(indexed_chunks):
            batch = indexed_chunks[i : i + cur_batch_size]
            audio_inputs = [(cwav, TARGET_SR) for _, (cwav, _) in batch]

            try:
                with torch.inference_mode():
                    outs = self.model.transcribe(audio=audio_inputs, language=lang)
            except torch.cuda.OutOfMemoryError as oom_err:
                logger.warning(
                    "CUDA OOM on %s with batch_size=%d. Releasing cache and halving batch size...",
                    self.device,
                    cur_batch_size,
                )
                if "cuda" in self.device:
                    torch.cuda.empty_cache()
                gc.collect()

                if cur_batch_size > 1:
                    cur_batch_size = max(1, cur_batch_size // 2)
                    logger.info("Retrying with batch_size=%d on %s", cur_batch_size, self.device)
                    continue
                raise oom_err
            except Exception as batch_err:
                logger.warning("Batch inference failed on %s: %s; falling back to single-chunk", self.device, batch_err)
                outs = []
                for _, (cwav, _) in batch:
                    with torch.inference_mode():
                        single_out = self.model.transcribe(audio=(cwav, TARGET_SR), language=lang)[0]
                        outs.append(single_out)

            for (orig_idx, (cwav, offset)), out in zip(batch, outs):
                text = (out.text or "").strip()
                detected_lang = getattr(out, "language", None)
                results.append((orig_idx, cwav, offset, text, detected_lang))

            if progress_cb:
                try:
                    progress_cb(len(batch))
                except Exception:
                    pass

            i += len(batch)

        return results

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

    _inference_lock = threading.Lock()

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
        self._workers: list[WorkerInstance] = []

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

        device_configs = resolve_devices_and_configs(work_mode, self.model_name)
        logger.info(
            "Initializing %d inference worker(s) for %s: %s",
            len(device_configs),
            self.model_name,
            device_configs,
        )

        self._workers = []
        for dev, dtype, batch_size in device_configs:
            if "cuda" in dev:
                torch.cuda.set_device(dev)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True

            load_kwargs: dict[str, Any] = {
                "dtype": dtype,
                "device_map": dev,
                "max_inference_batch_size": batch_size,
                "max_new_tokens": self.max_new_tokens,
            }
            try:
                import flash_attn  # noqa: F401
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                load_kwargs["attn_implementation"] = "sdpa"

            logger.info(
                "Loading %s worker from %s on %s (%s, batch_size=%d, attn=%s)",
                self.model_name,
                model_source,
                dev,
                dtype,
                batch_size,
                load_kwargs.get("attn_implementation", "sdpa"),
            )
            worker_model = Qwen3ASRModel.from_pretrained(
                model_source,
                **load_kwargs,
            )
            dev_idx = int(dev.split(":")[1]) if ":" in dev else 0 if "cuda" in dev else -1
            self._workers.append(
                WorkerInstance(
                    device=dev,
                    dtype=dtype,
                    batch_size=batch_size,
                    model=worker_model,
                    device_idx=dev_idx,
                )
            )

        if self._workers:
            self._model = self._workers[0].model

    async def unload(self) -> None:
        if self._workers:
            for w in self._workers:
                w.model = None
            self._workers.clear()

        if self._model is not None:
            self._model = None

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                    reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                    logger.info(
                        "CUDA memory device %d after model unload: allocated=%.2fMB, reserved=%.2fMB",
                        i,
                        allocated,
                        reserved,
                    )
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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
        from concurrent.futures import ThreadPoolExecutor
        from qwen_asr.inference.utils import (
            SAMPLE_RATE,
            normalize_audio_input,
            split_audio_into_chunks,
        )

        if (self._model is None and not self._workers) or not self.loaded:
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
        # Check if incoming audio is already a standard 16kHz mono audio (WAV or MP3) to avoid redundant conversion
        is_standard_audio = False
        try:
            import soundfile as sf
            with sf.SoundFile(audio_path) as info:
                if info.samplerate == TARGET_SR and info.channels == 1 and info.format in ("WAV", "MP3"):
                    is_standard_audio = True
        except Exception:
            is_standard_audio = False

        if is_standard_audio:
            _log(30, "Direct audio feed detected (standard 16kHz mono audio), skipping ffmpeg conversion...")
            wav = normalize_audio_input(audio_path)
            chunks = split_audio_into_chunks(wav, SAMPLE_RATE, max_chunk_sec=CHUNK_SEC)
        else:
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
                        "-f",
                        "s16le",
                        "pipe:1",
                    ],
                    capture_output=True,
                )
                if r.returncode != 0:
                    err_msg = r.stderr.decode("utf-8", errors="replace").strip() if r.stderr else "unknown error"
                    raise RuntimeError(f"ffmpeg failed for {audio_path}: {err_msg}")
                _log(30, "Preprocessing audio chunks and extracting features...")
                pcm_data = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
                wav = normalize_audio_input((pcm_data, TARGET_SR))
                chunks = split_audio_into_chunks(wav, SAMPLE_RATE, max_chunk_sec=CHUNK_SEC)
            except FileNotFoundError as fnf_err:
                raise RuntimeError(
                    "未在当前系统中检测到 ffmpeg 可执行程序。请安装 ffmpeg 并加入系统 PATH，"
                    "或者使用 cyphr 命令行客户端 (CLI) 在上传前自动完成音频格式转换。"
                ) from fnf_err

        duration = len(wav) / SAMPLE_RATE
        total = len(chunks)
        segments, texts, langs = [], [], []

        _log(30, "Waiting for model inference slot...")
        with self._inference_lock:
            # Check if mock model was injected (e.g. in unit tests)
            use_legacy_mock = False
            if not self._workers and self._model is not None:
                use_legacy_mock = True
            elif self._model is not None and len(self._workers) > 0 and self._model is not self._workers[0].model:
                use_legacy_mock = True

            if use_legacy_mock:
                import torch
                batch_size = int(os.getenv("QWEN3_ASR_BATCH_SIZE", "16" if torch.cuda.is_available() else "1"))
                batch_size = max(1, batch_size)
                for i in range(0, total, batch_size):
                    batch = chunks[i : i + batch_size]
                    cur_end = min(i + batch_size, total)
                    _log(30 + int(65 * cur_end / max(total, 1)), f"Running ASR batch inference ({cur_end}/{total})...")

                    audio_inputs = [(cwav, SAMPLE_RATE) for cwav, _ in batch]
                    try:
                        outs = self._model.transcribe(audio=audio_inputs, language=lang)
                    except Exception as batch_err:
                        logger.warning("Batch transcription failed, falling back to single-chunk: %s", batch_err)
                        outs = [
                            self._model.transcribe(audio=(cwav, SAMPLE_RATE), language=lang)[0]
                            for cwav, _ in batch
                        ]

                    for (cwav, offset), out in zip(batch, outs):
                        text = (out.text or "").strip()
                        detected_lang = getattr(out, "language", None)
                        if detected_lang and detected_lang not in langs:
                            langs.append(detected_lang)
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
            else:
                # Optimized multi-worker parallel inference
                num_workers = len(self._workers)
                completed_chunks = 0
                completed_lock = threading.Lock()

                def _progress_cb(count: int) -> None:
                    nonlocal completed_chunks
                    with completed_lock:
                        completed_chunks += count
                        cur = min(completed_chunks, total)
                        _log(30 + int(65 * cur / max(total, 1)), f"Running parallel ASR inference ({cur}/{total})...")

                indexed_chunks = list(enumerate(chunks))

                if num_workers <= 1:
                    worker = self._workers[0]
                    raw_results = worker.transcribe_chunks(indexed_chunks, lang=lang, progress_cb=_progress_cb)
                else:
                    # Distribute chunks across workers round-robin to balance load
                    worker_tasks: list[list[tuple[int, tuple[np.ndarray, float]]]] = [[] for _ in range(num_workers)]
                    for idx, chunk in enumerate(chunks):
                        worker_tasks[idx % num_workers].append((idx, chunk))

                    raw_results = []
                    with ThreadPoolExecutor(max_workers=num_workers) as pool:
                        futures = [
                            pool.submit(w.transcribe_chunks, w_chunks, lang, _progress_cb)
                            for w, w_chunks in zip(self._workers, worker_tasks)
                            if len(w_chunks) > 0
                        ]
                        for fut in futures:
                            raw_results.extend(fut.result())

                    raw_results.sort(key=lambda item: item[0])

                for orig_idx, cwav, offset, text, detected_lang in raw_results:
                    if detected_lang and detected_lang not in langs:
                        langs.append(detected_lang)
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
