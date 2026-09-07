# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os
import queue
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseEngine

logger = logging.getLogger(__name__)


class InsufficientVRAMError(RuntimeError):
    """Raised when GPU VRAM is insufficient to load the model or wake up to restore KV cache."""

    pass


MIN_LOAD_VRAM_MB = int(os.getenv("MIN_LOAD_VRAM_MB", "2048"))
MIN_WAKE_VRAM_MB = int(os.getenv("MIN_WAKE_VRAM_MB", "1536"))


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


MODEL_NAME_0_6B = "qwen3-asr-0.6b"
MODEL_NAME_1_7B = "qwen3-asr-1.7b"
MODEL_NAME = MODEL_NAME_0_6B
CHUNK_SEC = 30.0
TARGET_SR = 16000

HF_REPO_MAP = {
    MODEL_NAME_0_6B: "Qwen/Qwen3-ASR-0.6B",
    MODEL_NAME_1_7B: "Qwen/Qwen3-ASR-1.7B",
}

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


def resolve_device_and_dtype(work_mode: str = "gpu") -> tuple[str, Any, int]:
    """Resolve target inference device, dtype, and default batch size."""
    import torch

    env_device = os.getenv("QWEN3_ASR_DEVICE", "").strip().lower()
    req_device = env_device or work_mode.strip().lower()

    if req_device == "cpu":
        return "cpu", torch.float32, int(os.getenv("QWEN3_ASR_BATCH_SIZE", "1"))

    if req_device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested (%s) but torch.cuda is not available; falling back to CPU", req_device)
            return "cpu", torch.float32, int(os.getenv("QWEN3_ASR_BATCH_SIZE", "1"))
        device = req_device if ":" in req_device else "cuda:0"
        use_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        return device, dtype, int(os.getenv("QWEN3_ASR_BATCH_SIZE", "16"))

    if req_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float16, int(os.getenv("QWEN3_ASR_BATCH_SIZE", "4"))
        return "cpu", torch.float32, int(os.getenv("QWEN3_ASR_BATCH_SIZE", "1"))

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device_idx = int(os.getenv("CUDA_DEVICE_INDEX", "0"))
        device_idx = min(max(0, device_idx), torch.cuda.device_count() - 1)
        use_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        return f"cuda:{device_idx}", dtype, int(os.getenv("QWEN3_ASR_BATCH_SIZE", "16"))

    return "cpu", torch.float32, int(os.getenv("QWEN3_ASR_BATCH_SIZE", "1"))


def resolve_devices(work_mode: str = "gpu") -> list[int | str]:
    """Resolve list of target inference device identifiers (GPU indices or 'cpu'),
    filtering out any GPU devices with insufficient free VRAM for model loading.
    """
    import torch

    env_device = os.getenv("QWEN3_ASR_DEVICE", "").strip().lower()
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

    # Filter candidate GPUs by free memory against MIN_LOAD_VRAM_MB
    min_load_mb = int(os.getenv("MIN_LOAD_VRAM_MB", str(MIN_LOAD_VRAM_MB)))
    available_devices: list[int | str] = []
    for dev in candidate_devices:
        if str(dev).lower() == "cpu":
            available_devices.append(dev)
        else:
            free_mb = get_gpu_free_memory_mb(dev)
            if free_mb >= min_load_mb:
                available_devices.append(dev)
            else:
                logger.warning(
                    "[resolve_devices] Skipping GPU %s: insufficient free VRAM (%d MB < %d MB required)",
                    dev,
                    free_mb,
                    min_load_mb,
                )

    return available_devices


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
    """Qwen3-ASR vLLM inference engine yielding OpenAI verbose_json."""

    supports_concurrent_inference = True

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_dir: str | Path | None = None,
        max_new_tokens: int = 1024,
    ) -> None:
        super().__init__(model_name)
        self.model_dir = Path(model_dir) if model_dir else resolve_model_dir(model_name)
        self.max_new_tokens = max_new_tokens
        self.batch_size = max(1, min(8, int(os.getenv("QWEN3_ASR_BATCH_SIZE", "4"))))
        self.gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.60"))
        self.max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "16384"))
        self.enforce_eager = os.getenv("VLLM_ENFORCE_EAGER", "1").lower() in ("1", "true", "yes")
        self.sleep_idle_seconds = float(os.getenv("VLLM_SLEEP_IDLE_SECONDS", "5.0"))
        self._workers: list[Any] = []
        self._worker_queue: queue.Queue[Any] = queue.Queue()
        self._model = None
        self._lock = threading.Lock()
        self._sleep_timer: threading.Timer | None = None
        self._is_sleeping = False
        self._active_tasks = 0

    @property
    def is_sleeping(self) -> bool:
        with self._lock:
            return self._is_sleeping

    def enter_sleep(self) -> None:
        """Immediately enter Sleep Mode to offload KV cache without waiting for idle timer."""
        if self._workers:
            for w in self._workers:
                w.enter_sleep()
            with self._lock:
                self._is_sleeping = True
            return
        self._enter_sleep()

    def _cancel_sleep_timer_locked(self) -> None:
        if self._sleep_timer is not None:
            self._sleep_timer.cancel()
            self._sleep_timer = None

    def _arm_sleep_timer_locked(self) -> None:
        self._cancel_sleep_timer_locked()
        if self._model is not None and not self._is_sleeping and self._active_tasks == 0:
            self._sleep_timer = threading.Timer(self.sleep_idle_seconds, self._enter_sleep)
            self._sleep_timer.daemon = True
            self._sleep_timer.start()

    def _enter_sleep(self) -> None:
        with self._lock:
            if self._active_tasks > 0 or self._is_sleeping or self._model is None:
                return
            vllm_engine = getattr(self._model, "model", self._model)
            sleep_fn = getattr(vllm_engine, "sleep", None)
            if sleep_fn is not None:
                try:
                    logger.info(
                        "vLLM engine idle for %.1fs; entering Sleep Mode (level=1) to offload KV cache...",
                        self.sleep_idle_seconds,
                    )
                    sleep_fn(level=1)
                    self._is_sleeping = True
                except Exception as e:
                    logger.warning("Error entering vLLM Sleep Mode: %s", e)

    def _wake_up_locked(self) -> None:
        if not self._is_sleeping:
            return
        vllm_engine = getattr(self._model, "model", self._model)
        wake_fn = getattr(vllm_engine, "wake_up", None)
        if wake_fn is not None:
            try:
                logger.info("Waking up vLLM engine from Sleep Mode...")
                wake_fn()
            except Exception as e:
                logger.warning("Error waking up vLLM engine: %s", e)
        self._is_sleeping = False

    async def load(self, work_mode: str = "gpu") -> None:
        """Load weights using vLLM engine."""
        import asyncio

        await asyncio.to_thread(self._load_blocking, work_mode)
        self.loaded = True
        with self._lock:
            self._arm_sleep_timer_locked()

    def _cleanup_workers_locked(self) -> None:
        for w in self._workers:
            try:
                w.stop()
            except Exception as e:
                logger.warning("Error stopping worker %s: %s", getattr(w, "device_id", None), e)
        self._workers.clear()
        while not self._worker_queue.empty():
            try:
                self._worker_queue.get_nowait()
            except Exception:
                break

    def _resolve_model_source(self) -> str:
        if self.model_dir.joinpath("config.json").is_file():
            return str(self.model_dir.resolve())
        if self.model_name.lower() in HF_REPO_MAP:
            model_source = HF_REPO_MAP[self.model_name.lower()]
            logger.info(
                "Local weights not found at %s; loading directly via Hugging Face ID: %s",
                self.model_dir,
                model_source,
            )
            return model_source
        raise FileNotFoundError(
            f"Model package missing in {self.model_dir}. "
            f"Please download model '{self.model_name}' first using cyphr-installer."
        )

    def _load_blocking(self, work_mode: str = "gpu") -> None:
        from qwen_asr import Qwen3ASRModel

        model_source = self._resolve_model_source()
        devices = resolve_devices(work_mode)
        if not devices:
            min_load_mb = int(os.getenv("MIN_LOAD_VRAM_MB", str(MIN_LOAD_VRAM_MB)))
            raise InsufficientVRAMError(
                f"No GPU device has sufficient free VRAM to load model '{self.model_name}' (required: {min_load_mb} MB)"
            )

        logger.info(
            "Initializing Qwen3-ASR vLLM engine for %s from %s on devices %s (gpu_util=%.2f, max_model_len=%d, batch_size=%d)...",
            self.model_name,
            model_source,
            devices,
            self.gpu_memory_utilization,
            self.max_model_len,
            self.batch_size,
        )

        force_pool = os.getenv("VLLM_FORCE_WORKER_POOL", "").lower() in ("1", "true", "yes")
        use_worker_pool = (
            len(devices) > 1
            or force_pool
            or (len(devices) == 1 and devices[0] not in (0, "0", "cpu"))
        )
        if use_worker_pool:
            from .worker import WorkerProxy

            with self._lock:
                self._cleanup_workers_locked()
                for dev in devices:
                    wp = WorkerProxy(
                        device_id=dev,
                        model_source=model_source,
                        model_name=self.model_name,
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        max_model_len=self.max_model_len,
                        batch_size=self.batch_size,
                        enforce_eager=self.enforce_eager,
                        sleep_idle_seconds=self.sleep_idle_seconds,
                    )
                    wp.wait_ready()
                    self._workers.append(wp)
                    self._worker_queue.put(wp)

                if self._workers:
                    self._model = self._workers[0]
            return

        os.environ["HF_HUB_OFFLINE"] = "1"
        self._model = Qwen3ASRModel.LLM(
            model=model_source,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_inference_batch_size=self.batch_size,
            enable_sleep_mode=True,
            enforce_eager=self.enforce_eager,
        )

    async def unload(self) -> None:
        with self._lock:
            self._cleanup_workers_locked()
            self._cancel_sleep_timer_locked()
            self._is_sleeping = False
            self._model = None
            self.loaded = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as e:
            logger.warning("Failed during CUDA cache cleanup: %s", e)

    def is_active(self) -> bool:
        """Check if engine is active (performing inference or awake with KV cache allocated)."""
        with self._lock:
            if self._active_tasks > 0:
                return True
            if self._workers:
                return any(not getattr(w, "is_sleeping", False) for w in self._workers)
            return not self._is_sleeping

    def check_resources_available(self) -> bool:
        """Check if resources are available to load or wake up this engine."""
        if not self.loaded:
            devices = resolve_devices("gpu")
            return len(devices) > 0
        if self.is_active():
            return True
        min_wake_mb = int(os.getenv("MIN_WAKE_VRAM_MB", str(MIN_WAKE_VRAM_MB)))
        if self._workers:
            for w in self._workers:
                if get_gpu_free_memory_mb(w.device_id) >= min_wake_mb:
                    return True
            return False
        return get_gpu_free_memory_mb(0) >= min_wake_mb

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        task_type: str = "transcribe",
        log_callback: Callable[[int, str], Any] | None = None,
    ) -> dict[str, Any]:
        """Sync (runs in job_runner's executor): ffmpeg -> chunk -> vLLM transcribe -> verbose_json."""
        if self._workers:
            if not self.loaded:
                raise RuntimeError(f"Model '{self.model_name}' is not loaded")
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

            min_wake_mb = int(os.getenv("MIN_WAKE_VRAM_MB", str(MIN_WAKE_VRAM_MB)))
            selected_worker = None

            # Pre-wakeup probe: find candidate worker with sufficient VRAM to restore KV cache
            with self._lock:
                all_candidates = []
                while not self._worker_queue.empty():
                    try:
                        all_candidates.append(self._worker_queue.get_nowait())
                    except queue.Empty:
                        break

                for w in all_candidates:
                    if getattr(w, "is_sleeping", False):
                        free_mb = get_gpu_free_memory_mb(w.device_id)
                        if free_mb < min_wake_mb:
                            logger.warning(
                                "[Pre-Wakeup Probe] Worker on device %s has insufficient free VRAM (%d MB < %d MB); skipping",
                                w.device_id,
                                free_mb,
                                min_wake_mb,
                            )
                            continue
                    selected_worker = w
                    break

                for w in all_candidates:
                    if w is not selected_worker:
                        self._worker_queue.put(w)

            if selected_worker is None:
                all_sleeping = all(getattr(w, "is_sleeping", False) for w in self._workers)
                if all_sleeping:
                    raise InsufficientVRAMError(
                        f"All GPU workers have insufficient free VRAM to restore KV cache (required: {min_wake_mb} MB)"
                    )
                selected_worker = self._worker_queue.get(timeout=60.0)

            try:
                return selected_worker.execute_transcribe(
                    audio_path=audio_path,
                    language=language,
                    task_type=task_type,
                    log_callback=log_callback,
                )
            finally:
                self._worker_queue.put(selected_worker)

        from qwen_asr.inference.utils import (
            SAMPLE_RATE,
            normalize_audio_input,
            split_audio_into_chunks,
        )

        if self._model is None or not self.loaded:
            raise RuntimeError(f"Model '{self.model_name}' is not loaded")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        with self._lock:
            self._cancel_sleep_timer_locked()
            if self._is_sleeping:
                min_wake_mb = int(os.getenv("MIN_WAKE_VRAM_MB", str(MIN_WAKE_VRAM_MB)))
                free_mb = get_gpu_free_memory_mb(0)
                if free_mb < min_wake_mb:
                    raise InsufficientVRAMError(
                        f"GPU has insufficient free VRAM to restore KV cache ({free_mb} MB < {min_wake_mb} MB)"
                    )
                self._wake_up_locked()
            self._active_tasks += 1

        try:
            lang = LANG_MAP.get(language, language) if language else None

            def _log(progress: int, message: str) -> None:
                if log_callback is None:
                    return
                try:
                    log_callback(progress, message)
                except Exception as e:
                    logger.warning("log_callback failed: %s", e)

            _log(20, "Loading and decoding audio file...")
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

            _log(30, "Running vLLM ASR inference...")
            with self._lock:
                for i in range(0, total, self.batch_size):
                    batch = chunks[i : i + self.batch_size]
                    cur_end = min(i + self.batch_size, total)
                    _log(30 + int(65 * cur_end / max(total, 1)), f"Running vLLM ASR inference ({cur_end}/{total})...")

                    audio_inputs = [(cwav, SAMPLE_RATE) for cwav, _ in batch]
                    outs = self._model.transcribe(audio=audio_inputs, language=lang)

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

            _log(100, "Aligning timestamps and finalizing transcript...")
            return {
                "task": task_type or "transcribe",
                "language": langs[0] if langs else (lang or ""),
                "duration": round(duration, 2),
                "text": " ".join(texts),
                "segments": segments,
            }
        finally:
            with self._lock:
                self._active_tasks = max(0, self._active_tasks - 1)
                if self._active_tasks == 0:
                    self._arm_sleep_timer_locked()
