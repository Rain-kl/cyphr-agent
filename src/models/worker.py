# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing as mp
import os
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CHUNK_SEC = 30.0
TARGET_SR = 16000
SAMPLE_RATE = 16000

# ISO code -> Qwen3-ASR language name
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


def _process_audio_chunks(audio_path: str) -> tuple[np.ndarray, list[tuple[np.ndarray, float]]]:
    """Decode and chunk audio file into 30s segments with ffmpeg fallback."""
    from qwen_asr.inference.utils import (
        normalize_audio_input,
        split_audio_into_chunks,
    )

    is_standard = False
    try:
        import soundfile as sf

        with sf.SoundFile(audio_path) as info:
            if info.samplerate == TARGET_SR and info.channels == 1 and info.format in ("WAV", "MP3"):
                is_standard = True
    except Exception:
        is_standard = False

    if is_standard:
        wav = normalize_audio_input(audio_path)
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
            pcm_data = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            wav = normalize_audio_input((pcm_data, TARGET_SR))
        except FileNotFoundError as fnf_err:
            raise RuntimeError(
                "未在当前系统中检测到 ffmpeg 可执行程序。请安装 ffmpeg 并加入系统 PATH。"
            ) from fnf_err

    chunks = split_audio_into_chunks(wav, SAMPLE_RATE, max_chunk_sec=CHUNK_SEC)
    return wav, chunks


from ..resources.vram import calculate_dynamic_gpu_utilization


def asr_worker_process_main(
    device_id: int | str,
    model_source: str,
    model_name: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    batch_size: int,
    enforce_eager: bool,
    sleep_idle_seconds: float,
    cmd_queue: Any,
    resp_queue: Any,
) -> None:
    """Isolated child process running dedicated vLLM engine instance for one GPU."""
    if isinstance(device_id, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        resp_queue.put(("error", device_id, f"Failed to import qwen_asr: {e}"))
        return

    # Calculate dynamic VRAM utilization based on remaining free memory
    free_ratio = float(os.getenv("VLLM_FREE_MEMORY_RATIO", "0.70"))
    effective_gpu_util = calculate_dynamic_gpu_utilization(
        device_id=device_id,
        gpu_memory_utilization=gpu_memory_utilization,
        free_ratio=free_ratio,
    )

    logger.info(
        "[Worker Device %s] Initializing vLLM engine for %s (gpu_util=%.2f [effective=%.2f, ratio=%.2f], max_model_len=%d, batch_size=%d)...",
        device_id,
        model_name,
        gpu_memory_utilization,
        effective_gpu_util,
        free_ratio,
        max_model_len,
        batch_size,
    )

    try:
        model = Qwen3ASRModel.LLM(
            model=model_source,
            gpu_memory_utilization=effective_gpu_util,
            max_model_len=max_model_len,
            max_inference_batch_size=batch_size,
            enable_sleep_mode=True,
            enforce_eager=enforce_eager,
        )
    except Exception as e:
        logger.exception("[Worker Device %s] Model init failed: %s", device_id, e)
        resp_queue.put(("error", device_id, str(e)))
        return

    vllm_engine = getattr(model, "model", model)
    lock = threading.Lock()
    is_sleeping = False
    active_tasks = 0
    sleep_timer: threading.Timer | None = None

    def _cancel_timer_locked() -> None:
        nonlocal sleep_timer
        if sleep_timer is not None:
            sleep_timer.cancel()
            sleep_timer = None

    def _enter_sleep() -> None:
        nonlocal is_sleeping
        with lock:
            if active_tasks > 0 or is_sleeping:
                return
            sleep_fn = getattr(vllm_engine, "sleep", None)
            if sleep_fn is not None:
                try:
                    logger.info(
                        "[Worker Device %s] Idle for %.1fs; entering Sleep Mode (level=1) to offload KV cache...",
                        device_id,
                        sleep_idle_seconds,
                    )
                    sleep_fn(level=1)
                    is_sleeping = True
                except Exception as ex:
                    logger.warning("[Worker Device %s] Error entering Sleep Mode: %s", device_id, ex)

    def _arm_timer_locked() -> None:
        nonlocal sleep_timer
        _cancel_timer_locked()
        if not is_sleeping and active_tasks == 0 and sleep_idle_seconds > 0:
            sleep_timer = threading.Timer(sleep_idle_seconds, _enter_sleep)
            sleep_timer.daemon = True
            sleep_timer.start()

    def _wake_up_locked() -> None:
        nonlocal is_sleeping
        if not is_sleeping:
            return
        wake_fn = getattr(vllm_engine, "wake_up", None)
        if wake_fn is not None:
            try:
                logger.info("[Worker Device %s] Waking up engine from Sleep Mode...", device_id)
                wake_fn()
            except Exception as ex:
                logger.warning("[Worker Device %s] Error waking up engine: %s", device_id, ex)
        is_sleeping = False

    with lock:
        _arm_timer_locked()

    resp_queue.put(("ready", device_id))

    while True:
        try:
            cmd = cmd_queue.get()
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception:
            break

        if not cmd or not isinstance(cmd, tuple):
            continue

        action = cmd[0]
        if action == "stop":
            with lock:
                _cancel_timer_locked()
            break
        elif action == "sleep":
            with lock:
                _cancel_timer_locked()
                _enter_sleep()
            resp_queue.put(("slept", device_id))
        elif action == "wake_up":
            with lock:
                _wake_up_locked()
                _arm_timer_locked()
            resp_queue.put(("woken", device_id))
        elif action == "transcribe":
            task_id, audio_path, language, task_type = cmd[1], cmd[2], cmd[3], cmd[4]
            with lock:
                _cancel_timer_locked()
                _wake_up_locked()
                active_tasks += 1

            try:
                resp_queue.put(("progress", task_id, 20, f"Loading audio on GPU {device_id}..."))
                wav, chunks = _process_audio_chunks(audio_path)
                duration = len(wav) / SAMPLE_RATE
                total = len(chunks)
                lang = LANG_MAP.get(language, language) if language else None
                segments, texts, langs = [], [], []

                resp_queue.put(("progress", task_id, 30, f"Running vLLM inference on GPU {device_id}..."))
                for i in range(0, total, batch_size):
                    batch = chunks[i : i + batch_size]
                    cur_end = min(i + batch_size, total)
                    pct = 30 + int(65 * cur_end / max(total, 1))
                    resp_queue.put(
                        ("progress", task_id, pct, f"GPU {device_id} processing chunks ({cur_end}/{total})...")
                    )

                    audio_inputs = [(cwav, SAMPLE_RATE) for cwav, _ in batch]
                    outs = model.transcribe(audio=audio_inputs, language=lang)

                    for (cwav, offset), out in zip(batch, outs):
                        text = (out.text or "").strip()
                        det_lang = getattr(out, "language", None)
                        if det_lang and det_lang not in langs:
                            langs.append(det_lang)
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

                result_dict = {
                    "task": task_type or "transcribe",
                    "language": langs[0] if langs else (lang or ""),
                    "duration": round(duration, 2),
                    "text": " ".join(texts),
                    "segments": segments,
                }
                resp_queue.put(("result", task_id, result_dict))
            except Exception as e:
                logger.exception("[Worker Device %s] Transcription failed: %s", device_id, e)
                resp_queue.put(("error", task_id, str(e)))
            finally:
                with lock:
                    active_tasks = max(0, active_tasks - 1)
                    if active_tasks == 0:
                        _arm_timer_locked()

    with lock:
        _cancel_timer_locked()
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    logger.info("[Worker Device %s] Child process exiting cleanly...", device_id)


from ..workers.proxy import BaseWorkerProxy


class WorkerProxy(BaseWorkerProxy):
    """Manages child worker process lifecycle and communication for ASR inference."""

    def __init__(
        self,
        device_id: int | str,
        model_source: str,
        model_name: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        batch_size: int,
        enforce_eager: bool,
        sleep_idle_seconds: float,
    ) -> None:
        super().__init__(
            device_id=device_id,
            target_fn=asr_worker_process_main,
            args=(
                device_id,
                model_source,
                model_name,
                gpu_memory_utilization,
                max_model_len,
                batch_size,
                enforce_eager,
                sleep_idle_seconds,
            ),
        )

    def execute_transcribe(
        self,
        audio_path: str,
        language: str | None,
        task_type: str,
        log_callback: Callable[[int, str], Any] | None = None,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Send transcription task to worker and stream progress callbacks."""
        self.is_sleeping = False
        task_id = uuid.uuid4().hex[:8]
        self.cmd_queue.put(("transcribe", task_id, audio_path, language, task_type))

        start = time.time()
        while time.time() - start < timeout:
            if not self.process.is_alive():
                raise RuntimeError(
                    f"Worker for device {self.device_id} died while executing transcription {task_id}"
                )
            try:
                msg = self.resp_queue.get(timeout=1.0)
            except Exception:
                continue

            msg_type = msg[0]
            if msg_type == "progress" and msg[1] == task_id:
                if log_callback is not None:
                    try:
                        log_callback(msg[2], msg[3])
                    except Exception:
                        pass
            elif msg_type == "result" and msg[1] == task_id:
                return msg[2]
            elif msg_type == "error" and msg[1] == task_id:
                raise RuntimeError(f"Worker device {self.device_id} error: {msg[2]}")

        raise TimeoutError(f"Transcription on device {self.device_id} timed out after {timeout}s")
