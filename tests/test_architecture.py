# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.engine import BaseModelEngine
from src.models.registry import ModelRegistry
from src.resources import (
    calculate_dynamic_gpu_utilization,
    detect_supported_modes,
    get_gpu_free_memory_mb,
    resolve_devices,
)
from src.scheduler.handler import BaseTaskHandler
from src.scheduler.runner import JobRunner
from src.workers.proxy import BaseWorkerProxy


class DummyLLMEngine(BaseModelEngine):
    """A non-ASR model engine (e.g., LLM) verifying that ModelRegistry is model-agnostic."""

    def __init__(self, model_name: str = "dummy-llm-7b") -> None:
        super().__init__(model_name)
        self.is_sleeping = False
        self.generate_called = False

    async def load(self, work_mode: str = "gpu") -> None:
        self.loaded = True

    async def unload(self) -> None:
        self.loaded = False

    def enter_sleep(self) -> None:
        self.is_sleeping = True

    async def generate(self, prompt: str) -> str:
        self.generate_called = True
        return f"Response to: {prompt}"


class LLMTaskHandler(BaseTaskHandler):
    """Custom task handler for LLM chat/generate tasks."""

    def can_handle(self, task_type: str, payload: dict[str, Any]) -> bool:
        return task_type in ("chat", "generate", "llm")

    async def execute(
        self,
        payload: dict[str, Any],
        engine: Any,
        reporter: Any,
        inference_lock: asyncio.Lock,
    ) -> Any:
        job_id = int(payload["job_id"])
        prompt = payload.get("prompt", "")

        # Call the LLM-specific generate method
        async with inference_lock:
            result_text = await engine.generate(prompt)

        await reporter.report_completion(
            job_id=job_id,
            status="completed",
            duration_seconds=0.1,
            result_text=result_text,
            openai_response={"choices": [{"message": {"content": result_text}}]},
        )
        return result_text


@pytest.mark.asyncio
async def test_non_asr_model_in_registry() -> None:
    """Verify that any arbitrary model engine (e.g. LLM, TTS) can be registered,
    loaded, acquired, and unloaded by ModelRegistry without ASR dependencies.
    """
    registry = ModelRegistry(debug=False)
    registry.register("dummy-llm-7b", lambda: DummyLLMEngine("dummy-llm-7b"))

    assert "dummy-llm-7b" in registry.list_available_models()

    # Load model
    engine = await registry.load_model("dummy-llm-7b")
    assert engine.loaded is True
    assert isinstance(engine, DummyLLMEngine)
    assert registry.list_loaded_models() == ["dummy-llm-7b"]

    # Safe acquire with active reference counting
    async with registry.acquire_engine("dummy-llm-7b") as acquired:
        assert acquired is engine
        res = await acquired.generate("Hello world")
        assert res == "Response to: Hello world"

    # Unload model
    unloaded = await registry.unload_model("dummy-llm-7b")
    assert unloaded is True
    assert "dummy-llm-7b" not in registry.list_loaded_models()


@pytest.mark.asyncio
async def test_job_runner_pluggable_llm_task_handler() -> None:
    """Verify that JobRunner dispatches non-ASR jobs through pluggable task handlers
    without invoking media download or audio chunking.
    """
    mock_reporter = AsyncMock()
    registry = ModelRegistry(debug=False)
    registry.register("dummy-llm-7b", lambda: DummyLLMEngine("dummy-llm-7b"))

    job_runner = JobRunner(
        reporter=mock_reporter,
        registry=registry,
        max_concurrent_jobs=2,
    )
    # Register our LLM task handler
    job_runner.register_task_handler(LLMTaskHandler())

    payload = {
        "job_id": 9001,
        "task_type": "chat",
        "model_name": "dummy-llm-7b",
        "prompt": "What is the capital of France?",
    }

    task = job_runner.run_job(payload)
    await task

    # Assert reporter was called with completion settlement
    assert mock_reporter.download_media.call_count == 0  # No audio download!
    assert mock_reporter.report_completion.call_count == 1
    call_args = mock_reporter.report_completion.call_args[1]
    assert call_args["job_id"] == 9001
    assert call_args["status"] == "completed"
    assert call_args["result_text"] == "Response to: What is the capital of France?"


@pytest.mark.asyncio
async def test_job_runner_unsupported_task_type_reports_failure() -> None:
    """Verify JobRunner gracefully reports failure when no handler can process the task."""
    mock_reporter = AsyncMock()
    registry = ModelRegistry(debug=False)
    registry.register("dummy-llm-7b", lambda: DummyLLMEngine("dummy-llm-7b"))

    job_runner = JobRunner(
        reporter=mock_reporter,
        registry=registry,
    )

    payload = {
        "job_id": 9002,
        "task_type": "unknown_future_modality",
        "model_name": "dummy-llm-7b",
    }

    task = job_runner.run_job(payload)
    await task

    assert mock_reporter.report_completion.call_count == 1
    call_args = mock_reporter.report_completion.call_args[1]
    assert call_args["job_id"] == 9002
    assert call_args["status"] == "failed"
    assert "No task handler found" in call_args["error_msg"]


def test_resource_management_independence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that resource management (VRAM calculation, probes, hardware resolution)
    operates standalone and is usable by any model architecture.
    """
    # 1. Dynamic VRAM utilization
    util = calculate_dynamic_gpu_utilization(
        device_id=0,
        gpu_memory_utilization=0.90,
        free_ratio=0.70,
        mock_free_bytes=10 * 1024 * 1024 * 1024,  # 10 GB free
        mock_total_bytes=24 * 1024 * 1024 * 1024,  # 24 GB total
    )
    # (10 * 0.70) / 24 = 0.2917
    assert round(util, 4) == round(7.0 / 24.0, 4)

    # 2. Free VRAM probe
    free_mb = get_gpu_free_memory_mb("cpu")
    assert free_mb == 1000000

    # 3. Hardware detection
    modes, default_mode = detect_supported_modes()
    assert "cpu" in modes
    assert default_mode in ("cpu", "gpu")

    # 4. Device resolution
    monkeypatch.setenv("MODEL_DEVICE", "cpu")
    devices = resolve_devices("cpu")
    assert devices == ["cpu"]


def test_base_worker_proxy_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that BaseWorkerProxy starts, waits for ready, enters sleep, and terminates cleanly."""
    mock_proc = MagicMock()
    mock_proc.is_alive.return_value = True

    mock_queue = MagicMock()
    mock_queue.get.return_value = ("ready", 0)

    monkeypatch.setattr(
        "multiprocessing.get_context",
        lambda _: MagicMock(
            Process=lambda target, args, daemon: mock_proc,
            Queue=lambda: mock_queue,
        ),
    )

    def dummy_worker(device_id: int, cmd_q: Any, resp_q: Any) -> None:
        pass

    proxy = BaseWorkerProxy(device_id=0, target_fn=dummy_worker, args=(0,))
    assert proxy.is_sleeping is False

    proxy.wait_ready(timeout=1.0)
    assert proxy.process.start.call_count == 1

    proxy.enter_sleep()
    assert proxy.is_sleeping is True
    assert mock_queue.put.call_args[0][0] == ("sleep",)

    proxy.stop(timeout=1.0)
    assert mock_queue.put.call_args[0][0] == ("stop",)
