# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.config import AgentConfig, load_config
from src.job_runner import JobRunner
from src.models.base import BaseEngine
from src.models.mock_asr import MockASREngine
from src.models.registry import ModelRegistry
from src.monitor import SystemMonitor
from src.reporter import Reporter
from src.ws_client import AgentWebSocketClient

# =========================================================================
# 1. Config Tests
# =========================================================================


def test_config_defaults() -> None:
    config = AgentConfig()
    assert config.controller_url == "http://localhost:8080"
    assert config.agent_token == ""
    assert config.node_name == "agent-default"
    assert config.heartbeat_interval == 10
    assert config.max_concurrent_jobs == 2
    assert config.debug is False
    assert config.http_base_url == "http://localhost:8080"
    assert config.ws_url == "ws://localhost:8080/api/v1/agent/ws"


def test_config_ws_url_with_token_and_https() -> None:
    config = AgentConfig(
        controller_url="https://api.transcribe.io:8443/",
        agent_token="sec-token-123",
    )
    assert config.http_base_url == "https://api.transcribe.io:8443"
    assert config.ws_url == "wss://api.transcribe.io:8443/api/v1/agent/ws?token=sec-token-123"


def test_config_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_file = tmp_path / "custom_config.yaml"
    yaml_file.write_text(
        "controller_url: 'http://yaml-host:9000'\n"
        "agent_token: 'yaml-token'\n"
        "node_name: 'yaml-node'\n"
        "heartbeat_interval: 15\n"
        "media_dir: '/tmp/yaml-media'\n"
        "max_concurrent_jobs: 4\n"
        "debug: false\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("CONTROLLER_URL", "http://env-host:7000")
    monkeypatch.setenv("AGENT_TOKEN", "env-token")
    monkeypatch.setenv("HEARTBEAT_INTERVAL", "20")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "8")
    monkeypatch.setenv("DEBUG", "true")

    loaded = load_config(yaml_file)
    assert loaded.controller_url == "http://env-host:7000"
    assert loaded.agent_token == "env-token"
    assert loaded.node_name == "yaml-node"
    assert loaded.heartbeat_interval == 20
    assert loaded.media_dir == "/tmp/yaml-media"
    assert loaded.max_concurrent_jobs == 8
    assert loaded.debug is True


# =========================================================================
# 2. Monitor Tests
# =========================================================================


def test_monitor_collect() -> None:
    monitor = SystemMonitor()
    stats = monitor.collect()

    assert "cpu_percent" in stats
    assert "ram_percent" in stats
    assert "ram_used_mb" in stats
    assert "ram_total_mb" in stats
    assert "gpu_percent" in stats
    assert "gpu_memory_used_mb" in stats
    assert "gpu_memory_total_mb" in stats

    assert isinstance(stats["cpu_percent"], float)
    assert isinstance(stats["ram_percent"], float)
    assert isinstance(stats["ram_used_mb"], int)
    assert isinstance(stats["ram_total_mb"], int)
    assert stats["cpu_percent"] >= 0.0
    assert stats["ram_percent"] >= 0.0
    assert stats["ram_total_mb"] > 0
    assert stats["gpu_percent"] == 0.0 or stats["gpu_percent"] >= 0.0


def test_monitor_gpu_fallback() -> None:
    monitor = SystemMonitor()
    with patch("builtins.__import__", side_effect=ImportError("No module named pynvml")):
        gpu_pct, gpu_used, gpu_total = monitor._collect_gpu()
        assert gpu_pct == 0.0
        assert gpu_used == 0
        assert gpu_total == 0


# =========================================================================
# 3. Mock ASR Engine Tests
# =========================================================================


@pytest.mark.asyncio
async def test_mock_asr_engine_transcribe_and_progress(tmp_path: Path) -> None:
    dummy_audio = tmp_path / "sample.mp3"
    dummy_audio.write_bytes(b"dummy audio content")

    engine = MockASREngine(stage_delay=0.0)
    assert not engine.loaded
    await engine.load()
    assert engine.loaded

    reported_progress: list[tuple[int, str]] = []

    async def log_cb(p: int, msg: str) -> None:
        reported_progress.append((p, msg))

    res = await engine.transcribe(
        audio_path=str(dummy_audio),
        language="zh",
        task_type="transcribe",
        log_callback=log_cb,
    )

    # Validate progress callbacks
    progress_values = [p[0] for p in reported_progress]
    assert progress_values == [20, 30, 80, 100]

    # Validate OpenAI verbose_json compliant structure
    assert res["task"] == "transcribe"
    assert res["language"] == "zh"
    assert res["duration"] > 0
    assert len(res["text"]) > 0
    assert isinstance(res["segments"], list)
    assert len(res["segments"]) >= 2

    seg0 = res["segments"][0]
    assert seg0["id"] == 0
    assert seg0["start"] == 0.0
    assert seg0["end"] > 0
    assert "tokens" in seg0
    assert "temperature" in seg0
    assert "avg_logprob" in seg0


@pytest.mark.asyncio
async def test_mock_asr_engine_missing_file() -> None:
    engine = MockASREngine(stage_delay=0.0)
    with pytest.raises(FileNotFoundError):
        await engine.transcribe(audio_path="/non/existent/path/test.wav")


# =========================================================================
# 4. Model Registry Tests
# =========================================================================


@pytest.mark.asyncio
async def test_model_registry_default_no_mock() -> None:
    """By default, ModelRegistry must not preload any model, and mock-whisper-base is not available."""
    registry = ModelRegistry()
    assert registry.list_loaded_models() == []
    assert "mock-whisper-base" not in registry.list_loaded_models()
    assert "mock-whisper-base" not in registry.list_available_models()
    assert "mock-whisper-base" not in registry.list_downloaded_models()
    with pytest.raises(ValueError, match="debug mode"):
        await registry.load_model("mock-whisper-base")


@pytest.mark.asyncio
async def test_model_registry_mock_in_debug_mode() -> None:
    """When debug=True, mock-whisper-base is available but NOT preloaded by default."""
    registry = ModelRegistry(debug=True)
    assert registry.list_loaded_models() == []
    assert "mock-whisper-base" in registry.list_available_models()
    assert "mock-whisper-base" in registry.list_downloaded_models()

    # It can be loaded explicitly in debug mode
    engine = await registry.load_model("mock-whisper-base")
    assert engine.loaded
    assert "mock-whisper-base" in registry.list_loaded_models()


@pytest.mark.asyncio
async def test_model_registry_lifecycle() -> None:
    registry = ModelRegistry(preload_default=True, debug=True)
    assert "mock-whisper-base" in registry.list_loaded_models()
    assert "mock-whisper-base" in registry.list_available_models()
    assert "qwen3-asr-0.6b" in registry.list_available_models()
    assert "qwen3-asr-1.7b" in registry.list_available_models()
    assert "Qwen/Qwen3-ASR-1.7B" in registry.list_available_models()

    engine = registry.get_engine("mock-whisper-base")
    assert engine is not None
    assert engine.loaded

    # Unload
    unloaded = await registry.unload_model("mock-whisper-base")
    assert unloaded is True
    assert "mock-whisper-base" not in registry.list_loaded_models()

    # Reload
    reloaded_engine = await registry.load_model("mock-whisper-base")
    assert reloaded_engine is not None
    assert reloaded_engine.loaded
    assert "mock-whisper-base" in registry.list_loaded_models()

    # Unknown model error
    with pytest.raises(ValueError, match="Unknown or unregistered model"):
        await registry.load_model("nonexistent-model-xyz")


# =========================================================================
# 5. Reporter Tests
# =========================================================================


@pytest.mark.asyncio
async def test_reporter_download_logs_complete(tmp_path: Path) -> None:
    audio_content = b"RIFF-WAVE-AUDIO-BYTES"
    logs_received = []
    complete_received = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("Authorization")
        assert auth == "Bearer test-agent-token"

        if request.url.path == "/api/v1/agent/jobs/101/media":
            return httpx.Response(200, content=audio_content)
        elif request.url.path == "/api/v1/agent/jobs/101/logs":
            body = json.loads(request.content.decode("utf-8"))
            logs_received.append(body)
            return httpx.Response(200, json={"error_msg": "", "data": None})
        elif request.url.path == "/api/v1/agent/jobs/101/complete":
            body = json.loads(request.content.decode("utf-8"))
            complete_received.append(body)
            return httpx.Response(200, json={"error_msg": "", "data": None})
        return httpx.Response(404)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(mock_handler),
        headers={"Authorization": "Bearer test-agent-token"},
    )
    reporter = Reporter(
        base_url="http://test-controller:8080",
        agent_token="test-agent-token",
        client=client,
    )

    # 1. Download
    target_file = tmp_path / "downloaded.mp3"
    saved_path = await reporter.download_media(101, str(target_file))
    assert os.path.exists(saved_path)
    assert target_file.read_bytes() == audio_content

    # 2. Report logs
    await reporter.report_logs(
        101,
        progress=45,
        logs=[{"level": "info", "message": "Inference in progress"}],
    )
    assert len(logs_received) == 1
    assert logs_received[0]["progress"] == 45

    # 3. Report completion
    await reporter.report_completion(
        101,
        status="completed",
        duration_seconds=2.45,
        result_text="Mock Result",
        openai_response={"text": "Mock Result"},
    )
    assert len(complete_received) == 1
    assert complete_received[0]["status"] == "completed"
    assert complete_received[0]["duration_seconds"] == 2.45
    assert complete_received[0]["result_text"] == "Mock Result"

    await reporter.close()


# =========================================================================
# 6. Job Runner & Exception Shielding Tests
# =========================================================================


@pytest.mark.asyncio
async def test_job_runner_success(tmp_path: Path) -> None:
    media_dir = tmp_path / "agent_media"
    audio_content = b"VALID_AUDIO_FILE_DATA"

    complete_data: list[dict] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/media"):
            return httpx.Response(200, content=audio_content)
        elif request.url.path.endswith("/logs"):
            return httpx.Response(200, json={"error_msg": "", "data": None})
        elif request.url.path.endswith("/complete"):
            complete_data.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"error_msg": "", "data": None})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    reporter = Reporter("http://test", "token", client=client)

    registry = ModelRegistry(preload_default=False)
    registry.register(
        "mock-whisper-base",
        lambda: MockASREngine("mock-whisper-base", stage_delay=0.0),
    )

    runner = JobRunner(
        reporter=reporter,
        registry=registry,
        media_dir=str(media_dir),
        max_concurrent_jobs=2,
    )

    task = runner.run_job(
        {
            "job_id": 501,
            "model_name": "mock-whisper-base",
            "task_type": "transcribe",
            "media_path": "/api/v1/agent/jobs/501/media",
        }
    )
    await task

    # Assert settlement
    assert len(complete_data) == 1
    assert complete_data[0]["status"] == "completed"
    assert len(complete_data[0]["result_text"]) > 0
    assert complete_data[0]["openai_response"] is not None

    # Verify temp file cleanup
    remaining_files = list(media_dir.glob("job_501_*"))
    assert len(remaining_files) == 0

    assert runner.get_running_jobs_count() == 0


@pytest.mark.asyncio
async def test_job_runner_exception_shielding_no_crash(tmp_path: Path) -> None:
    """An error in any job must report status='failed' to the controller
    and never terminate the agent process."""
    media_dir = tmp_path / "agent_media"
    complete_data: list[dict] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/media"):
            # Simulate 500 download failure
            return httpx.Response(500, text="Internal Server Storage Error")
        elif request.url.path.endswith("/logs"):
            return httpx.Response(200, json={"error_msg": "", "data": None})
        elif request.url.path.endswith("/complete"):
            complete_data.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"error_msg": "", "data": None})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    reporter = Reporter("http://test", "token", client=client)

    registry = ModelRegistry(preload_default=True, debug=True)
    runner = JobRunner(
        reporter=reporter,
        registry=registry,
        media_dir=str(media_dir),
        max_concurrent_jobs=2,
    )

    # Launch failing job
    task = runner.run_job(
        {
            "job_id": 999,
            "model_name": "mock-whisper-base",
            "media_path": "/api/v1/agent/jobs/999/media",
        }
    )
    await task

    # Verify status="failed" reported and error recorded
    assert len(complete_data) == 1
    assert complete_data[0]["status"] == "failed"
    assert "500" in complete_data[0]["error_msg"] or "Error" in complete_data[0]["error_msg"]
    assert runner.get_running_jobs_count() == 0

    # Verify runner is completely healthy and can run subsequent jobs
    # Now provide a successful response for job 1000
    def mock_handler_success(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/media"):
            return httpx.Response(200, content=b"AUDIO")
        elif request.url.path.endswith("/logs"):
            return httpx.Response(200, json={"error_msg": "", "data": None})
        elif request.url.path.endswith("/complete"):
            complete_data.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"error_msg": "", "data": None})
        return httpx.Response(404)

    reporter.client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler_success))
    task2 = runner.run_job(
        {
            "job_id": 1000,
            "model_name": "mock-whisper-base",
            "media_path": "/api/v1/agent/jobs/1000/media",
        }
    )
    await task2
    assert len(complete_data) == 2
    assert complete_data[1]["status"] == "completed"


# =========================================================================
# 7. WebSocket Client Message Routing Tests
# =========================================================================


@pytest.mark.asyncio
async def test_ws_client_message_routing() -> None:
    config = AgentConfig()
    monitor = SystemMonitor()
    registry = ModelRegistry(preload_default=False)
    registry.register(
        "mock-whisper-base",
        lambda: MockASREngine("mock-whisper-base", stage_delay=0.0),
    )

    job_runner = MagicMock(spec=JobRunner)
    job_runner.get_running_jobs_count.return_value = 0

    client = AgentWebSocketClient(config, monitor, registry, job_runner)

    mock_ws = AsyncMock()

    # 1. Test dispatch_job
    dispatch_msg = {
        "type": "command",
        "action": "dispatch_job",
        "payload": {
            "job_id": 42,
            "model_name": "mock-whisper-base",
            "task_type": "transcribe",
            "media_path": "/api/v1/agent/jobs/42/media",
        },
    }
    await client._handle_message(mock_ws, dispatch_msg)
    job_runner.run_job.assert_called_once_with(dispatch_msg["payload"])

    # 2. Test load_model
    load_msg = {
        "type": "command",
        "action": "load_model",
        "payload": {"model_name": "mock-whisper-base"},
    }
    await client._handle_message(mock_ws, load_msg)
    assert "mock-whisper-base" in registry.list_loaded_models()
    assert mock_ws.send.called
    last_sent = json.loads(mock_ws.send.call_args[0][0])
    assert last_sent["type"] == "model_status"
    assert "mock-whisper-base" in last_sent["payload"]["loaded_models"]

    # 3. Test unload_model
    unload_msg = {
        "type": "command",
        "action": "unload_model",
        "payload": {"model_name": "mock-whisper-base"},
    }
    await client._handle_message(mock_ws, unload_msg)
    assert "mock-whisper-base" not in registry.list_loaded_models()
    last_sent = json.loads(mock_ws.send.call_args[0][0])
    assert last_sent["type"] == "model_status"
    assert "mock-whisper-base" not in last_sent["payload"]["loaded_models"]


@pytest.mark.asyncio
async def test_ws_client_message_loop_resilience() -> None:
    """Verify _message_loop handles invalid json, non-dict payloads, and handler exceptions without crashing."""
    config = AgentConfig()
    monitor = SystemMonitor()
    registry = ModelRegistry(preload_default=False)
    job_runner = MagicMock(spec=JobRunner)
    client = AgentWebSocketClient(config, monitor, registry, job_runner)

    class MockAsyncIterWS:
        def __init__(self, msgs: list[str]) -> None:
            self.msgs = msgs
            self.send = AsyncMock()

        def __aiter__(self):
            self._iter = iter(self.msgs)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    messages = [
        "not-a-valid-json",
        json.dumps([1, 2, 3]),  # list payload (non-dict)
        json.dumps("string payload"),  # string payload (non-dict)
        json.dumps({"type": "fail_action"}),  # raises exception
        json.dumps(
            {
                "type": "command",
                "action": "dispatch_job",
                "payload": {"job_id": 77},
            }
        ),
    ]
    mock_ws = MockAsyncIterWS(messages)

    orig_handle = client._handle_message

    async def mock_handle(ws, data):
        if data.get("type") == "fail_action":
            raise RuntimeError("simulated error in handler")
        return await orig_handle(ws, data)

    client._handle_message = mock_handle

    await client._message_loop(mock_ws)
    job_runner.run_job.assert_called_once_with({"job_id": 77})


@pytest.mark.asyncio
async def test_job_runner_sync_engine_gil_protection(tmp_path: Path) -> None:
    """Verify that synchronous / blocking CPU engines run safely via threadpool executor."""
    media_dir = tmp_path / "sync_media"
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"SYNC_AUDIO")

    complete_data: list[dict] = []
    logs_reported: list[dict] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/media"):
            return httpx.Response(200, content=b"SYNC_AUDIO")
        elif request.url.path.endswith("/logs"):
            logs_reported.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"error_msg": "", "data": None})
        elif request.url.path.endswith("/complete"):
            complete_data.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"error_msg": "", "data": None})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    reporter = Reporter("http://test", "token", client=client)

    # Define a pure synchronous engine (not async def)
    class PureSyncEngine(BaseEngine):
        def __init__(self) -> None:
            super().__init__("pure-sync-engine")
            self.loaded = True

        async def load(self) -> None:
            self.loaded = True

        async def unload(self) -> None:
            self.loaded = False

        def transcribe(
            self,
            audio_path: str,
            language: str | None = None,
            task_type: str = "transcribe",
            log_callback=None,
        ) -> dict:
            if log_callback:
                log_callback(75, "Sync inference running...")
            return {
                "task": task_type,
                "language": language or "en",
                "duration": 3.0,
                "text": "Sync engine transcribed text",
                "segments": [],
            }

    registry = ModelRegistry(preload_default=False)
    registry.register("pure-sync-engine", PureSyncEngine)
    await registry.load_model("pure-sync-engine")

    runner = JobRunner(
        reporter=reporter,
        registry=registry,
        media_dir=str(media_dir),
        max_concurrent_jobs=1,
    )

    task = runner.run_job(
        {
            "job_id": 777,
            "model_name": "pure-sync-engine",
            "media_path": "/api/v1/agent/jobs/777/media",
        }
    )
    await task

    assert len(complete_data) == 1
    assert complete_data[0]["status"] == "completed"
    assert complete_data[0]["result_text"] == "Sync engine transcribed text"
    # Ensure logs from sync callback were received
    assert any(log.get("progress") == 75 for log in logs_reported)


@pytest.mark.asyncio
async def test_ws_client_heartbeat_payload() -> None:
    config = AgentConfig(heartbeat_interval=1)
    monitor = SystemMonitor()
    registry = ModelRegistry(preload_default=True, debug=True)
    job_runner = MagicMock(spec=JobRunner)
    job_runner.get_running_jobs_count.return_value = 2

    client = AgentWebSocketClient(config, monitor, registry, job_runner)
    client._running = True

    mock_ws = AsyncMock()

    # Run one iteration of heartbeat loop
    async def stop_soon():
        await asyncio.sleep(0.05)
        client._running = False

    asyncio.create_task(stop_soon())
    await client._heartbeat_loop(mock_ws)

    assert mock_ws.send.called
    sent_payload = json.loads(mock_ws.send.call_args[0][0])
    assert sent_payload["type"] == "heartbeat"
    p = sent_payload["payload"]
    assert "mock-whisper-base" in p["loaded_models"]
    assert p["running_jobs"] == 2
    assert "cpu_percent" in p["system"]
    assert "ram_percent" in p["system"]
    assert "supported_modes" in p
    assert "current_mode" in p


@pytest.mark.asyncio
async def test_ws_client_default_heartbeat_payload_no_mock_model() -> None:
    """Default heartbeat without debug mode must not include mock-whisper-base."""
    config = AgentConfig(heartbeat_interval=1)
    monitor = SystemMonitor()
    registry = ModelRegistry()  # default: debug=False, preload_default=False
    job_runner = MagicMock(spec=JobRunner)
    job_runner.get_running_jobs_count.return_value = 0

    client = AgentWebSocketClient(config, monitor, registry, job_runner)
    client._running = True

    mock_ws = AsyncMock()

    async def stop_soon():
        await asyncio.sleep(0.05)
        client._running = False

    asyncio.create_task(stop_soon())
    await client._heartbeat_loop(mock_ws)

    assert mock_ws.send.called
    sent_payload = json.loads(mock_ws.send.call_args[0][0])
    p = sent_payload["payload"]
    assert p["loaded_models"] == []
    assert "mock-whisper-base" not in p["downloaded_models"]


@pytest.mark.asyncio
async def test_registry_work_mode_and_unload_all() -> None:
    """Test setting work mode, mode validation, and unload_all_models."""
    registry = ModelRegistry(preload_default=True, debug=True)
    assert len(registry.list_loaded_models()) == 1

    # Unload all models
    unloaded = await registry.unload_all_models()
    assert "mock-whisper-base" in unloaded
    assert len(registry.list_loaded_models()) == 0

    # Ensure cpu mode is always supported
    assert "cpu" in registry.get_supported_modes()
    await registry.set_work_mode("cpu")
    assert registry.get_current_mode() == "cpu"

    # Loading a model in CPU mode
    engine = await registry.load_model("mock-whisper-base")
    assert engine.loaded
    assert "mock-whisper-base" in registry.list_loaded_models()

    # Invalid mode raises ValueError
    with pytest.raises(ValueError):
        await registry.set_work_mode("invalid-mode-xyz")


@pytest.mark.asyncio
async def test_ws_client_work_mode_and_unload_all_handling() -> None:
    """Test WS client handling of set_work_mode and unload_all_models messages."""
    config = AgentConfig()
    monitor = SystemMonitor()
    registry = ModelRegistry(preload_default=True, debug=True)
    job_runner = MagicMock(spec=JobRunner)
    client = AgentWebSocketClient(config, monitor, registry, job_runner)

    mock_ws = AsyncMock()

    # 1. Test unload_all_models message
    await client._handle_message(
        mock_ws, {"type": "command", "action": "unload_all_models", "payload": {}}
    )
    assert len(registry.list_loaded_models()) == 0
    assert mock_ws.send.called
    status_msg = json.loads(mock_ws.send.call_args[0][0])
    assert status_msg["type"] == "model_status"
    assert status_msg["payload"]["loaded_models"] == []

    # 2. Test set_work_mode message
    mock_ws.reset_mock()
    await client._handle_message(
        mock_ws, {"type": "command", "action": "set_work_mode", "payload": {"mode": "cpu"}}
    )
    assert registry.get_current_mode() == "cpu"
    assert mock_ws.send.call_count >= 1


def test_qwen3_asr_missing_ffmpeg_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from src.models.qwen3_asr import Qwen3ASREngine

    engine = Qwen3ASREngine("qwen3-asr-0.6b")
    engine.loaded = True
    engine._model = MagicMock()

    # Create dummy non-wav file
    test_mp3 = tmp_path / "test.mp3"
    test_mp3.write_bytes(b"dummy mp3 data")

    def mock_subprocess_run(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    with pytest.raises(RuntimeError) as exc_info:
        engine.transcribe(str(test_mp3))

    assert "ffmpeg" in str(exc_info.value)
    assert "cyphr 命令行客户端" in str(exc_info.value)


@pytest.mark.asyncio
async def test_concurrent_jobs_inference_serialization(tmp_path: Path) -> None:
    """Verify multiple concurrent jobs serialize their inference phase without collision."""
    import time

    media_dir = tmp_path / "media"
    media_dir.mkdir()

    complete_data = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/media" in url_str:
            return httpx.Response(200, content=b"RIFFdummyWAVdata")
        if "/complete" in url_str:
            complete_data.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"status": "ok"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    reporter = Reporter("http://test", "token", client=client)

    active_inferences = 0
    max_active_inferences = 0

    class TrackingSyncEngine(BaseEngine):
        def __init__(self) -> None:
            super().__init__("tracking-engine")
            self.loaded = True

        async def load(self) -> None:
            self.loaded = True

        async def unload(self) -> None:
            self.loaded = False

        def transcribe(
            self,
            audio_path: str,
            language: str | None = None,
            task_type: str = "transcribe",
            log_callback=None,
        ) -> dict:
            nonlocal active_inferences, max_active_inferences
            active_inferences += 1
            if active_inferences > max_active_inferences:
                max_active_inferences = active_inferences
            time.sleep(0.05)
            active_inferences -= 1
            return {
                "task": task_type,
                "language": language or "en",
                "duration": 1.0,
                "text": f"Transcribed {audio_path}",
                "segments": [],
            }

    registry = ModelRegistry(preload_default=False)
    registry.register("tracking-engine", TrackingSyncEngine)
    await registry.load_model("tracking-engine")

    runner = JobRunner(
        reporter=reporter,
        registry=registry,
        media_dir=str(media_dir),
        max_concurrent_jobs=3,
    )

    t1 = runner.run_job(
        {
            "job_id": 101,
            "model_name": "tracking-engine",
            "media_path": "/api/v1/agent/jobs/101/media",
        }
    )
    t2 = runner.run_job(
        {
            "job_id": 102,
            "model_name": "tracking-engine",
            "media_path": "/api/v1/agent/jobs/102/media",
        }
    )
    t3 = runner.run_job(
        {
            "job_id": 103,
            "model_name": "tracking-engine",
            "media_path": "/api/v1/agent/jobs/103/media",
        }
    )

    await asyncio.gather(t1, t2, t3)

    assert len(complete_data) == 3
    # Max active inferences must be 1 because inference is serialized via _inference_lock
    assert max_active_inferences == 1


@pytest.mark.asyncio
async def test_concurrent_jobs_inference_parallel_when_supported(tmp_path: Path) -> None:
    """Verify engines with supports_concurrent_inference=True execute concurrently without serialization."""
    import threading
    import time

    media_dir = tmp_path / "media_par"
    media_dir.mkdir()

    complete_data = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/media" in url_str:
            return httpx.Response(200, content=b"RIFFdummyWAVdata")
        if "/complete" in url_str:
            complete_data.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"status": "ok"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    reporter = Reporter("http://test", "token", client=client)

    active_inferences = 0
    max_active_inferences = 0
    lock = threading.Lock()

    class ConcurrentSyncEngine(BaseEngine):
        supports_concurrent_inference = True

        def __init__(self) -> None:
            super().__init__("concurrent-engine")
            self.loaded = True

        async def load(self) -> None:
            self.loaded = True

        async def unload(self) -> None:
            self.loaded = False

        def transcribe(
            self,
            audio_path: str,
            language: str | None = None,
            task_type: str = "transcribe",
            log_callback=None,
        ) -> dict:
            nonlocal active_inferences, max_active_inferences
            with lock:
                active_inferences += 1
                if active_inferences > max_active_inferences:
                    max_active_inferences = active_inferences
            time.sleep(0.08)
            with lock:
                active_inferences -= 1
            return {
                "task": task_type,
                "language": language or "en",
                "duration": 1.0,
                "text": f"Transcribed {audio_path}",
                "segments": [],
            }

    registry = ModelRegistry(preload_default=False)
    registry.register("concurrent-engine", ConcurrentSyncEngine)
    await registry.load_model("concurrent-engine")

    runner = JobRunner(
        reporter=reporter,
        registry=registry,
        media_dir=str(media_dir),
        max_concurrent_jobs=3,
    )

    t1 = runner.run_job(
        {
            "job_id": 201,
            "model_name": "concurrent-engine",
            "media_path": "/api/v1/agent/jobs/201/media",
        }
    )
    t2 = runner.run_job(
        {
            "job_id": 202,
            "model_name": "concurrent-engine",
            "media_path": "/api/v1/agent/jobs/202/media",
        }
    )
    t3 = runner.run_job(
        {
            "job_id": 203,
            "model_name": "concurrent-engine",
            "media_path": "/api/v1/agent/jobs/203/media",
        }
    )

    await asyncio.gather(t1, t2, t3)

    assert len(complete_data) == 3
    # Max active inferences must be >= 2 because concurrent engine bypasses single-flight _inference_lock
    assert max_active_inferences >= 2


# =========================================================================
# 8. P0 Concurrency & Lifecycle Protection Tests
# =========================================================================


@pytest.mark.asyncio
async def test_concurrent_load_model_singleton() -> None:
    """Verify concurrent load_model calls for the same model serialize and only instantiate once."""
    init_call_count = 0
    load_call_count = 0

    class SlowLoadEngine(BaseEngine):
        def __init__(self) -> None:
            super().__init__("slow-model")
            nonlocal init_call_count
            init_call_count += 1

        async def load(self, work_mode: str = "cpu") -> None:
            nonlocal load_call_count
            load_call_count += 1
            await asyncio.sleep(0.05)  # simulate weight loading
            self.loaded = True

        async def unload(self) -> None:
            self.loaded = False

        async def transcribe(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"text": "ok"}

    registry = ModelRegistry(preload_default=False)
    registry.register("slow-model", SlowLoadEngine)

    # Concurrently launch 10 tasks all requesting to load "slow-model"
    results = await asyncio.gather(*[registry.load_model("slow-model") for _ in range(10)])

    assert init_call_count == 1
    assert load_call_count == 1
    # All 10 callers must receive the exact same engine instance
    for engine in results:
        assert engine is results[0]
        assert engine.loaded is True


@pytest.mark.asyncio
async def test_acquire_engine_protects_during_inference_and_drains_on_unload() -> None:
    """Verify active inferences are tracked and unload_model safely waits for inference drain."""
    inference_running = False
    inference_completed = False
    unloaded_occurred = False

    class SafeLifecycleEngine(BaseEngine):
        def __init__(self) -> None:
            super().__init__("safe-model")
            self.loaded = True

        async def load(self, work_mode: str = "cpu") -> None:
            self.loaded = True

        async def unload(self) -> None:
            nonlocal unloaded_occurred
            assert not inference_running, "unload occurred while inference was still running!"
            unloaded_occurred = True
            self.loaded = False

        async def transcribe(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"text": "done"}

    registry = ModelRegistry(preload_default=False)
    registry.register("safe-model", SafeLifecycleEngine)
    await registry.load_model("safe-model")

    async def run_simulated_inference() -> None:
        nonlocal inference_running, inference_completed
        async with registry.acquire_engine("safe-model"):
            inference_running = True
            await asyncio.sleep(0.1)  # simulate ongoing transcription
            inference_running = False
            inference_completed = True

    # Start inference in background
    inference_task = asyncio.create_task(run_simulated_inference())
    # Give it a moment to enter acquire_engine
    await asyncio.sleep(0.02)
    assert inference_running is True

    # Concurrently attempt to unload model
    unload_res = await registry.unload_model("safe-model", timeout=5.0)
    await inference_task

    assert unload_res is True
    assert inference_completed is True
    assert unloaded_occurred is True
    assert "safe-model" not in registry.list_loaded_models()


@pytest.mark.asyncio
async def test_dynamic_semaphore_concurrency_control() -> None:
    """Verify DynamicSemaphore handles dynamic capacity changes safely without permit drift."""
    from src.job_runner import DynamicSemaphore

    sem = DynamicSemaphore(initial_capacity=2)
    assert sem.capacity == 2

    active_tasks = 0
    max_concurrent = 0

    async def worker() -> None:
        nonlocal active_tasks, max_concurrent
        async with sem:
            active_tasks += 1
            if active_tasks > max_concurrent:
                max_concurrent = active_tasks
            await asyncio.sleep(0.05)
            active_tasks -= 1

    # Run 4 tasks with capacity 2
    tasks = [asyncio.create_task(worker()) for _ in range(4)]
    await asyncio.gather(*tasks)
    assert max_concurrent <= 2

    # Dynamically expand capacity to 5
    sem.set_capacity(5)
    assert sem.capacity == 5
    max_concurrent = 0
    tasks = [asyncio.create_task(worker()) for _ in range(5)]
    await asyncio.gather(*tasks)
    assert max_concurrent <= 5


@pytest.mark.asyncio
async def test_dynamic_semaphore_scale_down() -> None:
    """Verify DynamicSemaphore smoothly handles capacity reduction while tasks are running."""
    from src.job_runner import DynamicSemaphore

    sem = DynamicSemaphore(initial_capacity=4)
    active = 0
    post_scale_down_max = 0
    scaled_down = False

    async def worker() -> None:
        nonlocal active, post_scale_down_max
        async with sem:
            active += 1
            if scaled_down and active > post_scale_down_max:
                post_scale_down_max = active
            await asyncio.sleep(0.05)
            active -= 1

    # Start 4 workers with capacity 4
    tasks = [asyncio.create_task(worker()) for _ in range(4)]
    await asyncio.sleep(0.01)

    # Dynamically scale down to 1
    sem.set_capacity(1)
    scaled_down = True
    assert sem.capacity == 1

    # Add 2 more workers who should now be limited to capacity 1 once existing workers finish
    tasks.extend([asyncio.create_task(worker()) for _ in range(2)])
    await asyncio.gather(*tasks)

    # Once old workers drained, new workers must respect the new capacity limit
    assert post_scale_down_max <= 2


@pytest.mark.asyncio
async def test_acquire_engine_drain_timeout_force_unload() -> None:
    """Verify unload_model timeout forces unload when inference takes longer than timeout."""

    class LongInferenceEngine(BaseEngine):
        def __init__(self) -> None:
            super().__init__("long-model")
            self.loaded = True

        async def load(self, work_mode: str = "cpu") -> None:
            self.loaded = True

        async def unload(self) -> None:
            self.loaded = False

        async def transcribe(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"text": "done"}

    registry = ModelRegistry(preload_default=False)
    registry.register("long-model", LongInferenceEngine)
    await registry.load_model("long-model")

    inference_done = False

    async def slow_inference() -> None:
        nonlocal inference_done
        async with registry.acquire_engine("long-model"):
            await asyncio.sleep(0.2)
            inference_done = True

    t = asyncio.create_task(slow_inference())
    await asyncio.sleep(0.01)

    # Unload with a very short timeout (0.05s) - must force unload after timeout
    unloaded = await registry.unload_model("long-model", timeout=0.05)
    assert unloaded is True
    assert "long-model" not in registry.list_loaded_models()

    await t
    assert inference_done is True
    # Verify internal drain events and inference counts are cleaned up
    assert "long-model" not in registry._drain_events
    assert "long-model" not in registry._inference_counts


@pytest.mark.asyncio
async def test_acquire_engine_exception_resilience() -> None:
    """Verify acquire_engine properly decrements reference count even if inference raises exception."""

    class FaultyEngine(BaseEngine):
        def __init__(self) -> None:
            super().__init__("faulty-model")
            self.loaded = True

        async def load(self, work_mode: str = "cpu") -> None:
            self.loaded = True

        async def unload(self) -> None:
            self.loaded = False

        async def transcribe(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("Fatal CUDA hardware failure simulation")

    registry = ModelRegistry(preload_default=False)
    registry.register("faulty-model", FaultyEngine)
    await registry.load_model("faulty-model")

    with pytest.raises(RuntimeError, match="CUDA hardware failure"):
        async with registry.acquire_engine("faulty-model") as engine:
            await engine.transcribe("fake.wav")

    # Reference count must be completely cleaned up
    assert "faulty-model" not in registry._inference_counts

    # Subsequent unload must succeed immediately without waiting or hanging
    unloaded = await registry.unload_model("faulty-model", timeout=1.0)
    assert unloaded is True
    assert "faulty-model" not in registry.list_loaded_models()


# =========================================================================
# 9. P1 In-Memory Audio Pipeline & Multi-GPU Tests
# =========================================================================


def test_resolve_device_and_dtype_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from src.models.qwen3_asr import resolve_device_and_dtype

    # 1. CPU explicit
    dev, dtype, bs = resolve_device_and_dtype("cpu")
    assert dev == "cpu"
    assert dtype == torch.float32
    assert bs == 1

    # 2. Explicit QWEN3_ASR_DEVICE env
    monkeypatch.setenv("QWEN3_ASR_DEVICE", "cpu")
    dev, dtype, bs = resolve_device_and_dtype("gpu")
    assert dev == "cpu"
    monkeypatch.delenv("QWEN3_ASR_DEVICE")

    # 3. CUDA fallback to CPU if torch.cuda not available
    with patch("torch.cuda.is_available", return_value=False):
        dev, dtype, bs = resolve_device_and_dtype("cuda:0")
        assert dev == "cpu"
        assert dtype == torch.float32

    # 4. Multi-GPU device selection via CUDA_DEVICE_INDEX
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=4),
        patch("torch.cuda.is_bf16_supported", return_value=True),
    ):
        monkeypatch.setenv("CUDA_DEVICE_INDEX", "2")
        dev, dtype, bs = resolve_device_and_dtype("gpu")
        assert dev == "cuda:2"
        assert dtype == torch.bfloat16
        assert bs == 16


def test_registry_multi_gpu_discovery() -> None:
    from src.models.registry import detect_supported_modes

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=3),
    ):
        modes, default_mode = detect_supported_modes()
        assert "cpu" in modes
        assert "gpu" in modes
        assert "cuda:0" in modes
        assert "cuda:1" in modes
        assert "cuda:2" in modes
        assert default_mode == "gpu"


def test_qwen3_asr_in_memory_ffmpeg_pipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify non-standard audio uses in-memory pipe and never writes temporary wav files to disk."""
    import subprocess

    import numpy as np

    from src.models.qwen3_asr import Qwen3ASREngine

    engine = Qwen3ASREngine("qwen3-asr-0.6b")
    engine.loaded = True

    # Mock output object from qwen_asr model
    mock_out = MagicMock()
    mock_out.text = "In-memory streaming transcribed successfully"
    mock_out.language = "en"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = [mock_out]
    engine._model = mock_model

    test_mp3 = tmp_path / "stream_test.mp3"
    test_mp3.write_bytes(b"dummy mp3 header and frames")

    # Generate 1 second of fake 16kHz int16 PCM data
    sample_rate = 16000
    fake_pcm_samples = (np.sin(np.linspace(0, 2 * np.pi * 440, sample_rate)) * 16000).astype(
        np.int16
    )
    fake_pcm_bytes = fake_pcm_samples.tobytes()

    executed_cmd = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        nonlocal executed_cmd
        executed_cmd = cmd
        res = MagicMock()
        res.returncode = 0
        res.stdout = fake_pcm_bytes
        res.stderr = b""
        return res

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    initial_files = list(tmp_path.glob("*.wav"))

    result = engine.transcribe(str(test_mp3))

    # Verify ffmpeg was invoked with stdout pipe
    assert "-f" in executed_cmd
    assert "s16le" in executed_cmd
    assert "pipe:1" in executed_cmd
    assert str(test_mp3) in executed_cmd

    # Verify no temporary wav files were created on disk
    after_files = list(tmp_path.glob("*.wav"))
    assert len(after_files) == len(initial_files)

    # Verify transcribed text
    assert result["text"] == "In-memory streaming transcribed successfully"
    assert result["language"] == "en"


# =========================================================================
# 9. Qwen3-ASR vLLM Engine Configuration & Inference Tests
# =========================================================================


def test_qwen3_asr_vllm_engine_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Qwen3ASREngine default configuration and environment variable overrides."""
    from src.models.qwen3_asr import Qwen3ASREngine

    monkeypatch.delenv("VLLM_GPU_MEMORY_UTILIZATION", raising=False)
    monkeypatch.delenv("VLLM_MAX_MODEL_LEN", raising=False)
    monkeypatch.delenv("QWEN3_ASR_BATCH_SIZE", raising=False)

    engine_default = Qwen3ASREngine("qwen3-asr-0.6b")
    assert engine_default.gpu_memory_utilization == 0.60
    assert engine_default.max_model_len == 16384
    assert engine_default.batch_size == 4

    # Test environment variable overrides
    monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "0.75")
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "20480")
    monkeypatch.setenv("QWEN3_ASR_BATCH_SIZE", "6")

    engine_custom = Qwen3ASREngine("qwen3-asr-0.6b")
    assert engine_custom.gpu_memory_utilization == 0.75
    assert engine_custom.max_model_len == 20480
    assert engine_custom.batch_size == 6


def test_qwen3_asr_vllm_load_blocking_instantiation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify _load_blocking calls Qwen3ASRModel.LLM with appropriate arguments."""
    import json

    from src.models.qwen3_asr import Qwen3ASREngine

    model_dir = tmp_path / "mock_qwen3"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_asr"}))

    captured_kwargs = {}

    class MockLLM:
        @classmethod
        def LLM(cls, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            mock_inst = MagicMock()
            return mock_inst

    monkeypatch.setattr("qwen_asr.Qwen3ASRModel", MockLLM)
    monkeypatch.setattr("src.models.qwen3_asr.resolve_devices", lambda work_mode: [0])

    engine = Qwen3ASREngine("qwen3-asr-0.6b", model_dir=model_dir)
    engine._load_blocking(work_mode="gpu")

    assert captured_kwargs["model"] == str(model_dir.resolve())
    assert captured_kwargs["gpu_memory_utilization"] == 0.60
    assert captured_kwargs["max_model_len"] == 16384
    assert captured_kwargs["max_inference_batch_size"] == 4
    assert engine._model is not None


def test_qwen3_asr_vllm_transcribe_thread_safe_lock(tmp_path: Path) -> None:
    """Verify concurrent transcribe requests are serialized through engine lock."""
    import concurrent.futures
    import threading
    import time

    import numpy as np
    import soundfile as sf

    from src.models.qwen3_asr import Qwen3ASREngine

    engine = Qwen3ASREngine("qwen3-asr-0.6b")
    engine.loaded = True

    concurrent_calls = 0
    max_concurrent_observed = 0
    lock = threading.Lock()

    mock_m = MagicMock()

    def mock_transcribe(audio: list, language: str | None = None) -> list:
        nonlocal concurrent_calls, max_concurrent_observed
        with lock:
            concurrent_calls += 1
            if concurrent_calls > max_concurrent_observed:
                max_concurrent_observed = concurrent_calls
        time.sleep(0.05)
        outs = []
        for _ in audio:
            o = MagicMock()
            o.text = "vllm transcribed text"
            o.language = "zh"
            outs.append(o)
        with lock:
            concurrent_calls -= 1
        return outs

    mock_m.transcribe = mock_transcribe
    engine._model = mock_m

    dummy_audio = tmp_path / "test_vllm.wav"
    samplerate = 16000
    samples = np.zeros(int(samplerate * 1.0), dtype=np.float32)
    sf.write(str(dummy_audio), samples, samplerate)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futs = [executor.submit(engine.transcribe, str(dummy_audio)) for _ in range(3)]
        results = [f.result() for f in futs]

    assert len(results) == 3
    assert all("vllm transcribed text" in r["text"] for r in results)
    # The lock must guarantee max concurrent inside transcribe loop is 1
    assert max_concurrent_observed == 1


def test_qwen3_asr_sleep_mode_on_idle_and_auto_wake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify engine automatically enters Sleep Mode when idle and wakes up on new task."""
    import time

    import numpy as np
    import soundfile as sf

    from src.models.qwen3_asr import Qwen3ASREngine

    dummy_audio = tmp_path / "test_sleep.wav"
    samplerate = 16000
    sf.write(str(dummy_audio), np.zeros(int(samplerate * 0.5), dtype=np.float32), samplerate)

    monkeypatch.setenv("VLLM_SLEEP_IDLE_SECONDS", "0.1")
    engine = Qwen3ASREngine("qwen3-asr-0.6b")
    engine.loaded = True

    mock_llm = MagicMock()
    mock_llm.sleep = MagicMock()
    mock_llm.wake_up = MagicMock()

    mock_model = MagicMock()
    mock_model.model = mock_llm

    def mock_transcribe(audio: list, language: str | None = None) -> list:
        out = MagicMock()
        out.text = "wake up test"
        out.language = "en"
        return [out]

    mock_model.transcribe = mock_transcribe
    engine._model = mock_model

    # 1. Run transcription
    res1 = engine.transcribe(str(dummy_audio))
    assert res1["text"] == "wake up test"
    assert engine.is_sleeping is False
    assert mock_llm.sleep.call_count == 0

    # 2. Wait for idle timeout (0.1s + buffer)
    time.sleep(0.2)
    assert engine.is_sleeping is True
    assert mock_llm.sleep.call_count == 1
    mock_llm.sleep.assert_called_with(level=1)

    # 3. New task arrives while asleep -> auto wake-up
    res2 = engine.transcribe(str(dummy_audio))
    assert res2["text"] == "wake up test"
    assert engine.is_sleeping is False
    assert mock_llm.wake_up.call_count == 1


def test_qwen3_asr_sleep_timer_cancelled_by_rapid_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify new request cancels pending sleep timer before it enters sleep mode."""
    import time

    import numpy as np
    import soundfile as sf

    from src.models.qwen3_asr import Qwen3ASREngine

    dummy_audio = tmp_path / "test_rapid.wav"
    samplerate = 16000
    sf.write(str(dummy_audio), np.zeros(int(samplerate * 0.5), dtype=np.float32), samplerate)

    monkeypatch.setenv("VLLM_SLEEP_IDLE_SECONDS", "0.3")
    engine = Qwen3ASREngine("qwen3-asr-0.6b")
    engine.loaded = True

    mock_llm = MagicMock()
    mock_llm.sleep = MagicMock()
    mock_llm.wake_up = MagicMock()

    mock_model = MagicMock()
    mock_model.model = mock_llm

    def mock_transcribe(audio: list, language: str | None = None) -> list:
        out = MagicMock()
        out.text = "rapid test"
        out.language = "en"
        return [out]

    mock_model.transcribe = mock_transcribe
    engine._model = mock_model

    # Request 1
    engine.transcribe(str(dummy_audio))
    assert mock_llm.sleep.call_count == 0

    # Wait 0.1s (timer is armed for 0.3s)
    time.sleep(0.1)
    assert mock_llm.sleep.call_count == 0

    # Request 2 arrives before timer expires -> resets timer
    engine.transcribe(str(dummy_audio))
    assert mock_llm.sleep.call_count == 0

    # Wait another 0.1s
    time.sleep(0.1)
    assert mock_llm.sleep.call_count == 0
    assert engine.is_sleeping is False

    # Wait for timer to finally expire (0.3s from Request 2 + buffer)
    time.sleep(0.3)
    assert mock_llm.sleep.call_count == 1
    assert engine.is_sleeping is True


def test_qwen3_asr_enter_sleep_and_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default sleep_idle_seconds is 5.0 and enter_sleep triggers immediate sleep."""
    from src.models.qwen3_asr import Qwen3ASREngine

    monkeypatch.delenv("VLLM_SLEEP_IDLE_SECONDS", raising=False)
    engine = Qwen3ASREngine("qwen3-asr-0.6b")
    assert engine.sleep_idle_seconds == 5.0

    mock_llm = MagicMock()
    mock_llm.sleep = MagicMock()
    mock_model = MagicMock()
    mock_model.model = mock_llm
    engine._model = mock_model
    engine.loaded = True

    assert engine.is_sleeping is False
    engine.enter_sleep()
    assert engine.is_sleeping is True
    mock_llm.sleep.assert_called_once_with(level=1)


@pytest.mark.asyncio
async def test_registry_preemptively_sleeps_idle_models_on_new_model() -> None:
    """Verify acquiring a new model immediately offloads the KV cache of idle loaded models."""
    from src.models.registry import ModelRegistry

    registry = ModelRegistry()

    m1 = MagicMock()
    m1.loaded = True
    m1.load = AsyncMock()
    m1.enter_sleep = MagicMock()

    m2 = MagicMock()
    m2.loaded = True
    m2.load = AsyncMock()
    m2.enter_sleep = MagicMock()

    registry.register("model-1", lambda: m1)
    registry.register("model-2", lambda: m2)

    # Acquire model-1 first
    async with registry.acquire_engine("model-1") as eng1:
        assert eng1 is m1
        assert m1.enter_sleep.call_count == 0

    # Model-1 is now idle (0 active inferences). Now acquire model-2
    async with registry.acquire_engine("model-2") as eng2:
        assert eng2 is m2
        # m1 must have been immediately put to sleep!
        assert m1.enter_sleep.call_count == 1
        assert m2.enter_sleep.call_count == 0


@pytest.mark.asyncio
async def test_registry_local_idle_auto_unload() -> None:
    """Verify local idle timer unloads all models when no tasks run for auto_unload duration."""
    from src.models.registry import ModelRegistry

    registry = ModelRegistry()
    registry._idle_unload_delay = 0.1  # Test delay in seconds
    registry.set_auto_unload_minutes(1)

    callback_called = False

    async def on_unloaded() -> None:
        nonlocal callback_called
        callback_called = True

    registry.set_auto_unload_callback(on_unloaded)

    m1 = MagicMock()
    m1.loaded = True
    unload_called = False

    async def mock_unload() -> None:
        nonlocal unload_called
        unload_called = True
        m1.loaded = False

    m1.unload = mock_unload
    registry.register("auto-unload-model", lambda: m1)

    # Load model
    await registry.load_model("auto-unload-model")
    assert "auto-unload-model" in registry.list_loaded_models()

    # Wait for idle unload to trigger
    await asyncio.sleep(0.2)
    assert unload_called is True
    assert callback_called is True
    assert len(registry.list_loaded_models()) == 0


@pytest.mark.asyncio
async def test_ws_client_set_config_updates_auto_unload_minutes() -> None:
    """Verify set_config updates registry.auto_unload_minutes and triggers heartbeat."""
    from src.ws_client import AgentWebSocketClient

    config = AgentConfig()
    monitor = MagicMock()
    monitor.collect.return_value = {}
    registry = ModelRegistry()
    job_runner = MagicMock()
    job_runner.get_running_jobs_count.return_value = 0

    client = AgentWebSocketClient(config, monitor, registry, job_runner)

    mock_ws = AsyncMock()
    set_cfg_msg = {
        "type": "command",
        "action": "set_config",
        "payload": {
            "max_concurrent_jobs": 4,
            "auto_unload_minutes": 15,
        },
    }

    await client._handle_message(mock_ws, set_cfg_msg)

    assert registry.get_auto_unload_minutes() == 15
    job_runner.set_max_concurrent_jobs.assert_called_once_with(4)
    assert mock_ws.send.call_count >= 1


def test_resolve_devices_cpu_and_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resolve_devices parses CPU, explicit CUDA index, CUDA_VISIBLE_DEVICES, and multi-device counts."""
    from src.models.qwen3_asr import resolve_devices

    monkeypatch.delenv("QWEN3_ASR_DEVICE", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_INDEX", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    # 1. Explicit cpu
    assert resolve_devices("cpu") == ["cpu"]

    # 2. Explicit cuda:1
    assert resolve_devices("cuda:1") == [1]

    # 3. Environment QWEN3_ASR_DEVICE
    monkeypatch.setenv("QWEN3_ASR_DEVICE", "cuda:2")
    assert resolve_devices("gpu") == [2]
    monkeypatch.delenv("QWEN3_ASR_DEVICE", raising=False)

    # 4. Environment CUDA_DEVICE_INDEX
    monkeypatch.setenv("CUDA_DEVICE_INDEX", "3")
    assert resolve_devices("gpu") == [3]
    monkeypatch.delenv("CUDA_DEVICE_INDEX", raising=False)

    # 5. CUDA_VISIBLE_DEVICES list
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0, 1")
    assert resolve_devices("gpu") == [0, 1]


@pytest.mark.asyncio
async def test_qwen3_asr_multi_gpu_worker_pool_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify multi-device worker pool initialization, round-robin dispatch, sleep, and unload."""
    import json

    from src.models.qwen3_asr import Qwen3ASREngine

    dummy_audio = tmp_path / "pool_test.wav"
    dummy_audio.write_bytes(b"dummy")

    model_dir = tmp_path / "mock_qwen3_pool"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_asr"}))

    created_proxies = []

    class MockWorkerProxy:
        def __init__(self, device_id, **kwargs):
            self.device_id = device_id
            self.kwargs = kwargs
            self.wait_ready_called = False
            self.sleep_called = False
            self.stop_called = False
            created_proxies.append(self)

        def wait_ready(self, timeout: float = 60.0):
            self.wait_ready_called = True

        def execute_transcribe(self, audio_path, language, task_type, log_callback=None):
            return {
                "text": f"device_{self.device_id}_transcribed",
                "device": self.device_id,
            }

        def enter_sleep(self):
            self.sleep_called = True

        def stop(self, timeout: float = 3.0):
            self.stop_called = True

    monkeypatch.setattr("src.models.worker.WorkerProxy", MockWorkerProxy)
    monkeypatch.setattr("src.models.qwen3_asr.resolve_devices", lambda wm: [0, 1])

    engine = Qwen3ASREngine("qwen3-asr-0.6b", model_dir=model_dir)
    await engine.load(work_mode="gpu")

    assert len(created_proxies) == 2
    assert created_proxies[0].device_id == 0
    assert created_proxies[1].device_id == 1
    assert created_proxies[0].wait_ready_called is True
    assert created_proxies[1].wait_ready_called is True
    assert engine._worker_queue.qsize() == 2

    # First transcribe should use device 0
    res1 = engine.transcribe(str(dummy_audio))
    assert res1["text"] == "device_0_transcribed"
    assert engine._worker_queue.qsize() == 2

    # Second transcribe should use device 1
    res2 = engine.transcribe(str(dummy_audio))
    assert res2["text"] == "device_1_transcribed"

    # Test immediate sleep forwarding
    engine.enter_sleep()
    assert created_proxies[0].sleep_called is True
    assert created_proxies[1].sleep_called is True
    assert engine.is_sleeping is True

    # Test unload
    await engine.unload()
    assert created_proxies[0].stop_called is True
    assert created_proxies[1].stop_called is True
    assert len(engine._workers) == 0
    assert engine._worker_queue.empty()


def test_system_monitor_multi_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SystemMonitor aggregates multi-GPU VRAM and max utilization."""
    import sys

    from src.monitor import SystemMonitor

    monitor = SystemMonitor()

    class MockUtil:
        def __init__(self, gpu):
            self.gpu = gpu

    class MockMem:
        def __init__(self, used_mb, total_mb):
            self.used = used_mb * 1024 * 1024
            self.total = total_mb * 1024 * 1024

    mock_nvml = MagicMock()
    mock_nvml.nvmlDeviceGetCount.return_value = 2
    mock_nvml.nvmlDeviceGetHandleByIndex.side_effect = ["h0", "h1"]
    mock_nvml.nvmlDeviceGetUtilizationRates.side_effect = [MockUtil(25.0), MockUtil(80.0)]
    mock_nvml.nvmlDeviceGetMemoryInfo.side_effect = [MockMem(2000, 24000), MockMem(4000, 24000)]

    monkeypatch.setitem(sys.modules, "pynvml", mock_nvml)

    util, used_mb, total_mb = monitor._collect_gpu()
    assert util == 80.0
    assert used_mb == 6000
    assert total_mb == 48000


# =========================================================================
# 12. Dynamic VRAM Probe, Multi-GPU & Pull Mechanism Tests
# =========================================================================


def test_dynamic_vram_ratio_calculation() -> None:
    """Verify dynamic VRAM ratio calculates utilization based on remaining free memory."""
    from src.models.worker import calculate_dynamic_gpu_utilization

    # Scenario: Total 16GB, Free 16GB, free_ratio=0.70, max_util=0.60
    # Expected: (16 * 0.70) / 16 = 0.70, clamped to max_util 0.60
    util = calculate_dynamic_gpu_utilization(
        device_id=0,
        gpu_memory_utilization=0.60,
        free_ratio=0.70,
        mock_free_bytes=16 * 1024**3,
        mock_total_bytes=16 * 1024**3,
    )
    assert util == 0.60

    # Scenario: Total 16GB, Free 6GB (due to other processes), free_ratio=0.70, max_util=0.60
    # Expected: (6 * 0.70) / 16 = 0.2625
    util2 = calculate_dynamic_gpu_utilization(
        device_id=0,
        gpu_memory_utilization=0.60,
        free_ratio=0.70,
        mock_free_bytes=6 * 1024**3,
        mock_total_bytes=16 * 1024**3,
    )
    assert round(util2, 4) == 0.2625


@pytest.mark.asyncio
async def test_scenario_1_initial_load_card1_full_card2_idle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scenario 1:
    (a) Before load, Card 1 VRAM is 90% occupied (< 2048 MB free), Card 2 is 0% occupied.
    (b) Expected: Card 1 skips loading, Card 2 loads model and executes inference.
    """
    import torch

    from src.models.qwen3_asr import Qwen3ASREngine, resolve_devices

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def mock_mem_get_info(dev=0):
        dev_idx = int(dev) if isinstance(dev, (int, str)) and str(dev).isdigit() else 0
        if dev_idx == 0:
            return (1600 * 1024 * 1024, 16000 * 1024 * 1024)
        return (16000 * 1024 * 1024, 16000 * 1024 * 1024)

    monkeypatch.setattr(torch.cuda, "mem_get_info", mock_mem_get_info)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_INDEX", raising=False)
    monkeypatch.delenv("QWEN3_ASR_DEVICE", raising=False)

    # 1. resolve_devices should filter out dev 0 and return only [1]
    devices = resolve_devices("gpu")
    assert devices == [1]

    # 2. Engine load should only initialize worker on dev 1
    config_file = tmp_path / "config.json"
    config_file.write_text('{"model_type": "qwen3_asr"}')

    created_proxies = []

    class MockProxy:
        def __init__(self, device_id, **kwargs):
            self.device_id = device_id
            self.is_sleeping = False
            created_proxies.append(self)

        def wait_ready(self, timeout=60.0):
            pass

        def execute_transcribe(self, audio_path, language, task_type, log_callback=None):
            return {"task": "transcribe", "text": f"Transcribed on card {self.device_id}"}

        def enter_sleep(self):
            self.is_sleeping = True

        def stop(self, timeout=3.0):
            pass

    monkeypatch.setattr("src.models.worker.WorkerProxy", MockProxy)

    engine = Qwen3ASREngine(model_name="qwen3-asr-0.6b", model_dir=tmp_path)
    await engine.load("gpu")

    # Verify only 1 worker was created and it is on device 1 (Card 2)
    assert len(created_proxies) == 1
    assert created_proxies[0].device_id == 1
    assert len(engine._workers) == 1
    assert engine._workers[0].device_id == 1

    # Verify inference executes on Card 2
    dummy_audio = tmp_path / "audio.wav"
    dummy_audio.write_bytes(b"dummy audio")
    result = engine.transcribe(str(dummy_audio))
    assert result["text"] == "Transcribed on card 1"


def test_scenario_2_dynamic_scheduling_card1_external_task_card2_executes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scenario 2:
    (a) Both cards loaded and sleeping. External training occupies Card 1 up to 90% (< 1536 MB free).
    (b) Expected: Card 1 cannot restore KV cache and is skipped; Card 2 wakes up and executes inference.
    """
    import sys

    from src.models.qwen3_asr import Qwen3ASREngine

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    # Card 1 has 500 MB (< 1536 MB), Card 2 has 14000 MB (>= 1536 MB)
    def mock_mem_get_info(dev=0):
        dev_idx = int(dev) if isinstance(dev, (int, str)) and str(dev).isdigit() else 0
        if dev_idx == 0:
            return (500 * 1024 * 1024, 16000 * 1024 * 1024)
        return (14000 * 1024 * 1024, 16000 * 1024 * 1024)

    mock_torch.cuda.mem_get_info.side_effect = mock_mem_get_info
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    class MockProxy:
        def __init__(self, device_id):
            self.device_id = device_id
            self.is_sleeping = True  # Both sleeping initially
            self.executed = False

        def execute_transcribe(self, audio_path, language, task_type, log_callback=None):
            self.executed = True
            self.is_sleeping = False
            return {"task": "transcribe", "text": f"Transcribed by card {self.device_id}"}

    worker_card1 = MockProxy(device_id=0)
    worker_card2 = MockProxy(device_id=1)

    engine = Qwen3ASREngine(model_name="qwen3-asr-0.6b", model_dir=tmp_path)
    engine.loaded = True
    engine._workers = [worker_card1, worker_card2]
    engine._worker_queue.put(worker_card1)
    engine._worker_queue.put(worker_card2)

    dummy_audio = tmp_path / "audio.wav"
    dummy_audio.write_bytes(b"dummy audio")

    # Transcribe task arrives
    result = engine.transcribe(str(dummy_audio))

    # Expect: Card 1 was skipped (did not execute), Card 2 executed
    assert worker_card1.executed is False
    assert worker_card2.executed is True
    assert result["text"] == "Transcribed by card 1"


def test_scenario_3_both_cards_exceeded_reject_job_and_cooldown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scenario 3:
    (a) Both cards loaded and sleeping. External training occupies BOTH cards up to 90% (< 1536 MB free).
    (b) Expected: Neither can wake up -> InsufficientVRAMError raised.
    """
    import sys

    from src.models.qwen3_asr import InsufficientVRAMError, Qwen3ASREngine

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    # Both cards have 500 MB (< 1536 MB)
    mock_torch.cuda.mem_get_info.return_value = (500 * 1024 * 1024, 16000 * 1024 * 1024)
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    class MockProxy:
        def __init__(self, device_id):
            self.device_id = device_id
            self.is_sleeping = True

        def execute_transcribe(self, audio_path, language, task_type, log_callback=None):
            return {"task": "transcribe", "text": "ok"}

    worker_card1 = MockProxy(device_id=0)
    worker_card2 = MockProxy(device_id=1)

    engine = Qwen3ASREngine(model_name="qwen3-asr-0.6b", model_dir=tmp_path)
    engine.loaded = True
    engine._workers = [worker_card1, worker_card2]
    engine._worker_queue.put(worker_card1)
    engine._worker_queue.put(worker_card2)

    dummy_audio = tmp_path / "audio.wav"
    dummy_audio.write_bytes(b"dummy audio")

    with pytest.raises(InsufficientVRAMError) as exc_info:
        engine.transcribe(str(dummy_audio))

    assert "insufficient free vram" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_job_runner_reject_on_insufficient_vram(tmp_path: Path) -> None:
    """Verify JobRunner catches InsufficientVRAMError and invokes on_job_rejected instead of marking failed."""
    from src.models.qwen3_asr import InsufficientVRAMError

    mock_reporter = AsyncMock()
    mock_registry = MagicMock()

    class FaultyEngine(BaseEngine):
        supports_concurrent_inference = True

        async def load(self, work_mode: str = "gpu") -> None:
            self.loaded = True

        async def unload(self) -> None:
            self.loaded = False

        def transcribe(self, audio_path, language=None, task_type="transcribe", log_callback=None):
            raise InsufficientVRAMError("All GPU workers have insufficient free VRAM")

    faulty_engine = FaultyEngine("qwen3-asr-0.6b")
    faulty_engine.loaded = True

    @asynccontextmanager
    async def mock_acquire(model_name):
        yield faulty_engine

    mock_registry.acquire_engine = mock_acquire

    rejected_calls = []

    async def mock_on_rejected(job_id: int, reason: str) -> None:
        rejected_calls.append((job_id, reason))

    runner = JobRunner(
        reporter=mock_reporter,
        registry=mock_registry,
        media_dir=str(tmp_path),
        on_job_rejected=mock_on_rejected,
    )

    dummy_payload = {
        "job_id": 999999,
        "model_name": "qwen3-asr-0.6b",
        "media_path": "/api/v1/agent/jobs/999999/media",
    }

    # Run job
    task = runner.run_job(dummy_payload)
    await task

    # Verify on_job_rejected called
    assert len(rejected_calls) == 1
    assert rejected_calls[0][0] == 999999
    assert "insufficient free vram" in rejected_calls[0][1].lower()

    # Verify reporter.report_completion was NOT called with status="failed"
    assert mock_reporter.report_completion.call_count == 0


@pytest.mark.asyncio
async def test_pull_job_resource_check_optimization() -> None:
    """Verify resource check optimization:
    1. When running_jobs == 0: probes VRAM before pull.
    2. When running_jobs > 0 and model is active: skips VRAM probe.
    """
    mock_config = AgentConfig()
    mock_monitor = MagicMock()
    mock_registry = MagicMock()
    mock_job_runner = MagicMock()

    mock_ws = AsyncMock()
    client = AgentWebSocketClient(
        config=mock_config,
        monitor=mock_monitor,
        registry=mock_registry,
        job_runner=mock_job_runner,
    )
    client._running = True

    # Case 1: running_jobs == 0, resource check returns False -> NO pull_job sent
    mock_job_runner.get_running_jobs_count.return_value = 0
    mock_job_runner.max_concurrent_jobs = 2
    mock_registry.check_resources_available.return_value = False

    await client._check_and_pull_job(mock_ws)
    assert mock_ws.send.call_count == 0
    mock_registry.check_resources_available.assert_called_once()

    # Case 2: running_jobs == 0, resource check returns True -> pull_job sent
    mock_registry.check_resources_available.reset_mock()
    mock_registry.check_resources_available.return_value = True
    mock_registry.list_available_models.return_value = ["qwen3-asr-0.6b"]

    await client._check_and_pull_job(mock_ws)
    assert mock_ws.send.call_count == 1
    sent_data = json.loads(mock_ws.send.call_args[0][0])
    assert sent_data["type"] == "pull_job"
    assert sent_data["payload"]["supported_models"] == ["qwen3-asr-0.6b"]

    # Case 3: running_jobs == 1 (second job), model is active -> skips check_resources_available!
    mock_ws.send.reset_mock()
    mock_registry.check_resources_available.reset_mock()
    mock_job_runner.get_running_jobs_count.return_value = 1
    mock_registry.is_model_active.return_value = True

    await client._check_and_pull_job(mock_ws)
    assert mock_ws.send.call_count == 1
    # Crucial assertion: check_resources_available was NOT called!
    assert mock_registry.check_resources_available.call_count == 0


@pytest.mark.asyncio
async def test_failed_jobs_cooldown_and_exclusion() -> None:
    """Verify rejected job is placed on 1-minute cooldown and excluded in pull_job."""
    mock_config = AgentConfig()
    mock_monitor = MagicMock()
    mock_registry = MagicMock()
    mock_job_runner = MagicMock()

    mock_ws = AsyncMock()
    client = AgentWebSocketClient(
        config=mock_config,
        monitor=mock_monitor,
        registry=mock_registry,
        job_runner=mock_job_runner,
    )
    client._running = True
    client._current_ws = mock_ws
    mock_job_runner.get_running_jobs_count.return_value = 0
    mock_job_runner.max_concurrent_jobs = 2
    mock_registry.check_resources_available.return_value = True
    mock_registry.list_available_models.return_value = ["qwen3-asr-0.6b"]

    # Reject job 12345
    await client._on_job_rejected(12345, "insufficient_vram")

    # Verify reject_job sent over WS
    assert mock_ws.send.call_count == 1
    sent_reject = json.loads(mock_ws.send.call_args[0][0])
    assert sent_reject["type"] == "reject_job"
    assert sent_reject["payload"]["job_id"] == 12345

    # Verify cooldown exists
    assert 12345 in client._failed_jobs_cooldown
    assert client._failed_jobs_cooldown[12345] > time.time() + 50.0

    # Now pull job -> exclude_job_ids must contain 12345
    mock_ws.send.reset_mock()
    await client._check_and_pull_job(mock_ws)
    assert mock_ws.send.call_count == 1
    sent_pull = json.loads(mock_ws.send.call_args[0][0])
    assert sent_pull["type"] == "pull_job"
    assert 12345 in sent_pull["payload"]["exclude_job_ids"]

    # If cooldown expires (simulate past time)
    client._failed_jobs_cooldown[12345] = time.time() - 1.0
    mock_ws.send.reset_mock()
    await client._check_and_pull_job(mock_ws)
    sent_pull2 = json.loads(mock_ws.send.call_args[0][0])
    assert 12345 not in sent_pull2["payload"]["exclude_job_ids"]
