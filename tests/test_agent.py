# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
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
    assert config.http_base_url == "http://localhost:8080"
    assert config.ws_url == "ws://localhost:8080/api/v1/agent/ws"


def test_config_ws_url_with_token_and_https() -> None:
    config = AgentConfig(
        controller_url="https://api.transcribe.io:8443/",
        agent_token="sec-token-123",
    )
    assert config.http_base_url == "https://api.transcribe.io:8443"
    assert (
        config.ws_url
        == "wss://api.transcribe.io:8443/api/v1/agent/ws?token=sec-token-123"
    )


def test_config_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_file = tmp_path / "custom_config.yaml"
    yaml_file.write_text(
        "controller_url: 'http://yaml-host:9000'\n"
        "agent_token: 'yaml-token'\n"
        "node_name: 'yaml-node'\n"
        "heartbeat_interval: 15\n"
        "media_dir: '/tmp/yaml-media'\n"
        "max_concurrent_jobs: 4\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("CONTROLLER_URL", "http://env-host:7000")
    monkeypatch.setenv("AGENT_TOKEN", "env-token")
    monkeypatch.setenv("HEARTBEAT_INTERVAL", "20")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "8")

    config = load_config(yaml_file)
    assert config.controller_url == "http://env-host:7000"
    assert config.agent_token == "env-token"
    assert config.node_name == "yaml-node"  # from yaml since env not set
    assert config.heartbeat_interval == 20
    assert config.media_dir == "/tmp/yaml-media"
    assert config.max_concurrent_jobs == 8


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
    assert progress_values == [20, 50, 80, 100]

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
async def test_model_registry_lifecycle() -> None:
    registry = ModelRegistry(preload_default=True)
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

    task = runner.run_job({
        "job_id": 501,
        "model_name": "mock-whisper-base",
        "task_type": "transcribe",
        "media_path": "/api/v1/agent/jobs/501/media",
    })
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

    registry = ModelRegistry(preload_default=True)
    runner = JobRunner(
        reporter=reporter,
        registry=registry,
        media_dir=str(media_dir),
        max_concurrent_jobs=2,
    )

    # Launch failing job
    task = runner.run_job({
        "job_id": 999,
        "model_name": "mock-whisper-base",
        "media_path": "/api/v1/agent/jobs/999/media",
    })
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
    task2 = runner.run_job({
        "job_id": 1000,
        "model_name": "mock-whisper-base",
        "media_path": "/api/v1/agent/jobs/1000/media",
    })
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
        json.dumps({
            "type": "command",
            "action": "dispatch_job",
            "payload": {"job_id": 77},
        }),
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

    task = runner.run_job({
        "job_id": 777,
        "model_name": "pure-sync-engine",
        "media_path": "/api/v1/agent/jobs/777/media",
    })
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
    registry = ModelRegistry(preload_default=True)
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
async def test_registry_work_mode_and_unload_all() -> None:
    """Test setting work mode, mode validation, and unload_all_models."""
    registry = ModelRegistry(preload_default=True)
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
    registry = ModelRegistry(preload_default=True)
    job_runner = MagicMock(spec=JobRunner)
    client = AgentWebSocketClient(config, monitor, registry, job_runner)

    mock_ws = AsyncMock()

    # 1. Test unload_all_models message
    await client._handle_message(mock_ws, {"type": "command", "action": "unload_all_models", "payload": {}})
    assert len(registry.list_loaded_models()) == 0
    assert mock_ws.send.called
    status_msg = json.loads(mock_ws.send.call_args[0][0])
    assert status_msg["type"] == "model_status"
    assert status_msg["payload"]["loaded_models"] == []

    # 2. Test set_work_mode message
    mock_ws.reset_mock()
    await client._handle_message(mock_ws, {"type": "command", "action": "set_work_mode", "payload": {"mode": "cpu"}})
    assert registry.get_current_mode() == "cpu"
    assert mock_ws.send.call_count >= 1



def test_qwen3_asr_missing_ffmpeg_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.models.qwen3_asr import Qwen3ASREngine
    import subprocess

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

    t1 = runner.run_job({"job_id": 101, "model_name": "tracking-engine", "media_path": "/api/v1/agent/jobs/101/media"})
    t2 = runner.run_job({"job_id": 102, "model_name": "tracking-engine", "media_path": "/api/v1/agent/jobs/102/media"})
    t3 = runner.run_job({"job_id": 103, "model_name": "tracking-engine", "media_path": "/api/v1/agent/jobs/103/media"})

    await asyncio.gather(t1, t2, t3)

    assert len(complete_data) == 3
    # Max active inferences must be 1 because inference is serialized via _inference_lock
    assert max_active_inferences == 1


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
    from src.models.qwen3_asr import resolve_device_and_dtype
    import torch

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
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=4), \
         patch("torch.cuda.is_bf16_supported", return_value=True):
        monkeypatch.setenv("CUDA_DEVICE_INDEX", "2")
        dev, dtype, bs = resolve_device_and_dtype("gpu")
        assert dev == "cuda:2"
        assert dtype == torch.bfloat16
        assert bs == 16


def test_registry_multi_gpu_discovery() -> None:
    from src.models.registry import detect_supported_modes

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=3):
        modes, default_mode = detect_supported_modes()
        assert "cpu" in modes
        assert "gpu" in modes
        assert "cuda:0" in modes
        assert "cuda:1" in modes
        assert "cuda:2" in modes
        assert default_mode == "gpu"


# =========================================================================
# 9b. In-Memory Audio Pipeline (restored)
# =========================================================================

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
    fake_pcm_samples = (np.sin(np.linspace(0, 2 * np.pi * 440, sample_rate)) * 16000).astype(np.int16)
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
# 10. C1/C2 Multi-GPU Scheduling, Observability & Config (TDD)
# =========================================================================

def _force_torch_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force GpuScheduler/monitor to take the torch.cuda path (ignore pynvml)."""
    import sys

    monkeypatch.setitem(sys.modules, "pynvml", None)


def _mock_two_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock torch.cuda as 2x 16GB cards: cuda:0 ~95% used, cuda:1 ~10% used."""
    import sys

    _force_torch_only(monkeypatch)
    total = 16 * 1024 * 1024 * 1024
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)

    def _allocated(idx: int) -> int:
        return int(total * 0.95) if idx == 0 else int(total * 0.10)

    def _props(idx: int):
        m = MagicMock()
        m.total_memory = total
        return m

    monkeypatch.setattr("torch.cuda.memory_allocated", _allocated)
    monkeypatch.setattr("torch.cuda.get_device_properties", _props)


def test_gpu_scheduler_selects_empty_card(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.models.registry import GpuScheduler

    _mock_two_gpus(monkeypatch)
    monkeypatch.delenv("QWEN3_ASR_DEVICE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    sched = GpuScheduler()
    picked = sched.select_device()
    assert picked == "cuda:1"


def test_gpu_scheduler_single_gpu_fallback_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.models.registry import GpuScheduler

    _force_torch_only(monkeypatch)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    monkeypatch.delenv("QWEN3_ASR_DEVICE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    sched = GpuScheduler()
    assert sched.select_device() == "cpu"


def test_gpu_scheduler_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.models.registry import GpuScheduler

    _mock_two_gpus(monkeypatch)
    # Explicit device wins immediately
    monkeypatch.setenv("QWEN3_ASR_DEVICE", "cuda:0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert GpuScheduler().select_device() == "cuda:0"

    # CUDA_VISIBLE_DEVICES filters candidates
    monkeypatch.delenv("QWEN3_ASR_DEVICE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    picked = GpuScheduler().select_device()
    assert picked == "cuda:1"


@pytest.mark.asyncio
async def test_registry_per_device_engines_and_detailed() -> None:
    registry = ModelRegistry(preload_default=False)
    registry.register("mock-whisper-base", lambda: MockASREngine("mock-whisper-base", stage_delay=0.0))

    e0 = await registry.load_model("mock-whisper-base", device="cuda:0")
    e1 = await registry.load_model("mock-whisper-base", device="cuda:1")
    assert e0 is not e1

    # list_loaded_models stays deduped for heartbeat compat
    loaded = registry.list_loaded_models()
    assert loaded.count("mock-whisper-base") == 1

    detailed = registry.list_loaded_models_detailed()
    assert "mock-whisper-base@cuda:0" in detailed
    assert "mock-whisper-base@cuda:1" in detailed

    # acquire with explicit device returns the right instance
    async with registry.acquire_engine("mock-whisper-base", device="cuda:0") as eng:
        assert eng is e0

    # unload single device keeps the other
    assert await registry.unload_model("mock-whisper-base", device="cuda:0") is True
    assert "mock-whisper-base@cuda:1" in registry.list_loaded_models_detailed()
    assert "mock-whisper-base" in registry.list_loaded_models()


@pytest.mark.asyncio
async def test_registry_set_work_mode_multi_gpu_no_unload(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_two_gpus(monkeypatch)
    monkeypatch.delenv("QWEN3_ASR_DEVICE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    registry = ModelRegistry(preload_default=False)
    registry.register("mock-whisper-base", lambda: MockASREngine("mock-whisper-base", stage_delay=0.0))
    await registry.load_model("mock-whisper-base", device="cuda:0")

    await registry.set_work_mode("cuda:0,cuda:1")
    assert registry.get_current_mode() == "cuda:0,cuda:1"
    # 按需迁移：不再 unload 全部
    assert "mock-whisper-base" in registry.list_loaded_models()


def test_monitor_per_gpu_list_and_compat() -> None:
    import sys
    import types

    monitor = SystemMonitor()

    fake_pynvml = types.ModuleType("pynvml")
    mem0 = MagicMock(used=15 * 1024**3, total=16 * 1024**3)
    mem1 = MagicMock(used=2 * 1024**3, total=16 * 1024**3)
    util0 = MagicMock(gpu=95)
    util1 = MagicMock(gpu=10)
    fake_pynvml.nvmlInit = MagicMock()
    fake_pynvml.nvmlDeviceGetCount = MagicMock(return_value=2)
    fake_pynvml.nvmlDeviceGetHandleByIndex = MagicMock(side_effect=["h0", "h1"])
    fake_pynvml.nvmlDeviceGetUtilizationRates = MagicMock(side_effect=[util0, util1])
    fake_pynvml.nvmlDeviceGetMemoryInfo = MagicMock(side_effect=[mem0, mem1])

    with patch.dict(sys.modules, {"pynvml": fake_pynvml}):
        stats = monitor.collect()

    # 旧字段兼容
    assert "gpu_percent" in stats
    assert "gpu_memory_used_mb" in stats
    assert "gpu_memory_total_mb" in stats
    # 新增 per-gpu 列表
    assert "gpu_devices" in stats
    devs = stats["gpu_devices"]
    assert len(devs) == 2
    assert devs[0]["index"] == 0
    assert devs[0]["percent"] == 95
    assert devs[1]["index"] == 1
    assert devs[1]["percent"] == 10
    assert all("used_mb" in d and "total_mb" in d for d in devs)


def test_config_gpu_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    c = AgentConfig()
    assert c.gpu_devices is None
    assert c.max_concurrent_jobs_per_gpu == 2

    monkeypatch.setenv("GPU_DEVICES", "cuda:0,cuda:1")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS_PER_GPU", "3")
    c2 = load_config(None)
    assert c2.gpu_devices == "cuda:0,cuda:1"
    assert c2.max_concurrent_jobs_per_gpu == 3


@pytest.mark.asyncio
async def test_ws_heartbeat_multi_gpu_fields() -> None:
    config = AgentConfig()
    monitor = SystemMonitor()
    registry = ModelRegistry(preload_default=True)
    job_runner = MagicMock(spec=JobRunner)
    job_runner.get_running_jobs_count.return_value = 0
    client = AgentWebSocketClient(config, monitor, registry, job_runner)
    mock_ws = AsyncMock()
    await client._send_heartbeat(mock_ws)
    assert mock_ws.send.called
    payload = json.loads(mock_ws.send.call_args[0][0])["payload"]
    assert "gpu_devices" in payload
    assert "loaded_models_detailed" in payload
    assert isinstance(payload["gpu_devices"], list)
    assert isinstance(payload["loaded_models_detailed"], list)



# =========================================================================
# 10. A1/A2 TDD增量测试（先失败，后实现）
# =========================================================================

@pytest.mark.asyncio
async def test_different_models_parallel_inference(tmp_path: Path) -> None:
    """A1: 不同model可并行（max_active==2），同model仍串行由旧用例覆盖。"""
    import threading
    import time as _time

    media_dir = tmp_path / "media_multi"
    media_dir.mkdir()
    complete_data: list[dict] = []

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

    active = 0
    max_active = 0
    mtx = threading.Lock()

    def make_engine(name: str) -> type:
        class TrackingSyncEngine(BaseEngine):
            def __init__(self) -> None:
                super().__init__(name)
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
                nonlocal active, max_active
                with mtx:
                    active += 1
                    max_active = max(max_active, active)
                _time.sleep(0.05)
                with mtx:
                    active -= 1
                return {
                    "task": task_type,
                    "language": language or "en",
                    "duration": 1.0,
                    "text": f"ok-{name}",
                    "segments": [],
                }

        TrackingSyncEngine.__name__ = f"TrackingSyncEngine_{name}"
        return TrackingSyncEngine

    registry = ModelRegistry(preload_default=False)
    registry.register("model-a", make_engine("model-a"))
    registry.register("model-b", make_engine("model-b"))
    await registry.load_model("model-a")
    await registry.load_model("model-b")

    runner = JobRunner(
        reporter=reporter,
        registry=registry,
        media_dir=str(media_dir),
        max_concurrent_jobs=2,
    )

    t1 = runner.run_job({"job_id": 201, "model_name": "model-a", "media_path": "/m"})
    t2 = runner.run_job({"job_id": 202, "model_name": "model-b", "media_path": "/m"})
    await asyncio.gather(t1, t2)

    assert len(complete_data) == 2
    assert max_active == 2


@pytest.mark.asyncio
async def test_reporter_download_chunked_nonblocking_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A2: download_media 必须 aiter_bytes(chunk_size=65536) 且经 asyncio.to_thread 写盘。"""
    import asyncio as _asyncio

    seen: dict[str, Any] = {}
    to_thread_calls: list[tuple] = []
    orig_to_thread = _asyncio.to_thread

    async def spy_to_thread(func, /, *args, **kwargs):
        to_thread_calls.append((func, args, kwargs))
        return await orig_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(_asyncio, "to_thread", spy_to_thread)

    class FakeResp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        async def aiter_bytes(self, chunk_size=None):
            seen["chunk_size"] = chunk_size
            yield b"AAA"
            yield b"BBB"

    class FakeClient:
        def stream(self, method, url):
            seen["method"] = method
            return FakeResp()

    reporter = Reporter("http://test", "token", client=FakeClient())  # type: ignore[arg-type]
    target = tmp_path / "out.mp3"
    await reporter.download_media(7, str(target))

    assert seen["chunk_size"] == 65536
    assert target.read_bytes() == b"AAABBB"
    # 写盘必须走 to_thread（至少1次），避免阻塞 event loop
    assert len(to_thread_calls) >= 1


@pytest.mark.asyncio
async def test_engine_log_throttling(tmp_path: Path) -> None:
    """A2: engine_log_cb 节流：2s内、progress增量<10 不上报。"""
    media_dir = tmp_path / "throttle_media"
    media_dir.mkdir()

    report_progress: list[int] = []

    reporter = Reporter("http://test", "token", client=MagicMock())
    reporter.download_media = AsyncMock()  # type: ignore[method-assign]

    async def fake_report_logs(job_id: int, progress: int, logs: list[dict]) -> None:
        report_progress.append(progress)

    reporter.report_logs = fake_report_logs  # type: ignore[method-assign]
    reporter.report_completion = AsyncMock()  # type: ignore[method-assign]

    class NoisyEngine(BaseEngine):
        def __init__(self) -> None:
            super().__init__("noisy-model")
            self.loaded = True

        async def load(self) -> None:
            self.loaded = True

        async def unload(self) -> None:
            self.loaded = False

        async def transcribe(self, audio_path: str, language=None, task_type: str = "transcribe", log_callback=None) -> dict:
            for p in range(20, 40):  # 20次快速回调，每次+1
                await log_callback(p, f"step {p}")
            return {"text": "done"}

    registry = ModelRegistry(preload_default=False)
    registry.register("noisy-model", NoisyEngine)
    await registry.load_model("noisy-model")

    runner = JobRunner(reporter=reporter, registry=registry, media_dir=str(media_dir), max_concurrent_jobs=1)
    await runner._execute_job({"job_id": 301, "model_name": "noisy-model", "media_path": "/m"})

    # 无节流时 >= 22 次上报（5/10固定 + 20引擎回调）；节流后应显著减少
    assert len(report_progress) <= 8, f"throttling failed, reports={report_progress}"


@pytest.mark.asyncio
async def test_dynamic_semaphore_no_task_leak_on_set_capacity() -> None:
    """A2: 连续 set_capacity 不得泄漏后台任务，且扩容仍能唤醒等待者。"""
    from src.job_runner import DynamicSemaphore

    sem = DynamicSemaphore(initial_capacity=1)
    for _ in range(10):
        sem.set_capacity(2)
    await asyncio.sleep(0)  # 让已调度的 notify（如有）执行
    await asyncio.sleep(0)
    pending = [t for t in sem._bg_tasks if not t.done()]
    assert len(pending) <= 1, f"bg task leak: {len(pending)} pending"

    # 扩容唤醒能力：用新信号量验证（占满1槽，等待者阻塞，扩容到2后应被唤醒）
    from src.job_runner import DynamicSemaphore as _DS

    sem = _DS(initial_capacity=1)
    await sem.acquire()  # acquired=1，已满
    woke = False

    async def waiter() -> None:
        nonlocal woke
        await sem.acquire()
        woke = True
        await sem.release()

    wt = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    assert woke is False
    sem.set_capacity(2)
    await asyncio.wait_for(wt, timeout=2.0)
    assert woke is True
    await sem.release()


# =========================================================================
# 11. B1/B2/B3 TDD增量测试（先失败，后实现）
# =========================================================================

def test_qwen3_asr_b1_max_new_tokens_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1: 默认512，env QWEN3_ASR_MAX_NEW_TOKENS 覆盖并 clamp 到 128~1024。"""
    from src.models.qwen3_asr import Qwen3ASREngine, resolve_max_new_tokens

    monkeypatch.delenv("QWEN3_ASR_MAX_NEW_TOKENS", raising=False)
    assert resolve_max_new_tokens() == 512
    assert Qwen3ASREngine("qwen3-asr-0.6b").max_new_tokens == 512

    monkeypatch.setenv("QWEN3_ASR_MAX_NEW_TOKENS", "768")
    assert resolve_max_new_tokens() == 768
    assert Qwen3ASREngine("qwen3-asr-0.6b").max_new_tokens == 768

    # clamp 下界/上界
    monkeypatch.setenv("QWEN3_ASR_MAX_NEW_TOKENS", "64")
    assert resolve_max_new_tokens() == 128
    monkeypatch.setenv("QWEN3_ASR_MAX_NEW_TOKENS", "2048")
    assert resolve_max_new_tokens() == 1024
    # 非法值回落默认
    monkeypatch.setenv("QWEN3_ASR_MAX_NEW_TOKENS", "not-a-number")
    assert resolve_max_new_tokens() == 512


def _b2_make_pcm_bytes(seconds: float, sr: int = 16000, silent: bool = False) -> bytes:
    import numpy as np

    n = int(seconds * sr)
    if silent:
        pcm = np.zeros(n, dtype=np.int16)
    else:
        t = np.linspace(0, 2 * np.pi * 440 * seconds, n)
        pcm = (np.sin(t) * 16000).astype(np.int16)
    return pcm.tobytes()


def _b2_mock_ffmpeg(monkeypatch: pytest.MonkeyPatch, pcm_bytes: bytes) -> list:
    import subprocess

    executed: list = []

    def mock_run(cmd, *args, **kwargs):
        executed.append(cmd)
        res = MagicMock()
        res.returncode = 0
        res.stdout = pcm_bytes
        res.stderr = b""
        return res

    monkeypatch.setattr(subprocess, "run", mock_run)
    return executed


def _b2_make_engine_with_mock_model() -> Any:
    from src.models.qwen3_asr import Qwen3ASREngine

    engine = Qwen3ASREngine("qwen3-asr-0.6b")
    engine.loaded = True
    mock_model = MagicMock()
    engine._model = mock_model
    return engine


def test_qwen3_asr_b2_duration_batch_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B2: 4x~30s=120s 音频，预算60s 必须按秒切成多批（旧逻辑固定16下只有1批）。"""
    from src.models.qwen3_asr import resolve_max_batch_seconds

    monkeypatch.setenv("QWEN3_ASR_MAX_BATCH_SECONDS", "60")
    monkeypatch.setenv("QWEN3_ASR_BATCH_SIZE", "16")
    assert resolve_max_batch_seconds() == 60.0

    audio = tmp_path / "long.mp3"
    audio.write_bytes(b"dummy mp3")
    _b2_mock_ffmpeg(monkeypatch, _b2_make_pcm_bytes(120.0))

    engine = _b2_make_engine_with_mock_model()

    def fake_transcribe(audio=None, language=None, **kwargs):
        items = audio if isinstance(audio, list) else [audio]
        outs = []
        for _ in items:
            o = MagicMock()
            o.text = "hello"
            o.language = "en"
            outs.append(o)
        return outs

    engine._model.transcribe.side_effect = fake_transcribe

    result = engine.transcribe(str(audio))
    assert result["text"] != ""
    # 旧逻辑：1次调用带4个chunk；新逻辑：预算60s下必须拆成多批
    calls = engine._model.transcribe.call_args_list
    assert len(calls) > 1
    total_chunks = 0
    for call in calls:
        batch = call.kwargs.get("audio", call.args[0] if call.args else None)
        assert isinstance(batch, list)
        total_chunks += len(batch)
        # ~30s chunk + 60s预算 -> 每批最多2个chunk，且总时长不超预算太多（单chunk可超30s+搜索窗）
        assert len(batch) <= 2
        batch_sec = sum(len(cwav) / 16000 for cwav, _ in batch)
        assert batch_sec <= 66.0
    assert total_chunks == 4


def test_qwen3_asr_b2_oom_halving_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B2: 整批OOM时减半重试合并恢复（调用<5次），而非旧逻辑直接退化4次single-chunk（共5次）。"""
    monkeypatch.setenv("QWEN3_ASR_MAX_BATCH_SECONDS", "180")
    monkeypatch.setenv("QWEN3_ASR_BATCH_SIZE", "16")

    audio = tmp_path / "oom.mp3"
    audio.write_bytes(b"dummy mp3")
    _b2_mock_ffmpeg(monkeypatch, _b2_make_pcm_bytes(120.0))  # 4 x 30s chunks

    engine = _b2_make_engine_with_mock_model()

    def fake_transcribe(audio=None, language=None, **kwargs):
        items = audio if isinstance(audio, list) else [audio]
        if isinstance(audio, list) and len(items) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.5 GiB")
        outs = []
        for _ in items:
            o = MagicMock()
            o.text = "ok"
            o.language = "en"
            outs.append(o)
        return outs

    engine._model.transcribe.side_effect = fake_transcribe

    result = engine.transcribe(str(audio))
    assert result["text"] == "ok ok ok ok"
    # 减半重试：1次整批(4)失败 + 2次半批(2+2)成功；旧逻辑则是1+4=5次
    assert 2 <= engine._model.transcribe.call_count < 5


def test_qwen3_asr_b3_silence_chunks_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B3: 全静音chunk直接跳过，不进模型，segments为空。"""
    monkeypatch.setenv("QWEN3_ASR_SKIP_SILENCE", "1")

    audio = tmp_path / "silent.mp3"
    audio.write_bytes(b"dummy mp3")
    _b2_mock_ffmpeg(monkeypatch, _b2_make_pcm_bytes(5.0, silent=True))

    engine = _b2_make_engine_with_mock_model()
    result = engine.transcribe(str(audio))

    assert result["segments"] == []
    assert result["text"] == ""
    engine._model.transcribe.assert_not_called()


def test_qwen3_asr_b3_flac_fastpath_no_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B3: 16k单声道FLAC走soundfile快径，不调用ffmpeg。"""
    import subprocess

    import numpy as np

    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")

    sr = 16000
    wav_data = (np.sin(np.linspace(0, 2 * np.pi * 440, sr)) * 0.5).astype(np.float32)
    flac_path = tmp_path / "fast.flac"
    sf.write(str(flac_path), wav_data, sr, format="FLAC")

    ffmpeg_calls: list = []

    def mock_run(cmd, *args, **kwargs):
        ffmpeg_calls.append(cmd)
        res = MagicMock()
        res.returncode = 0
        res.stdout = b""
        res.stderr = b""
        return res

    monkeypatch.setattr(subprocess, "run", mock_run)

    engine = _b2_make_engine_with_mock_model()

    def fake_transcribe(audio=None, language=None, **kwargs):
        items = audio if isinstance(audio, list) else [audio]
        outs = []
        for _ in items:
            o = MagicMock()
            o.text = "flac ok"
            o.language = "en"
            outs.append(o)
        return outs

    engine._model.transcribe.side_effect = fake_transcribe

    result = engine.transcribe(str(flac_path))
    assert ffmpeg_calls == []
    assert result["text"] == "flac ok"


# =========================================================================
# 11. Dynamic capacity advertisement (TDD RED)
# =========================================================================

def _low_stats() -> dict:
    return {
        "cpu_percent": 10.0,
        "ram_percent": 20.0,
        "ram_used_mb": 100,
        "ram_total_mb": 1000,
        "gpu_percent": 5.0,
        "gpu_memory_used_mb": 100,
        "gpu_memory_total_mb": 8000,
        "gpu_devices": [],
    }


def _high_stats() -> dict:
    return {
        "cpu_percent": 95.0,
        "ram_percent": 95.0,
        "ram_used_mb": 950,
        "ram_total_mb": 1000,
        "gpu_percent": 99.0,
        "gpu_memory_used_mb": 7900,
        "gpu_memory_total_mb": 8000,
        "gpu_devices": [],
    }


def test_capacity_controller_starts_at_3() -> None:
    from src.job_runner import CapacityController

    c = CapacityController()
    assert c.capacity == 3


def test_capacity_controller_increases_when_saturated_and_idle() -> None:
    from src.job_runner import CapacityController

    c = CapacityController()
    # running == capacity (3/3) 且低负载，需连续2次才+1（防抖）
    assert c.update(_low_stats(), running_jobs=3) == 3
    assert c.update(_low_stats(), running_jobs=3) == 4


def test_capacity_controller_holds_when_unsaturated() -> None:
    from src.job_runner import CapacityController

    c = CapacityController()
    assert c.update(_low_stats(), running_jobs=1) == 3
    assert c.update(_low_stats(), running_jobs=1) == 3


def test_capacity_controller_decreases_on_high_load() -> None:
    from src.job_runner import CapacityController

    c = CapacityController()
    assert c.update(_high_stats(), running_jobs=3) == 3
    assert c.update(_high_stats(), running_jobs=3) == 2


def test_capacity_controller_clamps_min_max() -> None:
    from src.job_runner import CapacityController

    c = CapacityController(initial=8, min_capacity=1, max_capacity=8)
    for _ in range(6):
        c.update(_low_stats(), running_jobs=c.capacity)
    assert c.capacity == 8
    c2 = CapacityController(initial=1, min_capacity=1, max_capacity=8)
    for _ in range(6):
        c2.update(_high_stats(), running_jobs=1)
    assert c2.capacity == 1


def test_job_runner_dynamic_mode_reports_capacity(tmp_path: Path) -> None:
    from src.job_runner import JobRunner

    reporter = MagicMock()
    registry = MagicMock()
    runner = JobRunner(
        reporter=reporter,
        registry=registry,
        media_dir=str(tmp_path),
        max_concurrent_jobs=-1,
    )
    assert runner.is_dynamic is True
    assert runner.advertised_capacity == 3


def test_job_runner_static_mode_ignores_capacity(tmp_path: Path) -> None:
    from src.job_runner import JobRunner

    reporter = MagicMock()
    registry = MagicMock()
    runner = JobRunner(
        reporter=reporter,
        registry=registry,
        media_dir=str(tmp_path),
        max_concurrent_jobs=2,
    )
    assert runner.is_dynamic is False
    assert runner.advertised_capacity == 2


@pytest.mark.asyncio
async def test_heartbeat_includes_advertised_capacity() -> None:
    import json

    from src.config import AgentConfig
    from src.job_runner import JobRunner
    from src.monitor import SystemMonitor
    from src.models.registry import ModelRegistry
    from src.ws_client import AgentWebSocketClient

    config = AgentConfig(max_concurrent_jobs=-1)
    monitor = SystemMonitor()
    registry = ModelRegistry(preload_default=False)
    reporter = MagicMock()
    runner = JobRunner(reporter=reporter, registry=registry, media_dir="/tmp/x", max_concurrent_jobs=-1)
    client = AgentWebSocketClient(config, monitor, registry, runner)

    mock_ws = AsyncMock()
    await client._send_heartbeat(mock_ws)
    payload = json.loads(mock_ws.send.call_args[0][0])["payload"]
    assert payload["advertised_capacity"] == 3


# =========================================================================
# 12. Multi-replica parallel inference (TDD RED)
# =========================================================================

def test_pick_inference_device_prefers_idle_replica() -> None:
    """Loaded replica with zero in-flight inferences wins over busy one."""
    import asyncio

    from src.models.mock_asr import MockASREngine
    from src.models.registry import ModelRegistry

    async def _scenario() -> None:
        registry = ModelRegistry(preload_default=False)
        registry.register("replica-model", lambda: MockASREngine("replica-model", stage_delay=0.0))
        await registry.load_model("replica-model", device="cuda:0")
        await registry.load_model("replica-model", device="cuda:1")
        async with registry.acquire_engine("replica-model", device="cuda:0"):
            picked = registry.pick_inference_device("replica-model")
            assert picked == "cuda:1"

    asyncio.run(_scenario())


def test_pick_inference_device_unknown_model_resolves() -> None:
    from src.models.registry import ModelRegistry

    registry = ModelRegistry(preload_default=False)
    # No loaded replica: falls back to normal device resolution (cpu here).
    assert registry.pick_inference_device("never-loaded") == "cpu"


def test_qwen_engine_locks_are_per_instance() -> None:
    from src.models.qwen3_asr import Qwen3ASREngine

    a = Qwen3ASREngine("qwen3-asr-0.6b")
    b = Qwen3ASREngine("qwen3-asr-0.6b")
    assert a._inference_lock is not b._inference_lock


def test_inference_locks_are_per_device(tmp_path: Path) -> None:
    from src.job_runner import JobRunner

    runner = JobRunner(reporter=MagicMock(), registry=MagicMock(), media_dir=str(tmp_path))
    assert runner._get_inference_lock("m", "cuda:0") is runner._get_inference_lock("m", "cuda:0")
    assert runner._get_inference_lock("m", "cuda:0") is not runner._get_inference_lock("m", "cuda:1")


def test_capacity_decreases_when_inference_queued() -> None:
    from src.job_runner import CapacityController

    c = CapacityController()
    assert c.update(_low_stats(), running_jobs=3, queued=2) == 3
    assert c.update(_low_stats(), running_jobs=3, queued=2) == 2


def test_capacity_holds_without_queue_info_compat() -> None:
    from src.job_runner import CapacityController

    c = CapacityController()
    # Old two-arg callables keep working (queued defaults to 0).
    assert c.update(_low_stats(), running_jobs=1) == 3


@pytest.mark.asyncio
async def test_same_model_parallel_across_devices(tmp_path: Path) -> None:
    """Same model on two replicas must infer concurrently (fails on per-model lock)."""
    import asyncio
    import time

    import httpx

    from src.job_runner import JobRunner
    from src.models.base import BaseEngine
    from src.models.registry import ModelRegistry
    from src.reporter import Reporter

    active = 0
    max_active = 0

    class TrackingAsyncEngine(BaseEngine):
        def __init__(self) -> None:
            super().__init__("dual-model")
            self.loaded = True

        async def load(self, work_mode: str = "gpu") -> None:
            self.loaded = True

        async def unload(self) -> None:
            self.loaded = False

        async def transcribe(self, audio_path: str, language=None, task_type="transcribe", log_callback=None):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.1)
            active -= 1
            return {"task": task_type, "language": "en", "duration": 1.0, "text": "ok", "segments": []}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"AUDIO")

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    reporter = Reporter("http://test", "token", client=client)
    registry = ModelRegistry(preload_default=False)
    registry.register("dual-model", TrackingAsyncEngine)
    await registry.load_model("dual-model", device="cuda:0")
    await registry.load_model("dual-model", device="cuda:1")
    runner = JobRunner(reporter=reporter, registry=registry, media_dir=str(tmp_path), max_concurrent_jobs=2)

    t1 = runner.run_job({"job_id": 901, "model_name": "dual-model", "media_path": "x.mp3"})
    t2 = runner.run_job({"job_id": 902, "model_name": "dual-model", "media_path": "x.mp3"})
    await asyncio.gather(t1, t2)

    assert max_active == 2
    assert time.time() > 0  # wall clock sanity





