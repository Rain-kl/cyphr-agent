# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from pathlib import Path
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

