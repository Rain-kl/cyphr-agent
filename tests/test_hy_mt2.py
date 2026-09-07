# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import AgentConfig
from src.models.hy_mt2 import (
    ALIAS_HY_MT2_7B_GGUF_SHORT,
    MODEL_NAME_HY_MT2,
    MODEL_NAME_HY_MT2_7B_GGUF,
    HyMT2Engine,
    format_translation_prompt,
)
from src.models.registry import ModelRegistry
from src.ws_client import AgentWebSocketClient


def test_hy_mt2_prompt_injection() -> None:
    """Verify translation instruction prompt auto-injection."""
    # 1. Chinese target
    p1 = format_translation_prompt("Good morning, how are you?", target_lang="zh")
    assert "将以下文本翻译为中文" in p1
    assert "不要额外解释" in p1
    assert "Good morning, how are you?" in p1

    # 2. English target
    p2 = format_translation_prompt("今天天气真好", target_lang="en")
    assert "Translate the following text into English" in p2
    assert "without any additional explanation" in p2
    assert "今天天气真好" in p2

    # 3. Already formatted instruction should not be re-wrapped
    existing = "将以下文本翻译为日语，注意只需要输出翻译后的结果，不要额外解释：\n\n你好"  # noqa: RUF001
    p3 = format_translation_prompt(existing, target_lang="ja")
    assert p3 == existing


@pytest.mark.asyncio
async def test_hy_mt2_mock_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mock streaming generation yields deltas and ends with stop."""
    monkeypatch.setenv("HY_MT2_MOCK", "1")
    engine = HyMT2Engine(model_name=MODEL_NAME_HY_MT2)
    await engine.load(work_mode="cpu")
    assert engine.loaded is True

    chunks: list[str] = []
    final_reason = None

    async for delta, reason in engine.generate_stream(
        request_id="test-req-1",
        prompt="Hello world",
        target_lang="zh",
    ):
        if delta:
            chunks.append(delta)
        if reason:
            final_reason = reason

    full_text = "".join(chunks)
    assert len(chunks) > 0
    assert "你好" in full_text
    assert final_reason == "stop"

    await engine.unload()
    assert engine.loaded is False


@pytest.mark.asyncio
async def test_hy_mt2_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that aborting a request halts streaming early."""
    monkeypatch.setenv("HY_MT2_MOCK", "1")
    engine = HyMT2Engine(model_name=MODEL_NAME_HY_MT2)
    await engine.load(work_mode="cpu")

    gen = engine.generate_stream(
        request_id="test-abort-req",
        prompt="A very long text to translate",
        target_lang="zh",
    )

    # Receive first chunk
    first = await anext(gen)
    assert first is not None

    # Abort
    await engine.abort("test-abort-req")

    # Following iterations should terminate
    remaining = []
    async for item in gen:
        remaining.append(item)

    assert len(remaining) == 0
    await engine.unload()


@pytest.mark.asyncio
async def test_ws_client_chat_completion_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that AgentWebSocketClient handles chat_completion and streams chunks over WS."""
    monkeypatch.setenv("HY_MT2_MOCK", "1")

    config = AgentConfig()
    registry = ModelRegistry(debug=True)
    job_runner = MagicMock()
    monitor = MagicMock()

    client = AgentWebSocketClient(
        config=config,
        monitor=monitor,
        registry=registry,
        job_runner=job_runner,
    )
    client._running = True

    mock_ws = AsyncMock()
    client._current_ws = mock_ws

    msg = {
        "type": "chat_completion",
        "request_id": "chatcmpl-9999",
        "payload": {
            "model": MODEL_NAME_HY_MT2,
            "prompt": "Hello world",
            "target_lang": "zh",
            "stream": True,
        },
    }

    await client._handle_message(mock_ws, msg)
    if task := client._chat_tasks.get("chatcmpl-9999"):
        await task

    # Verify WebSocket sent chunks
    assert mock_ws.send.call_count >= 1

    sent_chunks = []
    for call in mock_ws.send.call_args_list:
        raw = call[0][0]
        data = json.loads(raw)
        assert data["type"] == "chat_chunk"
        assert data["request_id"] == "chatcmpl-9999"
        sent_chunks.append(data["payload"]["delta"])

    assert "你好" in "".join(sent_chunks)


@pytest.mark.asyncio
async def test_concurrent_chat_completion_multiplexing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify multiple concurrent chat completions stream multiplexed over the same WebSocket."""
    monkeypatch.setenv("HY_MT2_MOCK", "1")

    config = AgentConfig()
    registry = ModelRegistry(debug=True)
    job_runner = MagicMock()
    monitor = MagicMock()

    client = AgentWebSocketClient(
        config=config,
        monitor=monitor,
        registry=registry,
        job_runner=job_runner,
    )
    client._running = True

    mock_ws = AsyncMock()
    client._current_ws = mock_ws

    # Dispatch 3 concurrent chat completion requests
    req_ids = ["req-A", "req-B", "req-C"]
    for req_id in req_ids:
        msg = {
            "type": "chat_completion",
            "request_id": req_id,
            "payload": {
                "model": MODEL_NAME_HY_MT2,
                "prompt": f"Hello from {req_id}",
                "target_lang": "zh",
                "stream": True,
            },
        }
        await client._handle_message(mock_ws, msg)

    # Let concurrent streams run
    if tasks := list(client._chat_tasks.values()):
        await asyncio.gather(*tasks)

    received_by_req: dict[str, list[str]] = {r: [] for r in req_ids}
    for call in mock_ws.send.call_args_list:
        raw = call[0][0]
        data = json.loads(raw)
        if data.get("type") == "chat_chunk":
            r_id = data["request_id"]
            if r_id in received_by_req:
                received_by_req[r_id].append(data["payload"]["delta"])

    for req_id in req_ids:
        assert len(received_by_req[req_id]) > 0, f"Request {req_id} should have received chunks"


@pytest.mark.asyncio
async def test_hy_mt2_7b_gguf_mock_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Hy-MT2-7B-GGUF engine detects GGUF format and streams generation."""
    monkeypatch.setenv("HY_MT2_MOCK", "1")
    engine = HyMT2Engine(model_name=MODEL_NAME_HY_MT2_7B_GGUF)
    assert engine.is_gguf is True

    await engine.load(work_mode="cpu")
    assert engine.loaded is True

    chunks: list[str] = []
    final_reason = None
    async for delta, reason in engine.generate_stream(
        request_id="test-7b-req",
        prompt="Translating with 7B GGUF",
        target_lang="zh",
    ):
        if delta:
            chunks.append(delta)
        if reason:
            final_reason = reason

    assert len(chunks) > 0
    assert final_reason == "stop"


def test_registry_hy_mt2_7b_gguf() -> None:
    """Verify ModelRegistry registers 7B GGUF model and its aliases."""
    registry = ModelRegistry(debug=True)
    available = registry.list_available_models()

    assert MODEL_NAME_HY_MT2_7B_GGUF in available
    assert ALIAS_HY_MT2_7B_GGUF_SHORT in available

    engine = registry.get_engine(MODEL_NAME_HY_MT2_7B_GGUF)
    assert engine is None  # Not loaded yet
