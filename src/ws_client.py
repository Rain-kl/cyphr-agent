# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from .config import AgentConfig
from .job_runner import JobRunner
from .models.registry import ModelRegistry
from .monitor import SystemMonitor

logger = logging.getLogger(__name__)


class AgentWebSocketClient:
    """WebSocket client connecting worker agent to the central controller."""

    def __init__(
        self,
        config: AgentConfig,
        monitor: SystemMonitor,
        registry: ModelRegistry,
        job_runner: JobRunner,
    ) -> None:
        self.config = config
        self.monitor = monitor
        self.registry = registry
        self.job_runner = job_runner
        self._running = False
        self._current_ws: websockets.ClientConnection | None = None

    async def start(self) -> None:
        """Start the WebSocket connection loop with automatic reconnect and exponential backoff."""
        self._running = True
        backoff = 1.0
        max_backoff = 30.0

        while self._running:
            ws_url = self.config.ws_url
            logger.info("Connecting to controller WebSocket at %s", ws_url)

            try:
                async with websockets.connect(ws_url) as ws:
                    self._current_ws = ws
                    backoff = 1.0
                    logger.info("Connected to controller WebSocket successfully")

                    # Run heartbeat sender and incoming message consumer concurrently
                    heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))
                    try:
                        await self._message_loop(ws)
                    finally:
                        heartbeat_task.cancel()
                        try:
                            await heartbeat_task
                        except asyncio.CancelledError:
                            pass

            except (ConnectionClosed, OSError) as exc:
                if not self._running:
                    break
                logger.warning(
                    "WebSocket connection closed or error (%s). Reconnecting in %.1fs...",
                    exc,
                    backoff,
                )
            except Exception as exc:
                if not self._running:
                    break
                logger.exception(
                    "Unexpected WebSocket error (%s). Reconnecting in %.1fs...",
                    exc,
                    backoff,
                )
            finally:
                self._current_ws = None

            if self._running:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def stop(self) -> None:
        """Signal client to stop and disconnect."""
        self._running = False
        if self._current_ws is not None:
            try:
                await self._current_ws.close()
            except Exception:
                pass

    async def _heartbeat_loop(self, ws: websockets.ClientConnection) -> None:
        """Periodically transmit system stats and loaded models to controller."""
        while self._running:
            try:
                await self._send_heartbeat(ws)
            except Exception as exc:
                logger.warning("Failed to send heartbeat: %s", exc)
                break

            await asyncio.sleep(self.config.heartbeat_interval)

    async def _send_heartbeat(self, ws: websockets.ClientConnection) -> None:
        """Send a single heartbeat payload with hardware and model telemetry."""
        loaded = self.registry.list_loaded_models()
        downloaded = self.registry.list_downloaded_models()
        try:
            detailed = self.registry.list_loaded_models_detailed()
        except AttributeError:
            detailed = list(loaded)
        system = self.monitor.collect()
        gpu_devices = system.get("gpu_devices", [])
        payload = {
            "models": loaded,
            "loaded_models": loaded,
            "loaded_models_detailed": detailed,
            "downloaded_models": downloaded,
            "running_jobs": self.job_runner.get_running_jobs_count(),
            "supported_modes": self.registry.get_supported_modes(),
            "current_mode": self.registry.get_current_mode(),
            "gpu_devices": gpu_devices,
            "system": system,
        }
        heartbeat_msg = {
            "type": "heartbeat",
            "payload": payload,
        }
        await ws.send(json.dumps(heartbeat_msg))
        logger.debug("Heartbeat sent: %s", payload)

    async def _message_loop(self, ws: websockets.ClientConnection) -> None:
        """Receive and route signaling messages from controller."""
        async for raw_message in ws:
            try:
                data = json.loads(raw_message)
            except Exception as e:
                logger.error("Failed to decode JSON message from controller: %s", e)
                continue

            if not isinstance(data, dict):
                logger.warning("Ignoring non-dict WS payload: %s", type(data))
                continue
            try:
                await self._handle_message(ws, data)
            except Exception as exc:
                logger.exception("Error handling WS message: %s", exc)

    async def _handle_message(
        self,
        ws: websockets.ClientConnection,
        data: dict[str, Any],
    ) -> None:
        """Route message to appropriate handler based on action/type."""
        msg_type = data.get("type", "")
        action = data.get("action", "")
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        logger.info("Received WS message type=%s action=%s", msg_type, action)

        effective_action = action or msg_type

        if effective_action == "dispatch_job":
            self.job_runner.run_job(payload)

        elif effective_action == "load_model":
            model_name = payload.get("model_name", "")
            if model_name:
                try:
                    await self.registry.load_model(model_name)
                    await self._send_model_status(ws)
                except Exception as e:
                    logger.error("Failed to load model '%s': %s", model_name, e)
                    try:
                        err_msg = {
                            "type": "load_model_error",
                            "payload": {
                                "model_name": model_name,
                                "error": str(e),
                            },
                        }
                        await ws.send(json.dumps(err_msg))
                    except Exception:
                        pass

        elif effective_action == "unload_model":
            model_name = payload.get("model_name", "")
            if model_name:
                try:
                    await self.registry.unload_model(model_name)
                    await self._send_model_status(ws)
                except Exception as e:
                    logger.error("Failed to unload model '%s': %s", model_name, e)

        elif effective_action == "unload_all_models":
            try:
                await self.registry.unload_all_models()
                await self._send_model_status(ws)
            except Exception as e:
                logger.error("Failed to unload all models: %s", e)

        elif effective_action == "set_work_mode":
            mode = payload.get("mode", "")
            if mode:
                try:
                    await self.registry.set_work_mode(mode)
                    await self._send_model_status(ws)
                    await self._send_heartbeat(ws)
                except Exception as e:
                    logger.error("Failed to set work mode to '%s': %s", mode, e)

        elif effective_action == "set_config":
            max_jobs = payload.get("max_concurrent_jobs")
            if isinstance(max_jobs, int) and max_jobs > 0:
                self.job_runner.set_max_concurrent_jobs(max_jobs)
            mode = payload.get("work_mode")
            if mode:
                try:
                    await self.registry.set_work_mode(mode)
                    await self._send_model_status(ws)
                except Exception as e:
                    logger.error("Failed to set work mode from set_config: %s", e)
            await self._send_heartbeat(ws)

    async def _send_model_status(self, ws: websockets.ClientConnection) -> None:
        """Send immediate model status update back to controller."""
        loaded = self.registry.list_loaded_models()
        msg = {
            "type": "model_status",
            "payload": {
                "models": loaded,
                "loaded_models": loaded,
            },
        }
        await ws.send(json.dumps(msg))
