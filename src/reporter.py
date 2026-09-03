# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class Reporter:
    """HTTP reporter for media downloads, progress logs, and task completion settlements."""

    def __init__(
        self,
        base_url: str,
        agent_token: str,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_token = agent_token
        self._custom_client = client is not None
        headers = {}
        if agent_token:
            headers["Authorization"] = f"Bearer {agent_token}"

        self.client = client or httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
        )

    async def download_media(self, job_id: int, target_path: str) -> str:
        """Stream download audio media file for a job to local target path."""
        url = f"{self.base_url}/api/v1/agent/jobs/{job_id}/media"
        parent_dir = os.path.dirname(target_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        logger.info("Downloading media for job %d from %s", job_id, url)
        async with self.client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(target_path, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)

        logger.info("Downloaded media for job %d to %s", job_id, target_path)
        return target_path

    async def report_logs(
        self,
        job_id: int,
        progress: int,
        logs: list[dict[str, Any]],
    ) -> None:
        """Report batch execution logs and progress percentage to controller."""
        url = f"{self.base_url}/api/v1/agent/jobs/{job_id}/logs"
        payload = {
            "progress": progress,
            "logs": logs,
        }
        resp = await self.client.post(url, json=payload)
        resp.raise_for_status()

    async def report_completion(
        self,
        job_id: int,
        status: str,
        duration_seconds: float,
        result_text: str = "",
        openai_response: dict[str, Any] | None = None,
        error_msg: str = "",
    ) -> None:
        """Report final settlement status and transcription output to controller."""
        url = f"{self.base_url}/api/v1/agent/jobs/{job_id}/complete"
        payload = {
            "status": status,
            "duration_seconds": duration_seconds,
            "result_text": result_text,
            "openai_response": openai_response,
            "error_msg": error_msg,
        }
        logger.info(
            "Reporting completion for job %d: status=%s, duration=%.2fs",
            job_id,
            status,
            duration_seconds,
        )
        resp = await self.client.post(url, json=payload)
        resp.raise_for_status()

    async def close(self) -> None:
        """Close underlying HTTP client if not externally provided."""
        if not self._custom_client:
            await self.client.aclose()
