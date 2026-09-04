# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration model for the distributed Transcribe agent."""

    controller_url: str = Field(
        default="http://localhost:8080",
        description="Base URL of the controller service",
    )
    agent_token: str = Field(
        default="",
        description="Authentication token for connecting to the controller",
    )
    node_name: str = Field(
        default="agent-default",
        description="Human-readable name for this agent node",
    )
    heartbeat_interval: int = Field(
        default=10,
        description="Interval in seconds between heartbeats",
    )
    media_dir: str = Field(
        default="/tmp/transcribe/media",
        description="Local directory for temporary audio files",
    )
    max_concurrent_jobs: int = Field(
        default=2,
        description="Maximum number of concurrently running transcription jobs",
    )

    @property
    def http_base_url(self) -> str:
        """Normalized base HTTP URL without trailing slash."""
        return self.controller_url.rstrip("/")

    @property
    def ws_url(self) -> str:
        """Construct WebSocket endpoint URL with token authentication."""
        base = self.http_base_url
        if base.startswith("http://"):
            base = "ws://" + base[7:]
        elif base.startswith("https://"):
            base = "wss://" + base[8:]

        endpoint = f"{base}/api/v1/agent/ws"
        if self.agent_token:
            endpoint += f"?token={self.agent_token}"
        return endpoint


def load_config(config_path: str | Path | None = None) -> AgentConfig:
    """Load configuration from YAML file and apply environment variable overrides.

    Priority:
    1. Environment variables (highest)
    2. YAML configuration file
    3. Default values (lowest)
    """
    data: dict[str, object] = {}

    # Locate config file
    candidate_paths: list[Path] = []
    if config_path is not None:
        candidate_paths.append(Path(config_path))
    elif os.getenv("CONFIG_PATH"):
        candidate_paths.append(Path(os.getenv("CONFIG_PATH", "")))
    else:
        candidate_paths.extend([
            Path("config.yaml"),
            Path("backend/agent/config.yaml"),
            Path(__file__).resolve().parent.parent / "config.yaml",
        ])

    for p in candidate_paths:
        if p.is_file():
            try:
                with open(p, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        data = loaded
                break
            except Exception:
                pass

    # Environment variable overrides
    if env_url := os.getenv("CONTROLLER_URL"):
        data["controller_url"] = env_url
    if env_token := os.getenv("AGENT_TOKEN"):
        data["agent_token"] = env_token
    if env_node := os.getenv("NODE_NAME"):
        data["node_name"] = env_node
    if env_heartbeat := os.getenv("HEARTBEAT_INTERVAL"):
        try:
            data["heartbeat_interval"] = int(env_heartbeat)
        except ValueError:
            pass
    if env_media := os.getenv("MEDIA_DIR"):
        data["media_dir"] = env_media
    if env_max_jobs := os.getenv("MAX_CONCURRENT_JOBS"):
        try:
            data["max_concurrent_jobs"] = int(env_max_jobs)
        except ValueError:
            pass

    return AgentConfig(**data)
