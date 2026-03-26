"""Application settings loaded exclusively from environment variables.

Never hardcode secrets. All sensitive values must be in .env (gitignored).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Database ---
    database_url: PostgresDsn = Field(
        description="Async PostgreSQL DSN for feature store and scores."
    )
    audit_database_url: PostgresDsn = Field(
        description="Async PostgreSQL DSN for audit log (separate schema)."
    )

    # --- JWT ---
    jwt_private_key_path: Path = Field(description="Path to RS256 private key PEM file.")
    jwt_public_key_path: Path = Field(description="Path to RS256 public key PEM file.")
    jwt_algorithm: str = Field(default="RS256")
    jwt_access_token_expire_minutes: int = Field(default=60)

    # --- Pseudonymization ---
    org_salt: str = Field(description="Org-specific salt for SHA-256 UUID v5 pseudonymization.")

    # --- Encrypted Mapping Store ---
    vault_key: str = Field(description="AES-256 master key for encrypted ID mapping volume.")
    vault_path: Path = Field(default=Path("/vault/id_mapping.enc"))

    # --- Pipeline ---
    refresh_interval_hours: int = Field(default=48, ge=1, le=168)
    alert_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    cascade_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    cascade_alert_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    decay_factor: float = Field(default=0.60, ge=0.0, le=1.0)
    max_hops: int = Field(default=2, ge=1, le=5)
    retention_days: int = Field(default=90, ge=1)

    # --- Connectors ---
    adapter_mode: Literal["mock", "live"] = Field(
        default="mock",
        description="'mock' uses synthetic data. 'live' calls real APIs.",
    )
    msgraph_client_id: str = Field(default="")
    msgraph_client_secret: str = Field(default="")
    msgraph_tenant_id: str = Field(default="")
    slack_bot_token: str = Field(default="")
    slack_signing_secret: str = Field(default="")
    github_org_token: str = Field(default="")
    github_org_name: str = Field(default="")

    # --- Power BI ---
    powerbi_client_id: str = Field(default="")
    powerbi_client_secret: str = Field(default="")
    powerbi_tenant_id: str = Field(default="")
    powerbi_dataset_id: str = Field(default="")
    powerbi_workspace_id: str = Field(default="")
    powerbi_dashboard_url: str = Field(default="")

    # --- Teams Bot ---
    teams_app_id: str = Field(default="")
    teams_app_password: str = Field(default="")
    teams_hr_channel_id: str = Field(default="")
    teams_manager_channel_ids: dict[str, str] = Field(default_factory=dict)

    # --- Model Registry ---
    model_registry_path: Path = Field(default=Path("/models"))
    active_model_version: str = Field(default="latest")

    @field_validator("teams_manager_channel_ids", mode="before")
    @classmethod
    def parse_json_dict(cls, v: str | dict[str, str]) -> dict[str, str]:
        if isinstance(v, str):
            return json.loads(v)  # type: ignore[no-any-return]
        return v

    @property
    def jwt_private_key(self) -> str:
        return self.jwt_private_key_path.read_text()

    @property
    def jwt_public_key(self) -> str:
        return self.jwt_public_key_path.read_text()


# Singleton — import this throughout the application
settings: Settings = Settings()  # type: ignore[call-arg]
