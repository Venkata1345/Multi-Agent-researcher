"""Application settings via pydantic-settings.

Single source of truth for env-driven configuration. Phase 1 only needs the
model name and provider knobs declared; the LLM/MCP code that consumes them
arrives in later phases. Provider swap (OpenAI -> Anthropic) is intended to be a
one-line change to ``provider`` plus a model name.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Model provider ---
    provider: Literal["openai", "anthropic"] = "openai"
    research_model: str = Field(default="gpt-4o-mini")
    openai_api_key: str | None = None
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    # "function_calling" tolerates our full schema set (incl. HttpUrl/`format: uri`,
    # which OpenAI's strict `json_schema` rejects). Flip to "json_schema" only if
    # the message schemas are constrained to the strict-mode subset.
    structured_output_method: Literal["function_calling", "json_schema"] = (
        "function_calling"
    )

    # --- Observability ---
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "multi-agent-research"

    # --- Tools (Phase 3) ---
    tavily_api_key: str | None = None
    research_workspace: str = "./workspace"


def get_settings() -> Settings:
    """Load settings from the environment / ``.env``."""
    return Settings()
