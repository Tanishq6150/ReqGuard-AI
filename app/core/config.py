"""Application settings loaded from YAML and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

_ENV_FILE = ROOT_DIR / ".env"
load_dotenv(_ENV_FILE, override=True)


class ApplicationSettings(BaseModel):
    name: str = "ReqGuard AI"
    version: str = "1.0.0"
    max_requirement_length: int = 8000


class LLMSettings(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-3.5-flash-lite"
    temperature: float = 0.2
    timeout: int = 30
    max_retries: int = 3
    enrich_analysis: bool = True


class GenerationSettings(BaseModel):
    max_test_cases: int = 6
    include_positive: bool = True
    include_negative: bool = True
    include_boundary: bool = True
    include_edge_cases: bool = True
    include_security: bool = True


class NLPSettings(BaseModel):
    ambiguity_detection: bool = True
    completeness_analysis: bool = True
    similarity_threshold: float = 0.85
    embedding_model: str = "all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"


class APISettings(BaseModel):
    prefix: str = "/api/v1"


class Settings(BaseModel):
    application: ApplicationSettings = Field(default_factory=ApplicationSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    nlp: NLPSettings = Field(default_factory=NLPSettings)
    api: APISettings = Field(default_factory=APISettings)

    llm_api_key: str = ""
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


def _load_yaml() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read_api_key() -> str:
    return (
        os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    ).strip()


def get_settings() -> Settings:
    load_dotenv(_ENV_FILE, override=True)
    data = _load_yaml()
    settings = Settings.model_validate(data)
    provider = os.getenv("LLM_PROVIDER", settings.llm.provider)
    model = os.getenv("LLM_MODEL", settings.llm.model)
    settings.llm = settings.llm.model_copy(update={"provider": provider, "model": model})
    settings.llm_api_key = _read_api_key()
    settings.log_level = os.getenv("LOG_LEVEL", settings.log_level)
    settings.api_host = os.getenv("API_HOST", settings.api_host)
    settings.api_port = int(os.getenv("API_PORT", str(settings.api_port)))
    return settings
