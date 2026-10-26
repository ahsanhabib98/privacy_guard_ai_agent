from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class Settings:
    # Agent loop
    max_attempts: int = int(
        os.getenv("MAX_ATTEMPTS", "2")
    )

    # Privacy-safe episodic memory
    memory_db: str = os.getenv(
        "MEMORY_DB",
        ".agent_memory/experiences.sqlite3",
    )

    privacy_hmac_secret: str = os.getenv(
        "PRIVACY_HMAC_SECRET",
        "development-only-secret-change-me-please",
    )

    # Ollama + Qwen
    # 4B is the practical development default for a Mac.
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    )

    ollama_model: str = os.getenv(
        "OLLAMA_MODEL",
        "qwen3.5:4b",
    )

    ollama_keep_alive: str = os.getenv(
        "OLLAMA_KEEP_ALIVE",
        "10m",
    )

    # Local model settings
    llm_timeout_seconds: float = float(
        os.getenv("LLM_TIMEOUT_SECONDS", "900")
    )

    llm_temperature: float = float(
        os.getenv("LLM_TEMPERATURE", "0.0")
    )

    llm_max_tokens: int = int(
        os.getenv("LLM_MAX_TOKENS", "2500")
    )

    use_json_schema: bool = _env_bool(
        "USE_JSON_SCHEMA",
        True,
    )

    disable_thinking: bool = _env_bool(
        "DISABLE_THINKING",
        True,
    )

    def validate(self) -> None:
        if len(self.privacy_hmac_secret) < 24:
            raise ValueError(
                "PRIVACY_HMAC_SECRET must be at least 24 characters"
            )

        if self.max_attempts < 1:
            raise ValueError(
                "MAX_ATTEMPTS must be at least 1"
            )

        if self.llm_timeout_seconds <= 0:
            raise ValueError(
                "LLM_TIMEOUT_SECONDS must be positive"
            )

        if not self.ollama_base_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError(
                "OLLAMA_BASE_URL must start with http:// or https://"
            )


settings = Settings()