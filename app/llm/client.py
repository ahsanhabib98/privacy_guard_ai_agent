from __future__ import annotations

import json
import re
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMProtocolError(RuntimeError):
    """Ollama responded, but the model response was not usable structured output."""


class LLMConnectionError(RuntimeError):
    """Ollama/model server could not be reached or completed the request."""


def _strip_thinking(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def _extract_json(text: str) -> dict:
    """Parse a JSON object defensively without ever logging raw model content."""
    text = _strip_thinking(text)
    if not text:
        raise LLMProtocolError("Ollama returned empty structured output")

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # Defensive fallback if the model wrapped JSON in prose despite format/schema.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue

    raise LLMProtocolError("Model response did not contain a valid JSON object")


class QwenOllamaClient:
    """Native Ollama client for one locally hosted Qwen3.5 model."""

    def __init__(self, settings: Settings):
        self.settings = settings
        base = settings.ollama_base_url.rstrip("/")
        self.chat_url = base + "/api/chat"
        self.tags_url = base + "/api/tags"

    async def check_model(self) -> dict:
        """Check Ollama connectivity and whether the configured model is installed."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(self.tags_url)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            raise LLMConnectionError(
                f"Ollama is not reachable at {self.settings.ollama_base_url}. "
                "Start it with `ollama serve`."
            ) from exc

        names = {
            item.get("name")
            for item in body.get("models", [])
            if isinstance(item, dict)
        }
        configured = self.settings.ollama_model
        installed = configured in names

        # Ollama may surface :latest variants; allow prefix-equivalent local tags.
        if not installed:
            installed = any(
                isinstance(name, str)
                and (
                    name == configured
                    or name.startswith(configured + ":")
                    or configured.startswith(name + ":")
                )
                for name in names
            )

        return {
            "ollama": "ok",
            "configured_model": configured,
            "model_installed": installed,
            "installed_models": sorted(x for x in names if isinstance(x, str)),
        }

    async def structured_chat(
        self,
        *,
        system_prompt: str,
        user_payload: dict,
        output_model: type[T],
        schema_name: str,
    ) -> T:
        del schema_name  # Kept in the interface for compatibility/readability.

        schema = output_model.model_json_schema()
        payload = {
            "model": self.settings.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "stream": False,
            # Important for structured JSON calls with Qwen thinking models.
            "think": False if self.settings.disable_thinking else True,
            "format": schema if self.settings.use_json_schema else "json",
            "keep_alive": self.settings.ollama_keep_alive,
            "options": {
                "temperature": self.settings.llm_temperature,
                "num_predict": self.settings.llm_max_tokens,
            },
        }

        timeout = httpx.Timeout(
            connect=10.0,
            read=self.settings.llm_timeout_seconds,
            write=60.0,
            pool=10.0,
        )

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(self.chat_url, json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Ollama is not running at {self.settings.ollama_base_url}. "
                "Start it with `ollama serve`."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMConnectionError(
                "The local Qwen request timed out. Use a smaller model such as "
                "qwen3.5:4b or increase LLM_TIMEOUT_SECONDS."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                data = exc.response.json()
                detail = str(data.get("error", ""))
            except Exception:
                pass
            if "model" in detail.lower() and "not found" in detail.lower():
                raise LLMConnectionError(
                    f"Model {self.settings.ollama_model!r} is not installed. "
                    f"Run `ollama pull {self.settings.ollama_model}`."
                ) from exc
            raise LLMProtocolError(
                f"Ollama returned HTTP {exc.response.status_code}."
            ) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise LLMProtocolError("Ollama returned an invalid HTTP/JSON response") from exc

        message = body.get("message")
        if not isinstance(message, dict):
            raise LLMProtocolError("Unexpected Ollama response shape")

        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            # Do not echo the raw body because model/request content may be sensitive.
            done_reason = body.get("done_reason", "unknown")
            raise LLMProtocolError(
                f"Ollama returned empty message.content (done_reason={done_reason}). "
                "Structured calls should run with thinking disabled."
            )

        data = _extract_json(content)
        try:
            return output_model.model_validate(data)
        except ValidationError as exc:
            raise LLMProtocolError(
                "Structured output failed Pydantic schema validation"
            ) from exc
