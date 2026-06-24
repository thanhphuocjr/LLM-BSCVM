"""
llm_client.py
──────────────────────────────────────────────────────────────────────────────
Pluggable generative-LLM client for the LLM-BSCVM agent phases (Advisor,
Assessor, Fixer, Reporter ...).

The paper uses CodeLlama as the base generative model. This project standardises
on Gemini (configured in .env) for generation, but keeps the call behind a small
`LLMClient` protocol so a local CodeLlama (e.g. via Ollama) can be swapped in
later without touching the agents.

Default backend: Gemini (google-genai SDK), reading config from .env:
    GEMINI_API_KEY
    GEMINI_GENERATION_MODEL   (default: gemini-2.5-flash)
    GEMINI_TEMPERATURE        (default: 0.2)
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"
DEFAULT_BACKEND = "gemini"
DEFAULT_TEMPERATURE = 0.2


# ──────────────────────────────────────────────────────────────────────────────
# Minimal .env loader (avoids adding python-dotenv as a dependency)
# ──────────────────────────────────────────────────────────────────────────────
def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    """Populate os.environ from a .env file (existing env vars win)."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_retry_delay(error: object, default: float) -> float:
    """Pull the server-suggested retry delay (seconds) out of a 429 error."""
    text = str(error)
    match = re.search(r"retry(?:Delay)?[\"'\s:]+(?:in\s+)?(\d+(?:\.\d+)?)\s*s", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return default


def is_rate_limit_error(error: object) -> bool:
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate" in text.lower()


def strip_nonstandard_schema_keys(schema: Any) -> Any:
    """Remove keys (e.g. propertyOrdering) that some backends reject in a JSON schema."""
    if isinstance(schema, dict):
        return {
            key: strip_nonstandard_schema_keys(value)
            for key, value in schema.items()
            if key != "propertyOrdering"
        }
    if isinstance(schema, list):
        return [strip_nonstandard_schema_keys(item) for item in schema]
    return schema


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM response."""
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            return json.loads(text[start : end + 1])
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Client protocol
# ──────────────────────────────────────────────────────────────────────────────
class LLMClient(Protocol):
    model_name: str

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        *,
        as_json: bool = False,
        temperature: float | None = None,
    ) -> str: ...


# ──────────────────────────────────────────────────────────────────────────────
# Gemini backend
# ──────────────────────────────────────────────────────────────────────────────
class GeminiClient:
    """Generative client backed by Google Gemini (google-genai SDK)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
        max_retry_wait: float = 60.0,
    ) -> None:
        load_env_file()
        from google import genai  # imported lazily so the module loads without the SDK

        self._genai = genai
        from google.genai import types as genai_types

        self._types = genai_types

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env or pass api_key=..."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or os.environ.get(
            "GEMINI_GENERATION_MODEL", DEFAULT_GEMINI_MODEL
        )
        env_temp = os.environ.get("GEMINI_TEMPERATURE")
        self.temperature = (
            temperature
            if temperature is not None
            else (float(env_temp) if env_temp else DEFAULT_TEMPERATURE)
        )
        self.max_retries = max_retries
        self.max_retry_wait = max_retry_wait

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        *,
        as_json: bool = False,
        temperature: float | None = None,
        response_schema: Any | None = None,
    ) -> str:
        config = self._types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.temperature if temperature is None else temperature,
            response_mime_type="application/json" if (as_json or response_schema) else None,
            response_schema=response_schema,
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                text = (response.text or "").strip()
                if text:
                    return text
                last_error = RuntimeError("Empty response from Gemini.")
            except Exception as error:  # noqa: BLE001 - surfaced after retries
                last_error = error
            if attempt == self.max_retries - 1:
                break
            # On rate-limit errors, honour the server-suggested retry delay.
            if is_rate_limit_error(last_error):
                wait = min(parse_retry_delay(last_error, 5.0 * (attempt + 1)) + 1.0, self.max_retry_wait)
            else:
                wait = 1.5 * (attempt + 1)
            time.sleep(wait)
        raise RuntimeError(f"Gemini generation failed after {self.max_retries} attempts: {last_error}")

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        *,
        temperature: float | None = None,
        response_schema: Any | None = None,
    ) -> Any:
        raw = self.generate(
            prompt,
            system_instruction,
            as_json=True,
            temperature=temperature,
            response_schema=response_schema,
        )
        return extract_json(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Ollama backend (local models, e.g. Qwen)
# ──────────────────────────────────────────────────────────────────────────────
class OllamaClient:
    """Generative client backed by a local Ollama server (default model: Qwen)."""

    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
        temperature: float | None = None,
        max_retries: int = 2,
    ) -> None:
        load_env_file()
        import ollama  # imported lazily so the module loads without the SDK

        self.model_name = model_name or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        host = host or os.environ.get("OLLAMA_HOST")
        self.client = ollama.Client(host=host) if host else ollama.Client()
        env_temp = os.environ.get("OLLAMA_TEMPERATURE") or os.environ.get("GEMINI_TEMPERATURE")
        self.temperature = (
            temperature
            if temperature is not None
            else (float(env_temp) if env_temp else DEFAULT_TEMPERATURE)
        )
        self.max_retries = max_retries

    @staticmethod
    def _content(response: Any) -> str:
        # The ollama lib returns an object supporting both attribute and item access.
        try:
            return (response.message.content or "").strip()
        except AttributeError:
            return (response["message"]["content"] or "").strip()

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        *,
        as_json: bool = False,
        temperature: float | None = None,
        response_schema: Any | None = None,
    ) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        fmt: Any = None
        if response_schema is not None:
            fmt = strip_nonstandard_schema_keys(response_schema)
        elif as_json:
            fmt = "json"

        options = {"temperature": self.temperature if temperature is None else temperature}

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    format=fmt,
                    options=options,
                )
                text = self._content(response)
                if text:
                    return text
                last_error = RuntimeError("Empty response from Ollama.")
            except Exception as error:  # noqa: BLE001 - surfaced after retries
                last_error = error
            if attempt < self.max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Ollama generation failed after {self.max_retries} attempts: {last_error}")

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        *,
        temperature: float | None = None,
        response_schema: Any | None = None,
    ) -> Any:
        raw = self.generate(
            prompt,
            system_instruction,
            as_json=True,
            temperature=temperature,
            response_schema=response_schema,
        )
        return extract_json(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────
def resolve_backend(backend: str | None) -> str:
    """Resolve which backend to use. "auto"/None reads LLM_BACKEND from .env."""
    if backend and backend.lower() not in {"auto", "config", "default"}:
        return backend.lower()
    load_env_file()
    return os.environ.get("LLM_BACKEND", DEFAULT_BACKEND).strip().lower()


def build_llm_client(backend: str | None = "auto", **kwargs: Any) -> Any:
    """Return a generative LLM client for the requested (or configured) backend.

    backend="auto" (default) reads LLM_BACKEND from .env, e.g.:
        LLM_BACKEND=ollama   # local Qwen via Ollama
        LLM_BACKEND=gemini   # Google Gemini API
    """
    resolved = resolve_backend(backend)
    if resolved == "gemini":
        return GeminiClient(**kwargs)
    if resolved == "ollama":
        return OllamaClient(**kwargs)
    raise ValueError(f"Unsupported LLM backend: {resolved!r} (use 'gemini' or 'ollama').")
