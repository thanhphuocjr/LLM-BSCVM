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
    """Robust, best-effort JSON extraction from an LLM response.
    
    Handles markdown fences, unescaped control characters (newlines/tabs inside strings),
    and common JSON syntax slips from generative models.
    """
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    
    # 1. Direct parse with strict=False to allow control characters inside strings
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    # 2. Slice from outer braces
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        sliced = text[start : end + 1]
        try:
            return json.loads(sliced, strict=False)
        except json.JSONDecodeError:
            pass

        # 3. Clean common issues: unescaped raw newlines within JSON string values
        try:
            # Replace raw newlines in string properties with \n
            sanitized = re.sub(
                r'(?<!\\)\n',
                r'\\n',
                sliced
            )
            return json.loads(sanitized, strict=False)
        except Exception:
            pass

    # 4. Regex fallback dictionary for critical keys if full JSON parsing failed
    fallback_dict: dict[str, Any] = {}
    for key in (
        "fixed_code", "vulnerability_name", "root_cause", "impact",
        "risk_level", "cvss_score", "cvss_vector", "likelihood",
        "impact_rating", "justification", "status", "confidence",
        "bypass_scenario", "residual_risk", "analyzed_object",
        "contract_function", "executive_summary"
    ):
        pattern = rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"'
        m = re.search(pattern, text)
        if m:
            val = m.group(1).replace(r'\"', '"').replace(r'\n', '\n')
            fallback_dict[key] = val
        else:
            # Check for numeric or boolean
            num_m = re.search(rf'"{key}"\s*:\s*([0-9\.]+|true|false)', text, re.IGNORECASE)
            if num_m:
                raw_v = num_m.group(1).lower()
                if raw_v == "true":
                    fallback_dict[key] = True
                elif raw_v == "false":
                    fallback_dict[key] = False
                else:
                    try:
                        fallback_dict[key] = float(raw_v) if "." in raw_v else int(raw_v)
                    except ValueError:
                        pass

    # Regex for array fields (repair_steps, prevention, recommendations)
    for arr_key in ("repair_steps", "prevention", "recommendations", "changes"):
        arr_m = re.search(rf'"{arr_key}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if arr_m:
            items = re.findall(r'"((?:\\.|[^"\\])*)"', arr_m.group(1))
            fallback_dict[arr_key] = [it.replace(r'\"', '"').replace(r'\n', '\n') for it in items]

    if fallback_dict:
        return fallback_dict

    raise json.JSONDecodeError("Failed to extract valid JSON from LLM response", text, 0)


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
# Server GPU backend (Dedicated RTX 4070 Ti SUPER v13 API)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SERVER_GPU_URL = "http://192.168.91.176:6868"


class ServerGPUClient:
    """Generative client backed by the dedicated GPU Server running v13 FastAPI."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = "qwen25-v13-14b",
        temperature: float | None = None,
        timeout: int = 120,
    ) -> None:
        load_env_file()
        import urllib.request

        self._urllib = urllib.request
        self.base_url = (base_url or os.environ.get("SERVER_GPU_URL", DEFAULT_SERVER_GPU_URL)).rstrip("/")
        self.model_name = model_name or "qwen25-v13-14b"
        env_temp = os.environ.get("SERVER_GPU_TEMPERATURE") or os.environ.get("GEMINI_TEMPERATURE")
        self.temperature = (
            temperature
            if temperature is not None
            else (float(env_temp) if env_temp else DEFAULT_TEMPERATURE)
        )
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        *,
        as_json: bool = False,
        temperature: float | None = None,
        response_schema: Any | None = None,
    ) -> str:
        payload = {
            "prompt": prompt,
            "system_prompt": system_instruction or "",
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": 2048,
        }
        url = f"{self.base_url}/api/v1/generate"
        data = json.dumps(payload).encode("utf-8")
        req = self._urllib.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._urllib.urlopen(req, timeout=self.timeout) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return str(res.get("response", "")).strip()
        except Exception as exc:
            raise ConnectionError(
                f"Không thể kết nối tới GPU Server tại {url} ({exc}). "
                f"Vui lòng đảm bảo dịch vụ trên server đã được khởi chạy."
            ) from exc

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

    def detect(self, code: str, threshold: float = 0.700) -> dict[str, Any]:
        """Gọi endpoint phân loại nhị phân v13 trên GPU 0."""
        url = f"{self.base_url}/api/v1/detect"
        payload = {"code": code, "threshold": threshold}
        data = json.dumps(payload).encode("utf-8")
        req = self._urllib.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._urllib.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise ConnectionError(f"Lỗi khi gọi Detector v13 trên GPU Server ({exc})") from exc


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
        LLM_BACKEND=server_gpu  # GPU Server riêng (RTX 4070 Ti SUPER v13)
        LLM_BACKEND=ollama      # local Qwen via Ollama
        LLM_BACKEND=gemini      # Google Gemini API
    """
    resolved = resolve_backend(backend)
    if resolved in {"server_gpu", "remote_gpu", "server"}:
        return ServerGPUClient(**kwargs)
    if resolved == "gemini":
        return GeminiClient(**kwargs)
    if resolved == "ollama":
        return OllamaClient(**kwargs)
    raise ValueError(f"Unsupported LLM backend: {resolved!r} (use 'server_gpu', 'gemini' hoặc 'ollama').")
