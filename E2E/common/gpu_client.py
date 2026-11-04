"""
HTTP client for the GPU server (gpu_server/serve_v13_api.py).

One Qwen2.5-Coder-14B-Instruct instance on the server answers two kinds of request:
  * /api/v1/detect   - LoRA detector v13, P(vulnerable) of one function  (Phase 1C, Phase 5 re-check)
  * /api/v1/generate - the instruct model (adapter off), text / JSON     (Phase 2 Advisor, 3 Assessor, 4 Fixer)

The address comes from SERVER_GPU_URL in .env (optional shared secret SERVER_GPU_API_KEY).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from E2E.config import DEFAULT_DETECTOR_THRESHOLD

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_GPU_URL = "http://100.122.153.35:6868"
DEFAULT_TEMPERATURE = 0.2


def load_env_file(path: Path = PROJECT_ROOT / ".env", *, override: bool = False) -> None:
    """Populate os.environ from .env (no python-dotenv dependency).

    override=False: variables already set in the environment win (agents' LLM client).
    override=True:  .env wins and is re-read on every call (server URL / API key lookups).
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def get_server_url(server_url: str | None = None) -> str:
    if server_url:
        return server_url.rstrip("/")
    load_env_file(override=True)
    return os.environ.get("SERVER_GPU_URL", DEFAULT_SERVER_GPU_URL).rstrip("/")


def _auth_headers() -> dict[str, str]:
    """Optional shared secret for the GPU server (set SERVER_GPU_API_KEY on both sides)."""
    load_env_file(override=True)
    api_key = os.environ.get("SERVER_GPU_API_KEY", "").strip()
    return {"X-API-Key": api_key} if api_key else {}


def _post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **_auth_headers()},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_server_health(server_url: str | None = None, timeout: int = 3) -> dict[str, Any]:
    """Kiểm tra xem API server v13 trên GPU server có sẵn sàng hay không."""
    actual_url = get_server_url(server_url)
    url = f"{actual_url}/api/v1/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LLM-BSCVM-Local", **_auth_headers()})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "offline", "error": str(exc), "url": url}


def detect_via_server(
    code: str,
    threshold: float = DEFAULT_DETECTOR_THRESHOLD,
    server_url: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Gửi code tới endpoint /api/v1/detect trên GPU server (dùng code gốc để đồng bộ train-serve)."""
    actual_url = get_server_url(server_url)
    url = f"{actual_url}/api/v1/detect"
    payload = {"code": code, "threshold": threshold}
    return _post_json(url, payload, timeout=timeout)


def generate_via_server(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.2,
    max_tokens: int = 1500,
    server_url: str | None = None,
    timeout: int = 120,
) -> str:
    """Gửi prompt tới endpoint /api/v1/generate trên GPU server."""
    actual_url = get_server_url(server_url)
    url = f"{actual_url}/api/v1/generate"
    payload = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    res = _post_json(url, payload, timeout=timeout)
    return str(res.get("response", "")).strip()


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
        self._urllib = urllib.request
        self.base_url = (base_url or os.environ.get("SERVER_GPU_URL", DEFAULT_SERVER_GPU_URL)).rstrip("/")
        self.model_name = model_name or "qwen25-v13-14b"
        env_temp = os.environ.get("SERVER_GPU_TEMPERATURE")
        self.temperature = (
            temperature
            if temperature is not None
            else (float(env_temp) if env_temp else DEFAULT_TEMPERATURE)
        )
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("SERVER_GPU_API_KEY", "").strip()
        if api_key:
            headers["X-API-Key"] = api_key
        return headers

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
            headers=self._headers(),
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
