"""Client HTTP tới GPU server: /health, /detect (detector v13), /generate (Advisor, Assessor, Fixer)."""

import contextlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from E2E.common.regex_patterns import RE_JSON_FENCE, RE_UNESCAPED_NEWLINE, re_json_scalar_value, re_json_string_value
from E2E.config import DEFAULT_DETECTOR_THRESHOLD, PROJECT_ROOT

DEFAULT_SERVER_GPU_URL = "http://100.122.153.35:6868"
DEFAULT_TEMPERATURE = 0.2
# Các khoá Advisor/Assessor đọc từ JSON của model; dùng để "vớt" giá trị khi JSON bị hỏng.
_SALVAGE_KEYS = (
    "verdict", "guard_quote", "vulnerability_class", "root_cause", "attack_scenario", "impact", "fix_strategy",
    "attack_vector", "attack_complexity", "privileges_required", "user_interaction",
    "scope", "confidentiality", "integrity", "availability", "justification",
)


# Nạp biến từ file .env vào os.environ (giá trị trong .env thắng biến có sẵn, đọc lại ở mỗi lần gọi).
def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() and not key.startswith("#"):
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


# URL GPU server: tham số truyền vào > SERVER_GPU_URL trong .env > mặc định.
def get_server_url(server_url: str | None = None) -> str:
    if server_url:
        return server_url.rstrip("/")
    load_env_file()
    return os.environ.get("SERVER_GPU_URL", DEFAULT_SERVER_GPU_URL).rstrip("/")


# Gọi một endpoint của GPU server (GET khi payload=None, ngược lại POST JSON) và parse JSON trả về.
def _request_json(url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    load_env_file()
    api_key = os.environ.get("SERVER_GPU_API_KEY", "").strip()
    headers = {"Content-Type": "application/json", **({"X-API-Key": api_key} if api_key else {})}
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


# GET /api/v1/health; lỗi kết nối -> {"status": "offline"} (web app hiển thị, gpu_pipeline kiểm tra trước Phase 1C).
def check_server_health(server_url: str | None = None, timeout: int = 3) -> dict:
    url = f"{get_server_url(server_url)}/api/v1/health"
    try:
        return _request_json(url, timeout=timeout)
    except Exception as exc:
        return {"status": "offline", "error": str(exc), "url": url}


# [Phase 1C / 5] Gửi mã một hàm tới /api/v1/detect -> P(vulnerable) (gửi code gốc, giống lúc train).
def detect_via_server(code: str, threshold: float = DEFAULT_DETECTOR_THRESHOLD, server_url: str | None = None,
                      timeout: int = 60) -> dict:
    return _request_json(f"{get_server_url(server_url)}/api/v1/detect", {"code": code, "threshold": threshold}, timeout)


# [Phase 2/3/4] Gửi prompt tới /api/v1/generate (model instruct, tắt LoRA) và trả về câu trả lời.
def generate_via_server(prompt: str, system_prompt: str = "", temperature: float = DEFAULT_TEMPERATURE,
                        max_tokens: int = 1500, server_url: str | None = None, timeout: int = 120) -> str:
    payload = {"prompt": prompt, "system_prompt": system_prompt, "temperature": temperature, "max_tokens": max_tokens}
    response = _request_json(f"{get_server_url(server_url)}/api/v1/generate", payload, timeout)
    return str(response.get("response", "")).strip()


# Lấy JSON từ câu trả lời LLM: bỏ ```json fence -> parse -> cắt { ... } -> sửa xuống dòng thô -> vớt từng khoá.
def extract_json(text: str) -> Any:
    text = text.strip()
    if fenced := RE_JSON_FENCE.search(text):
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    sliced = text[start:end + 1] if 0 <= start < end else ""
    for candidate in (text, sliced, RE_UNESCAPED_NEWLINE.sub(r"\\n", sliced)):
        if candidate:
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(candidate, strict=False)
    salvaged: dict[str, Any] = {}
    for key in _SALVAGE_KEYS:
        if match := re_json_string_value(key).search(text):
            salvaged[key] = match.group(1).replace(r'\"', '"').replace(r'\n', '\n')
        elif match := re_json_scalar_value(key).search(text):
            raw = match.group(1).lower()
            if raw in {"true", "false"}:
                salvaged[key] = raw == "true"
            else:
                with contextlib.suppress(ValueError):
                    salvaged[key] = float(raw) if "." in raw else int(raw)
    if salvaged:
        return salvaged
    raise json.JSONDecodeError("Failed to extract valid JSON from LLM response", text, 0)


# Client LLM của Advisor và Assessor: generate_via_server + extract_json, lỗi mạng -> ConnectionError rõ ràng.
class ServerGPUClient:
    # Đọc URL server, temperature (SERVER_GPU_TEMPERATURE, mặc định 0.2) và timeout.
    def __init__(self, timeout: int = 120) -> None:
        self.base_url = get_server_url()
        self.temperature = float(os.environ.get("SERVER_GPU_TEMPERATURE") or DEFAULT_TEMPERATURE)
        self.timeout = timeout

    # Gửi prompt + system prompt, trả về văn bản.
    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        try:
            return generate_via_server(prompt, system_instruction or "", temperature=self.temperature, max_tokens=2048,
                                       server_url=self.base_url, timeout=self.timeout)
        except Exception as exc:
            raise ConnectionError(
                f"Không thể kết nối tới GPU Server tại {self.base_url}/api/v1/generate ({exc}). "
                f"Vui lòng đảm bảo dịch vụ trên server đã được khởi chạy."
            ) from exc

    # Như generate nhưng parse câu trả lời thành JSON (extract_json).
    def generate_json(self, prompt: str, system_instruction: str | None = None) -> Any:
        return extract_json(self.generate(prompt, system_instruction))
