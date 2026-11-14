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
import urllib.request
from pathlib import Path
from typing import Any

from E2E.common.regex_patterns import (
    RE_JSON_FENCE,
    RE_UNESCAPED_NEWLINE,
    re_json_scalar_value,
    re_json_string_value,
)
from E2E.config import DEFAULT_DETECTOR_THRESHOLD, PROJECT_ROOT

DEFAULT_SERVER_GPU_URL = "http://100.122.153.35:6868"
DEFAULT_TEMPERATURE = 0.2


# Nhiệm vụ: đọc file .env ở thư mục gốc và đưa biến vào os.environ (không cần thư viện python-dotenv).
# Giá trị trong .env THẮNG biến môi trường có sẵn và được đọc lại ở mỗi lần gọi.
# Mục đích: detector và các agent luôn dùng cùng một SERVER_GPU_URL / SERVER_GPU_API_KEY / SERVER_GPU_TEMPERATURE.
def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    """Populate os.environ from .env (no python-dotenv dependency); .env wins and is re-read on every call."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            os.environ[key] = value.strip().strip('"').strip("'")


# Trả về URL GPU server: tham số truyền vào > SERVER_GPU_URL trong .env > DEFAULT_SERVER_GPU_URL.
def get_server_url(server_url: str | None = None) -> str:
    if server_url:
        return server_url.rstrip("/")
    load_env_file()
    return os.environ.get("SERVER_GPU_URL", DEFAULT_SERVER_GPU_URL).rstrip("/")


# Tạo header X-API-Key nếu có SERVER_GPU_API_KEY (bí mật dùng chung giữa máy local và server).
def _auth_headers() -> dict[str, str]:
    """Optional shared secret for the GPU server (set SERVER_GPU_API_KEY on both sides)."""
    load_env_file()
    api_key = os.environ.get("SERVER_GPU_API_KEY", "").strip()
    return {"X-API-Key": api_key} if api_key else {}


# Gửi POST JSON tới một URL (kèm header xác thực) và parse JSON trả về. Hàm HTTP cấp thấp duy nhất cho POST.
def _post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **_auth_headers()},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# Nhiệm vụ: gọi GET /api/v1/health của GPU server; lỗi kết nối thì trả {"status": "offline", ...}.
# Mục đích: web app hiển thị trạng thái server; gpu_pipeline kiểm tra server sẵn sàng trước khi chạy Phase 1C.
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


# [Phase 1C / Phase 5] Gửi mã một hàm tới /api/v1/detect -> server trả P(vulnerable) và verdict
# theo ngưỡng. Gửi code gốc (không tiền xử lý) để giống hệt lúc train (train-serve consistency).
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


# [Phase 2/3/4] Gửi prompt tới /api/v1/generate (model instruct, tắt LoRA) và trả về chuỗi trả lời.
# Đường HTTP duy nhất để sinh văn bản: Fixer gọi trực tiếp (temperature 0), Advisor/Assessor gọi qua ServerGPUClient.
def generate_via_server(
    prompt: str,
    system_prompt: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
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


# Các khoá mà agent thực sự đọc từ JSON của model (dùng khi phải "vớt" JSON hỏng bằng regex).
#   Advisor (advisor.advise_one):  is_true_positive, root_cause, attack_scenario, impact, fix_strategy
#   Assessor (assessor.assess_one): 8 metric CVSS + justification
_SALVAGE_KEYS = (
    "is_true_positive", "root_cause", "attack_scenario", "impact", "fix_strategy",
    "attack_vector", "attack_complexity", "privileges_required", "user_interaction",
    "scope", "confidentiality", "integrity", "availability", "justification",
)


# Nhiệm vụ: lấy JSON từ câu trả lời của LLM một cách "chịu lỗi": bỏ ```json fence -> parse thẳng ->
# cắt từ { đầu tới } cuối -> thay xuống dòng thô bằng \n -> cuối cùng dùng regex vớt từng khoá mà
# Advisor/Assessor cần (_SALVAGE_KEYS).
# Mục đích: model 14B hay trả JSON sai cú pháp; hàm này giúp Advisor/Assessor vẫn dùng được kết quả.
def extract_json(text: str) -> Any:
    """Robust, best-effort JSON extraction from an LLM response.

    Handles markdown fences, unescaped control characters (newlines/tabs inside strings),
    and common JSON syntax slips from generative models.
    """
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    fenced = RE_JSON_FENCE.search(text)
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

        # 3. Raw newlines inside string values -> \n
        try:
            return json.loads(RE_UNESCAPED_NEWLINE.sub(r"\\n", sliced), strict=False)
        except Exception:
            pass

    # 4. Regex salvage of the keys the agents read (string, number or boolean values)
    salvaged: dict[str, Any] = {}
    for key in _SALVAGE_KEYS:
        string_match = re_json_string_value(key).search(text)
        if string_match:
            salvaged[key] = string_match.group(1).replace(r'\"', '"').replace(r'\n', '\n')
            continue
        scalar_match = re_json_scalar_value(key).search(text)
        if scalar_match:
            raw_value = scalar_match.group(1).lower()
            if raw_value in {"true", "false"}:
                salvaged[key] = raw_value == "true"
            else:
                try:
                    salvaged[key] = float(raw_value) if "." in raw_value else int(raw_value)
                except ValueError:
                    pass

    if salvaged:
        return salvaged

    raise json.JSONDecodeError("Failed to extract valid JSON from LLM response", text, 0)


# Client LLM dùng chung cho Advisor và Assessor: bọc generate_via_server + extract_json.
# Mục đích: các agent chỉ cần gọi .generate_json(), không cần biết chi tiết HTTP.
class ServerGPUClient:
    """Generative client backed by the dedicated GPU Server running v13 FastAPI."""

    # Đọc cấu hình: base_url (SERVER_GPU_URL), temperature (SERVER_GPU_TEMPERATURE, mặc định 0.2), timeout.
    def __init__(self, timeout: int = 120) -> None:
        self.base_url = get_server_url()  # also loads .env
        env_temp = os.environ.get("SERVER_GPU_TEMPERATURE")
        self.temperature = float(env_temp) if env_temp else DEFAULT_TEMPERATURE
        self.timeout = timeout

    # Gửi prompt + system prompt tới GPU server, trả về văn bản. Lỗi mạng -> ConnectionError có thông báo rõ.
    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        try:
            return generate_via_server(
                prompt,
                system_instruction or "",
                temperature=self.temperature,
                max_tokens=2048,
                server_url=self.base_url,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Không thể kết nối tới GPU Server tại {self.base_url}/api/v1/generate ({exc}). "
                f"Vui lòng đảm bảo dịch vụ trên server đã được khởi chạy."
            ) from exc

    # Như generate nhưng parse câu trả lời thành JSON bằng extract_json (agent cần output có cấu trúc).
    def generate_json(self, prompt: str, system_instruction: str | None = None) -> Any:
        return extract_json(self.generate(prompt, system_instruction))
