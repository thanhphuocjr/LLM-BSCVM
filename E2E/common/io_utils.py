"""Đọc/ghi JSON, ghi văn bản, SHA-256, thời gian UTC, in log tiến độ."""

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Thời điểm hiện tại theo UTC dạng ISO-8601 (created_at / completed_at của artifact).
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Băm SHA-256 một chuỗi (định danh mã nguồn trong run_id, kiểm tra kết quả GPU khớp source).
def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# In một dòng log kèm giờ "[HH:MM:SS] ..." (web app đọc các dòng "[PHASE x]" để cập nhật tiến độ).
def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# Đổi đệ quy dataclass / Path / set / tuple thành kiểu ghi được ra JSON.
def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return str(value) if isinstance(value, Path) else value


# Đọc một file JSON (UTF-8).
def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Ghi JSON "nguyên tử": ghi file tạm cùng thư mục rồi os.replace (dừng giữa chừng không để lại file hỏng).
def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
                                     delete=False) as f:
        f.write(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n")
    os.replace(f.name, path)
    return path


# Ghi một file văn bản UTF-8 (tự tạo thư mục cha), dùng cho báo cáo Markdown/HTML.
def write_text(path: str | Path, value: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path
