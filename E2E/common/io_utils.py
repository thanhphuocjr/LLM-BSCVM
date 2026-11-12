from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Trả về thời điểm hiện tại theo UTC dạng ISO-8601 (ghi vào created_at / completed_at của artifact).
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Băm SHA-256 một chuỗi. Dùng để định danh mã nguồn (run_id, kiểm tra kết quả GPU khớp đúng source).
def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# Chuyển đệ quy dataclass / Path / set / tuple thành kiểu JSON được (dict, list, str).
# Mục đích: write_json ghi được mọi đối tượng của pipeline (CodeUnit, Path...) mà không lỗi.
def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


# Đọc một file JSON (UTF-8) và trả về đối tượng Python.
def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Ghi JSON "nguyên tử": ghi vào file tạm cùng thư mục rồi os.replace sang tên thật.
# Mục đích: nếu tiến trình bị dừng giữa chừng (vd bấm Dừng trên web app) thì không để lại file JSON hỏng/nửa vời.
def write_json(path: str | Path, value: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(jsonable(value), ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, output_path)
    return output_path


# Ghi một file văn bản UTF-8 (tự tạo thư mục cha). Dùng cho báo cáo Markdown/HTML.
def write_text(path: str | Path, value: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(value, encoding="utf-8")
    return output_path

