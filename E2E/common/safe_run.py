"""Chạy một thành phần của pipeline; lỗi thành status=error thay vì dừng cả pipeline."""

import time
from typing import Callable


# Chạy action(), đo elapsed_ms; exception -> dict status="error" (fusion sẽ kết luận Inconclusive, không phải Safe).
def safe_run(component: str, action: Callable[[], dict]) -> dict:
    started = time.perf_counter()
    try:
        result = action()
        result.setdefault("component", component)
        result.setdefault("status", "ok")
    except Exception as error:
        result = {"component": component, "status": "error", "score": 0.0, "verdict": "Error",
                  "error": f"{type(error).__name__}: {error}"}
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result
