"""Run one pipeline component; an exception becomes a result with status=error (never aborts the run)."""

from __future__ import annotations

import time
from typing import Any, Callable


def safe_run(component: str, action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = action()
        result.setdefault("component", component)
        result.setdefault("status", "ok")
    except Exception as error:
        result = {
            "component": component,
            "status": "error",
            "score": 0.0,
            "verdict": "Error",
            "error": f"{type(error).__name__}: {error}",
        }
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result
