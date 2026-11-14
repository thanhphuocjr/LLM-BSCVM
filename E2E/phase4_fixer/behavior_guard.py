"""
Phase 4 - behaviour guard: a patch must keep what the original function does.

`validate_observable_behavior(original, replacement)` raises ValueError when the
replacement drops an ETH transfer (or its safe equivalent), a return statement, a state
write or an event of the original. Used by the Fixer before a patch is applied, and by
Phase 5 (behavior_verifier) for patched units the AST comparison could not cover.
"""

from __future__ import annotations

from E2E.common.regex_patterns import (
    RE_EMIT_EVENT,
    RE_ETH_VALUE_CALL,
    RE_LEGACY_EVENT_STATEMENT,
    RE_RETURN_KEYWORD,
    RE_SAFE_TRANSFER_CALL,
    RE_SAFE_TRANSFER_ETH_CALL,
    RE_SEND_CALL,
    RE_SEND_VALUE_CALL,
    RE_STATE_WRITE,
    RE_TRANSFER_CALL,
)
from E2E.phase0_parsing.code_units import mask_noncode
from E2E.phase0_parsing.contract_context import local_variable_names


_TRANSFER_EQUIVALENTS = {
    # kind: (pattern that detects the original call, patterns accepted as preserving it)
    "ETH value call": (RE_ETH_VALUE_CALL, (RE_ETH_VALUE_CALL, RE_SEND_VALUE_CALL, RE_SAFE_TRANSFER_ETH_CALL)),
    "ETH transfer": (
        RE_TRANSFER_CALL,
        (RE_TRANSFER_CALL, RE_SAFE_TRANSFER_CALL, RE_ETH_VALUE_CALL, RE_SEND_VALUE_CALL, RE_SAFE_TRANSFER_ETH_CALL),
    ),
    "ETH send": (
        RE_SEND_CALL,
        (RE_SEND_CALL, RE_TRANSFER_CALL, RE_ETH_VALUE_CALL, RE_SEND_VALUE_CALL),
    ),
}


# Tập tên biến state bị ghi trong hàm (=, +=, -=, ...), đã loại tham số / biến cục bộ (local_variable_names).
def _state_writes(code: str) -> set[str]:
    local = local_variable_names(code)
    return {
        match.group(1)
        for match in RE_STATE_WRITE.finditer(code)
        if match.group(1) not in local
    }


# Tập event được phát: "emit X(...)" hoặc câu lệnh "X(...);" kiểu Solidity cũ (< 0.4.21), không tính ép kiểu.
def _emitted_events(code: str) -> set[str]:
    """`emit X(...)`, or a legacy (<0.4.21) statement `X(...);` that is a call on its own.

    Type conversions such as `IERC20(token).transfer(...)` are not events.
    """
    events = set(RE_EMIT_EVENT.findall(code))
    for match in RE_LEGACY_EVENT_STATEMENT.finditer(code):
        depth = 0
        cursor = match.end() - 1
        while cursor < len(code):
            if code[cursor] == "(":
                depth += 1
            elif code[cursor] == ")":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        rest = code[cursor + 1 :].lstrip()
        if rest.startswith(";"):
            events.add(match.group(1))
    return events


# [GUARD HÀNH VI] So sánh hàm gốc và bản vá; raise ValueError nếu bản vá làm MẤT:
#   - chuyển ETH (call{value}/transfer/send - chấp nhận dạng an toàn tương đương như sendValue, safeTransfer),
#   - lệnh return, event đã phát, hoặc lệnh ghi biến state.
# Mục đích: chặn bản vá "sửa lỗi bằng cách xoá chức năng". Là kiểm tra cú pháp, không phải chứng minh tương đương.
# Dùng ở: Fixer (trước khi ghép bản vá) và Phase 5 (khi không so được bằng AST).
def validate_observable_behavior(original: str, replacement: str) -> None:
    """
    Guard bảo toàn hành vi (Phase 4 / Phase 5) — kiểm tra cú pháp các tín hiệu observable:
      - Chuyển giá trị: call{value}, transfer, send (chấp nhận dạng an toàn tương đương,
        ví dụ transfer -> safeTransfer, send -> call{value})
      - Event: emit X(...) hoặc câu lệnh X(...); kiểu 0.4
      - Ghi biến state (bỏ qua tham số và biến cục bộ không phải storage)
      - Lệnh return
    Đây là kiểm tra cú pháp, không phải chứng minh tương đương ngữ nghĩa.
    """
    original_code = mask_noncode(original, mask_strings=False)
    replacement_code = mask_noncode(replacement, mask_strings=False)

    for label, (detect, preserved_by) in _TRANSFER_EQUIVALENTS.items():
        if detect.search(original_code) and not any(
            pattern.search(replacement_code) for pattern in preserved_by
        ):
            raise ValueError(f"Generated patch removed observable behavior: {label}")

    if RE_RETURN_KEYWORD.search(original_code) and not RE_RETURN_KEYWORD.search(replacement_code):
        raise ValueError("Generated patch removed observable behavior: return statement")

    missing_events = _emitted_events(original_code) - _emitted_events(replacement_code)
    if missing_events:
        raise ValueError(
            f"Generated patch removed observable event emission: {', '.join(sorted(missing_events))}"
        )

    missing_writes = _state_writes(original_code) - _state_writes(replacement_code)
    if missing_writes:
        raise ValueError(f"Generated patch removed state variable mutation: {', '.join(sorted(missing_writes))}")
