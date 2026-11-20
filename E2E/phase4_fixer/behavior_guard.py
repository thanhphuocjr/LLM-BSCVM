"""Phase 4 - guard hành vi: bản vá phải giữ chuyển ETH, return, event và lệnh ghi state của hàm gốc."""

from E2E.common.regex_patterns import (
    RE_EMIT_EVENT, RE_ETH_VALUE_CALL, RE_LEGACY_EVENT_STATEMENT, RE_RETURN_KEYWORD, RE_SAFE_TRANSFER_CALL,
    RE_SAFE_TRANSFER_ETH_CALL, RE_SEND_CALL, RE_SEND_VALUE_CALL, RE_STATE_WRITE, RE_TRANSFER_CALL,
)
from E2E.phase0_parsing.code_units import mask_noncode, matching_close
from E2E.phase0_parsing.contract_context import local_variable_names

# Kiểu chuyển ETH: (regex nhận diện ở hàm gốc, các dạng được coi là vẫn giữ hành vi đó ở bản vá).
_TRANSFER_EQUIVALENTS = {
    "ETH value call": (RE_ETH_VALUE_CALL, (RE_ETH_VALUE_CALL, RE_SEND_VALUE_CALL, RE_SAFE_TRANSFER_ETH_CALL)),
    "ETH transfer": (RE_TRANSFER_CALL, (RE_TRANSFER_CALL, RE_SAFE_TRANSFER_CALL, RE_ETH_VALUE_CALL, RE_SEND_VALUE_CALL,
                                        RE_SAFE_TRANSFER_ETH_CALL)),
    "ETH send": (RE_SEND_CALL, (RE_SEND_CALL, RE_TRANSFER_CALL, RE_ETH_VALUE_CALL, RE_SEND_VALUE_CALL)),
}


# Tên biến state bị ghi trong hàm (=, +=, -=, ...), đã loại tham số / biến cục bộ.
def _state_writes(code: str) -> set[str]:
    local = local_variable_names(code)
    return {match.group(1) for match in RE_STATE_WRITE.finditer(code) if match.group(1) not in local}


# Event được phát: "emit X(...)" hoặc câu lệnh riêng "X(...);" kiểu Solidity < 0.4.21 (ép kiểu IERC20(x).f() thì không).
def _emitted_events(code: str) -> set[str]:
    events = set(RE_EMIT_EVENT.findall(code))
    for match in RE_LEGACY_EVENT_STATEMENT.finditer(code):
        close = matching_close(code, match.end() - 1, "(", ")")
        if close is not None and code[close + 1:].lstrip().startswith(";"):
            events.add(match.group(1))
    return events


# [GUARD HÀNH VI] Bản vá làm MẤT chuyển ETH, return, event hoặc lệnh ghi state của hàm gốc -> ValueError.
def validate_observable_behavior(original: str, replacement: str) -> None:
    original, replacement = mask_noncode(original, mask_strings=False), mask_noncode(replacement, mask_strings=False)
    for label, (detect, preserved_by) in _TRANSFER_EQUIVALENTS.items():
        if detect.search(original) and not any(pattern.search(replacement) for pattern in preserved_by):
            raise ValueError(f"Generated patch removed observable behavior: {label}")
    if RE_RETURN_KEYWORD.search(original) and not RE_RETURN_KEYWORD.search(replacement):
        raise ValueError("Generated patch removed observable behavior: return statement")
    if missing := _emitted_events(original) - _emitted_events(replacement):
        raise ValueError(f"Generated patch removed observable event emission: {', '.join(sorted(missing))}")
    if missing := _state_writes(original) - _state_writes(replacement):
        raise ValueError(f"Generated patch removed state variable mutation: {', '.join(sorted(missing))}")
