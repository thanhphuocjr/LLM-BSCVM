"""Phase 1A - cổng tín hiệu theo ngữ cảnh: luật tĩnh dễ báo nhầm chỉ "confirmed" khi mã có tín hiệu đúng SWC."""

from E2E.common.regex_patterns import (
    RE_ACCESS_CONTROL_GUARD, RE_ADMIN_WITHDRAW_FUNCTION, RE_ARITHMETIC_OPERATION, RE_BOOL_ASSIGNED_CALL_RESULT,
    RE_BOOL_TUPLE_RESULT, RE_CALL_INSIDE_REQUIRE, RE_CONTRACT_BALANCE, RE_ETHER_TRANSFER, RE_EXTERNAL_CALL,
    RE_FUNCTION_WITHOUT_VISIBILITY, RE_LOOP_BODY, RE_LOW_LEVEL_CALL, RE_NARROW_INT_CAST, RE_PRAGMA_SPEC_ANY_CASE,
    RE_REENTRANCY_GUARD, RE_SAFEMATH, RE_SELFDESTRUCT_CALL, RE_STATE_UPDATE, RE_UNCHECKED_BLOCK, RE_USER_BALANCE_GUARD,
    RE_VALUE_CALL_OR_SEND, RE_VERSION_NUMBER, re_result_check,
)


# Phiên bản (major, minor, patch) trong dòng pragma solidity (không có -> None).
def pragma_version(code: str) -> tuple[int, int, int] | None:
    pragma = RE_PRAGMA_SPEC_ANY_CASE.search(code)
    version = RE_VERSION_NUMBER.search(pragma.group(1)) if pragma else None
    return (int(version.group(1)), int(version.group(2)), int(version.group(3) or 0)) if version else None


# Pragma có cũ hơn phiên bản cho trước không (vd < 0.8.0 -> chưa có kiểm tra tràn số tự động).
def pragma_before(code: str, major: int, minor: int, patch: int = 0) -> bool:
    version = pragma_version(code)
    return bool(version and version < (major, minor, patch))


# Kết quả lời gọi cấp thấp trong câu lệnh có được kiểm tra không (require(...) / assert / if (!ok)).
def is_result_checked(full_code: str, statement: str) -> bool:
    names = RE_BOOL_TUPLE_RESULT.findall(statement) + RE_BOOL_ASSIGNED_CALL_RESULT.findall(statement)
    return bool(RE_CALL_INSIDE_REQUIRE.search(statement)) \
        or any(re_result_check(name).search(full_code) for name in names)


# Có cập nhật state SAU lời gọi gửi ETH (.call{value:} / .send) không -> dấu hiệu reentrancy; có nonReentrant -> False.
def has_state_update_after_external_value_call(code: str) -> bool:
    call = None if RE_REENTRANCY_GUARD.search(code) else RE_VALUE_CALL_OR_SEND.search(code)
    return bool(call and RE_STATE_UPDATE.search(code[call.end():]))


# Mã có cơ chế phân quyền không (onlyOwner, onlyRole, Ownable, require(msg.sender == ...)).
def has_access_control_guard(code: str) -> bool:
    return bool(RE_ACCESS_CONTROL_GUARD.search(code))


# [CỔNG TÍN HIỆU] Tín hiệu "có ngữ cảnh" của một SWC trong mã (vd SWC-104 chỉ khi kết quả .call KHÔNG được kiểm tra).
def signal_gate_hits(swc_id: str, code: str) -> list[str]:
    if swc_id == "SWC-100":
        return ["function without explicit visibility on pre-0.5 Solidity"] \
            if pragma_before(code, 0, 5) and RE_FUNCTION_WITHOUT_VISIBILITY.search(code) else []
    if swc_id == "SWC-101":
        checks = (
            (RE_UNCHECKED_BLOCK.search(code), "unchecked arithmetic block"),
            (pragma_before(code, 0, 8) and RE_ARITHMETIC_OPERATION.search(code) and not RE_SAFEMATH.search(code),
             "arithmetic on Solidity version without built-in overflow checks"),
            (RE_NARROW_INT_CAST.search(code), "narrow integer cast can truncate values"),
        )
        return [label for hit, label in checks if hit]
    if swc_id == "SWC-104":
        statements = [s.strip() for s in code.split(";") if RE_LOW_LEVEL_CALL.search(s)]
        unchecked = any(not is_result_checked(code, statement) for statement in statements)
        return ["unchecked low-level call return value"] if unchecked else []
    if swc_id == "SWC-105":
        drains = RE_ADMIN_WITHDRAW_FUNCTION.search(code) or RE_CONTRACT_BALANCE.search(code)
        return ["ether withdrawal path without access control"] if not has_access_control_guard(code) \
            and RE_ETHER_TRANSFER.search(code) and drains and not RE_USER_BALANCE_GUARD.search(code) else []
    if swc_id == "SWC-106":
        return ["selfdestruct without access control"] \
            if RE_SELFDESTRUCT_CALL.search(code) and not has_access_control_guard(code) else []
    if swc_id == "SWC-107":
        return ["external value call before state update"] if has_state_update_after_external_value_call(code) else []
    if swc_id == "SWC-113":
        return ["external call inside loop can block progress"] \
            if any(RE_EXTERNAL_CALL.search(body) for body in RE_LOOP_BODY.findall(code)) else []
    return []
