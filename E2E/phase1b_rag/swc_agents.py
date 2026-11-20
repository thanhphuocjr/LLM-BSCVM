"""Phase 1B - 37 SWC agent: regex tín hiệu + luật ngữ cảnh "gác cổng" từng SWC (dùng chung cho RAG và static)."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from E2E.common.regex_patterns import (
    RE_ACCESS_CONTROL_GUARD, RE_ADMIN_WITHDRAW_FUNCTION, RE_ANY_CODE_FENCE, RE_ARITHMETIC_OPERATION,
    RE_BOOL_ASSIGNED_CALL_RESULT, RE_BOOL_TUPLE_RESULT, RE_CALL_INSIDE_REQUIRE, RE_CAPITALIZED_FUNCTION,
    RE_CONTRACT_BALANCE, RE_DELEGATECALL_OR_CALLCODE, RE_ETHER_TRANSFER, RE_EXTERNAL_CALL, RE_FLOATING_PRAGMA,
    RE_FUNCTION_WITHOUT_VISIBILITY, RE_LOOP_BODY, RE_LOW_LEVEL_CALL, RE_NARROW_INT_CAST, RE_PRAGMA_SPEC_ANY_CASE,
    RE_REENTRANCY_GUARD, RE_SAFEMATH, RE_SELFDESTRUCT_CALL, RE_SENSITIVE_STATE_VALUE, RE_SINGLE_EQUALS, RE_STATE_UPDATE,
    RE_SWC_ID, RE_UNCHECKED_BLOCK, RE_USER_BALANCE_GUARD, RE_VALUE_CALL_OR_SEND, RE_VERSION_NUMBER, RE_WHITESPACE,
    re_condition_opening, re_result_check,
)
from E2E.config import KNOWLEDGE_BASE_DIR
from E2E.phase0_parsing.code_units import matching_close

DEFAULT_SWC_REGISTRY = KNOWLEDGE_BASE_DIR / "swc_registry.json"
# Mức nghiêm trọng theo SWC (SWC không có trong bảng = Medium).
SEVERITY_BY_SWC = {
    "SWC-104": "Critical", "SWC-105": "Critical", "SWC-106": "Critical", "SWC-107": "Critical", "SWC-112": "Critical",
    "SWC-115": "Critical", "SWC-101": "High", "SWC-113": "High", "SWC-114": "High", "SWC-121": "High",
    "SWC-122": "High", "SWC-124": "High", "SWC-126": "High", "SWC-128": "High",
}
# Số tín hiệu tối thiểu để agent kích hoạt (SWC ngoài bảng: 1 nếu có <= 1 pattern, ngược lại 2).
MIN_SIGNAL_HITS_BY_SWC = {**{f"SWC-{n}": 1 for n in (100, 101, 102, 103, 104, 105, 106, 107, 112, 113, 115, 118, 123,
                                                       129, 136)}, "SWC-120": 2, "SWC-121": 2, "SWC-122": 2}
# Regex tín hiệu chung của từng SWC (ghi nguyên chuỗi vào swc_registry.json khi build KB).
SIGNAL_PATTERNS_BY_SWC: dict[str, Sequence[str]] = {
    "SWC-100": (r"function\s+\w+\s*\([^)]*\)\s*(?:returns\s*\([^)]*\)\s*)?\{",),
    "SWC-101": (r"pragma\s+solidity\s+[^;]*0\.[0-7]\.", r"\bSafeMath\b", r"\bunchecked\s*\{",
                r"\buint(?:8|16|32|64|128)\s*\(", r"\bint(?:8|16|32|64|128)\s*\(", r"\w+\s*/\s*\w+\s*\*\s*\w+",
                r"\buint\w*\s+\w+\s*=\s*\w+\s*(?:\+|-|\*)\s*\w+", r"\b1e18\b|\b10\s*\*\*\s*18\b|\bdecimals\s*\("),
    "SWC-102": (r"pragma\s+solidity\s+[^;]*0\.[0-7]\.",),
    "SWC-103": (r"pragma\s+solidity\s+\^", r"pragma\s+solidity\s+>="),
    "SWC-104": (r"\.call\s*[\(\{]", r"\.delegatecall\s*\(", r"\.staticcall\s*\(", r"\.send\s*\("),
    "SWC-105": (r"function\s+(?:withdraw|drain|rescue|sweep|claim|payout)\b", r"\.call\s*\{[^}]*value\s*:",
                r"\.transfer\s*\(", r"\.send\s*\("),
    "SWC-106": (r"\bselfdestruct\s*\(", r"\bsuicide\s*\("),
    "SWC-107": (r"\.call\s*\{[^}]*value\s*:", r"\.send\s*\(", r"\bbalances?\s*\[.*?\]\s*(?:-=|=|\+=)"),
    "SWC-108": (r"^\s*(?:uint|int|bool|address|bytes|string|mapping)\s+(?!public|private|internal)\w+\s*(?:=|;)",),
    "SWC-109": (r"\b\w+\s+storage\s+\w+\s*;",),
    "SWC-110": (r"\bassert\s*\(",),
    "SWC-111": (r"\bsuicide\s*\(", r"\bsha3\s*\(", r"\bthrow\b", r"\bcallcode\s*\(", r"\bblock\.blockhash\s*\(",
                r"\bvar\s+\w+"),
    "SWC-112": (r"\.delegatecall\s*\(", r"\.callcode\s*\("),
    "SWC-113": (r"for\s*\([^)]*;\s*\w+\s*<\s*\w+\.length\s*;", r"for[^{]*\{[^}]*(?:\.transfer|\.send|\.call)",
                r"\.transfer\s*\(", r"\.send\s*\("),
    "SWC-114": (r"\bapprove\s*\(", r"\ballowance\b", r"\bamountOutMin\s*=\s*0\b|\bminOut\s*=\s*0\b",
                r"\btx\.gasprice\b", r"\bblock\.number\b", r"function\s+(?:bid|auction|buy|purchase)\b"),
    "SWC-115": (r"\btx\.origin\b",),
    "SWC-116": (r"\bblock\.timestamp\b|\bnow\b", r"\bblock\.number\b"),
    "SWC-117": (r"\becrecover\s*\(", r"\bsignature\b", r"\bv\s*,\s*r\s*,\s*s\b"),
    "SWC-118": (r"pragma\s+solidity\s+[^;]*0\.[0-4]\.", r"function\s+[A-Z]\w+\s*\("),
    "SWC-119": (),
    "SWC-120": (r"\bblockhash\s*\(", r"\bblock\.difficulty\b|\bblock\.prevrandao\b", r"\bkeccak256\s*\([^)]*block\.",
                r"\brandom\b|rand|entropy"),
    "SWC-121": (r"\becrecover\s*\(", r"\bnonce\b|\bnonces\b", r"\bchainid\b|\bchainId\b|DOMAIN_SEPARATOR"),
    "SWC-122": (r"\becrecover\s*\(", r"\bsignature\b", r"\bsigner\b", r"\baddress\s*\(\s*0\s*\)"),
    "SWC-123": (),
    "SWC-124": (r"\bassembly\s*\{[^}]*sstore\b", r"\.delegatecall\s*\(", r"\b\w+\s+storage\s+\w+\s*;"),
    "SWC-125": (r"contract\s+\w+\s+is\s+\w+\s*,\s*\w+",),
    "SWC-126": (r"\.call\s*\{[^}]*gas\s*:", r"\bgasleft\s*\(", r"\brelayer\b"),
    "SWC-127": (r"\bfunction\s*\([^)]*\)\s*(?:internal|external)\s+\w+",),
    "SWC-128": (r"for\s*\([^)]*;\s*\w+\s*<\s*\w+\.length\s*;", r"\bwhile\s*\(", r"\bpush\s*\("),
    "SWC-129": (r"if\s*\([^)]*[^=!<>]=[^=][^)]*\)", r"require\s*\([^)]*[^=!<>]=[^=][^)]*\)"),
    "SWC-130": ("‮",),
    "SWC-131": (),
    "SWC-132": (r"address\s*\(\s*this\s*\)\.balance", r"\bselfdestruct\s*\("),
    "SWC-133": (r"abi\.encodePacked\s*\(",),
    "SWC-134": (r"\.call\s*\{[^}]*gas\s*:\s*\d+",),
    "SWC-135": (r"\b\w+\s*==\s*\w+\s*;", r"\b\w+\s*!=\s*\w+\s*;"),
    "SWC-136": (r"\b(?:password|secret|privateKey|apiKey|seed)\b", r"\bprivate\b"),
}
# Từ khoá nhận diện một dòng là mã Solidity (extract_signal_code).
_SOLIDITY_TOKENS = (
    "pragma ", "contract ", "interface ", "library ", "function ", "modifier ", "constructor", "mapping", "require",
    "revert", "assert", "if ", "for ", "while ", ".call", ".delegatecall", ".staticcall", ".send", ".transfer",
    "selfdestruct", "suicide", "assembly", "tx.origin", "block.", "ecrecover", "abi.encodePacked", "approve",
    "unchecked", "SafeMath",
)


# Cấu hình MỘT SWC agent: mã SWC, tiêu đề, mô tả, cách sửa, mức độ, số tín hiệu tối thiểu, regex tín hiệu.
@dataclass(frozen=True)
class AgentConfig:
    swc_id: str
    title: str
    description: str
    remediation: str
    severity: str
    min_signal_hits: int
    signal_patterns: Sequence[str]


# Chuẩn hoá chuỗi bất kỳ về "SWC-xxx" ("swc-107: Reentrancy" -> "SWC-107").
def normalize_swc_id(value: object) -> str:
    match = RE_SWC_ID.search(str(value or "").upper())
    return match.group(0) if match else str(value or "").strip().upper()


# Đọc swc_registry.json (37 mục SWC đã làm giàu bằng dữ liệu), chỉ giữ mục hợp lệ.
def load_swc_registry(path: str | Path = DEFAULT_SWC_REGISTRY) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"SWC registry must be a JSON list: {path}")
    return [item for item in data if isinstance(item, dict) and normalize_swc_id(item.get("id"))]


# Dựng 37 AgentConfig từ registry + các bảng SIGNAL_PATTERNS_BY_SWC / SEVERITY_BY_SWC / MIN_SIGNAL_HITS_BY_SWC.
def build_agents(registry: list[dict]) -> list[AgentConfig]:
    agents = []
    for item in registry:
        swc_id = normalize_swc_id(item.get("id"))
        patterns = SIGNAL_PATTERNS_BY_SWC.get(swc_id, ())
        agents.append(AgentConfig(
            swc_id=swc_id, title=str(item.get("title") or swc_id).strip(),
            description=str(item.get("description") or "").strip(),
            remediation=str(item.get("remediation") or "").strip(),
            severity=SEVERITY_BY_SWC.get(swc_id, "Medium"),
            min_signal_hits=MIN_SIGNAL_HITS_BY_SWC.get(swc_id, 1 if len(patterns) <= 1 else 2),
            signal_patterns=patterns,
        ))
    return agents


# Chỉ giữ phần thực sự là mã Solidity (khối ```code``` + dòng có từ khoá Solidity) trước khi dò tín hiệu.
def extract_signal_code(code: str) -> str:
    lines = [line.strip() for line in code.splitlines()]
    code_lines = [line for line in lines if line and not line.startswith(("###", "#", "//"))
                  and any(token in line for token in _SOLIDITY_TOKENS) and any(char in line for char in "{}();=<>[]")]
    return "\n".join([*RE_ANY_CODE_FENCE.findall(code), *code_lines]).strip() or code


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


# Biểu thức điều kiện bên trong if(...) / require(...) (đếm ngoặc lồng nhau).
def condition_blocks(keyword: str, code: str) -> list[str]:
    blocks = []
    for match in re_condition_opening(keyword).finditer(code):
        close = matching_close(code, match.end() - 1, "(", ")")
        if close is not None:
            blocks.append(code[match.end():close])
    return blocks


# Có phép gán "=" (không phải ==, !=, >=, <=) trong điều kiện if/require không -> SWC-129.
def find_assignment_in_condition(code: str) -> bool:
    for condition in [*condition_blocks("if", code), *condition_blocks("require", code)]:
        if not any(op in RE_WHITESPACE.sub("", condition) for op in ("==", "!=", ">=", "<=")) \
                and RE_SINGLE_EQUALS.search(condition):
            return True
    return False


# Mã có cơ chế phân quyền không (onlyOwner, onlyRole, Ownable, require(msg.sender == ...)).
def has_access_control_guard(code: str) -> bool:
    return bool(RE_ACCESS_CONTROL_GUARD.search(code))


# Tín hiệu "có ngữ cảnh" cho SWC dễ báo nhầm (vd SWC-104 chỉ khi kết quả .call KHÔNG được kiểm tra); SWC khác -> None.
def find_custom_signal_hits(swc_id: str, code: str) -> list[str] | None:
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
    if swc_id == "SWC-102":
        return ["outdated Solidity compiler version"] if pragma_before(code, 0, 8) else []
    if swc_id == "SWC-103":
        return ["floating pragma version"] if RE_FLOATING_PRAGMA.search(code) else []
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
    if swc_id == "SWC-112":
        return ["delegatecall/callcode target is externally controllable or unguarded"] \
            if RE_DELEGATECALL_OR_CALLCODE.search(code) and not has_access_control_guard(code) else []
    if swc_id == "SWC-113":
        return ["external call inside loop can block progress"] \
            if any(RE_EXTERNAL_CALL.search(body) for body in RE_LOOP_BODY.findall(code)) else []
    if swc_id == "SWC-118":
        return ["old-style constructor-like function on pre-0.4.22 Solidity"] \
            if pragma_before(code, 0, 4, 22) and RE_CAPITALIZED_FUNCTION.search(code) else []
    if swc_id == "SWC-129":
        return ["assignment used as condition"] if find_assignment_in_condition(code) else []
    if swc_id == "SWC-136":
        return ["sensitive value stored in contract state"] if RE_SENSITIVE_STATE_VALUE.search(code) else []
    return None


# Tín hiệu của một SWC trong mã đã lọc: luật riêng (find_custom_signal_hits) nếu có, không thì regex chung của agent.
def find_signal_hits(signal_code: str, patterns: Sequence[str], swc_id: str | None = None) -> list[str]:
    custom = find_custom_signal_hits(swc_id, signal_code) if swc_id else None
    if custom is not None:
        return custom
    return [pattern for pattern in patterns
            if re.search(pattern, signal_code, re.IGNORECASE | re.DOTALL | re.MULTILINE)]
