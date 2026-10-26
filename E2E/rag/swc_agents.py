"""SWC signal agents shared by the static runner and the local RAG retriever.

One agent per SWC registry entry: regex signal patterns, per-SWC severity and the
custom context checks (guards, checked low-level calls, pragma) that gate a hit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from E2E.config import RUNTIME_DATA_DIR

DEFAULT_SWC_REGISTRY = str(RUNTIME_DATA_DIR / "swc_registry.json")


@dataclass(frozen=True)
class AgentConfig:
    swc_id: str
    title: str
    description: str
    remediation: str
    query: str
    severity: str
    threshold: float
    min_signal_hits: int
    signal_patterns: Sequence[str]

    @property
    def name(self) -> str:
        return self.swc_id.lower().replace("-", "_")

    @property
    def display_name(self) -> str:
        return f"{self.swc_id} Agent - {self.title}"



SEVERITY_BY_SWC = {
    "SWC-104": "Critical",
    "SWC-105": "Critical",
    "SWC-106": "Critical",
    "SWC-107": "Critical",
    "SWC-112": "Critical",
    "SWC-115": "Critical",
    "SWC-101": "High",
    "SWC-113": "High",
    "SWC-114": "High",
    "SWC-121": "High",
    "SWC-122": "High",
    "SWC-124": "High",
    "SWC-126": "High",
    "SWC-128": "High",
}

MIN_SIGNAL_HITS_BY_SWC = {
    "SWC-100": 1,
    "SWC-101": 1,
    "SWC-102": 1,
    "SWC-103": 1,
    "SWC-104": 1,
    "SWC-105": 1,
    "SWC-106": 1,
    "SWC-107": 1,
    "SWC-112": 1,
    "SWC-113": 1,
    "SWC-115": 1,
    "SWC-118": 1,
    "SWC-120": 2,
    "SWC-121": 2,
    "SWC-122": 2,
    "SWC-123": 1,
    "SWC-129": 1,
    "SWC-136": 1,
}

SIGNAL_PATTERNS_BY_SWC: Dict[str, Sequence[str]] = {
    "SWC-100": (r"function\s+\w+\s*\([^)]*\)\s*(?:returns\s*\([^)]*\)\s*)?\{",),
    "SWC-101": (
        r"pragma\s+solidity\s+[^;]*0\.[0-7]\.",
        r"\bSafeMath\b",
        r"\bunchecked\s*\{",
        r"\buint(?:8|16|32|64|128)\s*\(",
        r"\bint(?:8|16|32|64|128)\s*\(",
        r"\w+\s*/\s*\w+\s*\*\s*\w+",
        r"\buint\w*\s+\w+\s*=\s*\w+\s*(?:\+|-|\*)\s*\w+",
        r"\b1e18\b|\b10\s*\*\*\s*18\b|\bdecimals\s*\(",
    ),
    "SWC-102": (r"pragma\s+solidity\s+[^;]*0\.[0-7]\.",),
    "SWC-103": (r"pragma\s+solidity\s+\^", r"pragma\s+solidity\s+>="),
    "SWC-104": (
        r"\.call\s*[\(\{]",
        r"\.delegatecall\s*\(",
        r"\.staticcall\s*\(",
        r"\.send\s*\(",
    ),
    "SWC-105": (
        r"function\s+(?:withdraw|drain|rescue|sweep|claim|payout)\b",
        r"\.call\s*\{[^}]*value\s*:",
        r"\.transfer\s*\(",
        r"\.send\s*\(",
    ),
    "SWC-106": (r"\bselfdestruct\s*\(", r"\bsuicide\s*\("),
    "SWC-107": (
        r"\.call\s*\{[^}]*value\s*:",
        r"\.send\s*\(",
        r"\bbalances?\s*\[.*?\]\s*(?:-=|=|\+=)",
    ),
    "SWC-108": (
        r"^\s*(?:uint|int|bool|address|bytes|string|mapping)\s+(?!public|private|internal)\w+\s*(?:=|;)",
    ),
    "SWC-109": (r"\b\w+\s+storage\s+\w+\s*;",),
    "SWC-110": (r"\bassert\s*\(",),
    "SWC-111": (
        r"\bsuicide\s*\(",
        r"\bsha3\s*\(",
        r"\bthrow\b",
        r"\bcallcode\s*\(",
        r"\bblock\.blockhash\s*\(",
        r"\bvar\s+\w+",
    ),
    "SWC-112": (r"\.delegatecall\s*\(", r"\.callcode\s*\("),
    "SWC-113": (
        r"for\s*\([^)]*;\s*\w+\s*<\s*\w+\.length\s*;",
        r"for[^{]*\{[^}]*(?:\.transfer|\.send|\.call)",
        r"\.transfer\s*\(",
        r"\.send\s*\(",
    ),
    "SWC-114": (
        r"\bapprove\s*\(",
        r"\ballowance\b",
        r"\bamountOutMin\s*=\s*0\b|\bminOut\s*=\s*0\b",
        r"\btx\.gasprice\b",
        r"\bblock\.number\b",
        r"function\s+(?:bid|auction|buy|purchase)\b",
    ),
    "SWC-115": (r"\btx\.origin\b",),
    "SWC-116": (r"\bblock\.timestamp\b|\bnow\b", r"\bblock\.number\b"),
    "SWC-117": (r"\becrecover\s*\(", r"\bsignature\b", r"\bv\s*,\s*r\s*,\s*s\b"),
    "SWC-118": (r"pragma\s+solidity\s+[^;]*0\.[0-4]\.", r"function\s+[A-Z]\w+\s*\("),
    "SWC-119": (),
    "SWC-120": (
        r"\bblockhash\s*\(",
        r"\bblock\.difficulty\b|\bblock\.prevrandao\b",
        r"\bkeccak256\s*\([^)]*block\.",
        r"\brandom\b|rand|entropy",
    ),
    "SWC-121": (r"\becrecover\s*\(", r"\bnonce\b|\bnonces\b", r"\bchainid\b|\bchainId\b|DOMAIN_SEPARATOR"),
    "SWC-122": (r"\becrecover\s*\(", r"\bsignature\b", r"\bsigner\b", r"\baddress\s*\(\s*0\s*\)"),
    "SWC-123": (),
    "SWC-124": (r"\bassembly\s*\{[^}]*sstore\b", r"\.delegatecall\s*\(", r"\b\w+\s+storage\s+\w+\s*;"),
    "SWC-125": (r"contract\s+\w+\s+is\s+\w+\s*,\s*\w+",),
    "SWC-126": (r"\.call\s*\{[^}]*gas\s*:", r"\bgasleft\s*\(", r"\brelayer\b"),
    "SWC-127": (r"\bfunction\s*\([^)]*\)\s*(?:internal|external)\s+\w+",),
    "SWC-128": (r"for\s*\([^)]*;\s*\w+\s*<\s*\w+\.length\s*;", r"\bwhile\s*\(", r"\bpush\s*\("),
    "SWC-129": (r"if\s*\([^)]*[^=!<>]=[^=][^)]*\)", r"require\s*\([^)]*[^=!<>]=[^=][^)]*\)"),
    "SWC-130": ("\u202e",),
    "SWC-131": (),
    "SWC-132": (r"address\s*\(\s*this\s*\)\.balance", r"\bselfdestruct\s*\("),
    "SWC-133": (r"abi\.encodePacked\s*\(",),
    "SWC-134": (r"\.call\s*\{[^}]*gas\s*:\s*\d+",),
    "SWC-135": (r"\b\w+\s*==\s*\w+\s*;", r"\b\w+\s*!=\s*\w+\s*;"),
    "SWC-136": (r"\b(?:password|secret|privateKey|apiKey|seed)\b", r"\bprivate\b"),
}


def normalize_swc_id(value: object) -> str:
    match = re.search(r"SWC-\d{3}", str(value or "").upper())
    return match.group(0) if match else str(value or "").strip().upper()


def load_swc_registry(path: str = DEFAULT_SWC_REGISTRY) -> List[Dict]:
    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"SWC registry must be a JSON list: {registry_path}")
    return [item for item in data if isinstance(item, dict) and normalize_swc_id(item.get("id"))]


def build_agent_query(swc_id: str, title: str, description: str, remediation: str) -> str:
    return (
        f"Audit this Solidity code for {swc_id} - {title}. "
        f"Weakness description: {description[:700]} "
        f"Expected remediation or best practice: {remediation[:450]} "
        "Focus on whether the target code actually contains this weakness."
    )


def build_agents_from_registry(path: str = DEFAULT_SWC_REGISTRY) -> List[AgentConfig]:
    agents = []
    for item in load_swc_registry(path):
        swc_id = normalize_swc_id(item.get("id"))
        title = str(item.get("title") or swc_id).strip()
        description = str(item.get("description") or "").strip()
        remediation = str(item.get("remediation") or "").strip()
        patterns = SIGNAL_PATTERNS_BY_SWC.get(swc_id, ())
        agents.append(
            AgentConfig(
                swc_id=swc_id,
                title=title,
                description=description,
                remediation=remediation,
                query=build_agent_query(swc_id, title, description, remediation),
                severity=SEVERITY_BY_SWC.get(swc_id, "Medium"),
                threshold=0.58 if patterns else 0.72,
                min_signal_hits=MIN_SIGNAL_HITS_BY_SWC.get(swc_id, 1 if len(patterns) <= 1 else 2),
                signal_patterns=patterns,
            )
        )
    return agents


def extract_signal_code(code: str) -> str:
    fenced_blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", code, re.IGNORECASE | re.DOTALL)
    code_lines = []
    solidity_tokens = (
        "pragma ",
        "contract ",
        "interface ",
        "library ",
        "function ",
        "modifier ",
        "constructor",
        "mapping",
        "require",
        "revert",
        "assert",
        "if ",
        "for ",
        "while ",
        ".call",
        ".delegatecall",
        ".staticcall",
        ".send",
        ".transfer",
        "selfdestruct",
        "suicide",
        "assembly",
        "tx.origin",
        "block.",
        "ecrecover",
        "abi.encodePacked",
        "approve",
        "unchecked",
        "SafeMath",
    )
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("###", "#", "//")):
            continue
        if any(token in stripped for token in solidity_tokens) and any(char in stripped for char in "{}();=<>[]"):
            code_lines.append(stripped)
    return "\n".join([*fenced_blocks, *code_lines]).strip() or code


def pragma_version(code: str) -> tuple[int, int, int] | None:
    match = re.search(r"pragma\s+solidity\s+([^;]+);", code, re.IGNORECASE)
    if not match:
        return None
    version = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", match.group(1))
    if not version:
        return None
    return (
        int(version.group(1)),
        int(version.group(2)),
        int(version.group(3) or 0),
    )


def pragma_before(code: str, major: int, minor: int, patch: int = 0) -> bool:
    version = pragma_version(code)
    return bool(version and version < (major, minor, patch))


def has_arithmetic_operation(code: str) -> bool:
    return bool(
        re.search(
            r"(?:\+\+|--|[+\-*]=|=\s*[^;]*(?:\+|-|\*)[^;]*;|\b(?:add|sub|mul)\s*\()",
            code,
            re.IGNORECASE | re.DOTALL,
        )
    )


def low_level_call_statements(code: str) -> List[str]:
    return [
        statement.strip()
        for statement in re.split(r";", code)
        if re.search(r"\.(?:call|delegatecall|staticcall|send)\s*[\(\{]", statement, re.IGNORECASE | re.DOTALL)
    ]


def statement_result_names(statement: str) -> List[str]:
    names = re.findall(r"\(\s*bool\s+(\w+)\s*,?", statement, re.IGNORECASE)
    names.extend(
        re.findall(
            r"\bbool\s+(\w+)\s*=\s*[^;]*\.(?:call|delegatecall|staticcall|send)\s*[\(\{]",
            statement,
            re.IGNORECASE | re.DOTALL,
        )
    )
    return names


def is_result_checked(full_code: str, statement: str) -> bool:
    if re.search(r"\b(?:require|assert)\s*\([^;]*\.(?:send|call|delegatecall|staticcall)\s*[\(\{]", statement, re.IGNORECASE | re.DOTALL):
        return True
    for name in statement_result_names(statement):
        if re.search(
            rf"(?:require\s*\(\s*{re.escape(name)}\b|assert\s*\(\s*{re.escape(name)}\b|if\s*\(\s*!\s*{re.escape(name)}\b|if\s*\(\s*{re.escape(name)}\s*==\s*false)",
            full_code,
            re.IGNORECASE,
        ):
            return True
    return False


def has_state_update_after_external_value_call(code: str) -> bool:
    if re.search(r"\bnonReentrant\b|ReentrancyGuard", code, re.IGNORECASE):
        return False
    call_match = re.search(r"\.(?:call\s*\{[^}]*value\s*:|send\s*\()", code, re.IGNORECASE | re.DOTALL)
    if not call_match:
        return False
    after_call = code[call_match.end() :]
    return bool(
        re.search(
            r"(?:balances?\s*\[[^\]]+\]\s*(?:-=|=|\+=)|\bdelete\s+\w+\s*\[|\b\w+\s*(?:-=|\+=|=)\s*\w+)",
            after_call,
            re.IGNORECASE | re.DOTALL,
        )
    )


def loop_bodies(code: str) -> List[str]:
    return re.findall(r"\b(?:for|while)\s*\([^)]*\)\s*\{(.*?)\}", code, re.IGNORECASE | re.DOTALL)


def condition_blocks(keyword: str, code: str) -> List[str]:
    blocks = []
    pattern = re.compile(rf"\b{keyword}\s*\(", re.IGNORECASE)
    for match in pattern.finditer(code):
        start = match.end() - 1
        depth = 0
        for index in range(start, len(code)):
            char = code[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    blocks.append(code[start + 1 : index])
                    break
    return blocks


def find_assignment_in_condition(code: str) -> bool:
    for condition in [*condition_blocks("if", code), *condition_blocks("require", code)]:
        compact = re.sub(r"\s+", "", condition)
        if any(operator in compact for operator in ("==", "!=", ">=", "<=")):
            continue
        if re.search(r"(?<![=!<>])=(?![=>])", condition):
            return True
    return False


def find_custom_signal_hits(swc_id: str, code: str) -> List[str] | None:
    if swc_id == "SWC-100":
        if pragma_before(code, 0, 5) and re.search(
            r"\bfunction\s+\w+\s*\([^)]*\)\s*(?![^{};]*(?:public|private|internal|external))",
            code,
            re.IGNORECASE | re.DOTALL,
        ):
            return ["function without explicit visibility on pre-0.5 Solidity"]
        return []

    if swc_id == "SWC-101":
        hits = []
        if re.search(r"\bunchecked\s*\{", code):
            hits.append("unchecked arithmetic block")
        if pragma_before(code, 0, 8) and has_arithmetic_operation(code) and not re.search(r"\bSafeMath\b", code):
            hits.append("arithmetic on Solidity version without built-in overflow checks")
        if re.search(r"\b(?:u?int)(?:8|16|32|64|128)\s*\(", code):
            hits.append("narrow integer cast can truncate values")
        return hits

    if swc_id == "SWC-102":
        return ["outdated Solidity compiler version"] if pragma_before(code, 0, 8) else []

    if swc_id == "SWC-103":
        return ["floating pragma version"] if re.search(r"pragma\s+solidity\s+[^;]*(?:\^|>=|>|\*)", code, re.IGNORECASE) else []

    if swc_id == "SWC-104":
        unchecked = [
            statement
            for statement in low_level_call_statements(code)
            if not is_result_checked(code, statement)
        ]
        return ["unchecked low-level call return value"] if unchecked else []

    if swc_id == "SWC-105":
        if has_access_control_guard(code):
            return []
        has_ether_transfer = re.search(r"\.(?:call\s*\{[^}]*value\s*:|send\s*\(|transfer\s*\()", code, re.IGNORECASE | re.DOTALL)
        has_admin_like_withdraw = re.search(r"\bfunction\s+(?:drain|sweep|rescue|withdrawAll|emergencyWithdraw)\b", code, re.IGNORECASE)
        drains_contract_balance = re.search(r"address\s*\(\s*this\s*\)\.balance|\bthis\.balance\b", code, re.IGNORECASE)
        has_user_balance_guard = re.search(r"balances?\s*\[\s*msg\.sender\s*\].{0,80}(?:>=|>)", code, re.IGNORECASE | re.DOTALL)
        if has_ether_transfer and (has_admin_like_withdraw or drains_contract_balance) and not has_user_balance_guard:
            return ["ether withdrawal path without access control"]
        return []

    if swc_id == "SWC-106":
        if re.search(r"\b(?:selfdestruct|suicide)\s*\(", code, re.IGNORECASE) and not has_access_control_guard(code):
            return ["selfdestruct without access control"]
        return []

    if swc_id == "SWC-107":
        return ["external value call before state update"] if has_state_update_after_external_value_call(code) else []

    if swc_id == "SWC-112":
        if re.search(r"\.\s*(?:delegatecall|callcode)\s*\(", code, re.IGNORECASE) and not has_access_control_guard(code):
            return ["delegatecall/callcode target is externally controllable or unguarded"]
        return []

    if swc_id == "SWC-113":
        for body in loop_bodies(code):
            if re.search(r"\.(?:transfer|send|call)\s*[\(\{]", body, re.IGNORECASE | re.DOTALL):
                return ["external call inside loop can block progress"]
        return []

    if swc_id == "SWC-118":
        if pragma_before(code, 0, 4, 22) and re.search(r"\bfunction\s+[A-Z]\w+\s*\(", code):
            return ["old-style constructor-like function on pre-0.4.22 Solidity"]
        return []

    if swc_id == "SWC-123":
        return []

    if swc_id == "SWC-129":
        return ["assignment used as condition"] if find_assignment_in_condition(code) else []

    if swc_id == "SWC-136":
        if re.search(r"\b(?:string|bytes\d*|uint\d*|address)\s+(?:private\s+)?\w*(?:password|secret|privateKey|apiKey|seed)\w*", code, re.IGNORECASE):
            return ["sensitive value stored in contract state"]
        return []

    return None


def find_signal_hits(code: str, patterns: Sequence[str], swc_id: str | None = None) -> List[str]:
    signal_code = extract_signal_code(code)
    custom_hits = find_custom_signal_hits(swc_id, signal_code) if swc_id else None
    if custom_hits is not None:
        return custom_hits
    return [
        pattern
        for pattern in patterns
        if re.search(pattern, signal_code, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    ]


def has_access_control_guard(code: str) -> bool:
    return bool(
        re.search(
            r"\bonlyOwner\b|onlyRole|onlyAdmin|AccessControl|Ownable|require\s*\(\s*msg\.sender\s*==",
            code,
            re.IGNORECASE,
        )
    )
