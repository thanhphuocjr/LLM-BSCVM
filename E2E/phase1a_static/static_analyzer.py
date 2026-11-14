"""
Phase 1A - bộ phân tích tĩnh dựa trên luật (không cần compiler, không gọi Slither).

Phát hiện các lỗ hổng phổ biến trong Solidity bằng cách kết hợp:
  1. Bảng 27 luật regex có trọng số (VULNERABILITY_REGISTRY): mỗi mẫu dương cộng trọng số vào raw_score,
     mẫu âm (dấu hiệu phòng vệ) và guard modifier trừ điểm; luật "confirmed" khi raw_score - trừ > 0.25.
  2. 25 kiểm tra ngữ nghĩa (_run_semantic_checks), cài lại logic tương tự các detector của Slither.
Kết quả: {"findings": {vuln_type: VulnerabilityFinding}}; static_runner.py chuẩn hoá thành finding của pipeline.
Regex dùng trong các hàm của file này (RE_XXX, re_xxx(name)) nằm ở E2E/common/regex_patterns.py;
bảng luật VULNERABILITY_REGISTRY vẫn ở đây vì mỗi regex đi kèm trọng số và mô tả.

Mức rủi ro của 27 luật:
  Critical : Reentrancy, Access Control (gồm tx.origin, selfdestruct, rút ETH), Unchecked Calls, Storage Collision
  High     : Integer Overflow/Underflow, DoS (gas limit, revert), Flash Loan, Front-Running, Signature Replay,
             Price Oracle Manipulation
  Medium   : Timestamp / Block Info Dependence, Transaction Ordering, Gas Griefing, Unsafe Type Casting,
             Incorrect Unary, Tautological Comparison, Divide Before Multiply
  Low      : Default Visibility, Floating/Outdated Pragma, Naming Conventions, Redundant Code, Deprecated Usage
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from E2E.common.regex_patterns import (
    RE_ABI_ENCODE_CALL,
    RE_ABSTRACT_AT_END,
    RE_ARRAY_LENGTH_ASSIGNMENT,
    RE_ARRAY_SUFFIX,
    RE_ASSEMBLY_BLOCK,
    RE_BALANCE_ASSIGNMENT,
    RE_BALANCE_SOURCE,
    RE_BOOLEAN_CONSTANT_OPERAND,
    RE_CAST_PREFIX,
    RE_CONTRACT_WITH_BODY,
    RE_DECIMAL_NUMBER,
    RE_DELEGATECALL,
    RE_DELETE_STATEMENT,
    RE_ENCODE_PACKED_CALL,
    RE_ENUM_OPENING,
    RE_ERC20_TRANSFER_CALL,
    RE_FUNCTION_HEADER_PREFIX,
    RE_FUNCTION_WITH_BODY,
    RE_GUARD_MODIFIER_NAME,
    RE_IDENTIFIER,
    RE_IF_CONSTANT_CONDITION,
    RE_INHERITANCE_LIST,
    RE_INTEGER_LITERAL,
    RE_INTEGER_TYPE,
    RE_INTERNAL_OR_PRIVATE,
    RE_IS_KEYWORD,
    RE_LEADING_IDENTIFIER,
    RE_LOOP_OPENING,
    RE_MAPPING_KEYWORD,
    RE_MAPPING_VALUE_TYPE,
    RE_MODIFIER_NAME,
    RE_MODIFIER_WITH_BODY,
    RE_MSG_VALUE,
    RE_MSG_VALUE_ZERO_CHECK,
    RE_ORDER_COMPARISON,
    RE_OUTGOING_ETHER,
    RE_PARAMETER,
    RE_PAYABLE,
    RE_PINNED_PRAGMA_0_4_OR_0_5,
    RE_PRAGMA_0_4,
    RE_PRAGMA_0_4_0_TO_0_4_4,
    RE_PRAGMA_ABIENCODERV2,
    RE_PRAGMA_STATEMENT,
    RE_PUBLIC_MAPPING_VALUE_TYPE,
    RE_PUBLIC_OR_EXTERNAL,
    RE_RETURNS_CLAUSE,
    RE_SENDER_CHECKS,
    RE_SIGNED_INT_ARRAY_TYPE,
    RE_SPECIAL_FUNCTION_WITH_BODY,
    RE_STATE_DECLARATION,
    RE_STATE_MAPPING,
    RE_STATE_TYPE_AND_NAME,
    RE_STORAGE_ARRAY_LOCAL,
    RE_STRICT_EQUALITY,
    RE_STRUCT_OPENING,
    RE_SYMBOL_DECLARATION,
    RE_WHITESPACE,
    RE_WORD_TOKEN,
    RE_XOR_EXPRESSION,
    RE_YUL_RETURN_CALL,
    RE_YUL_SHIFT_WITH_CONSTANT_VALUE,
    re_assignment_from,
    re_assignment_to,
    re_call,
    re_call_target,
    re_enum_conversion,
    re_ether_destination,
    re_indexed_write,
    re_state_write,
    re_word,
    re_zero_address_require,
    re_zero_address_revert_if,
)
from E2E.phase0_parsing.code_units import mask_noncode

# ──────────────────────────────────────────────────────────────────────────────
# Kiểu dữ liệu nội bộ
# ──────────────────────────────────────────────────────────────────────────────

# Một lần khớp pattern: dòng, đoạn text khớp, ngữ cảnh xung quanh và trọng số đóng góp vào điểm.
@dataclass
class PatternMatch:
    """Kết quả một lần khớp pattern."""
    line: int
    matched_text: str
    context: str          # ~3 dòng xung quanh để hiển thị
    pattern_weight: float # trọng số đóng góp vào score


# Kết quả cho MỘT loại lỗ hổng (vd reentrancy, SWC-107): danh sách các lần khớp, mức rủi ro,
# đã được xác nhận (confirmed) hay chưa, ghi chú theo dòng.
@dataclass
class VulnerabilityFinding:
    """Kết quả phát hiện một loại lỗ hổng."""
    vuln_id: str          # vd: "SWC-107"
    vuln_type: str        # vd: "reentrancy"
    risk_level: str       # Critical / High / Medium / Low
    matches: List[PatternMatch] = field(default_factory=list)
    confirmed: bool = False   # True nếu context-check xác nhận thêm
    notes: List[str] = field(default_factory=list)

    # Điểm thô = tổng trọng số các lần khớp, chặn trần 1.0.
    @property
    def raw_score(self) -> float:
        if not self.matches:
            return 0.0
        return min(1.0, sum(m.pattern_weight for m in self.matches))


# Ranh giới một hàm (tên, tham số, header, vị trí bắt đầu/thân/kết thúc) - dùng cho các kiểm tra ngữ nghĩa.
@dataclass(frozen=True)
class SolidityFunction:
    """Lightweight function boundary used by semantic checks."""

    name: str
    params: str
    header: str
    start: int
    body_start: int
    end: int


# Ranh giới một contract (tên, header kế thừa, vị trí thân) - dùng cho kiểm tra cấp contract.
@dataclass(frozen=True)
class SolidityContract:
    """Lightweight contract boundary used by contract-level checks."""

    name: str
    header: str
    start: int
    body_start: int
    end: int


# Ranh giới một modifier kèm thân - dùng để nhận diện modifier bảo vệ quyền truy cập.
@dataclass(frozen=True)
class SolidityModifier:
    """Lightweight modifier boundary used by access-control checks."""

    name: str
    params: str
    header: str
    body: str
    start: int
    body_start: int
    end: int



# ──────────────────────────────────────────────────────────────────────────────
# Bảng pattern toàn diện
# Cấu trúc mỗi entry:
#   vuln_id      : SWC registry ID (https://swcregistry.io/)
#   vuln_type    : tên ngắn dùng trong code
#   risk_level   : mức rủi ro
#   patterns     : list[Tuple[regex_str, weight, description]]
#                  weight = đóng góp của pattern này vào raw_score
#   negative_patterns : dấu hiệu phòng vệ, trừ điểm (_apply_negative_patterns)
# ──────────────────────────────────────────────────────────────────────────────

VULNERABILITY_REGISTRY: List[Dict] = [

    # ──────────────────────────────── CRITICAL ────────────────────────────────

    {
        "vuln_id": "SWC-107",
        "vuln_type": "reentrancy",
        "risk_level": "Critical",
        "patterns": [
            # 1. Low-level call với ETH value trước khi cập nhật state
            (r'\.call\s*\{[^}]*value\s*:', 0.50,
             "Low-level call gửi ETH - nguy cơ reentrancy kinh điển"),
            (r'\.call\s*\.\s*value\s*\(', 0.50,
             "Low-level call.value() gửi ETH - nguy cơ reentrancy kinh điển (Solidity <0.7)"),
            # 2. External call không rõ giá trị
            (r'\.call\s*\(', 0.25,
             "External call - kiểm tra xem state có được cập nhật trước không"),
            # 3. Cross-function reentrancy: gọi hàm external trong modifier/callback
            (r'(function\s+\w+[^{]*\bexternal\b[^{]*\{[^}]*\.call)', 0.30,
             "External function thực hiện low-level call"),
            # 4. Delegate call - cross-contract reentrancy
            (r'\.delegatecall\s*\(', 0.40,
             "delegatecall - nguy cơ reentrancy và storage collision"),
            # 5. Withdraw pattern thiếu kiểm soát
            (r'function\s+withdraw[^{]*\{(?:(?!require|revert|nonReentrant|\}).)*\.call', 0.50,
             "Hàm withdraw gọi external mà không có bảo vệ reentrancy"),
        ],
        "negative_patterns": [
            (r'nonReentrant', 0.40,      "Có modifier nonReentrant"),
            (r'ReentrancyGuard', 0.35,   "Kế thừa ReentrancyGuard"),
            (r'_status\s*==\s*_NOT_ENTERED', 0.25, "Kiểm tra reentrancy thủ công"),
        ],
    },

    {
        "vuln_id": "SWC-105",
        "vuln_type": "unprotected_ether_withdrawal",
        "risk_level": "Critical",
        "patterns": [
            (r'function\s+\w+[^{]*\bpublic\b[^{]*\{(?:(?!require|onlyOwner|modifier|\}).)*selfdestruct',
             0.60, "selfdestruct không được bảo vệ"),
            (r'\bselfdestruct\s*\(', 0.50,
             "Gọi selfdestruct - cần kiểm tra quyền truy cập"),
            (r'\bsuicide\s*\(', 0.50,
             "Gọi suicide (alias của selfdestruct) - deprecated"),
            # Hàm rút tiền công khai không có modifier
            (r'function\s+(?:withdraw|drain|rescue)[^{]*\bpublic\b(?!\s+\bview\b)(?!\s+\bpure\b)[^{]*\{(?:(?!require|onlyOwner|onlyAdmin|modifier|\}).){0,200}\.call',
             0.55, "Hàm rút tiền public không có kiểm soát truy cập"),
        ],
        "negative_patterns": [
            (r'onlyOwner|onlyAdmin|onlyRole|onlyMinter|onlyFounder|oO\b|oOOrContract\b', 0.45, "Có access modifier"),
            (r'require\s*\(\s*msg\.sender\s*==', 0.40, "Có kiểm tra msg.sender"),
            (r'require\s*\(\s*[^;]*==\s*msg\.sender', 0.40, "Có kiểm tra msg.sender"),
            (r'if\s*\([^)]*msg\.sender\s*!=[^)]*\)\s*(?:revert\b|throw\b)', 0.40, "Idiom 0.4 if msg.sender != revert"),
            (r'\b(?:internal|private)\b', 0.45, "Visibility internal hoặc private"),
        ],
    },

    {
        "vuln_id": "SWC-106",
        "vuln_type": "unprotected_selfdestruct",
        "risk_level": "Critical",
        "patterns": [
            (r'selfdestruct\s*\(\s*(?:msg\.sender|tx\.origin|_?owner)\s*\)', 0.50,
             "selfdestruct gửi ETH về owner - kiểm tra bảo vệ"),
            (r'function\s+\w+[^)]*\)[^{]*\{[^}]*selfdestruct', 0.45,
             "Hàm chứa selfdestruct"),
        ],
        "negative_patterns": [
            (r'(?:onlyOwner|onlyAdmin|onlyRole|onlyFounder|oO\b|oOOrContract\b)', 0.45, "Có access control"),
            (r'require\s*\(.*msg\.sender', 0.40, "Có kiểm tra msg.sender"),
            (r'if\s*\([^)]*msg\.sender\s*!=[^)]*\)\s*(?:revert\b|throw\b)', 0.40, "Idiom 0.4 if msg.sender != revert"),
            (r'\b(?:internal|private)\b', 0.45, "Visibility internal hoặc private"),
        ],
    },

    {
        "vuln_id": "SWC-115",
        "vuln_type": "tx_origin_authentication",
        "risk_level": "Critical",
        "patterns": [
            (r'tx\.origin\s*==', 0.70,
             "Xác thực dựa trên tx.origin - dễ bị phishing attack"),
            (r'require\s*\(\s*tx\.origin', 0.65,
             "require kiểm tra tx.origin - không an toàn"),
            (r'tx\.origin\s*!=', 0.60,
             "So sánh tx.origin"),
            (r'if\s*\(\s*tx\.origin', 0.55,
             "Điều kiện dựa trên tx.origin"),
        ],
        "negative_patterns": [],
    },

    {
        "vuln_id": "SWC-100",
        "vuln_type": "function_default_visibility",
        "risk_level": "Low",
        "patterns": [
            # Hàm không có visibility specifier (Solidity < 0.5.0)
            (r'function\s+\w+\s*\([^)]*\)\s*(?:returns\s*\([^)]*\)\s*)?\{',
             0.30, "Hàm có thể thiếu visibility specifier"),
            # Hàm internal/private nhưng khai báo sai
            (r'function\s+_\w+\s*\([^)]*\)\s*(?:public|external)', 0.45,
             "Hàm bắt đầu bằng _ (quy ước internal) nhưng khai báo public/external"),
        ],
        "negative_patterns": [
            (r'function\s+\w+\s*\([^)]*\)\s*(?:public|external|internal|private)',
             0.40, "Hàm có visibility specifier rõ ràng"),
        ],
    },

    # ─────────────────────────────── ACCESS CONTROL ───────────────────────────

    {
        "vuln_id": "SWC-101",
        "vuln_type": "access_control",
        "risk_level": "Critical",
        "patterns": [
            # Hàm nhạy cảm không có modifier bảo vệ
            (r'function\s+(?:setOwner|transferOwnership|updateOwner)\s*\([^)]*\)\s*(?:public|external)(?!\s+\w*[Oo]nly)',
             0.55, "Thay đổi owner không có access control"),
            (r'function\s+(?:mint|burn|pause|unpause|upgrade)\s*\([^)]*\)\s*(?:public|external)(?:(?!onlyOwner|onlyAdmin|onlyRole|require|\}).){0,100}\{',
             0.50, "Hàm đặc quyền (mint/burn/pause) có thể thiếu access control"),
            # Modifier rỗng hoặc luôn pass
            (r'modifier\s+\w+\s*\([^)]*\)\s*\{\s*_;\s*\}', 0.60,
             "Modifier rỗng - không có logic kiểm tra"),
            (r'modifier\s+\w+\s*\([^)]*\)\s*\{\s*//[^\n]*\n\s*_;\s*\}', 0.55,
             "Modifier chỉ có comment - nghi ngờ placeholder"),
            # Role-based access sai
            (r'hasRole\s*\([^)]*\)\s*==\s*false', 0.40,
             "Kiểm tra hasRole ngược - logic sai tiềm ẩn"),
        ],
        "negative_patterns": [
            (r'require\s*\(\s*msg\.sender\s*==', 0.40, "Kiểm tra msg.sender"),
            (r'require\s*\(\s*[^;]*==\s*msg\.sender', 0.40, "Kiểm tra msg.sender"),
            (r'if\s*\([^)]*msg\.sender\s*!=[^)]*\)\s*(?:revert\b|throw\b)', 0.40, "Idiom 0.4 if msg.sender != revert"),
            (r'onlyOwner|onlyAdmin|onlyRole|onlyMinter|onlyFounder|oO\b|oOOrContract\b', 0.45, "Có access modifier"),
            (r'AccessControl|Ownable', 0.20, "Dùng thư viện access control"),
            (r'\b(?:internal|private)\b', 0.45, "Visibility internal hoặc private"),
        ],
    },

    # ──────────────────────────── UNCHECKED CALLS ─────────────────────────────

    {
        "vuln_id": "SWC-104",
        "vuln_type": "unchecked_low_level_calls",
        "risk_level": "Critical",
        "patterns": [
            # .call() không kiểm tra return value
            (r'(?<!\(bool\s)(?<!\bbool\b\s)\.\s*call\s*[\({](?:[^;]*;)', 0.45,
             "low-level call() không kiểm tra return value"),
            # Gán kết quả call nhưng không revert
            (r'(?:bool\s+\w+\s*,?\s*)?\.\s*call\b(?:(?!require|revert|if\s*\().){0,100};',
             0.40, "Kết quả call có thể không được xử lý"),
            # .send() không kiểm tra
            (r'\.send\s*\([^)]*\)\s*;', 0.40,
             ".send() không kiểm tra return value bool"),
            # staticcall không kiểm tra
            (r'\.staticcall\s*\(', 0.30,
             "staticcall - kiểm tra return value"),
            # Assembly call
            (r'assembly\s*\{[^}]*call\b', 0.35,
             "Assembly-level call - cần kiểm tra cẩn thận"),
        ],
        "negative_patterns": [
            (r'\(bool\s+\w+,\s*\)\s*=\s*', 0.35, "Kết quả được destructure đúng cách"),
            (r'require\s*\(\s*\w+(?:success|Success|ok|Ok)\b', 0.30,
             "Kiểm tra success flag"),
        ],
    },

    # ─────────────────────────── INTEGER OVERFLOW ─────────────────────────────

    {
        "vuln_id": "SWC-101",  # reused ID for arithmetic
        "vuln_type": "integer_overflow_underflow",
        "risk_level": "High",
        "patterns": [
            # Phép toán trực tiếp không có SafeMath (Solidity < 0.8.0)
            (r'pragma\s+solidity\s+[^;]*0\.[0-7]\.', 0.15,
             "Phiên bản < 0.8 không có overflow check tự động"),
            # Cộng/trừ trực tiếp trên uint
            (r'uint\w*\s+\w+\s*=\s*\w+\s*\+\s*\w+(?!\s*SafeMath)', 0.30,
             "Cộng uint không dùng SafeMath (nguy hiểm với Solidity <0.8)"),
            (r'uint\w*\s+\w+\s*=\s*\w+\s*-\s*\w+(?!\s*SafeMath)', 0.30,
             "Trừ uint - nguy cơ underflow"),
            # unchecked block trong Solidity >= 0.8
            (r'\bunchecked\s*\{', 0.50,
             "Khối unchecked - tắt overflow check, phải review kỹ"),
            # Ép kiểu thu hẹp
            (r'uint(?:8|16|32|64|128)\s*\(\s*\w+\s*\)', 0.35,
             "Downcast uint - có thể mất dữ liệu nếu giá trị vượt range"),
            (r'int(?:8|16|32|64|128)\s*\(\s*\w+\s*\)', 0.35,
             "Downcast int - có thể mất dữ liệu"),
            # Exponentiation có thể overflow
            (r'\w+\s*\*\*\s*\w+', 0.20,
             "Phép lũy thừa - kiểm tra overflow"),
        ],
        "negative_patterns": [
            (r'using\s+SafeMath\s+for', 0.35, "Dùng SafeMath"),
            (r'pragma\s+solidity\s+\^?0\.8', 0.40,
             "Solidity 0.8+ có overflow check tự động"),
        ],
    },

    # ─────────────────────────────── DENIAL OF SERVICE ───────────────────────

    {
        "vuln_id": "SWC-113",
        "vuln_type": "dos_gas_limit",
        "risk_level": "High",
        "patterns": [
            # Unbounded loop trên dynamic array
            (r'for\s*\([^)]*;\s*\w+\s*<\s*\w+\.length\s*;', 0.45,
             "Vòng lặp trên array động - nguy cơ DoS nếu array lớn"),
            (r'for\s*\([^)]*;\s*i\s*<\s*\w+\.length', 0.40,
             "Iterate toàn bộ array - gas cost không giới hạn"),
            # Transfer trong vòng lặp
            (r'for[^{]*\{[^}]*\.transfer\s*\(', 0.55,
             "Transfer ETH trong vòng lặp - có thể fail cả batch nếu 1 địa chỉ revert"),
            (r'for[^{]*\{[^}]*\.call\s*[\({]', 0.50,
             "External call trong vòng lặp"),
            # Push payment pattern nguy hiểm
            (r'for[^{]*\{[^}]*(?:payable|\.transfer|\.send)', 0.45,
             "Push payment pattern trong loop - nguy cơ DoS"),
            # Storage write trong loop
            (r'for[^{]*\{[^}]*\w+\[', 0.30,
             "Ghi vào storage mapping/array trong vòng lặp"),
            # Revert trong constructor làm contract không deploy được
            (r'constructor[^{]*\{[^}]*require\s*\((?:(?!msg\.sender).)*\)', 0.25,
             "require trong constructor - kiểm tra điều kiện không thể thay đổi"),
        ],
        "negative_patterns": [
            (r'gasleft\s*\(\s*\)', 0.15, "Kiểm tra gas còn lại"),
        ],
    },

    {
        "vuln_id": "SWC-128",
        "vuln_type": "dos_revert_griefing",
        "risk_level": "High",
        "patterns": [
            # Contract từ chối nhận ETH
            (r'receive\s*\(\s*\)\s*external\s*payable\s*\{[^}]*revert', 0.50,
             "receive() luôn revert - contract không nhận ETH được"),
            (r'fallback\s*\(\s*\)\s*external\s*(?:payable\s*)?\{[^}]*revert', 0.45,
             "fallback() revert - nguy cơ griefing"),
            # Gọi external contract có thể fail toàn bộ transaction
            (r'(?:\.call|\.transfer|\.send)\s*[({][^)]*\)\s*;(?:(?!if|require|revert).){0,50}', 0.35,
             "External call fail không được xử lý có thể DoS toàn bộ hàm"),
        ],
        "negative_patterns": [],
    },

    # ──────────────────────────────── FLASH LOAN ──────────────────────────────

    {
        "vuln_id": "SWC-FLASHLOAN",  # custom
        "vuln_type": "flash_loan_vulnerability",
        "risk_level": "High",
        "patterns": [
            # Sử dụng price oracle dễ bị thao túng
            (r'getPrice\s*\(|getReserve\s*\(|getAmountsOut\s*\(', 0.40,
             "Dùng on-chain price query - dễ bị flash loan thao túng"),
            # Tính giá dựa trên balance của contract
            (r'address\s*\(\s*this\s*\)\.balance', 0.45,
             "Dùng contract balance làm oracle giá - dễ bị flash loan"),
            (r'token\.balanceOf\s*\(\s*address\s*\(\s*this\s*\)\s*\)', 0.40,
             "Dùng token balance của contract - dễ bị flash loan"),
            # Uniswap/Sushiswap spot price
            (r'IUniswapV[23](?:Pair|Router)\s*\(\s*\w+\s*\)\.getReserves\s*\(', 0.45,
             "Dùng Uniswap spot price (getReserves) - không an toàn, dùng TWAP"),
            (r'\.getReserves\s*\(\s*\)', 0.40,
             "Lấy reserves trực tiếp - thao túng được bằng flash loan"),
            # Không kiểm tra reentrancy trong callback
            (r'function\s+(?:executeOperation|onFlashLoan|uniswapV2Call|pancakeCall)\s*\(', 0.35,
             "Flash loan callback - kiểm tra reentrancy guard"),
        ],
        "negative_patterns": [
            (r'TWAP|twap|timeWeightedAverage|consult\s*\(', 0.40,
             "Dùng TWAP oracle - an toàn hơn spot price"),
            (r'Chainlink|AggregatorV3|latestRoundData', 0.35,
             "Dùng Chainlink oracle - an toàn hơn"),
        ],
    },

    # ─────────────────────────────── FRONT-RUNNING ────────────────────────────

    {
        "vuln_id": "SWC-114",
        "vuln_type": "front_running",
        "risk_level": "High",
        "patterns": [
            # Approve pattern dễ bị front-run
            (r'function\s+approve\s*\([^)]*\)[^{]*\{(?:(?!increaseAllowance|decreaseAllowance).){0,300}',
             0.35, "approve() pattern - nên dùng increaseAllowance/decreaseAllowance"),
            # Race condition trong allowance
            (r'allowance\s*\[\s*\w+\s*\]\s*\[\s*\w+\s*\]\s*=', 0.30,
             "Set allowance trực tiếp - nguy cơ front-running"),
            # Transaction ordering dependency
            (r'block\.number\s*[+\-]\s*\d+', 0.25,
             "Logic dựa trên block number relative - dễ bị front-run"),
            # Commit-reveal thiếu
            (r'function\s+(?:bid|auction|buy|purchase)\s*\([^)]*uint[^)]*\)', 0.30,
             "Hàm đấu giá/mua - kiểm tra có commit-reveal không"),
            # Giá gas không hạn chế
            (r'tx\.gasprice', 0.25,
             "Logic phụ thuộc tx.gasprice - dễ bị miner thao túng"),
        ],
        "negative_patterns": [
            (r'commit\w*\[|reveal\w*\(|CommitReveal', 0.35,
             "Có commit-reveal scheme"),
        ],
    },

    # ──────────────────────────── TIMESTAMP DEPENDENCE ────────────────────────

    {
        "vuln_id": "SWC-116",
        "vuln_type": "timestamp_dependence",
        "risk_level": "Medium",
        "patterns": [
            (r'block\.timestamp\s*(?:<|>|<=|>=|==|!=)', 0.45,
             "So sánh block.timestamp - miner có thể điều chỉnh ~15s"),
            (r'\bnow\b\s*(?:<|>|<=|>=|==|!=)', 0.40,
             "Dùng 'now' để so sánh - deprecated, alias của block.timestamp"),
            (r'block\.timestamp\s*[%+\-]', 0.35,
             "Tính toán dựa trên timestamp"),
            (r'require\s*\([^)]*block\.timestamp', 0.40,
             "require dựa trên timestamp"),
            (r'if\s*\([^)]*block\.timestamp[^)]*\)', 0.35,
             "Điều kiện if dựa trên timestamp"),
            # Entropy từ timestamp (cực kỳ nguy hiểm)
            (r'uint\s*\(\s*keccak256\s*\([^)]*block\.timestamp', 0.60,
             "Dùng timestamp để tạo random - KHÔNG AN TOÀN"),
            (r'block\.timestamp\s*%\s*\d+', 0.55,
             "Modulo timestamp để random - predictable bởi miner"),
        ],
        "negative_patterns": [],
    },

    # ────────────────────────── BLOCK INFO DEPENDENCE ─────────────────────────

    {
        "vuln_id": "SWC-120",
        "vuln_type": "block_info_dependence",
        "risk_level": "Medium",
        "patterns": [
            # Dùng blockhash làm nguồn random
            (r'blockhash\s*\(', 0.55,
             "blockhash làm random - chỉ valid với 256 block gần nhất, predictable"),
            (r'block\.blockhash\s*\(', 0.55,
             "block.blockhash - deprecated"),
            (r'block\.difficulty\s*[%+\-*]', 0.45,
             "block.difficulty làm random source - miner có thể chọn"),
            (r'block\.prevrandao', 0.30,
             "block.prevrandao (post-merge) - vẫn có thể bị manipulate bởi validator"),
            # Random kém chất lượng
            (r'keccak256\s*\([^)]*block\.(?:number|difficulty|coinbase)', 0.50,
             "Hash của block metadata làm random - không an toàn"),
            (r'uint\s*\(\s*keccak256\s*\([^)]*abi\.encodePacked\s*\([^)]*block\.', 0.45,
             "Entropy từ block info - predictable"),
        ],
        "negative_patterns": [
            (r'VRF|vrfCoordinator|requestRandomness|Chainlink', 0.50,
             "Dùng Chainlink VRF - an toàn"),
        ],
    },

    # ─────────────────────────── TRANSACTION ORDERING ─────────────────────────

    {
        "vuln_id": "SWC-114",
        "vuln_type": "transaction_ordering_dependency",
        "risk_level": "Medium",
        "patterns": [
            # State phụ thuộc vào thứ tự tx
            (r'mapping\s*\(\s*address\s*=>\s*uint\w*\s*\)\s*public\s+(?:balances|amounts)', 0.25,
             "Public balance mapping - kiểm tra race condition"),
            # Check-then-act không atomic
            (r'if\s*\([^)]*balances?\[', 0.30,
             "Điều kiện kiểm tra balance - có thể bị race condition"),
            # Global price/state dễ bị sandwich attack
            (r'price\s*=\s*[^;]+;\s*[^;]*transfer\s*\(', 0.40,
             "Cập nhật giá rồi transfer trong cùng tx - sandwich attack"),
        ],
        "negative_patterns": [],
    },

    # ────────────────────────────── GAS LIMIT ─────────────────────────────────

    {
        "vuln_id": "SWC-126",
        "vuln_type": "insufficient_gas_griefing",
        "risk_level": "Medium",
        "patterns": [
            # Forward all gas (nguy hiểm)
            (r'\.call\s*\{[^}]*\}', 0.20,
             "External call - kiểm tra gas forwarding"),
            # Hardcode gas limit thấp
            (r'\.call\s*\{[^}]*gas\s*:\s*(?:[0-9]{1,4})\b', 0.45,
             "Hardcode gas limit nhỏ - có thể không đủ gas cho callee"),
            # Gas stipend của transfer/send
            (r'\.transfer\s*\(\s*\w+\s*\)', 0.20,
             ".transfer() forward 2300 gas - có thể fail với complex receiver"),
            # Loop gas estimation
            (r'for\s*\([^;]*;\s*[^;]*;\s*[^)]*\)\s*\{[^}]*(?:sstore|SSTORE|\[\s*\w+\s*\]\s*=)', 0.40,
             "SSTORE trong vòng lặp - gas cost tuyến tính với số phần tử"),
        ],
        "negative_patterns": [],
    },

    # ────────────────────────────── UNSAFE CAST ───────────────────────────────

    {
        "vuln_id": "SWC-CAST",
        "vuln_type": "unsafe_type_casting",
        "risk_level": "Medium",
        "patterns": [
            # uint256 -> uint8/16/32/64/128
            (r'uint(?:8|16|32|64|128)\s*\(\s*(?!uint(?:8|16|32|64|128))\w+\s*\)', 0.40,
             "Explicit downcast uint256 -> smaller - kiểm tra overflow"),
            # int -> uint không check âm
            (r'uint\w*\s*\(\s*int\w*\s*\w+\s*\)', 0.45,
             "Cast int sang uint - nếu giá trị âm sẽ wrap thành số lớn"),
            (r'int\w*\s*\(\s*uint\w*\s*\w+\s*\)', 0.35,
             "Cast uint sang int - kiểm tra overflow"),
            # address -> uint
            (r'uint(?:256|160)?\s*\(\s*address\s*\(', 0.30,
             "Cast address sang uint"),
            # bytes -> uint/int không kiểm tra
            (r'uint\w*\s*\(\s*bytes\w*\s*\w+\s*\)', 0.35,
             "Cast bytes sang uint"),
            # Dùng assembly cho cast nguy hiểm
            (r'assembly\s*\{[^}]*(?:shr|shl|and)\s+0x', 0.30,
             "Assembly bitshift/mask - kiểm tra logic cẩn thận"),
        ],
        "negative_patterns": [
            (r'SafeCast|toUint\d+\s*\(', 0.40,
             "Dùng SafeCast library"),
        ],
    },

    # ─────────────────────────── COMPILER VERSION ─────────────────────────────

    {
        "vuln_id": "SWC-103",
        "vuln_type": "floating_pragma",
        "risk_level": "Low",
        "patterns": [
            # Floating pragma
            (r'pragma\s+solidity\s+\^0\.[0-7]\.', 0.50,
             "Floating pragma với phiên bản < 0.8 - nhiều known bug"),
            (r'pragma\s+solidity\s+>=\s*0\.[0-4]\.', 0.45,
             "Cho phép compile với phiên bản rất cũ"),
            # Pragma quá cũ cố định
            (r'pragma\s+solidity\s+0\.[0-6]\.', 0.40,
             "Pragma cố định phiên bản cũ < 0.7"),
            (r'pragma\s+solidity\s+0\.4\.', 0.60,
             "Pragma 0.4.x - rất cũ, nhiều lỗ hổng đã biết"),
            (r'pragma\s+solidity\s+0\.5\.', 0.40,
             "Pragma 0.5.x - thiếu nhiều bảo vệ của 0.8+"),
            # Không có SPDX license
            (r'\A(?![\s\S]*SPDX-License-Identifier)', 0.20,
             "Không có SPDX license identifier"),
        ],
        "negative_patterns": [
            (r'pragma\s+solidity\s+\^?0\.8\.', 0.50,
             "Dùng Solidity 0.8+ - an toàn hơn"),
        ],
    },

    # ─────────────────────────── NAMING CONVENTIONS ───────────────────────────

    {
        "vuln_id": "SWC-NAMING",
        "vuln_type": "naming_convention",
        "risk_level": "Low",
        "patterns": [
            # Biến state không có underscore prefix
            (r'^\s*(?:uint|int|bool|address|bytes|string|mapping)\s+(?!_)\w+\s*;', 0.15,
             "Biến state nên có tiền tố _ để phân biệt với local variable"),
            # Hàm internal/private không có underscore
            (r'function\s+(?!_)[a-z]\w+\s*\([^)]*\)\s*(?:internal|private)', 0.10,
             "Hàm internal/private nên có tiền tố _"),
            # Constant không SCREAMING_SNAKE_CASE
            (r'(?:uint|int|bytes|address)\s+(?:constant|immutable)\s+[a-z]\w+', 0.10,
             "Hằng số nên dùng SCREAMING_SNAKE_CASE"),
            # Event không PascalCase
            (r'event\s+[a-z]\w+\s*\(', 0.10,
             "Event nên dùng PascalCase"),
        ],
        "negative_patterns": [],
    },

    # ──────────────────────────── REDUNDANT CODE ──────────────────────────────

    {
        "vuln_id": "SWC-REDUNDANT",
        "vuln_type": "redundant_code",
        "risk_level": "Low",
        "patterns": [
            # Tautology trong require
            (r'require\s*\(\s*true\s*\)', 0.60,
             "require(true) - vô nghĩa"),
            # Dead code
            (r'return\s*;[^}]*\w+\s*=', 0.40,
             "Code sau return statement - dead code"),
            # Emit event không cần thiết
            (r'emit\s+\w+\s*\([^)]*\)\s*;\s*emit\s+\w+\s*\([^)]*\)\s*;', 0.20,
             "Emit hai event liên tiếp - có thể gộp"),
            # Unused import (cơ bản)
            (r'import\s+"[^"]+\.sol"\s*;', 0.05,
             "Import - kiểm tra có thực sự dùng không"),
            # Floating point qua uint với divisor nhỏ
            (r'uint\w*\s+\w+\s*=\s*\d+\s*/\s*\d+\s*;', 0.30,
             "Phép chia integer literal - kết quả luôn truncate, kiểm tra intent"),
        ],
        "negative_patterns": [],
    },

    # ──────────────────────────── SIGNATURE REPLAY ────────────────────────────

    {
        "vuln_id": "SWC-121",
        "vuln_type": "signature_replay",
        "risk_level": "High",
        "patterns": [
            # ecrecover không có nonce
            (r'ecrecover\s*\(', 0.45,
             "ecrecover - kiểm tra nonce để tránh replay attack"),
            # Thiếu chain ID trong signature data
            (r'keccak256\s*\([^)]*abi\.encodePacked\s*\([^)]*\)\s*\)[^)]*ecrecover', 0.40,
             "Signature không có chain ID - cross-chain replay"),
            # Thiếu nonce tracking
            (r'ecrecover[^;]{0,300}mapping[^;]*nonce', 0.20,
             "Kiểm tra nonce tracking sau ecrecover"),
        ],
        "negative_patterns": [
            (r'nonce\s*\+\+|nonces\s*\[|_useNonce', 0.40,
             "Có nonce tracking"),
            (r'block\.chainid|chainId|DOMAIN_SEPARATOR', 0.35,
             "Có chain ID trong signature"),
        ],
    },

    # ───────────────────────── PRICE ORACLE MANIPULATION ─────────────────────

    {
        "vuln_id": "SWC-ORACLE",
        "vuln_type": "price_oracle_manipulation",
        "risk_level": "High",
        "patterns": [
            # Tính toán ratio từ balance
            (r'balanceOf\s*\([^)]*\)\s*/\s*totalSupply', 0.50,
             "Tính price ratio từ balanceOf/totalSupply - dễ bị flash loan"),
            (r'reserve\d*\s*[*/]\s*reserve\d*', 0.45,
             "Tính price từ reserves - spot price có thể bị thao túng"),
            # AMM price calculation
            (r'getAmountOut\s*\(|getAmountIn\s*\(', 0.40,
             "Dùng AMM price function - kiểm tra slippage protection"),
            # Không có slippage protection
            (r'amountOutMin\s*=\s*0|minOut\s*=\s*0|slippage\s*=\s*0', 0.55,
             "Slippage protection = 0 - sandwich attack dễ dàng"),
        ],
        "negative_patterns": [
            (r'TWAP|twap|observe\s*\(|consult\s*\(', 0.40,
             "Dùng TWAP"),
            (r'amountOutMin\s*>', 0.25, "Có slippage protection"),
        ],
    },

    # ──────────────────────────── STORAGE COLLISION ───────────────────────────

    {
        "vuln_id": "SWC-STORAGE",
        "vuln_type": "storage_collision",
        "risk_level": "Critical",
        "patterns": [
            # delegatecall vào address không tin tưởng
            (r'\.delegatecall\s*\([^)]*\)', 0.55,
             "delegatecall - nguy cơ storage collision và unauthorized code"),
            # Proxy pattern không dùng EIP-1967 slot
            (r'bytes32\s+(?:private\s+)?constant\s+\w+IMPLEMENTATION\w*\s*=\s*keccak256', 0.25,
             "Implementation slot - kiểm tra EIP-1967 compliance"),
            # Unstructured storage
            (r'assembly\s*\{[^}]*sload\s*\(', 0.30,
             "Assembly sload - kiểm tra storage slot collision"),
            (r'assembly\s*\{[^}]*sstore\s*\(', 0.30,
             "Assembly sstore - kiểm tra storage slot collision"),
        ],
        "negative_patterns": [
            (r'EIP1967|eip1967|0x360894', 0.35,
             "Dùng EIP-1967 storage slot"),
        ],
    },

    # ───────────────────── SLITHER-DERIVED LOGIC CHECKS ──────────────────────

    {
        "vuln_id": "SWC-111",
        "vuln_type": "deprecated_solidity_usage",
        "risk_level": "Low",
        "patterns": [
            (r'\bmsg\.gas\b', 0.35, "msg.gas đã bị thay thế bởi gasleft()"),
            (r'\bsha3\s*\(', 0.35, "sha3() đã bị thay thế bởi keccak256()"),
            (r'\bthrow\s*;', 0.40, "throw đã bị thay thế bởi revert()/require()"),
            (r'\.callcode\s*\(', 0.45, "callcode đã bị thay thế bởi delegatecall"),
        ],
        "negative_patterns": [],
    },

    {
        "vuln_id": "SWC-129",
        "vuln_type": "incorrect_unary_expression",
        "risk_level": "Medium",
        "patterns": [
            (r'(?<![+\-*/%&|^])=\s*\+\s*(?:\d+|[A-Za-z_]\w*)', 0.55,
             "Biểu thức '= +' thường là lỗi gõ nhầm của '+='"),
        ],
        "negative_patterns": [],
    },

    {
        "vuln_id": "SWC-135",
        "vuln_type": "tautological_comparison",
        "risk_level": "Medium",
        "patterns": [
            (r'\b([A-Za-z_]\w*)\s*(?:==|!=|<=|>=|<|>)\s*\1\b', 0.55,
             "Một giá trị đang được so sánh với chính nó"),
        ],
        "negative_patterns": [],
    },

    {
        "vuln_id": "STATIC-DIVIDE-BEFORE-MULTIPLY",
        "vuln_type": "divide_before_multiply",
        "risk_level": "Medium",
        "patterns": [
            (r'\b[A-Za-z_]\w*\s*/\s*[A-Za-z_]\w*\s*\*\s*[A-Za-z_]\w*\b',
             0.35, "Chia số nguyên trước khi nhân có thể làm mất độ chính xác"),
        ],
        "negative_patterns": [],
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

# Đổi vị trí ký tự thành số dòng (bắt đầu từ 1).
def _line_at(code: str, pos: int) -> int:
    """Trả về số dòng của vị trí character trong chuỗi."""
    return code.count('\n', 0, pos) + 1


# Cắt ~120 ký tự quanh vị trí khớp để làm bằng chứng (evidence) hiển thị trong báo cáo.
def _extract_context(code: str, pos: int, window: int = 120) -> str:
    """Trích đoạn code xung quanh vị trí pos."""
    start = max(0, pos - window)
    end   = min(len(code), pos + window)
    snippet = code[start:end]
    return snippet.replace('\n', ' ').strip()


# Che cả comment lẫn chuỗi -> mã "sạch" để các regex luật chạy trên đó.
def _clean_code(code: str) -> str:
    """Return Solidity code with comments and strings masked (offsets preserved)."""
    return mask_noncode(code)


# Tìm vị trí ngoặc đóng khớp với ngoặc mở tại open_pos ({}, (), ...).
def _balanced_end(
    code: str,
    open_pos: int,
    opening: str = "{",
    closing: str = "}",
) -> Optional[int]:
    """Return the matching closing delimiter position."""
    if open_pos < 0 or open_pos >= len(code) or code[open_pos] != opening:
        return None
    depth = 0
    for pos in range(open_pos, len(code)):
        char = code[pos]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return pos
    return None


# Duyệt mọi hàm có thân (function/constructor/receive/fallback) mà không cần compiler.
def _iter_functions(code: str) -> Iterator[SolidityFunction]:
    """Yield implemented functions without requiring a Solidity compiler."""
    clean = _clean_code(code)
    functions: List[SolidityFunction] = []
    for pattern in (RE_FUNCTION_WITH_BODY, RE_SPECIAL_FUNCTION_WITH_BODY):
        for match in pattern.finditer(clean):
            open_pos = match.end() - 1
            close_pos = _balanced_end(clean, open_pos)
            if close_pos is None:
                continue
            functions.append(SolidityFunction(
                name=match.group("name"),
                params=code[match.start("params"):match.end("params")],
                header=code[match.start("header"):match.end("header")],
                start=match.start(),
                body_start=open_pos + 1,
                end=close_pos,
            ))
    yield from sorted(functions, key=lambda item: item.start)


# Duyệt mọi contract (ranh giới thân) để chạy các kiểm tra cấp contract.
def _iter_contracts(code: str) -> Iterator[SolidityContract]:
    """Yield concrete contract boundaries for contract-level checks."""
    clean = _clean_code(code)
    for match in RE_CONTRACT_WITH_BODY.finditer(clean):
        open_pos = match.end() - 1
        close_pos = _balanced_end(clean, open_pos)
        if close_pos is None:
            continue
        yield SolidityContract(
            name=match.group("name"),
            header=code[match.start("header"):match.end("header")],
            start=match.start(),
            body_start=open_pos + 1,
            end=close_pos,
        )


# Duyệt mọi modifier kèm thân của nó.
def _iter_modifiers(code: str) -> Iterator[SolidityModifier]:
    """Yield modifier boundaries without requiring a Solidity compiler."""
    clean = _clean_code(code)
    for match in RE_MODIFIER_WITH_BODY.finditer(clean):
        open_pos = match.end() - 1
        close_pos = _balanced_end(clean, open_pos)
        if close_pos is None:
            continue
        yield SolidityModifier(
            name=match.group("name"),
            params=match.group("params") or "",
            header=match.group("header") or "",
            body=clean[open_pos + 1:close_pos],
            start=match.start(),
            body_start=open_pos + 1,
            end=close_pos,
        )


# Thân hàm/modifier có ít nhất một kiểm tra người gọi trong RE_SENDER_CHECKS không: require(msg.sender == ...),
# require(... == msg.sender), if (msg.sender != ...) revert/throw, require(owners[msg.sender]...),
# require(isOwner(...)/hasRole(...)/authorized(...)).
# Dùng chung cho extract_guard_modifiers (thân modifier) và is_function_guarded (thân hàm).
def _has_sender_check(body: str) -> bool:
    return any(pattern.search(body) for pattern in RE_SENDER_CHECKS)


# Nhiệm vụ: tìm tên các modifier có chức năng phân quyền: tên kiểu onlyXxx/auth/restricted, HOẶC thân modifier
# có require(msg.sender == ...), if (msg.sender != ...) revert, require(owners[msg.sender]) ...
# Mục đích: hàm có guard modifier thì không bị báo lỗi access control / withdraw / selfdestruct (giảm false positive).
def extract_guard_modifiers(code: str) -> set[str]:
    """
    Trích xuất tên các modifier có chức năng bảo vệ truy cập (guard modifier).
    Nhận diện:
      - Các tên thông dụng: onlyOwner, onlyAdmin, onlyRole, onlyMinter, onlyFounder, etc.
      - Modifier tự đặt tên (như oO, oOOrContract) mà thân có:
        * require(msg.sender == ...) hoặc require(... == msg.sender)
        * require(msg.sender == ... || ...)
        * if (msg.sender != ...) revert/throw
        * if (... != msg.sender) revert/throw
        * require(owners[msg.sender]...) hoặc require(isOwner[msg.sender]...)
    """
    clean = _clean_code(code)
    guards: set[str] = set()
    for m in RE_MODIFIER_NAME.finditer(clean):
        name = m.group(1)
        if RE_GUARD_MODIFIER_NAME.match(name):
            guards.add(name)
    for mod in _iter_modifiers(code):
        if _has_sender_check(mod.body):
            guards.add(mod.name)
    return guards


# Nhiệm vụ: xác định một hàm đã được bảo vệ quyền truy cập chưa: private/internal, có guard modifier trong header,
# hoặc trong thân có kiểm tra msg.sender. Dùng để bỏ qua các match SWC-101/105/106 trong hàm đã được bảo vệ.
def is_function_guarded(
    header: str,
    body: str,
    guard_modifiers: Optional[set[str]] = None,
) -> bool:
    """
    Kiểm tra xem một hàm có được bảo vệ quyền truy cập hay không.
    Trả về True nếu:
      - Visibility là private hoặc internal
      - Header hàm có chứa guard modifier (oO, oOOrContract, onlyOwner, ...)
      - Thân hàm có inline guard check:
        require(msg.sender == ...)
        if (msg.sender != ...) revert/throw
    """
    if RE_INTERNAL_OR_PRIVATE.search(header):
        return True
    active_guards = (guard_modifiers or set()) | {
        "onlyOwner", "onlyAdmin", "onlyRole", "onlyMinter",
        "onlyFounder", "authorized", "auth", "restricted"
    }
    for mod in active_guards:
        if re_word(mod).search(header):
            return True
    for token in RE_WORD_TOKEN.findall(header):
        if token.startswith("only") and len(token) > 4:
            return True
    return _has_sender_check(body)


# Tách danh sách đối số theo dấu phẩy ở mức ngoài cùng (không cắt trong ngoặc/chuỗi lồng nhau).
def _split_top_level_arguments(value: str) -> List[str]:
    """Split a Solidity argument list while respecting nested expressions."""
    arguments: List[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    closing_to_opening = {")": "(", "]": "[", "}": "{"}
    quote: Optional[str] = None
    escaped = False
    for pos, char in enumerate(value):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in depths:
            depths[char] += 1
        elif char in closing_to_opening:
            opening = closing_to_opening[char]
            depths[opening] = max(0, depths[opening] - 1)
        elif char == "," and not any(depths.values()):
            arguments.append(value[start:pos].strip())
            start = pos + 1
    tail = value[start:].strip()
    if tail or arguments:
        arguments.append(tail)
    return arguments


# Thu thập ánh xạ đơn giản tên biến -> kiểu (để các kiểm tra biết biến là address, bytes, mảng...).
def _collect_symbol_types(code: str) -> Dict[str, str]:
    """Collect simple variable-to-type bindings used by focused checks."""
    clean = _clean_code(code)
    ignored_types = {
        "contract", "function", "event", "error", "modifier", "return",
        "returns", "import", "pragma", "using", "struct", "enum",
    }
    bindings: Dict[str, str] = {}
    for match in RE_SYMBOL_DECLARATION.finditer(clean):
        type_name = RE_WHITESPACE.sub('', match.group("type"))
        if type_name.lower() in ignored_types:
            continue
        bindings[match.group("name")] = type_name
    return bindings


# Đối số có phải kiểu động (string, bytes, mảng) không - dùng cho kiểm tra va chạm abi.encodePacked.
def _is_dynamic_abi_argument(argument: str, symbol_types: Dict[str, str]) -> bool:
    value = argument.strip()
    if not value:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return True
    identifier = RE_IDENTIFIER.fullmatch(value)
    if not identifier:
        return False
    type_name = symbol_types.get(identifier.group(), "").replace(" ", "")
    return type_name in {"string", "bytes"} or "[]" in type_name


# Đối tượng nhận lời gọi có vẻ là token ERC20 không (ép kiểu IERC20(...) hoặc biến kiểu ERC20).
def _looks_like_erc20(receiver: str, symbol_types: Dict[str, str]) -> bool:
    compact = RE_WHITESPACE.sub('', receiver)
    cast = RE_CAST_PREFIX.match(compact)
    if cast:
        return "erc20" in cast.group("type").lower()
    type_name = symbol_types.get(compact, "").lower()
    return "erc20" in type_name


# Phân tích danh sách tham số hàm -> {tên: (kiểu, vị trí dữ liệu memory/storage/calldata)}.
def _parse_parameter_details(params: str) -> Dict[str, Tuple[str, str]]:
    """Return parameter name -> (type, data location)."""
    details: Dict[str, Tuple[str, str]] = {}
    for parameter in _split_top_level_arguments(params):
        match = RE_PARAMETER.match(parameter)
        if not match:
            continue
        details[match.group("name")] = (
            RE_WHITESPACE.sub('', match.group("type")),
            match.group("location") or "",
        )
    return details


# Duyệt các câu lệnh kết thúc bằng ";" ở mức ngoài cùng của một contract (= khai báo biến trạng thái).
def _contract_top_level_statements(
    code: str, contract: SolidityContract
) -> Iterator[Tuple[int, str]]:
    """Yield semicolon-terminated declarations for one contract."""
    clean = _clean_code(code)
    depth = 0
    statement_start = contract.body_start
    for pos in range(contract.body_start, contract.end):
        char = clean[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
            if depth == 0:
                statement_start = pos + 1
        elif char == ";" and depth == 0:
            yield statement_start, code[statement_start:pos + 1]
            statement_start = pos + 1


# Như trên nhưng cho mọi contract trong file.
def _top_level_contract_statements(code: str) -> Iterator[Tuple[int, str]]:
    """Yield semicolon-terminated state declarations outside nested blocks."""
    for contract in _iter_contracts(code):
        yield from _contract_top_level_statements(code, contract)


# Biến trạng thái của một contract -> (kiểu, vị trí, có khởi tạo hay không).
def _collect_contract_state_declarations(
    code: str, contract: SolidityContract
) -> Dict[str, Tuple[str, int, bool]]:
    """Return state name -> (type, source position, has initializer)."""
    result: Dict[str, Tuple[str, int, bool]] = {}
    for position, statement in _contract_top_level_statements(code, contract):
        clean_statement = _clean_code(statement)
        match = RE_STATE_DECLARATION.match(clean_statement)
        if match:
            leading = len(clean_statement) - len(clean_statement.lstrip())
            result[match.group("name")] = (
                RE_WHITESPACE.sub('', match.group("type")),
                position + leading,
                "=" in match.group("tail"),
            )
            continue
        mapping_match = RE_STATE_MAPPING.match(clean_statement)
        if mapping_match:
            leading = len(clean_statement) - len(clean_statement.lstrip())
            result[mapping_match.group("name")] = (
                "mapping",
                position + leading,
                False,
            )
    return result


# Biến trạng thái (mọi contract) -> kiểu.
def _collect_state_symbol_types(code: str) -> Dict[str, str]:
    """Collect simple top-level state variable declarations."""
    result: Dict[str, str] = {}
    for _, statement in _top_level_contract_statements(code):
        match = RE_STATE_TYPE_AND_NAME.match(_clean_code(statement))
        if match:
            result[match.group("name")] = RE_WHITESPACE.sub('', match.group("type"))
    return result


# Tên mapping -> kiểu giá trị (vd mapping(address => User) users -> "User").
def _collect_mapping_value_types(code: str) -> Dict[str, str]:
    """Collect top-level mapping name -> direct value type."""
    result: Dict[str, str] = {}
    for _, statement in _top_level_contract_statements(code):
        match = RE_MAPPING_VALUE_TYPE.match(_clean_code(statement))
        if match:
            result[match.group("name")] = match.group("value")
    return result


# Mỗi struct -> (tập kiểu thành viên, có chứa mapping trực tiếp không).
def _collect_struct_members(code: str) -> Dict[str, Tuple[set[str], bool]]:
    """Return struct member types and whether each struct directly has a mapping."""
    clean = _clean_code(code)
    structs: Dict[str, Tuple[set[str], bool]] = {}
    for match in RE_STRUCT_OPENING.finditer(clean):
        open_pos = match.end() - 1
        close_pos = _balanced_end(clean, open_pos)
        if close_pos is None:
            continue
        body = code[open_pos + 1:close_pos]
        member_types = set(_collect_symbol_types(body).values())
        structs[match.group("name")] = (
            member_types,
            bool(RE_MAPPING_KEYWORD.search(_clean_code(body))),
        )
    return structs


# Đệ quy kiểm tra struct có chứa mapping (trực tiếp hoặc qua struct lồng) - dùng cho kiểm tra delete mapping.
def _struct_contains_mapping(
    struct_name: str,
    structs: Dict[str, Tuple[set[str], bool]],
    visiting: Optional[set[str]] = None,
) -> bool:
    """Recursively determine whether a struct contains a mapping."""
    if struct_name not in structs:
        return False
    seen = set(visiting or ())
    if struct_name in seen:
        return False
    seen.add(struct_name)
    member_types, direct_mapping = structs[struct_name]
    if direct_mapping:
        return True
    return any(
        _struct_contains_mapping(RE_ARRAY_SUFFIX.sub('', type_name), structs, seen)
        for type_name in member_types
    )


# Duyệt vùng mã của các vòng lặp for/while trong một khoảng - dùng cho các kiểm tra "trong vòng lặp".
def _iter_loop_ranges(code: str, start: int, end: int) -> Iterator[Tuple[int, int]]:
    """Yield for/while ranges inside a source interval."""
    clean = _clean_code(code)
    for match in RE_LOOP_OPENING.finditer(clean, start, end):
        open_paren = clean.find("(", match.start(), match.end())
        close_paren = _balanced_end(clean, open_paren, "(", ")")
        if close_paren is None or close_paren >= end:
            continue
        body_pos = close_paren + 1
        while body_pos < end and clean[body_pos].isspace():
            body_pos += 1
        if body_pos < end and clean[body_pos] == "{":
            close_body = _balanced_end(clean, body_pos)
            if close_body is not None and close_body <= end:
                yield match.start(), close_body + 1
        else:
            semicolon = clean.find(";", body_pos, end)
            if semicolon >= 0:
                yield match.start(), semicolon + 1


# Đóng gói danh sách hit (vị trí, text, trọng số, mô tả) thành một VulnerabilityFinding đã confirmed.
# Mọi hàm _detect_* bên dưới đều trả về kết quả qua hàm này.
def _build_semantic_finding(
    code: str,
    vuln_id: str,
    vuln_type: str,
    risk_level: str,
    hits: List[Tuple[int, str, float, str]],
) -> Optional[VulnerabilityFinding]:
    if not hits:
        return None
    finding = VulnerabilityFinding(
        vuln_id=vuln_id,
        vuln_type=vuln_type,
        risk_level=risk_level,
        confirmed=True,
    )
    for pos, matched_text, weight, description in hits:
        line = _line_at(code, pos)
        finding.matches.append(PatternMatch(
            line=line,
            matched_text=matched_text[:80],
            context=_extract_context(code, pos),
            pattern_weight=weight,
        ))
        finding.notes.append(f"L{line}: {description}")
    return finding


# [Kiểm tra ngữ nghĩa] SWC-130: ký tự Unicode điều khiển hướng chữ (RLO U+202E...) dùng để giấu mã độc.
def _detect_bidirectional_controls(code: str) -> Optional[VulnerabilityFinding]:
    controls = {
        "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
        "\u2066", "\u2067", "\u2068", "\u2069",
    }
    hits = [
        (
            pos,
            f"U+{ord(char):04X}",
            0.80,
            "Ký tự điều khiển hướng văn bản có thể che giấu thứ tự token hiển thị",
        )
        for pos, char in enumerate(code)
        if char in controls
    ]
    return _build_semantic_finding(
        code, "SWC-130", "right_to_left_override", "High", hits
    )


# [Kiểm tra ngữ nghĩa] Dùng "^" (XOR) với hằng số khi có ý định lũy thừa "**" (vd 2^256). Đã che pragma ^0.x.
def _detect_incorrect_exponentiation(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    # Mask pragma statements to avoid false positives on caret version specifiers (e.g. pragma solidity ^0.4.25;)
    clean = RE_PRAGMA_STATEMENT.sub(lambda m: ' ' * len(m.group(0)), clean)
    hits: List[Tuple[int, str, float, str]] = []
    for match in RE_XOR_EXPRESSION.finditer(clean):
        operands = (match.group("left"), match.group("right"))
        if not any(RE_DECIMAL_NUMBER.fullmatch(value) for value in operands):
            continue
        hits.append((
            match.start(),
            code[match.start():match.end()],
            0.70,
            "Toán tử ^ là XOR; nếu muốn lũy thừa phải dùng **",
        ))
    return _build_semantic_finding(
        code,
        "STATIC-INCORRECT-EXP",
        "incorrect_exponentiation_operator",
        "High",
        hits,
    )


# [Kiểm tra ngữ nghĩa] SWC-133: abi.encodePacked với >= 2 đối số động liền nhau -> va chạm hash.
def _detect_encode_packed_collision(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    symbol_types = _collect_symbol_types(code)
    hits: List[Tuple[int, str, float, str]] = []
    for match in RE_ENCODE_PACKED_CALL.finditer(clean):
        open_pos = clean.find("(", match.start(), match.end())
        close_pos = _balanced_end(clean, open_pos, "(", ")")
        if close_pos is None:
            continue
        arguments = _split_top_level_arguments(code[open_pos + 1:close_pos])
        dynamic_flags = [
            _is_dynamic_abi_argument(argument, symbol_types)
            for argument in arguments
        ]
        has_adjacent_dynamic = any(
            left and right
            for left, right in zip(dynamic_flags, dynamic_flags[1:])
        )
        if not has_adjacent_dynamic:
            continue
        hits.append((
            match.start(),
            code[match.start():close_pos + 1],
            0.70,
            "abi.encodePacked ghép các đối số kiểu động liền nhau mà không có ranh giới",
        ))
    return _build_semantic_finding(
        code,
        "SWC-133",
        "abi_encode_packed_collision",
        "High",
        hits,
    )


# Duyệt các lời gọi transfer/transferFrom/approve trên đối tượng ERC20 trong từng hàm (dùng cho 2 kiểm tra dưới).
def _iter_erc20_calls(
    code: str,
) -> Iterator[Tuple[SolidityFunction, re.Match, str, List[str], int]]:
    clean = _clean_code(code)
    symbol_types = _collect_symbol_types(code)
    for function in _iter_functions(code):
        for match in RE_ERC20_TRANSFER_CALL.finditer(clean, function.body_start, function.end):
            receiver = match.group("receiver")
            if not _looks_like_erc20(receiver, symbol_types):
                continue
            open_pos = clean.find("(", match.start("method"), match.end())
            close_pos = _balanced_end(clean, open_pos, "(", ")")
            if close_pos is None or close_pos > function.end:
                continue
            arguments = _split_top_level_arguments(code[open_pos + 1:close_pos])
            yield function, match, match.group("method"), arguments, close_pos


# [Kiểm tra ngữ nghĩa] transferFrom(from, ...) với "from" là tham số do caller truyền vào,
# không ràng buộc với msg.sender -> ai cũng rút được token của người khác đã approve.
def _detect_arbitrary_erc20_transfer_from(code: str) -> Optional[VulnerabilityFinding]:
    hits: List[Tuple[int, str, float, str]] = []
    for function, match, method, arguments, close_pos in _iter_erc20_calls(code):
        if method != "transferFrom" or len(arguments) < 3:
            continue
        parameter_types = _collect_symbol_types(function.params)
        source = arguments[0].strip()
        if parameter_types.get(source, "").lower() != "address":
            continue
        hits.append((
            match.start(),
            code[match.start():close_pos + 1],
            0.65,
            f"Nguồn token '{source}' do caller cung cấp và không ràng buộc với msg.sender",
        ))
    return _build_semantic_finding(
        code,
        "STATIC-ARBITRARY-ERC20",
        "arbitrary_erc20_transfer_from",
        "High",
        hits,
    )


# [Kiểm tra ngữ nghĩa] Gọi token.transfer/transferFrom mà bỏ qua giá trị bool trả về.
def _detect_unchecked_erc20_transfer(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    hits: List[Tuple[int, str, float, str]] = []
    for function, match, method, _, close_pos in _iter_erc20_calls(code):
        boundaries = [
            clean.rfind(delimiter, function.body_start, match.start())
            for delimiter in (";", "{", "}")
        ]
        nearest_boundary = max(boundaries, default=-1)
        statement_start = (
            nearest_boundary
            if nearest_boundary >= function.body_start
            else function.body_start - 1
        )
        prefix = clean[statement_start + 1:match.start()].strip()
        suffix_pos = close_pos + 1
        while suffix_pos < function.end and clean[suffix_pos].isspace():
            suffix_pos += 1
        if prefix or suffix_pos >= function.end or clean[suffix_pos] != ";":
            continue
        hits.append((
            match.start(),
            code[match.start():close_pos + 1],
            0.60,
            f"Giá trị bool trả về từ ERC20 {method}() bị bỏ qua",
        ))
    return _build_semantic_finding(
        code,
        "STATIC-UNCHECKED-ERC20",
        "unchecked_erc20_transfer",
        "High",
        hits,
    )


# [Kiểm tra ngữ nghĩa] Contract nhận ETH (payable) nhưng không có cách nào rút ETH ra -> ETH bị khoá.
def _detect_locked_ether(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    functions = list(_iter_functions(code))
    hits: List[Tuple[int, str, float, str]] = []
    for contract in _iter_contracts(code):
        prefix = clean[max(0, contract.start - 16):contract.start]
        if RE_ABSTRACT_AT_END.search(prefix) or RE_IS_KEYWORD.search(contract.header):
            continue
        contract_functions = [
            function for function in functions
            if contract.body_start <= function.start < contract.end
        ]
        payable_entries = [
            function for function in contract_functions
            if RE_PAYABLE.search(function.header)
            and (
                function.name in {"constructor", "receive", "fallback"}
                or RE_PUBLIC_OR_EXTERNAL.search(function.header)
            )
        ]
        if not payable_entries:
            continue
        body = clean[contract.body_start:contract.end]
        if RE_OUTGOING_ETHER.search(body):
            continue
        entry = payable_entries[0]
        hits.append((
            entry.start,
            code[entry.start:entry.body_start].strip(),
            0.55,
            f"Contract '{contract.name}' nhận được Ether nhưng không có đường gửi Ether ra",
        ))
    return _build_semantic_finding(
        code, "STATIC-LOCKED-ETHER", "locked_ether", "Medium", hits
    )


# [Kiểm tra ngữ nghĩa] Dùng msg.value trong vòng lặp -> cùng một khoản ETH bị tính nhiều lần.
def _detect_msg_value_in_loop(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    hits: List[Tuple[int, str, float, str]] = []
    for function in _iter_functions(code):
        if not RE_PAYABLE.search(function.header):
            continue
        for loop_start, loop_end in _iter_loop_ranges(
            code, function.body_start, function.end
        ):
            loop = clean[loop_start:loop_end]
            residual = RE_MSG_VALUE_ZERO_CHECK.sub("", loop)
            value_match = RE_MSG_VALUE.search(residual)
            if not value_match:
                continue
            pos = loop_start + value_match.start()
            hits.append((
                pos,
                "msg.value",
                0.55,
                f"Hàm payable '{function.name}' tái sử dụng msg.value trong vòng lặp",
            ))
    return _build_semantic_finding(
        code,
        "STATIC-MSG-VALUE-LOOP",
        "msg_value_in_loop",
        "Medium",
        hits,
    )


# [Kiểm tra ngữ nghĩa] SWC-112: hàm payable gọi delegatecall trong vòng lặp (msg.value được tái sử dụng).
def _detect_delegatecall_in_loop(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    hits: List[Tuple[int, str, float, str]] = []
    for function in _iter_functions(code):
        if not (
            RE_PAYABLE.search(function.header)
            and RE_PUBLIC_OR_EXTERNAL.search(function.header)
        ):
            continue
        for loop_start, loop_end in _iter_loop_ranges(
            code, function.body_start, function.end
        ):
            call = RE_DELEGATECALL.search(clean[loop_start:loop_end])
            if not call:
                continue
            pos = loop_start + call.start()
            hits.append((
                pos,
                code[pos:pos + len(call.group())],
                0.65,
                f"Hàm payable '{function.name}' thực hiện delegatecall trong vòng lặp",
            ))
    return _build_semantic_finding(
        code,
        "SWC-112",
        "delegatecall_in_loop",
        "High",
        hits,
    )


# Tìm tham số address được ghi vào state hoặc dùng làm đích chuyển tiền mà không kiểm tra != address(0).
# Chỉ báo khi tham số thực sự "chảy" vào state/lời gọi (giống Slither missing-zero-check), tránh báo tràn lan.
def _missing_zero_address_hits(
    code: str,
) -> List[Tuple[int, str, float, str]]:
    clean = _clean_code(code)
    hits: List[Tuple[int, str, float, str]] = []
    for function in _iter_functions(code):
        parameter_types = _collect_symbol_types(function.params)
        address_parameters = [
            name for name, type_name in parameter_types.items()
            if type_name.lower() == "address"
        ]
        if not address_parameters:
            continue
        body = clean[function.body_start:function.end]
        local_names = set(_collect_symbol_types(body))
        for parameter in address_parameters:
            require_check = re_zero_address_require(parameter).search(body)
            reverting_if = re_zero_address_revert_if(parameter).search(body)
            if require_check or reverting_if:
                continue

            sensitive_pos: Optional[int] = None
            assignment = re_assignment_from(parameter).search(body)
            if assignment:
                lhs = RE_IDENTIFIER.match(assignment.group("lhs"))
                lhs_name = lhs.group() if lhs else ""
                if lhs_name not in local_names and lhs_name not in parameter_types:
                    sensitive_pos = assignment.start()

            destination = re_call_target(parameter).search(body)
            if sensitive_pos is None and destination:
                sensitive_pos = destination.start()
            if sensitive_pos is None:
                continue
            pos = function.body_start + sensitive_pos
            hits.append((
                pos,
                parameter,
                0.40,
                f"Address '{parameter}' được ghi vào state/dùng làm call target mà chưa kiểm tra zero address",
            ))
    return hits


# [Kiểm tra ngữ nghĩa] Đóng gói kết quả _missing_zero_address_hits thành finding.
def _detect_missing_zero_address(code: str) -> Optional[VulnerabilityFinding]:
    return _build_semantic_finding(
        code,
        "STATIC-MISSING-ZERO-ADDRESS",
        "missing_zero_address_validation",
        "Low",
        _missing_zero_address_hits(code),
    )


# [Kiểm tra ngữ nghĩa] Trong assembly (Yul), shl/shr bị đảo thứ tự tham số (shift là biến, value là hằng).
def _detect_incorrect_shift_in_assembly(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    hits: List[Tuple[int, str, float, str]] = []
    for assembly in RE_ASSEMBLY_BLOCK.finditer(clean):
        open_pos = assembly.end() - 1
        close_pos = _balanced_end(clean, open_pos)
        if close_pos is None:
            continue
        for shift in RE_YUL_SHIFT_WITH_CONSTANT_VALUE.finditer(clean, open_pos + 1, close_pos):
            hits.append((
                shift.start(),
                code[shift.start():shift.end()],
                0.75,
                "Tham số shift trong Yul có vẻ bị đảo: shift amount là biến còn value là hằng",
            ))
    return _build_semantic_finding(
        code,
        "STATIC-INCORRECT-SHIFT",
        "incorrect_shift_in_assembly",
        "High",
        hits,
    )


# [Kiểm tra ngữ nghĩa] Gán độ dài mảng storage (arr.length = ...) bằng giá trị do người dùng điều khiển.
def _detect_controlled_array_length(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    state_arrays = {
        name for name, type_name in _collect_state_symbol_types(code).items()
        if "[]" in type_name
    }
    hits: List[Tuple[int, str, float, str]] = []
    if not state_arrays:
        return None
    for function in _iter_functions(code):
        if not RE_PUBLIC_OR_EXTERNAL.search(function.header):
            continue
        parameters = set(_parse_parameter_details(function.params))
        if not parameters:
            continue
        for assignment in RE_ARRAY_LENGTH_ASSIGNMENT.finditer(
            clean, function.body_start, function.end
        ):
            if assignment.group("array") not in state_arrays:
                continue
            rhs = assignment.group("rhs")
            tainted_by_parameter = next(
                (
                    parameter for parameter in parameters
                    if re_word(parameter).search(rhs)
                ),
                None,
            )
            if not tainted_by_parameter:
                continue
            hits.append((
                assignment.start(),
                code[assignment.start():assignment.end()],
                0.70,
                f"Độ dài storage array phụ thuộc input '{tainted_by_parameter}' của caller",
            ))
    return _build_semantic_finding(
        code,
        "STATIC-CONTROLLED-ARRAY-LENGTH",
        "controlled_array_length",
        "High",
        hits,
    )


# [Kiểm tra ngữ nghĩa] Truyền mảng storage vào hàm nhận tham số "memory" rồi sửa -> sửa bản sao, state không đổi.
def _detect_array_by_reference(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    functions = list(_iter_functions(code))
    modifying_parameters: List[Tuple[SolidityFunction, int, int, str]] = []
    for function in functions:
        details = _parse_parameter_details(function.params)
        for index, (name, (type_name, location)) in enumerate(details.items()):
            if "[" not in type_name or location == "storage":
                continue
            body = clean[function.body_start:function.end]
            if re_indexed_write(name).search(body):
                modifying_parameters.append((function, len(details), index, name))

    state_arrays = {
        name for name, type_name in _collect_state_symbol_types(code).items()
        if "[" in type_name
    }
    if not modifying_parameters or not state_arrays:
        return None

    hits: List[Tuple[int, str, float, str]] = []
    seen_positions: set[int] = set()
    for caller in functions:
        caller_body = clean[caller.body_start:caller.end]
        storage_arrays = set(state_arrays)
        storage_arrays.update(
            match.group("name")
            for match in RE_STORAGE_ARRAY_LOCAL.finditer(caller_body)
        )
        for callee, arity, parameter_index, parameter_name in modifying_parameters:
            for call in re_call(callee.name).finditer(clean, caller.body_start, caller.end):
                open_pos = clean.find("(", call.start(), call.end())
                close_pos = _balanced_end(clean, open_pos, "(", ")")
                if close_pos is None or close_pos > caller.end:
                    continue
                arguments = _split_top_level_arguments(code[open_pos + 1:close_pos])
                if len(arguments) != arity or parameter_index >= len(arguments):
                    continue
                argument = arguments[parameter_index].strip()
                if argument not in storage_arrays or call.start() in seen_positions:
                    continue
                seen_positions.add(call.start())
                hits.append((
                    call.start(),
                    code[call.start():close_pos + 1],
                    0.70,
                    f"Storage array '{argument}' được truyền vào '{callee.name}' qua tham số '{parameter_name}' không phải storage",
                ))
    return _build_semantic_finding(
        code,
        "STATIC-ARRAY-BY-REFERENCE",
        "array_by_reference",
        "High",
        hits,
    )


# [Kiểm tra ngữ nghĩa] Contract có cả constructor() lẫn hàm trùng tên contract (kiểu cũ) -> nhập nhằng constructor.
def _detect_multiple_constructors(code: str) -> Optional[VulnerabilityFinding]:
    functions = list(_iter_functions(code))
    hits: List[Tuple[int, str, float, str]] = []
    for contract in _iter_contracts(code):
        declared = [
            function for function in functions
            if contract.body_start <= function.start < contract.end
        ]
        modern = [function for function in declared if function.name == "constructor"]
        legacy = [function for function in declared if function.name == contract.name]
        if not modern or not legacy:
            continue
        first = modern[0]
        hits.append((
            first.start,
            code[first.start:first.body_start].strip(),
            0.80,
            f"Contract '{contract.name}' khai báo đồng thời constructor() và function {contract.name}()",
        ))
    return _build_semantic_finding(
        code,
        "STATIC-MULTIPLE-CONSTRUCTORS",
        "multiple_constructor_schemes",
        "High",
        hits,
    )


# [Kiểm tra ngữ nghĩa] Mapping public trỏ tới struct chứa mảng/struct lồng (getter tự sinh lỗi với solc cũ).
def _detect_public_mapping_nested(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    if not RE_PRAGMA_0_4.search(clean):
        return None
    structs = _collect_struct_members(code)
    if not structs:
        return None
    hits: List[Tuple[int, str, float, str]] = []
    for match in RE_PUBLIC_MAPPING_VALUE_TYPE.finditer(clean):
        value_type = match.group("value")
        if value_type not in structs:
            continue
        member_types, _ = structs[value_type]
        if not any(RE_ARRAY_SUFFIX.sub('', item) in structs for item in member_types):
            continue
        hits.append((
            match.start(),
            code[match.start():match.end()],
            0.75,
            f"Public mapping '{match.group('name')}' trả về struct lồng nhau sai trên Solidity 0.4.x",
        ))
    return _build_semantic_finding(
        code,
        "STATIC-PUBLIC-MAPPING-NESTED",
        "public_mapping_with_nested_struct",
        "High",
        hits,
    )


# [Kiểm tra ngữ nghĩa] delete một struct có chứa mapping -> mapping bên trong KHÔNG bị xoá.
def _detect_mapping_deletion(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    structs = _collect_struct_members(code)
    if not structs:
        return None
    symbol_types = _collect_symbol_types(code)
    mapping_values = _collect_mapping_value_types(code)
    hits: List[Tuple[int, str, float, str]] = []
    for match in RE_DELETE_STATEMENT.finditer(clean):
        base = match.group("base")
        type_name = mapping_values.get(base) or symbol_types.get(base, "")
        struct_name = RE_ARRAY_SUFFIX.sub('', type_name)
        if not _struct_contains_mapping(struct_name, structs):
            continue
        hits.append((
            match.start(),
            code[match.start():match.end()],
            0.65,
            f"delete không xóa dữ liệu mapping nằm trong struct '{struct_name}'",
        ))
    return _build_semantic_finding(
        code,
        "STATIC-MAPPING-DELETION",
        "mapping_deletion_in_struct",
        "Medium",
        hits,
    )


# Khoảng giá trị [min, max] của kiểu uintN/intN.
def _integer_type_range(type_name: str) -> Optional[Tuple[int, int]]:
    match = RE_INTEGER_TYPE.fullmatch(type_name)
    if not match:
        return None
    bits = int(match.group("bits") or "256")
    if bits < 8 or bits > 256 or bits % 8:
        return None
    if match.group("unsigned"):
        return 0, (2 ** bits) - 1
    return -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1


# Phép so sánh với hằng có luôn đúng/luôn sai theo khoảng giá trị của kiểu không.
def _range_makes_comparison_constant(
    low: int, high: int, operator: str, constant: int
) -> bool:
    if operator == "<":
        return high < constant or low >= constant
    if operator == ">":
        return low > constant or high <= constant
    if operator == "<=":
        return high <= constant or low > constant
    if operator == ">=":
        return low >= constant or high < constant
    return False


# [Kiểm tra ngữ nghĩa] So sánh luôn đúng/luôn sai do kiểu (vd uint >= 0, uint8 < 300).
def _detect_type_based_tautology(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    flip = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}
    hits: List[Tuple[int, str, float, str]] = []
    state_types = _collect_state_symbol_types(code)
    for function in _iter_functions(code):
        symbol_types = dict(state_types)
        symbol_types.update({
            name: details[0]
            for name, details in _parse_parameter_details(function.params).items()
        })
        symbol_types.update(
            _collect_symbol_types(code[function.body_start:function.end])
        )
        for match in RE_ORDER_COMPARISON.finditer(clean, function.body_start, function.end):
            left, right = match.group("left"), match.group("right")
            variable: Optional[str] = None
            constant_text: Optional[str] = None
            operator = match.group("operator")
            if left in symbol_types and RE_INTEGER_LITERAL.fullmatch(right):
                variable, constant_text = left, right
            elif right in symbol_types and RE_INTEGER_LITERAL.fullmatch(left):
                variable, constant_text = right, left
                operator = flip[operator]
            if variable is None or constant_text is None:
                continue
            value_range = _integer_type_range(symbol_types[variable])
            if value_range is None:
                continue
            constant = int(constant_text, 0)
            if not _range_makes_comparison_constant(*value_range, operator, constant):
                continue
            hits.append((
                match.start(),
                code[match.start():match.end()],
                0.65,
                f"So sánh luôn đúng hoặc luôn sai theo miền giá trị của {symbol_types[variable]}",
            ))
    return _build_semantic_finding(
        code,
        "STATIC-TYPE-TAUTOLOGY",
        "type_based_tautology",
        "Medium",
        hits,
    )


# [Kiểm tra ngữ nghĩa] Dùng hằng true/false trong điều kiện (if (x || true)) làm điều kiện cố định/dư thừa.
def _detect_boolean_constant_misuse(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    hits: List[Tuple[int, str, float, str]] = []
    seen: set[int] = set()
    for pattern in (RE_IF_CONSTANT_CONDITION, RE_BOOLEAN_CONSTANT_OPERAND):
        for match in pattern.finditer(clean):
            if match.start() in seen:
                continue
            seen.add(match.start())
            hits.append((
                match.start(),
                code[match.start():match.end()],
                0.55,
                "Boolean constant làm điều kiện trở nên cố định hoặc dư thừa",
            ))
    return _build_semantic_finding(
        code,
        "STATIC-BOOLEAN-CONSTANT",
        "boolean_constant_misuse",
        "Medium",
        hits,
    )


# [Kiểm tra ngữ nghĩa] So sánh bằng chặt (==) với this.balance / balanceOf -> kẻ tấn công gửi ETH để phá điều kiện.
def _detect_dangerous_balance_equality(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    hits: List[Tuple[int, str, float, str]] = []
    for function in _iter_functions(code):
        body = clean[function.body_start:function.end]
        tainted_names = {
            match.group("name")
            for match in RE_BALANCE_ASSIGNMENT.finditer(body)
        }
        for equality in RE_STRICT_EQUALITY.finditer(body):
            absolute = function.body_start + equality.start()
            left_boundaries = [
                clean.rfind(delimiter, function.body_start, absolute)
                for delimiter in (";", "{", "}")
            ]
            start = max(left_boundaries)
            start = start + 1 if start >= function.body_start else function.body_start
            right_positions = [
                position for position in (
                    clean.find(";", absolute, function.end),
                    clean.find("{", absolute, function.end),
                    clean.find("}", absolute, function.end),
                ) if position >= 0
            ]
            end = min(right_positions) if right_positions else function.end
            expression = clean[start:end]
            if not RE_BALANCE_SOURCE.search(expression) and not any(
                re_word(name).search(expression)
                for name in tainted_names
            ):
                continue
            hits.append((
                absolute,
                "==",
                0.60,
                "Strict equality trên Ether/token balance có thể bị phá vỡ bởi thay đổi balance ngoài dự kiến",
            ))
    return _build_semantic_finding(
        code,
        "STATIC-DANGEROUS-BALANCE-EQUALITY",
        "dangerous_balance_equality",
        "Medium",
        hits,
    )


# [Kiểm tra ngữ nghĩa] Ép số nguyên sang enum không kiểm tra phạm vi (solc < 0.4.5 không revert).
def _detect_dangerous_enum_conversion(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    if not RE_PRAGMA_0_4_0_TO_0_4_4.search(clean):
        return None
    enum_names = {
        match.group("name")
        for match in RE_ENUM_OPENING.finditer(clean)
    }
    hits: List[Tuple[int, str, float, str]] = []
    for function in _iter_functions(code):
        for enum_name in enum_names:
            for conversion in re_enum_conversion(enum_name).finditer(
                clean, function.body_start, function.end
            ):
                hits.append((
                    conversion.start(),
                    code[conversion.start():conversion.end()],
                    0.65,
                    f"Ép kiểu sang enum '{enum_name}' có thể nhận giá trị ngoài miền trên solc <= 0.4.4",
                ))
    return _build_semantic_finding(
        code,
        "STATIC-ENUM-CONVERSION",
        "dangerous_enum_conversion",
        "Medium",
        hits,
    )


# Tìm các lệnh Yul return(offset, size) bên trong một hàm.
def _assembly_return_positions(
    code: str, function: SolidityFunction
) -> List[Tuple[int, int]]:
    """Return Yul return(offset, size) ranges inside one function."""
    clean = _clean_code(code)
    results: List[Tuple[int, int]] = []
    for assembly in RE_ASSEMBLY_BLOCK.finditer(
        clean, function.body_start, function.end
    ):
        open_pos = assembly.end() - 1
        close_pos = _balanced_end(clean, open_pos)
        if close_pos is None or close_pos > function.end:
            continue
        for call in RE_YUL_RETURN_CALL.finditer(clean, open_pos, close_pos):
            results.append((call.start(), call.end()))
    return results


# [Kiểm tra ngữ nghĩa] Hàm internal/private có 2 giá trị trả về mà dùng Yul return(...) -> dừng TOÀN BỘ execution; thường ý định là "leave".
def _detect_return_instead_of_leave(code: str) -> Optional[VulnerabilityFinding]:
    hits: List[Tuple[int, str, float, str]] = []
    for function in _iter_functions(code):
        if not RE_INTERNAL_OR_PRIVATE.search(function.header):
            continue
        returns = RE_RETURNS_CLAUSE.search(function.header)
        if not returns or len(_split_top_level_arguments(returns.group("values"))) != 2:
            continue
        for start, end in _assembly_return_positions(code, function):
            hits.append((
                start,
                code[start:end],
                0.65,
                f"Hàm '{function.name}' có hai return value nhưng Yul return() dừng toàn bộ execution; thường phải dùng leave",
            ))
    return _build_semantic_finding(
        code,
        "STATIC-RETURN-INSTEAD-OF-LEAVE",
        "return_instead_of_leave",
        "High",
        hits,
    )


# [Kiểm tra ngữ nghĩa] Gọi một hàm có Yul return() bên trong -> execution dừng luôn, code phía sau lời gọi không chạy.
def _detect_incorrect_assembly_return(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    functions = list(_iter_functions(code))
    halting_names = {
        function.name for function in functions
        if _assembly_return_positions(code, function)
    }
    if not halting_names:
        return None

    changed = True
    while changed:
        changed = False
        for function in functions:
            body = clean[function.body_start:function.end]
            if function.name in halting_names:
                continue
            if any(
                re_call(name).search(body)
                for name in halting_names
            ):
                halting_names.add(function.name)
                changed = True

    hits: List[Tuple[int, str, float, str]] = []
    seen: set[int] = set()
    for function in functions:
        for target in halting_names:
            for call in re_call(target).finditer(clean, function.body_start, function.end):
                open_pos = clean.find("(", call.start(), call.end())
                close_pos = _balanced_end(clean, open_pos, "(", ")")
                if close_pos is None or close_pos > function.end:
                    continue
                trailing = clean[close_pos + 1:function.end].strip(" \t\r\n;")
                if not trailing or call.start() in seen:
                    continue
                seen.add(call.start())
                hits.append((
                    call.start(),
                    code[call.start():close_pos + 1],
                    0.65,
                    f"Lời gọi '{target}' có thể dừng execution bằng Yul return() trước phần code phía sau",
                ))
    return _build_semantic_finding(
        code,
        "STATIC-INCORRECT-ASSEMBLY-RETURN",
        "incorrect_assembly_return",
        "High",
        hits,
    )


# Pragma có pin đúng các bản solc dính bug encode storage array năm 2019 không.
def _uses_exact_vulnerable_storage_array_compiler(code: str) -> bool:
    """Recognize explicitly pinned solc versions affected by 2019 storage bugs."""
    clean = _clean_code(code)
    versions = [
        (int(match.group("minor")), int(match.group("patch")))
        for match in RE_PINNED_PRAGMA_0_4_OR_0_5.finditer(clean)
    ]
    return any(
        (minor == 4 and 7 <= patch <= 25)
        or (minor == 5 and 0 <= patch <= 9)
        for minor, patch in versions
    )


# [Kiểm tra ngữ nghĩa] ABIEncoderV2 encode sai mảng struct trong storage trên các solc bị lỗi.
def _detect_storage_abiencoderv2_array(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    if not (
        _uses_exact_vulnerable_storage_array_compiler(code)
        and RE_PRAGMA_ABIENCODERV2.search(clean)
    ):
        return None
    structs = _collect_struct_members(code)
    state_types = _collect_state_symbol_types(code)
    vulnerable_arrays = {
        name for name, type_name in state_types.items()
        if type_name.count("[") >= 2
        or ("[" in type_name and RE_ARRAY_SUFFIX.sub('', type_name) in structs)
    }
    hits: List[Tuple[int, str, float, str]] = []
    for call in RE_ABI_ENCODE_CALL.finditer(clean):
        open_pos = clean.find("(", call.start(), call.end())
        close_pos = _balanced_end(clean, open_pos, "(", ")")
        if close_pos is None:
            continue
        arguments = _split_top_level_arguments(code[open_pos + 1:close_pos])
        affected = next(
            (argument.strip() for argument in arguments if argument.strip() in vulnerable_arrays),
            None,
        )
        if not affected:
            continue
        hits.append((
            call.start(),
            code[call.start():close_pos + 1],
            0.75,
            f"ABIEncoderV2 có thể encode sai storage array '{affected}' trên compiler đã pin",
        ))
    return _build_semantic_finding(
        code,
        "STATIC-ABIENCODERV2-STORAGE-ARRAY",
        "storage_abiencoderv2_array",
        "High",
        hits,
    )


# Lấy kiểu gốc intN của một kiểu mảng số nguyên có dấu.
def _signed_integer_base(type_name: str) -> Optional[str]:
    match = RE_SIGNED_INT_ARRAY_TYPE.match(type_name)
    return match.group("base") if match else None


# [Kiểm tra ngữ nghĩa] Gán toàn bộ mảng số nguyên có dấu (int[]) trong storage trên solc đã pin dính bug -> ghi sai giá trị.
def _detect_storage_signed_integer_array(code: str) -> Optional[VulnerabilityFinding]:
    if not _uses_exact_vulnerable_storage_array_compiler(code):
        return None
    clean = _clean_code(code)
    state_types = _collect_state_symbol_types(code)
    symbol_types = _collect_symbol_types(code)
    vulnerable_arrays = {
        name: type_name for name, type_name in state_types.items()
        if _signed_integer_base(type_name)
    }
    hits: List[Tuple[int, str, float, str]] = []
    for name, target_type in vulnerable_arrays.items():
        for assignment in re_assignment_to(name).finditer(clean):
            rhs = assignment.group("rhs").strip()
            vulnerable = rhs.startswith("[")
            if not vulnerable and RE_IDENTIFIER.fullmatch(rhs):
                source_type = symbol_types.get(rhs, "")
                source_base = _signed_integer_base(source_type)
                vulnerable = bool(
                    source_base and source_base != _signed_integer_base(target_type)
                )
            if not vulnerable:
                continue
            hits.append((
                assignment.start(),
                code[assignment.start():assignment.end()],
                0.65,
                f"Gán toàn bộ signed storage array '{name}' kích hoạt compiler bug của solc đã pin",
            ))
    return _build_semantic_finding(
        code,
        "STATIC-SIGNED-STORAGE-ARRAY",
        "signed_storage_array_compiler_bug",
        "High",
        hits,
    )


# Lấy danh sách contract cha trực tiếp từ header "contract A is B, C".
def _direct_base_names(header: str) -> List[str]:
    inheritance = RE_INHERITANCE_LIST.search(header)
    if not inheritance:
        return []
    names = []
    for base in _split_top_level_arguments(inheritance.group("bases")):
        match = RE_LEADING_IDENTIFIER.match(base)
        if match:
            names.append(match.group(1))
    return names


# [Kiểm tra ngữ nghĩa] Contract con khai báo lại biến trạng thái trùng tên contract cha (che biến, sai logic).
def _detect_state_variable_shadowing(code: str) -> Optional[VulnerabilityFinding]:
    contracts = {contract.name: contract for contract in _iter_contracts(code)}
    declarations = {
        name: _collect_contract_state_declarations(code, contract)
        for name, contract in contracts.items()
    }

    # Đệ quy gom mọi biến trạng thái được kế thừa từ các contract cha (có chống vòng lặp).
    def inherited_states(contract_name: str, visiting: Optional[set[str]] = None) -> set[str]:
        if contract_name not in contracts:
            return set()
        seen = set(visiting or ())
        if contract_name in seen:
            return set()
        seen.add(contract_name)
        states: set[str] = set()
        for base in _direct_base_names(contracts[contract_name].header):
            states.update(declarations.get(base, {}))
            states.update(inherited_states(base, seen))
        return states

    hits: List[Tuple[int, str, float, str]] = []
    for contract_name, state_variables in declarations.items():
        inherited = inherited_states(contract_name)
        for variable, (_, position, _) in state_variables.items():
            if variable == "__gap" or variable not in inherited:
                continue
            hits.append((
                position,
                variable,
                0.75,
                f"State variable '{variable}' trong '{contract_name}' che khuất biến cùng tên từ base contract",
            ))
    return _build_semantic_finding(
        code,
        "STATIC-STATE-SHADOWING",
        "state_variable_shadowing",
        "High",
        hits,
    )


# [Kiểm tra ngữ nghĩa] Biến address trạng thái không bao giờ được gán mà vẫn được dùng làm đích chuyển tiền.
def _detect_uninitialized_state_address(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    contracts = list(_iter_contracts(code))
    inherited_base_names = {
        base for contract in contracts for base in _direct_base_names(contract.header)
    }
    hits: List[Tuple[int, str, float, str]] = []
    for contract in contracts:
        if contract.name in inherited_base_names or _direct_base_names(contract.header):
            continue
        body = clean[contract.body_start:contract.end]
        for name, (type_name, position, initialized) in (
            _collect_contract_state_declarations(code, contract).items()
        ):
            if type_name != "address" or initialized:
                continue
            write = re_state_write(name).search(body)
            if write:
                continue
            dangerous_read = re_ether_destination(name).search(body)
            if not dangerous_read:
                continue
            hits.append((
                position,
                name,
                0.75,
                f"State address '{name}' không bao giờ được khởi tạo nhưng được dùng làm đích chuyển Ether/call",
            ))
    return _build_semantic_finding(
        code,
        "STATIC-UNINITIALIZED-STATE-ADDRESS",
        "uninitialized_state_address",
        "High",
        hits,
    )


# Chạy lần lượt 25 kiểm tra ngữ nghĩa _detect_* ở trên, trả về danh sách finding phát hiện được.
# Mục đích: bổ sung các lỗi mà regex đơn thuần không bắt được (cài lại logic tương tự Slither, không gọi Slither).
def _run_semantic_checks(code: str) -> List[VulnerabilityFinding]:
    detectors = (
        _detect_bidirectional_controls,
        _detect_incorrect_exponentiation,
        _detect_encode_packed_collision,
        _detect_arbitrary_erc20_transfer_from,
        _detect_unchecked_erc20_transfer,
        _detect_locked_ether,
        _detect_msg_value_in_loop,
        _detect_delegatecall_in_loop,
        _detect_missing_zero_address,
        _detect_incorrect_shift_in_assembly,
        _detect_controlled_array_length,
        _detect_array_by_reference,
        _detect_multiple_constructors,
        _detect_public_mapping_nested,
        _detect_mapping_deletion,
        _detect_type_based_tautology,
        _detect_boolean_constant_misuse,
        _detect_dangerous_balance_equality,
        _detect_dangerous_enum_conversion,
        _detect_return_instead_of_leave,
        _detect_incorrect_assembly_return,
        _detect_storage_abiencoderv2_array,
        _detect_storage_signed_integer_array,
        _detect_state_variable_shadowing,
        _detect_uninitialized_state_address,
    )
    findings: List[VulnerabilityFinding] = []
    for detector in detectors:
        finding = detector(code)
        if finding is not None:
            findings.append(finding)
    return findings


# ──────────────────────────────────────────────────────────────────────────────
# StaticAnalyzer – class chính
# ──────────────────────────────────────────────────────────────────────────────

# Bộ phân tích tĩnh chính: chạy bảng luật VULNERABILITY_REGISTRY (regex có trọng số + negative pattern)
# + 25 kiểm tra ngữ nghĩa, trả về các finding theo loại lỗ hổng (static_runner chuẩn hoá và tính điểm).
class StaticAnalyzer:
    """
    Phân tích tĩnh Smart Contract Solidity.

    Cách dùng:
        findings = StaticAnalyzer().analyze(solidity_code)["findings"]   # Dict[str, VulnerabilityFinding]
    """

    # Gắn bảng luật VULNERABILITY_REGISTRY.
    def __init__(self):
        self._registry = VULNERABILITY_REGISTRY

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    # [HÀM CHÍNH] Phân tích một đoạn mã (cả contract hoặc một hàm):
    # (1) với mỗi luật: khớp regex, trừ điểm theo negative pattern / guard modifier, confirmed nếu điểm > 0.25;
    # (2) cộng thêm kết quả 25 kiểm tra ngữ nghĩa. Cùng vuln_type thì gộp các lần khớp.
    def analyze(self, contract_code: str, guard_modifiers: Optional[set[str]] = None) -> Dict:
        """Phân tích contract_code, trả {"findings": Dict[vuln_type, VulnerabilityFinding]}."""
        clean = _clean_code(contract_code)
        if guard_modifiers is None:
            guard_modifiers = extract_guard_modifiers(contract_code)
        functions = list(_iter_functions(clean))

        findings: Dict[str, VulnerabilityFinding] = {}

        for entry in self._registry:
            finding = self._run_pattern_check(
                entry,
                clean,
                contract_code,
                guard_modifiers=guard_modifiers,
                functions=functions,
            )
            if finding.matches:
                # Áp dụng negative patterns để giảm score
                neg_reduction = self._apply_negative_patterns(
                    entry, clean, finding.raw_score, guard_modifiers=guard_modifiers
                )
                finding.confirmed = (
                    finding.raw_score - neg_reduction
                ) > 0.25
                key = entry["vuln_type"]
                # Nếu cùng vuln_type xuất hiện nhiều lần, merge
                if key in findings:
                    findings[key].matches.extend(finding.matches)
                    findings[key].notes.extend(finding.notes)
                else:
                    findings[key] = finding

        # Focused semantic checks derived from detector behavior, implemented
        # locally without importing or invoking Slither.
        for finding in _run_semantic_checks(contract_code):
            key = finding.vuln_type
            if key in findings:
                findings[key].matches.extend(finding.matches)
                findings[key].notes.extend(finding.notes)
                findings[key].confirmed = (
                    findings[key].confirmed or finding.confirmed
                )
            else:
                findings[key] = finding

        return {"findings": findings}

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    # Chạy các regex của MỘT luật trên mã sạch. Với SWC-101/105/106, bỏ qua các match nằm trong hàm
    # đã được bảo vệ (is_function_guarded) hoặc hàm burn token của chính msg.sender (giảm false positive).
    def _run_pattern_check(
        self,
        entry: Dict,
        clean_code: str,
        original_code: str,
        guard_modifiers: Optional[set[str]] = None,
        functions: Optional[List[SolidityFunction]] = None,
    ) -> VulnerabilityFinding:
        finding = VulnerabilityFinding(
            vuln_id=entry["vuln_id"],
            vuln_type=entry["vuln_type"],
            risk_level=entry["risk_level"],
        )

        vuln_id = entry.get("vuln_id")
        is_access_sensitive = vuln_id in {"SWC-101", "SWC-105", "SWC-106"}

        for pat_tuple in entry.get("patterns", []):
            regex, weight, desc = pat_tuple
            compiled = re.compile(regex, re.DOTALL | re.MULTILINE)
            for m in compiled.finditer(clean_code):
                # Bỏ qua finding nếu nằm trong hàm được bảo vệ hoặc burn số dư chính mình
                if is_access_sensitive:
                    m_pos = m.start()
                    containing_func = (
                        next((f for f in functions if f.start <= m_pos <= f.end), None)
                        if functions
                        else None
                    )
                    if containing_func:
                        func_body = clean_code[containing_func.body_start:containing_func.end]
                        if is_function_guarded(containing_func.header, func_body, guard_modifiers):
                            continue
                        if containing_func.name == "burn" and (
                            "balances[msg.sender]" in func_body
                            or "burner = msg.sender" in func_body
                            or "require(_value <= balances[msg.sender])" in func_body
                        ):
                            continue
                    elif not functions and clean_code.strip().startswith("function"):
                        hdr_m = RE_FUNCTION_HEADER_PREFIX.search(clean_code)
                        if hdr_m:
                            hdr = clean_code[:hdr_m.end()]
                            bdy = clean_code[hdr_m.end():]
                            if is_function_guarded(hdr, bdy, guard_modifiers):
                                continue
                            if "function burn" in hdr and (
                                "balances[msg.sender]" in bdy
                                or "burner = msg.sender" in bdy
                            ):
                                continue

                line = _line_at(clean_code, m.start())
                context = _extract_context(original_code, m.start())
                finding.matches.append(PatternMatch(
                    line=line,
                            matched_text=m.group()[:80],
                    context=context,
                    pattern_weight=weight,
                ))
                finding.notes.append(f"L{line}: {desc}")

        return finding

    # Tính lượng trừ điểm: mỗi negative pattern khớp (vd có nonReentrant, SafeMath, require kiểm tra) trừ theo trọng số;
    # có guard modifier thì trừ thêm 0.45 cho SWC-101/105/106.
    def _apply_negative_patterns(
        self,
        entry: Dict,
        clean_code: str,
        raw_score: float,
        guard_modifiers: Optional[set[str]] = None,
    ) -> float:
        """
        Áp dụng negative patterns và trả về lượng giảm trừ cho raw_score.
        """
        reduction = 0.0
        for pat_tuple in entry.get("negative_patterns", []):
            regex, weight, _ = pat_tuple
            if re.search(regex, clean_code, re.DOTALL):
                reduction += weight

        if guard_modifiers and entry.get("vuln_id") in {"SWC-101", "SWC-105", "SWC-106"}:
            for mod in guard_modifiers:
                if re_word(mod).search(clean_code):
                    reduction += 0.45
                    break

        return min(raw_score, reduction)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone runner (dùng để test nhanh)
# ──────────────────────────────────────────────────────────────────────────────

# Hàm tiện ích: tạo StaticAnalyzer và gọi analyze(). Được static_runner.run_static gọi cho cả contract và từng hàm.
def run_static_analysis(
    contract_code: str,
    guard_modifiers: Optional[set[str]] = None,
) -> Dict:
    """Return {"findings": Dict[vuln_type, VulnerabilityFinding]} for one contract or function."""
    return StaticAnalyzer().analyze(contract_code, guard_modifiers=guard_modifiers)
