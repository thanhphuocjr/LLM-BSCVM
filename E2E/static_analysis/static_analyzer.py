"""
static_analyzer.py
──────────────────────────────────────────────────────────────────────────────
Module phân tích tĩnh (Static Analysis) dành cho Detector Agent trong
framework LLM-BSCVM.

Tác giả  : Smart Contract Security Expert
Mô tả    : Phát hiện các lỗ hổng phổ biến trong Solidity bằng cách kết hợp:
           1. Pattern matching dựa trên Regex (nhanh, deterministic)
           2. Phân tích ngữ cảnh (context-aware) để giảm false positive
           3. Phân tích luồng dữ liệu đơn giản (CEI, state-change ordering)
           4. Phân tích metadata (pragma, imports, tên biến/hàm)

Các lỗ hổng được bao phủ (ánh xạ theo RISK_LEVELS trong Config):
  Critical : Reentrancy, Access Control, Unchecked External Calls
  High     : Integer Overflow/Underflow, DoS, Flash Loan, Front-Running
  Medium   : Timestamp Dependence, Block Info Dependence, Gas Limit,
             Transaction Ordering Dependency (TOD), Unsafe Type Casting
  Low      : Outdated Compiler, Naming Conventions, Redundant Code
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Kiểu dữ liệu nội bộ
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PatternMatch:
    """Kết quả một lần khớp pattern."""
    line: int
    column: int
    matched_text: str
    context: str          # ~3 dòng xung quanh để hiển thị
    pattern_weight: float # trọng số đóng góp vào score


@dataclass
class VulnerabilityFinding:
    """Kết quả phát hiện một loại lỗ hổng."""
    vuln_id: str          # vd: "SWC-107"
    vuln_type: str        # vd: "reentrancy"
    risk_level: str       # Critical / High / Medium / Low
    type_weight: float    # trọng số loại lỗ hổng (dùng khi tổng hợp score)
    matches: List[PatternMatch] = field(default_factory=list)
    confirmed: bool = False   # True nếu context-check xác nhận thêm
    notes: List[str] = field(default_factory=list)

    @property
    def raw_score(self) -> float:
        if not self.matches:
            return 0.0
        return min(1.0, sum(m.pattern_weight for m in self.matches))

    @property
    def weighted_score(self) -> float:
        return self.raw_score * self.type_weight


@dataclass(frozen=True)
class SolidityFunction:
    """Lightweight function boundary used by semantic checks."""

    name: str
    params: str
    header: str
    start: int
    body_start: int
    end: int


@dataclass(frozen=True)
class SolidityContract:
    """Lightweight contract boundary used by contract-level checks."""

    name: str
    header: str
    start: int
    body_start: int
    end: int


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
#   type_weight  : trọng số loại (tổng không cần = 1, sẽ chuẩn hoá)
#   patterns     : list[Tuple[regex_str, weight, description]]
#                  weight = đóng góp của pattern này vào raw_score
# ──────────────────────────────────────────────────────────────────────────────

VULNERABILITY_REGISTRY: List[Dict] = [

    # ──────────────────────────────── CRITICAL ────────────────────────────────

    {
        "vuln_id": "SWC-107",
        "vuln_type": "reentrancy",
        "risk_level": "Critical",
        "type_weight": 0.30,
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
        "context_checks": [
            "check_call_before_state_update",
        ],
    },

    {
        "vuln_id": "SWC-105",
        "vuln_type": "unprotected_ether_withdrawal",
        "risk_level": "Critical",
        "type_weight": 0.25,
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
        "type_weight": 0.25,
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
        "type_weight": 0.22,
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
        "type_weight": 0.05,
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
        "type_weight": 0.28,
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
        "type_weight": 0.25,
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
        "type_weight": 0.22,
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
        "type_weight": 0.18,
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
        "type_weight": 0.15,
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
        "type_weight": 0.20,
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
        "type_weight": 0.15,
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
        "type_weight": 0.12,
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
        "type_weight": 0.10,
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
        "type_weight": 0.10,
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
        "type_weight": 0.10,
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
        "type_weight": 0.10,
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
        "type_weight": 0.05,
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
        "type_weight": 0.03,
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
        "type_weight": 0.03,
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
        "type_weight": 0.18,
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
        "type_weight": 0.18,
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
        "type_weight": 0.20,
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
        "type_weight": 0.05,
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
        "type_weight": 0.10,
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
        "type_weight": 0.10,
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
        "type_weight": 0.08,
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

def _get_line_col(code: str, pos: int) -> Tuple[int, int]:
    """Trả về (line, column) từ vị trí character trong chuỗi."""
    prefix = code[:pos]
    line = prefix.count('\n') + 1
    col  = pos - prefix.rfind('\n')
    return line, col


def _extract_context(code: str, pos: int, window: int = 120) -> str:
    """Trích đoạn code xung quanh vị trí pos."""
    start = max(0, pos - window)
    end   = min(len(code), pos + window)
    snippet = code[start:end]
    return snippet.replace('\n', ' ').strip()


def _strip_comments(code: str) -> str:
    """Mask comments while preserving source offsets and line numbers."""
    return re.sub(
        r'/\*[\s\S]*?\*/|//[^\n]*',
        lambda match: _mask_text(match.group()),
        code,
    )


def _strip_strings(code: str) -> str:
    """Mask string literals while preserving source offsets and line numbers."""
    code = re.sub(
        r'"[^"\\]*(?:\\.[^"\\]*)*"',
        lambda match: _mask_text(match.group()),
        code,
    )
    return re.sub(
        r"'[^'\\]*(?:\\.[^'\\]*)*'",
        lambda match: _mask_text(match.group()),
        code,
    )


def _mask_text(value: str) -> str:
    """Replace non-newline characters with spaces so offsets stay stable."""
    return "".join("\n" if char == "\n" else " " for char in value)


def _clean_code(code: str) -> str:
    """Return Solidity code with comments and strings masked."""
    return _strip_comments(_strip_strings(code))


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


def _iter_functions(code: str) -> Iterator[SolidityFunction]:
    """Yield implemented functions without requiring a Solidity compiler."""
    clean = _clean_code(code)
    patterns = (
        re.compile(
            r'\bfunction\s+(?P<name>[A-Za-z_]\w*)\s*'
            r'\((?P<params>[^)]*)\)(?P<header>[^;{}]*)\{',
            re.DOTALL,
        ),
        re.compile(
            r'\b(?P<name>constructor|receive|fallback)\s*'
            r'\((?P<params>[^)]*)\)(?P<header>[^;{}]*)\{',
            re.DOTALL,
        ),
    )
    functions: List[SolidityFunction] = []
    for pattern in patterns:
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


def _iter_contracts(code: str) -> Iterator[SolidityContract]:
    """Yield concrete contract boundaries for contract-level checks."""
    clean = _clean_code(code)
    pattern = re.compile(
        r'\bcontract\s+(?P<name>[A-Za-z_]\w*)'
        r'(?P<header>[^;{}]*)\{',
        re.DOTALL,
    )
    for match in pattern.finditer(clean):
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


def _iter_modifiers(code: str) -> Iterator[SolidityModifier]:
    """Yield modifier boundaries without requiring a Solidity compiler."""
    clean = _clean_code(code)
    pattern = re.compile(
        r'\bmodifier\s+(?P<name>[A-Za-z_]\w*)\s*(?:\((?P<params>[^)]*)\))?(?P<header>[^;{}]*)\{',
        re.DOTALL,
    )
    for match in pattern.finditer(clean):
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
    for m in re.finditer(r'\bmodifier\s+([A-Za-z0-9_]+)', clean):
        name = m.group(1)
        if re.match(r'^(?:only[A-Z]\w*|auth|authorized|restricted)$', name):
            guards.add(name)
    for mod in _iter_modifiers(code):
        body = mod.body
        if (
            re.search(r'require\s*\([^;]*msg\.sender\s*==', body)
            or re.search(r'require\s*\([^;]*==\s*msg\.sender', body)
            or re.search(r'if\s*\([^)]*msg\.sender\s*!=[^)]*\)\s*(?:revert\b|throw\b)', body)
            or re.search(r'if\s*\([^)]*!=\s*msg\.sender[^)]*\)\s*(?:revert\b|throw\b)', body)
            or re.search(r'require\s*\([^;]*\[msg\.sender\]', body)
            or re.search(r'require\s*\([^;]*(?:isOwner|hasRole|authorized)\s*\(', body)
        ):
            guards.add(mod.name)
    return guards


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
    if re.search(r'\b(?:private|internal)\b', header):
        return True
    active_guards = (guard_modifiers or set()) | {
        "onlyOwner", "onlyAdmin", "onlyRole", "onlyMinter",
        "onlyFounder", "authorized", "auth", "restricted"
    }
    for mod in active_guards:
        if re.search(rf'\b{re.escape(mod)}\b', header):
            return True
    for token in re.findall(r'\b([A-Za-z0-9_]+)\b', header):
        if token.startswith("only") and len(token) > 4:
            return True
    if (
        re.search(r'require\s*\([^;]*msg\.sender\s*==', body)
        or re.search(r'require\s*\([^;]*==\s*msg\.sender', body)
        or re.search(r'if\s*\([^)]*msg\.sender\s*!=[^)]*\)\s*(?:revert\b|throw\b)', body)
        or re.search(r'if\s*\([^)]*!=\s*msg\.sender[^)]*\)\s*(?:revert\b|throw\b)', body)
        or re.search(r'require\s*\([^;]*\[msg\.sender\]', body)
        or re.search(r'require\s*\([^;]*(?:isOwner|hasRole|authorized)\s*\(', body)
    ):
        return True
    return False


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


def _collect_symbol_types(code: str) -> Dict[str, str]:
    """Collect simple variable-to-type bindings used by focused checks."""
    clean = _clean_code(code)
    declaration = re.compile(
        r'\b(?P<type>[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])*)\s+'
        r'(?:(?:public|private|internal|external|constant|immutable|memory|'
        r'calldata|storage|payable|indexed)\s+)*'
        r'(?P<name>[A-Za-z_]\w*)\b'
    )
    ignored_types = {
        "contract", "function", "event", "error", "modifier", "return",
        "returns", "import", "pragma", "using", "struct", "enum",
    }
    bindings: Dict[str, str] = {}
    for match in declaration.finditer(clean):
        type_name = re.sub(r'\s+', '', match.group("type"))
        if type_name.lower() in ignored_types:
            continue
        bindings[match.group("name")] = type_name
    return bindings


def _is_dynamic_abi_argument(argument: str, symbol_types: Dict[str, str]) -> bool:
    value = argument.strip()
    if not value:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return True
    identifier = re.fullmatch(r'([A-Za-z_]\w*)', value)
    if not identifier:
        return False
    type_name = symbol_types.get(identifier.group(1), "").replace(" ", "")
    return type_name in {"string", "bytes"} or "[]" in type_name


def _looks_like_erc20(receiver: str, symbol_types: Dict[str, str]) -> bool:
    compact = re.sub(r'\s+', '', receiver)
    cast = re.match(r'(?P<type>[A-Za-z_]\w*)\(', compact)
    if cast:
        return "erc20" in cast.group("type").lower()
    type_name = symbol_types.get(compact, "").lower()
    return "erc20" in type_name


def _parse_parameter_details(params: str) -> Dict[str, Tuple[str, str]]:
    """Return parameter name -> (type, data location)."""
    details: Dict[str, Tuple[str, str]] = {}
    pattern = re.compile(
        r'^\s*(?P<type>[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])*)\s*'
        r'(?P<location>memory|storage|calldata|payable)?\s*'
        r'(?P<name>[A-Za-z_]\w*)\s*$'
    )
    for parameter in _split_top_level_arguments(params):
        match = pattern.match(parameter)
        if not match:
            continue
        details[match.group("name")] = (
            re.sub(r'\s+', '', match.group("type")),
            match.group("location") or "",
        )
    return details


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


def _top_level_contract_statements(code: str) -> Iterator[Tuple[int, str]]:
    """Yield semicolon-terminated state declarations outside nested blocks."""
    for contract in _iter_contracts(code):
        yield from _contract_top_level_statements(code, contract)


def _collect_contract_state_declarations(
    code: str, contract: SolidityContract
) -> Dict[str, Tuple[str, int, bool]]:
    """Return state name -> (type, source position, has initializer)."""
    result: Dict[str, Tuple[str, int, bool]] = {}
    standard = re.compile(
        r'^\s*(?P<type>[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])*)\s+'
        r'(?:(?:public|private|internal|constant|immutable|payable)\s+)*'
        r'(?P<name>[A-Za-z_]\w*)\b(?P<tail>[^;]*);'
    )
    mapping = re.compile(
        r'^\s*mapping\s*\([^;]+\)\s*'
        r'(?:(?:public|private|internal)\s+)*(?P<name>[A-Za-z_]\w*)\b'
    )
    for position, statement in _contract_top_level_statements(code, contract):
        clean_statement = _clean_code(statement)
        match = standard.match(clean_statement)
        if match:
            leading = len(clean_statement) - len(clean_statement.lstrip())
            result[match.group("name")] = (
                re.sub(r'\s+', '', match.group("type")),
                position + leading,
                "=" in match.group("tail"),
            )
            continue
        mapping_match = mapping.match(clean_statement)
        if mapping_match:
            leading = len(clean_statement) - len(clean_statement.lstrip())
            result[mapping_match.group("name")] = (
                "mapping",
                position + leading,
                False,
            )
    return result


def _collect_state_symbol_types(code: str) -> Dict[str, str]:
    """Collect simple top-level state variable declarations."""
    result: Dict[str, str] = {}
    declaration = re.compile(
        r'^\s*(?P<type>[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])*)\s+'
        r'(?:(?:public|private|internal|constant|immutable|payable)\s+)*'
        r'(?P<name>[A-Za-z_]\w*)\b'
    )
    for _, statement in _top_level_contract_statements(code):
        match = declaration.match(_clean_code(statement))
        if match:
            result[match.group("name")] = re.sub(r'\s+', '', match.group("type"))
    return result


def _collect_mapping_value_types(code: str) -> Dict[str, str]:
    """Collect top-level mapping name -> direct value type."""
    result: Dict[str, str] = {}
    pattern = re.compile(
        r'^\s*mapping\s*\([^=;]+=>\s*(?P<value>[A-Za-z_]\w*)\s*\)\s*'
        r'(?:(?:public|private|internal)\s+)*(?P<name>[A-Za-z_]\w*)\b'
    )
    for _, statement in _top_level_contract_statements(code):
        match = pattern.match(_clean_code(statement))
        if match:
            result[match.group("name")] = match.group("value")
    return result


def _collect_struct_members(code: str) -> Dict[str, Tuple[set[str], bool]]:
    """Return struct member types and whether each struct directly has a mapping."""
    clean = _clean_code(code)
    structs: Dict[str, Tuple[set[str], bool]] = {}
    for match in re.finditer(r'\bstruct\s+(?P<name>[A-Za-z_]\w*)\s*\{', clean):
        open_pos = match.end() - 1
        close_pos = _balanced_end(clean, open_pos)
        if close_pos is None:
            continue
        body = code[open_pos + 1:close_pos]
        member_types = set(_collect_symbol_types(body).values())
        structs[match.group("name")] = (
            member_types,
            bool(re.search(r'\bmapping\s*\(', _clean_code(body))),
        )
    return structs


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
        _struct_contains_mapping(re.sub(r'\[.*$', '', type_name), structs, seen)
        for type_name in member_types
    )


def _iter_loop_ranges(code: str, start: int, end: int) -> Iterator[Tuple[int, int]]:
    """Yield for/while ranges inside a source interval."""
    clean = _clean_code(code)
    loop_pattern = re.compile(r'\b(?:for|while)\s*\(')
    for match in loop_pattern.finditer(clean, start, end):
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


def _build_semantic_finding(
    code: str,
    vuln_id: str,
    vuln_type: str,
    risk_level: str,
    type_weight: float,
    hits: List[Tuple[int, str, float, str]],
) -> Optional[VulnerabilityFinding]:
    if not hits:
        return None
    finding = VulnerabilityFinding(
        vuln_id=vuln_id,
        vuln_type=vuln_type,
        risk_level=risk_level,
        type_weight=type_weight,
        confirmed=True,
    )
    for pos, matched_text, weight, description in hits:
        line, col = _get_line_col(code, pos)
        finding.matches.append(PatternMatch(
            line=line,
            column=col,
            matched_text=matched_text[:80],
            context=_extract_context(code, pos),
            pattern_weight=weight,
        ))
        finding.notes.append(f"L{line}: {description}")
    return finding


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
        code, "SWC-130", "right_to_left_override", "High", 0.18, hits
    )


def _detect_incorrect_exponentiation(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    # Mask pragma statements to avoid false positives on caret version specifiers (e.g. pragma solidity ^0.4.25;)
    clean = re.sub(r'pragma\s+solidity\s+[^;]+;', lambda m: ' ' * len(m.group(0)), clean)
    operand = r'(?:0x[0-9A-Fa-f]+|\d+|[A-Za-z_]\w*)'
    pattern = re.compile(
        rf'(?P<left>{operand})\s*\^(?!=)\s*(?P<right>{operand})'
    )
    hits: List[Tuple[int, str, float, str]] = []
    for match in pattern.finditer(clean):
        operands = (match.group("left"), match.group("right"))
        if not any(re.fullmatch(r'\d+', value) for value in operands):
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
        0.16,
        hits,
    )


def _detect_encode_packed_collision(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    symbol_types = _collect_symbol_types(code)
    hits: List[Tuple[int, str, float, str]] = []
    for match in re.finditer(r'\babi\.encodePacked\s*\(', clean):
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
        0.18,
        hits,
    )


def _iter_erc20_calls(
    code: str,
) -> Iterator[Tuple[SolidityFunction, re.Match, str, List[str], int]]:
    clean = _clean_code(code)
    symbol_types = _collect_symbol_types(code)
    call_pattern = re.compile(
        r'(?P<receiver>(?:[A-Za-z_]\w*|'
        r'[A-Za-z_]\w*\s*\([^()]*\)))\s*\.\s*'
        r'(?P<method>transferFrom|transfer)\s*\('
    )
    for function in _iter_functions(code):
        for match in call_pattern.finditer(clean, function.body_start, function.end):
            receiver = match.group("receiver")
            if not _looks_like_erc20(receiver, symbol_types):
                continue
            open_pos = clean.find("(", match.start("method"), match.end())
            close_pos = _balanced_end(clean, open_pos, "(", ")")
            if close_pos is None or close_pos > function.end:
                continue
            arguments = _split_top_level_arguments(code[open_pos + 1:close_pos])
            yield function, match, match.group("method"), arguments, close_pos


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
        0.16,
        hits,
    )


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
        0.15,
        hits,
    )


def _detect_locked_ether(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    functions = list(_iter_functions(code))
    hits: List[Tuple[int, str, float, str]] = []
    outgoing_ether = re.compile(
        r'\.call\s*\{[^}]*\bvalue\s*:'
        r'|\.send\s*\('
        r'|\.transfer\s*\('
        r'|\b(?:selfdestruct|suicide)\s*\('
        r'|\bassembly\s*\{[^}]*\bcall\s*\(',
        re.DOTALL,
    )
    for contract in _iter_contracts(code):
        prefix = clean[max(0, contract.start - 16):contract.start]
        if re.search(r'\babstract\s*$', prefix) or re.search(r'\bis\b', contract.header):
            continue
        contract_functions = [
            function for function in functions
            if contract.body_start <= function.start < contract.end
        ]
        payable_entries = [
            function for function in contract_functions
            if re.search(r'\bpayable\b', function.header)
            and (
                function.name in {"constructor", "receive", "fallback"}
                or re.search(r'\b(?:public|external)\b', function.header)
            )
        ]
        if not payable_entries:
            continue
        body = clean[contract.body_start:contract.end]
        if outgoing_ether.search(body):
            continue
        entry = payable_entries[0]
        hits.append((
            entry.start,
            code[entry.start:entry.body_start].strip(),
            0.55,
            f"Contract '{contract.name}' nhận được Ether nhưng không có đường gửi Ether ra",
        ))
    return _build_semantic_finding(
        code, "STATIC-LOCKED-ETHER", "locked_ether", "Medium", 0.12, hits
    )


def _detect_msg_value_in_loop(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    hits: List[Tuple[int, str, float, str]] = []
    zero_comparison = re.compile(
        r'msg\.value\s*(?:==|!=)\s*0|0\s*(?:==|!=)\s*msg\.value'
    )
    for function in _iter_functions(code):
        if not re.search(r'\bpayable\b', function.header):
            continue
        for loop_start, loop_end in _iter_loop_ranges(
            code, function.body_start, function.end
        ):
            loop = clean[loop_start:loop_end]
            residual = zero_comparison.sub("", loop)
            value_match = re.search(r'\bmsg\.value\b', residual)
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
        0.11,
        hits,
    )


def _detect_delegatecall_in_loop(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    hits: List[Tuple[int, str, float, str]] = []
    for function in _iter_functions(code):
        if not (
            re.search(r'\bpayable\b', function.header)
            and re.search(r'\b(?:public|external)\b', function.header)
        ):
            continue
        for loop_start, loop_end in _iter_loop_ranges(
            code, function.body_start, function.end
        ):
            call = re.search(r'\.delegatecall\s*\(', clean[loop_start:loop_end])
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
        0.16,
        hits,
    )


def _missing_zero_address_hits(
    code: str,
) -> List[Tuple[int, str, float, str]]:
    clean = _clean_code(code)
    hits: List[Tuple[int, str, float, str]] = []
    zero = r'address\s*\(\s*(?:0|0x0+)\s*\)'
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
            escaped = re.escape(parameter)
            require_check = re.search(
                rf'\b(?:require|assert)\s*\([^;]*'
                rf'(?:\b{escaped}\b\s*!=\s*{zero}|{zero}\s*!=\s*\b{escaped}\b)',
                body,
            )
            reverting_if = re.search(
                rf'\bif\s*\([^;{{}}]*'
                rf'(?:\b{escaped}\b\s*==\s*{zero}|{zero}\s*==\s*\b{escaped}\b)'
                rf'[^;{{}}]*\)\s*(?:\{{\s*)?(?:revert|return)\b',
                body,
                re.DOTALL,
            )
            if require_check or reverting_if:
                continue

            sensitive_pos: Optional[int] = None
            assignment = re.search(
                rf'(?P<lhs>\b[A-Za-z_]\w*(?:\s*\[[^\]]+\]|\.[A-Za-z_]\w*)?)'
                rf'\s*=\s*\b{escaped}\b\s*;',
                body,
            )
            if assignment:
                lhs = re.match(r'[A-Za-z_]\w*', assignment.group("lhs"))
                lhs_name = lhs.group() if lhs else ""
                if lhs_name not in local_names and lhs_name not in parameter_types:
                    sensitive_pos = assignment.start()

            destination = re.search(
                rf'(?:\b{escaped}\b\s*\.\s*(?:call|send|transfer)\b'
                rf'|payable\s*\(\s*\b{escaped}\b\s*\)\s*\.'
                rf'|[A-Za-z_]\w*\s*\(\s*\b{escaped}\b\s*\)\s*\.)',
                body,
            )
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


def _detect_missing_zero_address(code: str) -> Optional[VulnerabilityFinding]:
    return _build_semantic_finding(
        code,
        "STATIC-MISSING-ZERO-ADDRESS",
        "missing_zero_address_validation",
        "Low",
        0.06,
        _missing_zero_address_hits(code),
    )


def _detect_incorrect_shift_in_assembly(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    hits: List[Tuple[int, str, float, str]] = []
    assembly_pattern = re.compile(r'\bassembly(?:\s*\([^)]*\))?\s*\{')
    shift_pattern = re.compile(
        r'\b(?:shl|shr|sar)\s*\(\s*'
        r'(?P<shift>[A-Za-z_]\w*)\s*,\s*'
        r'(?P<value>0x[0-9A-Fa-f]+|\d+)\s*\)'
    )
    for assembly in assembly_pattern.finditer(clean):
        open_pos = assembly.end() - 1
        close_pos = _balanced_end(clean, open_pos)
        if close_pos is None:
            continue
        for shift in shift_pattern.finditer(clean, open_pos + 1, close_pos):
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
        0.17,
        hits,
    )


def _detect_controlled_array_length(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    state_arrays = {
        name for name, type_name in _collect_state_symbol_types(code).items()
        if "[]" in type_name
    }
    hits: List[Tuple[int, str, float, str]] = []
    if not state_arrays:
        return None
    assignment_pattern = re.compile(
        r'\b(?P<array>[A-Za-z_]\w*)\.length\s*=\s*(?P<rhs>[^;]+);'
    )
    for function in _iter_functions(code):
        if not re.search(r'\b(?:public|external)\b', function.header):
            continue
        parameters = set(_parse_parameter_details(function.params))
        if not parameters:
            continue
        for assignment in assignment_pattern.finditer(
            clean, function.body_start, function.end
        ):
            if assignment.group("array") not in state_arrays:
                continue
            rhs = assignment.group("rhs")
            tainted_by_parameter = next(
                (
                    parameter for parameter in parameters
                    if re.search(rf'\b{re.escape(parameter)}\b', rhs)
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
        0.17,
        hits,
    )


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
            if re.search(
                rf'\b{re.escape(name)}\s*\[[^\]]+\]\s*'
                rf'(?:=|\+=|-=|\*=|/=|%=|\+\+|--)',
                body,
            ):
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
            for match in re.finditer(
                r'\b[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])+\s+storage\s+'
                r'(?P<name>[A-Za-z_]\w*)\b',
                caller_body,
            )
        )
        for callee, arity, parameter_index, parameter_name in modifying_parameters:
            call_pattern = re.compile(rf'\b{re.escape(callee.name)}\s*\(')
            for call in call_pattern.finditer(clean, caller.body_start, caller.end):
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
        0.17,
        hits,
    )


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
        0.18,
        hits,
    )


def _detect_public_mapping_nested(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    if not re.search(r'\bpragma\s+solidity\s+[^;]*\b0\.4\.', clean):
        return None
    structs = _collect_struct_members(code)
    if not structs:
        return None
    hits: List[Tuple[int, str, float, str]] = []
    pattern = re.compile(
        r'\bmapping\s*\([^=;]+=>\s*(?P<value>[A-Za-z_]\w*)\s*\)\s*'
        r'public\s+(?P<name>[A-Za-z_]\w*)\b'
    )
    for match in pattern.finditer(clean):
        value_type = match.group("value")
        if value_type not in structs:
            continue
        member_types, _ = structs[value_type]
        if not any(re.sub(r'\[.*$', '', item) in structs for item in member_types):
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
        0.17,
        hits,
    )


def _detect_mapping_deletion(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    structs = _collect_struct_members(code)
    if not structs:
        return None
    symbol_types = _collect_symbol_types(code)
    mapping_values = _collect_mapping_value_types(code)
    hits: List[Tuple[int, str, float, str]] = []
    pattern = re.compile(
        r'\bdelete\s+(?P<target>(?P<base>[A-Za-z_]\w*)'
        r'(?:\s*\[[^\]]+\])?)\s*;'
    )
    for match in pattern.finditer(clean):
        base = match.group("base")
        type_name = mapping_values.get(base) or symbol_types.get(base, "")
        struct_name = re.sub(r'\[.*$', '', type_name)
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
        0.12,
        hits,
    )


def _integer_type_range(type_name: str) -> Optional[Tuple[int, int]]:
    match = re.fullmatch(r'(?P<unsigned>u?)int(?P<bits>\d*)', type_name)
    if not match:
        return None
    bits = int(match.group("bits") or "256")
    if bits < 8 or bits > 256 or bits % 8:
        return None
    if match.group("unsigned"):
        return 0, (2 ** bits) - 1
    return -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1


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


def _detect_type_based_tautology(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    token = r'(?:[A-Za-z_]\w*|-?(?:0x[0-9A-Fa-f]+|\d+))'
    pattern = re.compile(
        rf'(?P<left>{token})\s*(?P<operator><=|>=|<|>)\s*(?P<right>{token})'
    )
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
        for match in pattern.finditer(clean, function.body_start, function.end):
            left, right = match.group("left"), match.group("right")
            variable: Optional[str] = None
            constant_text: Optional[str] = None
            operator = match.group("operator")
            if left in symbol_types and re.fullmatch(r'-?(?:0x[0-9A-Fa-f]+|\d+)', right):
                variable, constant_text = left, right
            elif right in symbol_types and re.fullmatch(r'-?(?:0x[0-9A-Fa-f]+|\d+)', left):
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
        0.12,
        hits,
    )


def _detect_boolean_constant_misuse(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    hits: List[Tuple[int, str, float, str]] = []
    seen: set[int] = set()
    patterns = (
        re.compile(r'\bif\s*\(\s*(?:true|false)\s*\)'),
        re.compile(
            r'(?:\b(?:true|false)\b\s*(?:&&|\|\|)'
            r'|(?:&&|\|\|)\s*\b(?:true|false)\b)'
        ),
    )
    for pattern in patterns:
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
        0.10,
        hits,
    )


def _detect_dangerous_balance_equality(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    balance_source = re.compile(r'\.balance\b|\.balanceOf\s*\(')
    hits: List[Tuple[int, str, float, str]] = []
    for function in _iter_functions(code):
        body = clean[function.body_start:function.end]
        tainted_names = {
            match.group("name")
            for match in re.finditer(
                r'\b(?:u?int\d*\s+)?(?P<name>[A-Za-z_]\w*)\s*=\s*'
                r'[^;]*(?:\.balance\b|\.balanceOf\s*\()[^;]*;',
                body,
            )
        }
        for equality in re.finditer(r'==', body):
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
            if not balance_source.search(expression) and not any(
                re.search(rf'\b{re.escape(name)}\b', expression)
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
        0.12,
        hits,
    )


def _detect_dangerous_enum_conversion(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    if not re.search(r'\bpragma\s+solidity\s+(?:=\s*)?0\.4\.[0-4]\s*;', clean):
        return None
    enum_names = {
        match.group("name")
        for match in re.finditer(r'\benum\s+(?P<name>[A-Za-z_]\w*)\s*\{', clean)
    }
    hits: List[Tuple[int, str, float, str]] = []
    for function in _iter_functions(code):
        for enum_name in enum_names:
            pattern = re.compile(rf'\b{re.escape(enum_name)}\s*\([^)]*\)')
            for conversion in pattern.finditer(clean, function.body_start, function.end):
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
        0.11,
        hits,
    )


def _assembly_return_positions(
    code: str, function: SolidityFunction
) -> List[Tuple[int, int]]:
    """Return Yul return(offset, size) ranges inside one function."""
    clean = _clean_code(code)
    results: List[Tuple[int, int]] = []
    assembly_pattern = re.compile(r'\bassembly(?:\s*\([^)]*\))?\s*\{')
    for assembly in assembly_pattern.finditer(
        clean, function.body_start, function.end
    ):
        open_pos = assembly.end() - 1
        close_pos = _balanced_end(clean, open_pos)
        if close_pos is None or close_pos > function.end:
            continue
        return_pattern = re.compile(r'\breturn\s*\([^,]+,[^)]+\)')
        for call in return_pattern.finditer(clean, open_pos, close_pos):
            results.append((call.start(), call.end()))
    return results


def _detect_return_instead_of_leave(code: str) -> Optional[VulnerabilityFinding]:
    hits: List[Tuple[int, str, float, str]] = []
    for function in _iter_functions(code):
        if not re.search(r'\b(?:internal|private)\b', function.header):
            continue
        returns = re.search(r'\breturns\s*\((?P<values>[^)]*)\)', function.header)
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
        0.16,
        hits,
    )


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
                re.search(rf'\b{re.escape(name)}\s*\(', body)
                for name in halting_names
            ):
                halting_names.add(function.name)
                changed = True

    hits: List[Tuple[int, str, float, str]] = []
    seen: set[int] = set()
    for function in functions:
        for target in halting_names:
            call_pattern = re.compile(rf'\b{re.escape(target)}\s*\(')
            for call in call_pattern.finditer(clean, function.body_start, function.end):
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
        0.16,
        hits,
    )


def _uses_exact_vulnerable_storage_array_compiler(code: str) -> bool:
    """Recognize explicitly pinned solc versions affected by 2019 storage bugs."""
    clean = _clean_code(code)
    versions = [
        (int(match.group("minor")), int(match.group("patch")))
        for match in re.finditer(
            r'\bpragma\s+solidity\s+(?:=\s*)?0\.(?P<minor>[45])\.'
            r'(?P<patch>\d+)\s*;',
            clean,
        )
    ]
    return any(
        (minor == 4 and 7 <= patch <= 25)
        or (minor == 5 and 0 <= patch <= 9)
        for minor, patch in versions
    )


def _detect_storage_abiencoderv2_array(code: str) -> Optional[VulnerabilityFinding]:
    clean = _clean_code(code)
    if not (
        _uses_exact_vulnerable_storage_array_compiler(code)
        and re.search(r'\bpragma\s+experimental\s+ABIEncoderV2\s*;', clean)
    ):
        return None
    structs = _collect_struct_members(code)
    state_types = _collect_state_symbol_types(code)
    vulnerable_arrays = {
        name for name, type_name in state_types.items()
        if type_name.count("[") >= 2
        or ("[" in type_name and re.sub(r'\[.*$', '', type_name) in structs)
    }
    hits: List[Tuple[int, str, float, str]] = []
    for call in re.finditer(r'\babi\.encode\s*\(', clean):
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
        0.18,
        hits,
    )


def _signed_integer_base(type_name: str) -> Optional[str]:
    match = re.match(r'(?P<base>int\d*)\[', type_name)
    return match.group("base") if match else None


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
        pattern = re.compile(rf'\b{re.escape(name)}\s*=\s*(?P<rhs>[^;]+);')
        for assignment in pattern.finditer(clean):
            rhs = assignment.group("rhs").strip()
            vulnerable = rhs.startswith("[")
            if not vulnerable and re.fullmatch(r'[A-Za-z_]\w*', rhs):
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
        0.17,
        hits,
    )


def _direct_base_names(header: str) -> List[str]:
    inheritance = re.search(r'\bis\s+(?P<bases>[\s\S]+)$', header)
    if not inheritance:
        return []
    names = []
    for base in _split_top_level_arguments(inheritance.group("bases")):
        match = re.match(r'\s*([A-Za-z_]\w*)', base)
        if match:
            names.append(match.group(1))
    return names


def _detect_state_variable_shadowing(code: str) -> Optional[VulnerabilityFinding]:
    contracts = {contract.name: contract for contract in _iter_contracts(code)}
    declarations = {
        name: _collect_contract_state_declarations(code, contract)
        for name, contract in contracts.items()
    }

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
        0.18,
        hits,
    )


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
            escaped = re.escape(name)
            write = re.search(
                rf'\b{escaped}\b\s*(?:=(?!=)|\+=|-=|\+\+|--)', body
            )
            if write:
                continue
            dangerous_read = re.search(
                rf'(?:\b{escaped}\b\s*\.\s*(?:transfer|send|call)\b'
                rf'|payable\s*\(\s*\b{escaped}\b\s*\)\s*\.'
                rf'|(?:selfdestruct|suicide)\s*\(\s*\b{escaped}\b)',
                body,
            )
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
        0.18,
        hits,
    )


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
# Context checks nâng cao (CEI pattern, state-update ordering)
# ──────────────────────────────────────────────────────────────────────────────

def check_call_before_state_update(code: str) -> List[str]:
    """
    Phát hiện vi phạm CEI (Checks-Effects-Interactions):
    Nếu có external call trước khi cập nhật state variable cùng function,
    đây là dấu hiệu reentrancy.
    Trả về danh sách mô tả vi phạm.
    """
    violations: List[str] = []
    # Tìm tất cả function bodies
    func_pattern = re.compile(
        r'function\s+(\w+)\s*\([^)]*\)[^{]*\{', re.DOTALL
    )
    for func_match in func_pattern.finditer(code):
        func_name = func_match.group(1)
        start = func_match.end()
        # Tìm phần thân hàm (đơn giản: đến dấu } cân bằng)
        depth, body_start = 1, start
        for i, ch in enumerate(code[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    body = code[body_start:i]
                    break
        else:
            continue

        clean = _clean_code(body)

        # Tìm vị trí external call
        call_m = re.search(
            r'\.call\s*\{[^}]*value|\.call\s*\(|\.transfer\s*\(|\.send\s*\(',
            clean
        )
        if not call_m:
            continue

        # Tìm state change sau call
        after_call = clean[call_m.end():]
        state_change = re.search(
            r'\b(?:balances?|amounts?|stakes?|deposits?|locked|totalSupply)\b'
            r'\s*(?:\[.*?\]\s*)?(?:-=|=(?!=)|\+=)',
            after_call
        )
        if state_change:
            violations.append(
                f"Hàm '{func_name}': cập nhật state SAU external call "
                f"(vi phạm CEI pattern)"
            )
    return violations


def check_missing_zero_address(code: str) -> List[str]:
    """
    Report only address parameters that reach state or a call destination.

    This deliberately avoids flagging every address parameter. The usage-based
    filter mirrors the useful part of Slither's missing-zero-check detector and
    keeps ordinary lookup/query parameters out of the result.
    """
    return [description for _, _, _, description in _missing_zero_address_hits(code)]


# ──────────────────────────────────────────────────────────────────────────────
# StaticAnalyzer – class chính
# ──────────────────────────────────────────────────────────────────────────────

class StaticAnalyzer:
    """
    Phân tích tĩnh Smart Contract Solidity.

    Cách dùng:
        analyzer = StaticAnalyzer()
        result   = analyzer.analyze(solidity_code)
        score    = result["static_score"]        # float [0, 1]
        findings = result["findings"]            # Dict[str, VulnerabilityFinding]
    """

    RISK_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}

    def __init__(self):
        # Build registry một lần
        self._registry = VULNERABILITY_REGISTRY
        logger.info(
            f"[StaticAnalyzer] Loaded {len(self._registry)} vulnerability patterns"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(self, contract_code: str, guard_modifiers: Optional[set[str]] = None) -> Dict:
        """
        Phân tích contract_code và trả về dict:
        {
            "static_score"   : float,          # 0.0 – 1.0
            "verdict"        : str,             # "Vulnerable" | "Safe"
            "risk_level"     : str,             # mức rủi ro cao nhất tìm thấy
            "findings"       : Dict[str, VulnerabilityFinding],
            "high_risk_list" : List[Dict],      # sorted by risk
            "context_issues" : List[str],       # từ advanced context checks
            "summary"        : str,
        }
        """
        clean = _clean_code(contract_code)
        if guard_modifiers is None:
            guard_modifiers = extract_guard_modifiers(contract_code)
        functions = list(_iter_functions(clean))

        findings: Dict[str, VulnerabilityFinding] = {}
        total_score = 0.0

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
                total_score += max(0.0, finding.weighted_score - neg_reduction * entry["type_weight"])
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
            total_score += finding.weighted_score
            key = finding.vuln_type
            if key in findings:
                findings[key].matches.extend(finding.matches)
                findings[key].notes.extend(finding.notes)
                findings[key].confirmed = (
                    findings[key].confirmed or finding.confirmed
                )
            else:
                findings[key] = finding

        # Context checks nâng cao
        context_issues = check_call_before_state_update(contract_code)
        context_issues += check_missing_zero_address(contract_code)

        # Nếu context checks phát hiện thêm vấn đề → tăng nhẹ score
        if context_issues:
            total_score = min(1.0, total_score + 0.05 * len(context_issues))

        # Chuẩn hoá score
        static_score = min(1.0, total_score)

        # Mức rủi ro cao nhất
        max_risk = self._get_max_risk(findings)

        # Danh sách high-risk findings để export
        high_risk_list = self._build_high_risk_list(findings)

        verdict = "Vulnerable" if static_score > 0.25 or high_risk_list else "Safe"

        summary = self._build_summary(findings, context_issues, static_score, verdict)

        return {
            "static_score":   static_score,
            "verdict":        verdict,
            "risk_level":     max_risk,
            "findings":       findings,
            "high_risk_list": high_risk_list,
            "context_issues": context_issues,
            "summary":        summary,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

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
            type_weight=entry["type_weight"],
        )

        vuln_id = entry.get("vuln_id")
        is_access_sensitive = vuln_id in {"SWC-101", "SWC-105", "SWC-106"}

        for pat_tuple in entry.get("patterns", []):
            regex, weight, desc = pat_tuple
            try:
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
                            hdr_m = re.search(r'function\s+[^{]*\{', clean_code)
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

                    line, col = _get_line_col(clean_code, m.start())
                    context = _extract_context(original_code, m.start())
                    finding.matches.append(PatternMatch(
                        line=line,
                        column=col,
                        matched_text=m.group()[:80],
                        context=context,
                        pattern_weight=weight,
                    ))
                    finding.notes.append(f"L{line}: {desc}")
            except re.error as e:
                logger.debug(f"[StaticAnalyzer] Regex error in {entry['vuln_type']}: {e}")

        return finding

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
            try:
                if re.search(regex, clean_code, re.DOTALL):
                    reduction += weight
            except re.error:
                pass

        if guard_modifiers and entry.get("vuln_id") in {"SWC-101", "SWC-105", "SWC-106"}:
            for mod in guard_modifiers:
                if re.search(rf'\b{re.escape(mod)}\b', clean_code):
                    reduction += 0.45
                    break

        return min(raw_score, reduction)

    def _get_max_risk(self, findings: Dict[str, VulnerabilityFinding]) -> str:
        if not findings:
            return "None"
        return max(
            (f.risk_level for f in findings.values()),
            key=lambda r: self.RISK_ORDER.get(r, 0),
            default="None"
        )

    def _build_high_risk_list(
        self, findings: Dict[str, VulnerabilityFinding]
    ) -> List[Dict]:
        result = []
        for f in findings.values():
            if self.RISK_ORDER.get(f.risk_level, 0) >= 3:  # Critical or High
                for m in f.matches:
                    result.append({
                        "vuln_id":    f.vuln_id,
                        "vuln_type":  f.vuln_type,
                        "risk_level": f.risk_level,
                        "line":       m.line,
                        "context":    m.context,
                        "note":       f.notes[f.matches.index(m)] if f.notes else "",
                    })
        return sorted(
            result,
            key=lambda x: self.RISK_ORDER.get(x["risk_level"], 0),
            reverse=True,
        )

    def _build_summary(
        self,
        findings: Dict[str, VulnerabilityFinding],
        context_issues: List[str],
        score: float,
        verdict: str,
    ) -> str:
        lines = [f"[Static Analysis] Verdict: {verdict}  |  Score: {score:.3f}"]
        if not findings and not context_issues:
            lines.append("  → Không phát hiện dấu hiệu lỗ hổng rõ ràng.")
        else:
            by_level: Dict[str, List[str]] = {}
            for f in findings.values():
                by_level.setdefault(f.risk_level, []).append(
                    f"{f.vuln_type} ({len(f.matches)} match(es))"
                )
            for level in ["Critical", "High", "Medium", "Low"]:
                if level in by_level:
                    lines.append(f"  [{level}] " + ", ".join(by_level[level]))
            if context_issues:
                lines.append("  [Context Issues]")
                for ci in context_issues:
                    lines.append(f"    - {ci}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone runner (dùng để test nhanh)
# ──────────────────────────────────────────────────────────────────────────────

def run_static_analysis(
    contract_code: str,
    guard_modifiers: Optional[set[str]] = None,
) -> Dict:
    """
    Convenience wrapper cho Detector Agent.

    Returns:
        {
            "static_score"   : float,
            "verdict"        : "Vulnerable" | "Safe",
            "risk_level"     : str,
            "findings"       : Dict,
            "high_risk_list" : List[Dict],
            "context_issues" : List[str],
            "summary"        : str,
        }
    """
    analyzer = StaticAnalyzer()
    return analyzer.analyze(contract_code, guard_modifiers=guard_modifiers)



# ──────────────────────────────────────────────────────────────────────────────
# Tích hợp vào VulnerabilityConfig (cập nhật VULNERABILITY_PATTERNS tương thích)
# ──────────────────────────────────────────────────────────────────────────────

def build_compat_patterns() -> Dict:
    """
    Tạo VULNERABILITY_PATTERNS tương thích với định dạng của
    VulnerabilityConfig trong vulnerability_detector.py gốc,
    để có thể dùng thay thế trực tiếp.

    Format output:
        {
            vuln_type: {
                "weight"    : float,
                "risk_level": str,
                "patterns"  : List[Tuple[regex, weight]]
            }
        }
    """
    compat: Dict = {}
    for entry in VULNERABILITY_REGISTRY:
        key = entry["vuln_type"]
        compat[key] = {
            "weight":     entry["type_weight"],
            "risk_level": entry["risk_level"],
            "patterns":   [
                (p[0], p[1])          # chỉ lấy (regex, weight)
                for p in entry["patterns"]
            ],
        }
    return compat


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _SAMPLE_VULNERABLE = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.6.0;

contract VulnerableBank {
    mapping(address => uint256) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    // BUG: Reentrancy - state updated AFTER external call
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        // External call BEFORE state update  -> CEI violation
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        balances[msg.sender] -= amount;   // too late!
    }

    // BUG: tx.origin authentication
    function adminWithdraw() public {
        require(tx.origin == owner, "Not owner");
        selfdestruct(payable(tx.origin));
    }

    // BUG: timestamp dependence for random
    function random() public view returns (uint256) {
        return uint256(keccak256(abi.encodePacked(block.timestamp))) % 100;
    }

    address public owner;
}
"""

    logging.basicConfig(level=logging.WARNING)
    result = run_static_analysis(_SAMPLE_VULNERABLE)
    print(result["summary"])
    print(f"\nTotal findings: {len(result['findings'])}")
    print(f"High-risk items: {len(result['high_risk_list'])}")
    for item in result["high_risk_list"]:
        print(f"  [{item['risk_level']}] {item['vuln_type']} @ L{item['line']}: {item['note']}")
    if result["context_issues"]:
        print("\nContext Issues:")
        for ci in result["context_issues"]:
            print(f"  - {ci}")
