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
from typing import Dict, List, Optional, Tuple

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
            (r'\.call\s*\{[^}]*value\s*:', 0.45,
             "Low-level call gửi ETH - nguy cơ reentrancy kinh điển"),
            # 2. transfer / send (ít nguy hiểm hơn nhưng vẫn cần kiểm tra)
            (r'\.\s*transfer\s*\(', 0.20,
             ".transfer() - vẫn có thể dẫn đến reentrancy nếu logic sai"),
            (r'\.\s*send\s*\(', 0.20,
             ".send() - trả về bool nhưng vẫn cần CEI pattern"),
            # 3. External call không rõ giá trị
            (r'\.call\s*\(', 0.35,
             "External call - kiểm tra xem state có được cập nhật trước không"),
            # 4. Cross-function reentrancy: gọi hàm external trong modifier/callback
            (r'(function\s+\w+[^{]*\bexternal\b[^{]*\{[^}]*\.call)', 0.30,
             "External function thực hiện low-level call"),
            # 5. Delegate call - cross-contract reentrancy
            (r'\.delegatecall\s*\(', 0.40,
             "delegatecall - nguy cơ reentrancy và storage collision"),
            # 6. Withdraw pattern thiếu kiểm soát
            (r'function\s+withdraw[^{]*\{(?:(?!require|revert|nonReentrant).)*\.call', 0.50,
             "Hàm withdraw gọi external mà không có bảo vệ reentrancy"),
            # 7. Thiếu ReentrancyGuard / nonReentrant modifier
            # (pattern negative - sẽ được xử lý trong context check)
        ],
        "negative_patterns": [
            # Nếu có những pattern này thì giảm score
            (r'nonReentrant', 0.40,      "Có modifier nonReentrant"),
            (r'ReentrancyGuard', 0.35,   "Kế thừa ReentrancyGuard"),
            (r'_status\s*==\s*_NOT_ENTERED', 0.25, "Kiểm tra reentrancy thủ công"),
        ],
        "context_checks": [
            # Kiểm tra CEI pattern vi phạm: call trước khi update state
            "check_call_before_state_update",
        ],
    },

    {
        "vuln_id": "SWC-105",
        "vuln_type": "unprotected_ether_withdrawal",
        "risk_level": "Critical",
        "type_weight": 0.25,
        "patterns": [
            (r'function\s+\w+[^{]*\bpublic\b[^{]*\{(?:(?!require|onlyOwner|modifier).)*selfdestruct',
             0.60, "selfdestruct không được bảo vệ"),
            (r'\bselfdestruct\s*\(', 0.50,
             "Gọi selfdestruct - cần kiểm tra quyền truy cập"),
            (r'\bsuicide\s*\(', 0.50,
             "Gọi suicide (alias của selfdestruct) - deprecated"),
            # Hàm rút tiền công khai không có modifier
            (r'function\s+(?:withdraw|drain|rescue)[^{]*\bpublic\b(?!\s+\bview\b)(?!\s+\bpure\b)[^{]*\{(?:(?!require|onlyOwner|onlyAdmin|modifier).){0,200}\.call',
             0.55, "Hàm rút tiền public không có kiểm soát truy cập"),
        ],
        "negative_patterns": [
            (r'onlyOwner', 0.30, "Có kiểm soát onlyOwner"),
            (r'require\s*\(\s*msg\.sender\s*==', 0.30, "Có kiểm tra msg.sender"),
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
            (r'(?:onlyOwner|require\s*\(.*msg\.sender)', 0.35, "Có access control"),
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
        "risk_level": "Critical",
        "type_weight": 0.18,
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
            (r'function\s+(?:mint|burn|pause|unpause|upgrade)\s*\([^)]*\)\s*(?:public|external)(?:(?!onlyOwner|onlyAdmin|onlyRole|require).){0,100}\{',
             0.50, "Hàm đặc quyền (mint/burn/pause) có thể thiếu access control"),
            # Modifier rỗng hoặc luôn pass
            (r'modifier\s+\w+\s*\([^)]*\)\s*\{\s*_;\s*\}', 0.60,
             "Modifier rỗng - không có logic kiểm tra"),
            (r'modifier\s+\w+\s*\([^)]*\)\s*\{\s*//[^\n]*\n\s*_;\s*\}', 0.55,
             "Modifier chỉ có comment - nghi ngờ placeholder"),
            # Không kiểm tra zero address
            (r'function\s+\w+[^{]*address\s+\w+[^{]*\{(?:(?!require\s*\(\s*\w+\s*!=\s*address\s*\(\s*0).){0,200}\w+\s*=\s*\w+',
             0.30, "Gán address mà không kiểm tra zero address"),
            # Role-based access sai
            (r'hasRole\s*\([^)]*\)\s*==\s*false', 0.40,
             "Kiểm tra hasRole ngược - logic sai tiềm ẩn"),
        ],
        "negative_patterns": [
            (r'require\s*\(\s*msg\.sender\s*==\s*owner', 0.25, "Kiểm tra owner"),
            (r'onlyOwner|onlyAdmin|onlyRole|onlyMinter', 0.30, "Có access modifier"),
            (r'AccessControl|Ownable', 0.20, "Dùng thư viện access control"),
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
            # Nhân trước khi chia (precision loss)
            (r'\w+\s*/\s*\w+\s*\*\s*\w+', 0.25,
             "Chia trước rồi nhân - mất độ chính xác (nên nhân trước)"),
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
        "vuln_type": "outdated_compiler_version",
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
            (r'require\s*\(\s*\w+\s*==\s*\w+\s*\)', 0.25,
             "require với so sánh bằng tự thân - kiểm tra logic"),
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

    # ──────────────────────────── SHORT ADDRESS ───────────────────────────────

    {
        "vuln_id": "SWC-103",
        "vuln_type": "short_address_attack",
        "risk_level": "Medium",
        "type_weight": 0.08,
        "patterns": [
            # Transfer không validate độ dài address
            (r'function\s+transfer\s*\(\s*address\s+\w+\s*,\s*uint\w*\s*\w+\s*\)', 0.20,
             "Transfer function - kiểm tra short address padding"),
            # ABI encoding dễ bị short address
            (r'abi\.encodePacked\s*\([^)]*address', 0.25,
             "encodePacked với address - dễ bị hash collision"),
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
    """Loại bỏ comment để tránh false positive."""
    # Bỏ block comment /* ... */
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # Bỏ single-line comment //
    code = re.sub(r'//[^\n]*', '', code)
    return code


def _strip_strings(code: str) -> str:
    """Loại bỏ string literals để tránh false positive."""
    code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
    code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code)
    return code


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

        clean = _strip_comments(body)

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
    Phát hiện hàm nhận address parameter nhưng không kiểm tra != address(0).
    """
    violations: List[str] = []
    func_pattern = re.compile(
        r'function\s+(\w+)\s*\(([^)]*)\)[^{]*\{', re.DOTALL
    )
    for m in func_pattern.finditer(code):
        func_name = m.group(1)
        params = m.group(2)
        # Hàm có address param?
        if not re.search(r'\baddress\b', params):
            continue
        # Lấy body
        start = m.end()
        depth, body = 1, ""
        for i, ch in enumerate(code[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    body = code[start:i]
                    break
        if not body:
            continue
        clean = _strip_comments(body)
        # Có kiểm tra zero address không?
        has_zero_check = bool(re.search(
            r'require\s*\([^)]*!=\s*address\s*\(\s*0\s*\)', clean
        ))
        if not has_zero_check:
            violations.append(
                f"Hàm '{func_name}': nhận address nhưng thiếu kiểm tra zero address"
            )
    return violations


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

    def analyze(self, contract_code: str) -> Dict:
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
        clean = _strip_comments(_strip_strings(contract_code))

        findings: Dict[str, VulnerabilityFinding] = {}
        total_score = 0.0

        for entry in self._registry:
            finding = self._run_pattern_check(entry, clean, contract_code)
            if finding.matches:
                # Áp dụng negative patterns để giảm score
                neg_reduction = self._apply_negative_patterns(
                    entry, clean, finding.raw_score
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
    ) -> VulnerabilityFinding:
        finding = VulnerabilityFinding(
            vuln_id=entry["vuln_id"],
            vuln_type=entry["vuln_type"],
            risk_level=entry["risk_level"],
            type_weight=entry["type_weight"],
        )

        for pat_tuple in entry.get("patterns", []):
            regex, weight, desc = pat_tuple
            try:
                compiled = re.compile(regex, re.DOTALL | re.MULTILINE)
                for m in compiled.finditer(clean_code):
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

def run_static_analysis(contract_code: str) -> Dict:
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
    return analyzer.analyze(contract_code)


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