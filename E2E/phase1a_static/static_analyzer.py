"""Phase 1A - bộ luật phân tích tĩnh: 27 luật regex có trọng số (VULNERABILITY_REGISTRY) + 25 kiểm tra ngữ nghĩa."""

import re
from dataclasses import dataclass, field
from typing import Iterator

from E2E.common.regex_patterns import (
    RE_ABI_ENCODE_CALL, RE_ABSTRACT_AT_END, RE_ARRAY_LENGTH_ASSIGNMENT, RE_ARRAY_SUFFIX, RE_ASSEMBLY_BLOCK,
    RE_BALANCE_ASSIGNMENT, RE_BALANCE_SOURCE, RE_BOOLEAN_CONSTANT_OPERAND, RE_CAST_PREFIX, RE_CONTRACT_WITH_BODY,
    RE_DECIMAL_NUMBER, RE_DELEGATECALL, RE_DELETE_STATEMENT, RE_ENCODE_PACKED_CALL, RE_ENUM_OPENING,
    RE_ERC20_TRANSFER_CALL, RE_FUNCTION_HEADER_PREFIX, RE_FUNCTION_WITH_BODY, RE_GUARD_MODIFIER_NAME, RE_IDENTIFIER,
    RE_IF_CONSTANT_CONDITION, RE_INHERITANCE_LIST, RE_INTEGER_LITERAL, RE_INTEGER_TYPE, RE_INTERNAL_OR_PRIVATE,
    RE_IS_KEYWORD, RE_LEADING_IDENTIFIER, RE_LOOP_OPENING, RE_MAPPING_KEYWORD, RE_MAPPING_VALUE_TYPE, RE_MODIFIER_NAME,
    RE_MODIFIER_WITH_BODY, RE_MSG_VALUE, RE_MSG_VALUE_ZERO_CHECK, RE_ORDER_COMPARISON, RE_OUTGOING_ETHER, RE_PARAMETER,
    RE_PAYABLE, RE_PINNED_PRAGMA_0_4_OR_0_5, RE_PRAGMA_0_4, RE_PRAGMA_0_4_0_TO_0_4_4, RE_PRAGMA_ABIENCODERV2,
    RE_PRAGMA_STATEMENT, RE_PUBLIC_MAPPING_VALUE_TYPE, RE_PUBLIC_OR_EXTERNAL, RE_RETURNS_CLAUSE, RE_SENDER_CHECKS,
    RE_SIGNED_INT_ARRAY_TYPE, RE_SPECIAL_FUNCTION_WITH_BODY, RE_STATE_DECLARATION, RE_STATE_MAPPING,
    RE_STATE_TYPE_AND_NAME, RE_STORAGE_ARRAY_LOCAL, RE_STRICT_EQUALITY, RE_STRUCT_OPENING, RE_SYMBOL_DECLARATION,
    RE_WHITESPACE, RE_WORD_TOKEN, RE_XOR_EXPRESSION, RE_YUL_RETURN_CALL, RE_YUL_SHIFT_WITH_CONSTANT_VALUE,
    re_assignment_from, re_assignment_to, re_call, re_call_target, re_enum_conversion, re_ether_destination,
    re_indexed_write, re_state_write, re_word, re_zero_address_require, re_zero_address_revert_if,
)
from E2E.phase0_parsing.code_units import line_number, mask_noncode, matching_close
from E2E.phase0_parsing.contract_context import statements_at_depth


# Một lần khớp pattern: dòng, đoạn khớp, ngữ cảnh xung quanh, trọng số đóng góp vào điểm.
@dataclass
class PatternMatch:
    line: int
    matched_text: str
    context: str
    pattern_weight: float


# Kết quả cho MỘT loại lỗ hổng (vd reentrancy / SWC-107): các lần khớp, mức rủi ro, confirmed, ghi chú theo dòng.
@dataclass
class VulnerabilityFinding:
    vuln_id: str
    vuln_type: str
    risk_level: str
    matches: list[PatternMatch] = field(default_factory=list)
    confirmed: bool = False
    notes: list[str] = field(default_factory=list)

    # Điểm thô = tổng trọng số các lần khớp, chặn trần 1.0.
    @property
    def raw_score(self) -> float:
        return min(1.0, sum((match.pattern_weight for match in self.matches), 0.0))


# Một khối có thân (hàm / contract / modifier): tên, tham số, header, vị trí bắt đầu / mở thân / dấu "}", thân đã che.
@dataclass(frozen=True)
class Block:
    name: str
    params: str
    header: str
    start: int
    body_start: int
    end: int
    body: str


# Mỗi luật: vuln_id, vuln_type, risk_level, patterns = [(regex, trọng số, mô tả)], negative_patterns = dấu hiệu phòng vệ.
VULNERABILITY_REGISTRY: list[dict] = [
    {"vuln_id": "SWC-107", "vuln_type": "reentrancy", "risk_level": "Critical",
     "patterns": [
        (r'\.call\s*\{[^}]*value\s*:', 0.50, "Low-level call gửi ETH - nguy cơ reentrancy kinh điển"),
        (r'\.call\s*\.\s*value\s*\(', 0.50, "Low-level call.value() gửi ETH - nguy cơ reentrancy kinh điển (Solidity <0.7)"),
        (r'\.call\s*\(', 0.25, "External call - kiểm tra xem state có được cập nhật trước không"),
        (r'(function\s+\w+[^{]*\bexternal\b[^{]*\{[^}]*\.call)', 0.30, "External function thực hiện low-level call"),
        (r'\.delegatecall\s*\(', 0.40, "delegatecall - nguy cơ reentrancy và storage collision"),
        (r'function\s+withdraw[^{]*\{(?:(?!require|revert|nonReentrant|\}).)*\.call', 0.50, "Hàm withdraw gọi external mà không có bảo vệ reentrancy"),
     ],
     "negative_patterns": [
        (r'nonReentrant', 0.40, "Có modifier nonReentrant"),
        (r'ReentrancyGuard', 0.35, "Kế thừa ReentrancyGuard"),
        (r'_status\s*==\s*_NOT_ENTERED', 0.25, "Kiểm tra reentrancy thủ công"),
     ]},
    {"vuln_id": "SWC-105", "vuln_type": "unprotected_ether_withdrawal", "risk_level": "Critical",
     "patterns": [
        (r'function\s+\w+[^{]*\bpublic\b[^{]*\{(?:(?!require|onlyOwner|modifier|\}).)*selfdestruct', 0.60, "selfdestruct không được bảo vệ"),
        (r'\bselfdestruct\s*\(', 0.50, "Gọi selfdestruct - cần kiểm tra quyền truy cập"),
        (r'\bsuicide\s*\(', 0.50, "Gọi suicide (alias của selfdestruct) - deprecated"),
        (r'function\s+(?:withdraw|drain|rescue)[^{]*\bpublic\b(?!\s+\bview\b)(?!\s+\bpure\b)[^{]*\{(?:(?!require|onlyOwner|onlyAdmin|modifier|\}).){0,200}\.call', 0.55, "Hàm rút tiền public không có kiểm soát truy cập"),
     ],
     "negative_patterns": [
        (r'onlyOwner|onlyAdmin|onlyRole|onlyMinter|onlyFounder|oO\b|oOOrContract\b', 0.45, "Có access modifier"),
        (r'require\s*\(\s*msg\.sender\s*==', 0.40, "Có kiểm tra msg.sender"),
        (r'require\s*\(\s*[^;]*==\s*msg\.sender', 0.40, "Có kiểm tra msg.sender"),
        (r'if\s*\([^)]*msg\.sender\s*!=[^)]*\)\s*(?:revert\b|throw\b)', 0.40, "Idiom 0.4 if msg.sender != revert"),
        (r'\b(?:internal|private)\b', 0.45, "Visibility internal hoặc private"),
     ]},
    {"vuln_id": "SWC-106", "vuln_type": "unprotected_selfdestruct", "risk_level": "Critical",
     "patterns": [
        (r'selfdestruct\s*\(\s*(?:msg\.sender|tx\.origin|_?owner)\s*\)', 0.50, "selfdestruct gửi ETH về owner - kiểm tra bảo vệ"),
        (r'function\s+\w+[^)]*\)[^{]*\{[^}]*selfdestruct', 0.45, "Hàm chứa selfdestruct"),
     ],
     "negative_patterns": [
        (r'(?:onlyOwner|onlyAdmin|onlyRole|onlyFounder|oO\b|oOOrContract\b)', 0.45, "Có access control"),
        (r'require\s*\(.*msg\.sender', 0.40, "Có kiểm tra msg.sender"),
        (r'if\s*\([^)]*msg\.sender\s*!=[^)]*\)\s*(?:revert\b|throw\b)', 0.40, "Idiom 0.4 if msg.sender != revert"),
        (r'\b(?:internal|private)\b', 0.45, "Visibility internal hoặc private"),
     ]},
    {"vuln_id": "SWC-115", "vuln_type": "tx_origin_authentication", "risk_level": "Critical",
     "patterns": [
        (r'tx\.origin\s*==', 0.70, "Xác thực dựa trên tx.origin - dễ bị phishing attack"),
        (r'require\s*\(\s*tx\.origin', 0.65, "require kiểm tra tx.origin - không an toàn"),
        (r'tx\.origin\s*!=', 0.60, "So sánh tx.origin"),
        (r'if\s*\(\s*tx\.origin', 0.55, "Điều kiện dựa trên tx.origin"),
     ],
     "negative_patterns": []},
    {"vuln_id": "SWC-100", "vuln_type": "function_default_visibility", "risk_level": "Low",
     "patterns": [
        (r'function\s+\w+\s*\([^)]*\)\s*(?:returns\s*\([^)]*\)\s*)?\{', 0.30, "Hàm có thể thiếu visibility specifier"),
        (r'function\s+_\w+\s*\([^)]*\)\s*(?:public|external)', 0.45, "Hàm bắt đầu bằng _ (quy ước internal) nhưng khai báo public/external"),
     ],
     "negative_patterns": [
        (r'function\s+\w+\s*\([^)]*\)\s*(?:public|external|internal|private)', 0.40, "Hàm có visibility specifier rõ ràng"),
     ]},
    {"vuln_id": "SWC-101", "vuln_type": "access_control", "risk_level": "Critical",
     "patterns": [
        (r'function\s+(?:setOwner|transferOwnership|updateOwner)\s*\([^)]*\)\s*(?:public|external)(?!\s+\w*[Oo]nly)', 0.55, "Thay đổi owner không có access control"),
        (r'function\s+(?:mint|burn|pause|unpause|upgrade)\s*\([^)]*\)\s*(?:public|external)(?:(?!onlyOwner|onlyAdmin|onlyRole|require|\}).){0,100}\{', 0.50, "Hàm đặc quyền (mint/burn/pause) có thể thiếu access control"),
        (r'modifier\s+\w+\s*\([^)]*\)\s*\{\s*_;\s*\}', 0.60, "Modifier rỗng - không có logic kiểm tra"),
        (r'modifier\s+\w+\s*\([^)]*\)\s*\{\s*//[^\n]*\n\s*_;\s*\}', 0.55, "Modifier chỉ có comment - nghi ngờ placeholder"),
        (r'hasRole\s*\([^)]*\)\s*==\s*false', 0.40, "Kiểm tra hasRole ngược - logic sai tiềm ẩn"),
     ],
     "negative_patterns": [
        (r'require\s*\(\s*msg\.sender\s*==', 0.40, "Kiểm tra msg.sender"),
        (r'require\s*\(\s*[^;]*==\s*msg\.sender', 0.40, "Kiểm tra msg.sender"),
        (r'if\s*\([^)]*msg\.sender\s*!=[^)]*\)\s*(?:revert\b|throw\b)', 0.40, "Idiom 0.4 if msg.sender != revert"),
        (r'onlyOwner|onlyAdmin|onlyRole|onlyMinter|onlyFounder|oO\b|oOOrContract\b', 0.45, "Có access modifier"),
        (r'AccessControl|Ownable', 0.20, "Dùng thư viện access control"),
        (r'\b(?:internal|private)\b', 0.45, "Visibility internal hoặc private"),
     ]},
    {"vuln_id": "SWC-104", "vuln_type": "unchecked_low_level_calls", "risk_level": "Critical",
     "patterns": [
        (r'(?<!\(bool\s)(?<!\bbool\b\s)\.\s*call\s*[\({](?:[^;]*;)', 0.45, "low-level call() không kiểm tra return value"),
        (r'(?:bool\s+\w+\s*,?\s*)?\.\s*call\b(?:(?!require|revert|if\s*\().){0,100};', 0.40, "Kết quả call có thể không được xử lý"),
        (r'\.send\s*\([^)]*\)\s*;', 0.40, ".send() không kiểm tra return value bool"),
        (r'\.staticcall\s*\(', 0.30, "staticcall - kiểm tra return value"),
        (r'assembly\s*\{[^}]*call\b', 0.35, "Assembly-level call - cần kiểm tra cẩn thận"),
     ],
     "negative_patterns": [
        (r'\(bool\s+\w+,\s*\)\s*=\s*', 0.35, "Kết quả được destructure đúng cách"),
        (r'require\s*\(\s*\w+(?:success|Success|ok|Ok)\b', 0.30, "Kiểm tra success flag"),
     ]},
    {"vuln_id": "SWC-101", "vuln_type": "integer_overflow_underflow", "risk_level": "High",
     "patterns": [
        (r'pragma\s+solidity\s+[^;]*0\.[0-7]\.', 0.15, "Phiên bản < 0.8 không có overflow check tự động"),
        (r'uint\w*\s+\w+\s*=\s*\w+\s*\+\s*\w+(?!\s*SafeMath)', 0.30, "Cộng uint không dùng SafeMath (nguy hiểm với Solidity <0.8)"),
        (r'uint\w*\s+\w+\s*=\s*\w+\s*-\s*\w+(?!\s*SafeMath)', 0.30, "Trừ uint - nguy cơ underflow"),
        (r'\bunchecked\s*\{', 0.50, "Khối unchecked - tắt overflow check, phải review kỹ"),
        (r'uint(?:8|16|32|64|128)\s*\(\s*\w+\s*\)', 0.35, "Downcast uint - có thể mất dữ liệu nếu giá trị vượt range"),
        (r'int(?:8|16|32|64|128)\s*\(\s*\w+\s*\)', 0.35, "Downcast int - có thể mất dữ liệu"),
        (r'\w+\s*\*\*\s*\w+', 0.20, "Phép lũy thừa - kiểm tra overflow"),
     ],
     "negative_patterns": [
        (r'using\s+SafeMath\s+for', 0.35, "Dùng SafeMath"),
        (r'pragma\s+solidity\s+\^?0\.8', 0.40, "Solidity 0.8+ có overflow check tự động"),
     ]},
    {"vuln_id": "SWC-113", "vuln_type": "dos_gas_limit", "risk_level": "High",
     "patterns": [
        (r'for\s*\([^)]*;\s*\w+\s*<\s*\w+\.length\s*;', 0.45, "Vòng lặp trên array động - nguy cơ DoS nếu array lớn"),
        (r'for\s*\([^)]*;\s*i\s*<\s*\w+\.length', 0.40, "Iterate toàn bộ array - gas cost không giới hạn"),
        (r'for[^{]*\{[^}]*\.transfer\s*\(', 0.55, "Transfer ETH trong vòng lặp - có thể fail cả batch nếu 1 địa chỉ revert"),
        (r'for[^{]*\{[^}]*\.call\s*[\({]', 0.50, "External call trong vòng lặp"),
        (r'for[^{]*\{[^}]*(?:payable|\.transfer|\.send)', 0.45, "Push payment pattern trong loop - nguy cơ DoS"),
        (r'for[^{]*\{[^}]*\w+\[', 0.30, "Ghi vào storage mapping/array trong vòng lặp"),
        (r'constructor[^{]*\{[^}]*require\s*\((?:(?!msg\.sender).)*\)', 0.25, "require trong constructor - kiểm tra điều kiện không thể thay đổi"),
     ],
     "negative_patterns": [
        (r'gasleft\s*\(\s*\)', 0.15, "Kiểm tra gas còn lại"),
     ]},
    {"vuln_id": "SWC-128", "vuln_type": "dos_revert_griefing", "risk_level": "High",
     "patterns": [
        (r'receive\s*\(\s*\)\s*external\s*payable\s*\{[^}]*revert', 0.50, "receive() luôn revert - contract không nhận ETH được"),
        (r'fallback\s*\(\s*\)\s*external\s*(?:payable\s*)?\{[^}]*revert', 0.45, "fallback() revert - nguy cơ griefing"),
        (r'(?:\.call|\.transfer|\.send)\s*[({][^)]*\)\s*;(?:(?!if|require|revert).){0,50}', 0.35, "External call fail không được xử lý có thể DoS toàn bộ hàm"),
     ],
     "negative_patterns": []},
    {"vuln_id": "SWC-FLASHLOAN", "vuln_type": "flash_loan_vulnerability", "risk_level": "High",
     "patterns": [
        (r'getPrice\s*\(|getReserve\s*\(|getAmountsOut\s*\(', 0.40, "Dùng on-chain price query - dễ bị flash loan thao túng"),
        (r'address\s*\(\s*this\s*\)\.balance', 0.45, "Dùng contract balance làm oracle giá - dễ bị flash loan"),
        (r'token\.balanceOf\s*\(\s*address\s*\(\s*this\s*\)\s*\)', 0.40, "Dùng token balance của contract - dễ bị flash loan"),
        (r'IUniswapV[23](?:Pair|Router)\s*\(\s*\w+\s*\)\.getReserves\s*\(', 0.45, "Dùng Uniswap spot price (getReserves) - không an toàn, dùng TWAP"),
        (r'\.getReserves\s*\(\s*\)', 0.40, "Lấy reserves trực tiếp - thao túng được bằng flash loan"),
        (r'function\s+(?:executeOperation|onFlashLoan|uniswapV2Call|pancakeCall)\s*\(', 0.35, "Flash loan callback - kiểm tra reentrancy guard"),
     ],
     "negative_patterns": [
        (r'TWAP|twap|timeWeightedAverage|consult\s*\(', 0.40, "Dùng TWAP oracle - an toàn hơn spot price"),
        (r'Chainlink|AggregatorV3|latestRoundData', 0.35, "Dùng Chainlink oracle - an toàn hơn"),
     ]},
    {"vuln_id": "SWC-114", "vuln_type": "front_running", "risk_level": "High",
     "patterns": [
        (r'function\s+approve\s*\([^)]*\)[^{]*\{(?:(?!increaseAllowance|decreaseAllowance).){0,300}', 0.35, "approve() pattern - nên dùng increaseAllowance/decreaseAllowance"),
        (r'allowance\s*\[\s*\w+\s*\]\s*\[\s*\w+\s*\]\s*=', 0.30, "Set allowance trực tiếp - nguy cơ front-running"),
        (r'block\.number\s*[+\-]\s*\d+', 0.25, "Logic dựa trên block number relative - dễ bị front-run"),
        (r'function\s+(?:bid|auction|buy|purchase)\s*\([^)]*uint[^)]*\)', 0.30, "Hàm đấu giá/mua - kiểm tra có commit-reveal không"),
        (r'tx\.gasprice', 0.25, "Logic phụ thuộc tx.gasprice - dễ bị miner thao túng"),
     ],
     "negative_patterns": [
        (r'commit\w*\[|reveal\w*\(|CommitReveal', 0.35, "Có commit-reveal scheme"),
     ]},
    {"vuln_id": "SWC-116", "vuln_type": "timestamp_dependence", "risk_level": "Medium",
     "patterns": [
        (r'block\.timestamp\s*(?:<|>|<=|>=|==|!=)', 0.45, "So sánh block.timestamp - miner có thể điều chỉnh ~15s"),
        (r'\bnow\b\s*(?:<|>|<=|>=|==|!=)', 0.40, "Dùng 'now' để so sánh - deprecated, alias của block.timestamp"),
        (r'block\.timestamp\s*[%+\-]', 0.35, "Tính toán dựa trên timestamp"),
        (r'require\s*\([^)]*block\.timestamp', 0.40, "require dựa trên timestamp"),
        (r'if\s*\([^)]*block\.timestamp[^)]*\)', 0.35, "Điều kiện if dựa trên timestamp"),
        (r'uint\s*\(\s*keccak256\s*\([^)]*block\.timestamp', 0.60, "Dùng timestamp để tạo random - KHÔNG AN TOÀN"),
        (r'block\.timestamp\s*%\s*\d+', 0.55, "Modulo timestamp để random - predictable bởi miner"),
     ],
     "negative_patterns": []},
    {"vuln_id": "SWC-120", "vuln_type": "block_info_dependence", "risk_level": "Medium",
     "patterns": [
        (r'blockhash\s*\(', 0.55, "blockhash làm random - chỉ valid với 256 block gần nhất, predictable"),
        (r'block\.blockhash\s*\(', 0.55, "block.blockhash - deprecated"),
        (r'block\.difficulty\s*[%+\-*]', 0.45, "block.difficulty làm random source - miner có thể chọn"),
        (r'block\.prevrandao', 0.30, "block.prevrandao (post-merge) - vẫn có thể bị manipulate bởi validator"),
        (r'keccak256\s*\([^)]*block\.(?:number|difficulty|coinbase)', 0.50, "Hash của block metadata làm random - không an toàn"),
        (r'uint\s*\(\s*keccak256\s*\([^)]*abi\.encodePacked\s*\([^)]*block\.', 0.45, "Entropy từ block info - predictable"),
     ],
     "negative_patterns": [
        (r'VRF|vrfCoordinator|requestRandomness|Chainlink', 0.50, "Dùng Chainlink VRF - an toàn"),
     ]},
    {"vuln_id": "SWC-114", "vuln_type": "transaction_ordering_dependency", "risk_level": "Medium",
     "patterns": [
        (r'mapping\s*\(\s*address\s*=>\s*uint\w*\s*\)\s*public\s+(?:balances|amounts)', 0.25, "Public balance mapping - kiểm tra race condition"),
        (r'if\s*\([^)]*balances?\[', 0.30, "Điều kiện kiểm tra balance - có thể bị race condition"),
        (r'price\s*=\s*[^;]+;\s*[^;]*transfer\s*\(', 0.40, "Cập nhật giá rồi transfer trong cùng tx - sandwich attack"),
     ],
     "negative_patterns": []},
    {"vuln_id": "SWC-126", "vuln_type": "insufficient_gas_griefing", "risk_level": "Medium",
     "patterns": [
        (r'\.call\s*\{[^}]*\}', 0.20, "External call - kiểm tra gas forwarding"),
        (r'\.call\s*\{[^}]*gas\s*:\s*(?:[0-9]{1,4})\b', 0.45, "Hardcode gas limit nhỏ - có thể không đủ gas cho callee"),
        (r'\.transfer\s*\(\s*\w+\s*\)', 0.20, ".transfer() forward 2300 gas - có thể fail với complex receiver"),
        (r'for\s*\([^;]*;\s*[^;]*;\s*[^)]*\)\s*\{[^}]*(?:sstore|SSTORE|\[\s*\w+\s*\]\s*=)', 0.40, "SSTORE trong vòng lặp - gas cost tuyến tính với số phần tử"),
     ],
     "negative_patterns": []},
    {"vuln_id": "SWC-CAST", "vuln_type": "unsafe_type_casting", "risk_level": "Medium",
     "patterns": [
        (r'uint(?:8|16|32|64|128)\s*\(\s*(?!uint(?:8|16|32|64|128))\w+\s*\)', 0.40, "Explicit downcast uint256 -> smaller - kiểm tra overflow"),
        (r'uint\w*\s*\(\s*int\w*\s*\w+\s*\)', 0.45, "Cast int sang uint - nếu giá trị âm sẽ wrap thành số lớn"),
        (r'int\w*\s*\(\s*uint\w*\s*\w+\s*\)', 0.35, "Cast uint sang int - kiểm tra overflow"),
        (r'uint(?:256|160)?\s*\(\s*address\s*\(', 0.30, "Cast address sang uint"),
        (r'uint\w*\s*\(\s*bytes\w*\s*\w+\s*\)', 0.35, "Cast bytes sang uint"),
        (r'assembly\s*\{[^}]*(?:shr|shl|and)\s+0x', 0.30, "Assembly bitshift/mask - kiểm tra logic cẩn thận"),
     ],
     "negative_patterns": [
        (r'SafeCast|toUint\d+\s*\(', 0.40, "Dùng SafeCast library"),
     ]},
    {"vuln_id": "SWC-103", "vuln_type": "floating_pragma", "risk_level": "Low",
     "patterns": [
        (r'pragma\s+solidity\s+\^0\.[0-7]\.', 0.50, "Floating pragma với phiên bản < 0.8 - nhiều known bug"),
        (r'pragma\s+solidity\s+>=\s*0\.[0-4]\.', 0.45, "Cho phép compile với phiên bản rất cũ"),
        (r'pragma\s+solidity\s+0\.[0-6]\.', 0.40, "Pragma cố định phiên bản cũ < 0.7"),
        (r'pragma\s+solidity\s+0\.4\.', 0.60, "Pragma 0.4.x - rất cũ, nhiều lỗ hổng đã biết"),
        (r'pragma\s+solidity\s+0\.5\.', 0.40, "Pragma 0.5.x - thiếu nhiều bảo vệ của 0.8+"),
        (r'\A(?![\s\S]*SPDX-License-Identifier)', 0.20, "Không có SPDX license identifier"),
     ],
     "negative_patterns": [
        (r'pragma\s+solidity\s+\^?0\.8\.', 0.50, "Dùng Solidity 0.8+ - an toàn hơn"),
     ]},
    {"vuln_id": "SWC-NAMING", "vuln_type": "naming_convention", "risk_level": "Low",
     "patterns": [
        (r'^\s*(?:uint|int|bool|address|bytes|string|mapping)\s+(?!_)\w+\s*;', 0.15, "Biến state nên có tiền tố _ để phân biệt với local variable"),
        (r'function\s+(?!_)[a-z]\w+\s*\([^)]*\)\s*(?:internal|private)', 0.10, "Hàm internal/private nên có tiền tố _"),
        (r'(?:uint|int|bytes|address)\s+(?:constant|immutable)\s+[a-z]\w+', 0.10, "Hằng số nên dùng SCREAMING_SNAKE_CASE"),
        (r'event\s+[a-z]\w+\s*\(', 0.10, "Event nên dùng PascalCase"),
     ],
     "negative_patterns": []},
    {"vuln_id": "SWC-REDUNDANT", "vuln_type": "redundant_code", "risk_level": "Low",
     "patterns": [
        (r'require\s*\(\s*true\s*\)', 0.60, "require(true) - vô nghĩa"),
        (r'return\s*;[^}]*\w+\s*=', 0.40, "Code sau return statement - dead code"),
        (r'emit\s+\w+\s*\([^)]*\)\s*;\s*emit\s+\w+\s*\([^)]*\)\s*;', 0.20, "Emit hai event liên tiếp - có thể gộp"),
        (r'import\s+"[^"]+\.sol"\s*;', 0.05, "Import - kiểm tra có thực sự dùng không"),
        (r'uint\w*\s+\w+\s*=\s*\d+\s*/\s*\d+\s*;', 0.30, "Phép chia integer literal - kết quả luôn truncate, kiểm tra intent"),
     ],
     "negative_patterns": []},
    {"vuln_id": "SWC-121", "vuln_type": "signature_replay", "risk_level": "High",
     "patterns": [
        (r'ecrecover\s*\(', 0.45, "ecrecover - kiểm tra nonce để tránh replay attack"),
        (r'keccak256\s*\([^)]*abi\.encodePacked\s*\([^)]*\)\s*\)[^)]*ecrecover', 0.40, "Signature không có chain ID - cross-chain replay"),
        (r'ecrecover[^;]{0,300}mapping[^;]*nonce', 0.20, "Kiểm tra nonce tracking sau ecrecover"),
     ],
     "negative_patterns": [
        (r'nonce\s*\+\+|nonces\s*\[|_useNonce', 0.40, "Có nonce tracking"),
        (r'block\.chainid|chainId|DOMAIN_SEPARATOR', 0.35, "Có chain ID trong signature"),
     ]},
    {"vuln_id": "SWC-ORACLE", "vuln_type": "price_oracle_manipulation", "risk_level": "High",
     "patterns": [
        (r'balanceOf\s*\([^)]*\)\s*/\s*totalSupply', 0.50, "Tính price ratio từ balanceOf/totalSupply - dễ bị flash loan"),
        (r'reserve\d*\s*[*/]\s*reserve\d*', 0.45, "Tính price từ reserves - spot price có thể bị thao túng"),
        (r'getAmountOut\s*\(|getAmountIn\s*\(', 0.40, "Dùng AMM price function - kiểm tra slippage protection"),
        (r'amountOutMin\s*=\s*0|minOut\s*=\s*0|slippage\s*=\s*0', 0.55, "Slippage protection = 0 - sandwich attack dễ dàng"),
     ],
     "negative_patterns": [
        (r'TWAP|twap|observe\s*\(|consult\s*\(', 0.40, "Dùng TWAP"),
        (r'amountOutMin\s*>', 0.25, "Có slippage protection"),
     ]},
    {"vuln_id": "SWC-STORAGE", "vuln_type": "storage_collision", "risk_level": "Critical",
     "patterns": [
        (r'\.delegatecall\s*\([^)]*\)', 0.55, "delegatecall - nguy cơ storage collision và unauthorized code"),
        (r'bytes32\s+(?:private\s+)?constant\s+\w+IMPLEMENTATION\w*\s*=\s*keccak256', 0.25, "Implementation slot - kiểm tra EIP-1967 compliance"),
        (r'assembly\s*\{[^}]*sload\s*\(', 0.30, "Assembly sload - kiểm tra storage slot collision"),
        (r'assembly\s*\{[^}]*sstore\s*\(', 0.30, "Assembly sstore - kiểm tra storage slot collision"),
     ],
     "negative_patterns": [
        (r'EIP1967|eip1967|0x360894', 0.35, "Dùng EIP-1967 storage slot"),
     ]},
    {"vuln_id": "SWC-111", "vuln_type": "deprecated_solidity_usage", "risk_level": "Low",
     "patterns": [
        (r'\bmsg\.gas\b', 0.35, "msg.gas đã bị thay thế bởi gasleft()"),
        (r'\bsha3\s*\(', 0.35, "sha3() đã bị thay thế bởi keccak256()"),
        (r'\bthrow\s*;', 0.40, "throw đã bị thay thế bởi revert()/require()"),
        (r'\.callcode\s*\(', 0.45, "callcode đã bị thay thế bởi delegatecall"),
     ],
     "negative_patterns": []},
    {"vuln_id": "SWC-129", "vuln_type": "incorrect_unary_expression", "risk_level": "Medium",
     "patterns": [
        (r'(?<![+\-*/%&|^])=\s*\+\s*(?:\d+|[A-Za-z_]\w*)', 0.55, "Biểu thức '= +' thường là lỗi gõ nhầm của '+='"),
     ],
     "negative_patterns": []},
    {"vuln_id": "SWC-135", "vuln_type": "tautological_comparison", "risk_level": "Medium",
     "patterns": [
        (r'\b([A-Za-z_]\w*)\s*(?:==|!=|<=|>=|<|>)\s*\1\b', 0.55, "Một giá trị đang được so sánh với chính nó"),
     ],
     "negative_patterns": []},
    {"vuln_id": "STATIC-DIVIDE-BEFORE-MULTIPLY", "vuln_type": "divide_before_multiply", "risk_level": "Medium",
     "patterns": [
        (r'\b[A-Za-z_]\w*\s*/\s*[A-Za-z_]\w*\s*\*\s*[A-Za-z_]\w*\b', 0.35, "Chia số nguyên trước khi nhân có thể làm mất độ chính xác"),
     ],
     "negative_patterns": []},
]

_DEFAULT_GUARDS = {"onlyOwner", "onlyAdmin", "onlyRole", "onlyMinter", "onlyFounder", "authorized", "auth",
                   "restricted"}
_IGNORED_TYPES = {"contract", "function", "event", "error", "modifier", "return", "returns", "import", "pragma",
                  "using", "struct", "enum"}
_CLOSING_TO_OPENING = {")": "(", "]": "[", "}": "{"}
_FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}
_BIDI_CONTROLS = set("‪‫‬‭‮⁦⁧⁨⁩")
_ACCESS_SWCS = {"SWC-101", "SWC-105", "SWC-106"}
_BURN_OWN_BALANCE = ("balances[msg.sender]", "burner = msg.sender")


# Cắt ~120 ký tự quanh vị trí khớp làm bằng chứng (evidence) trong báo cáo.
def _extract_context(code: str, pos: int, window: int = 120) -> str:
    return code[max(0, pos - window):pos + window].replace("\n", " ").strip()


# Các khối có thân khớp `patterns` (tìm "}" khớp trên mã đã che comment/chuỗi), sắp theo vị trí.
def _iter_blocks(code: str, *patterns: re.Pattern) -> list[Block]:
    clean, blocks = mask_noncode(code), []
    for pattern in patterns:
        for match in pattern.finditer(clean):
            close = matching_close(clean, match.end() - 1)
            if close is not None:
                params = code[match.start("params"):match.end("params")] if "params" in pattern.groupindex else ""
                blocks.append(Block(match.group("name"), params, code[match.start("header"):match.end("header")],
                                    match.start(), match.end(), close, clean[match.end():close]))
    return sorted(blocks, key=lambda block: block.start)


# Mọi hàm có thân: function / constructor / receive / fallback.
def _functions(code: str) -> list[Block]:
    return _iter_blocks(code, RE_FUNCTION_WITH_BODY, RE_SPECIAL_FUNCTION_WITH_BODY)


# Mọi contract có thân.
def _contracts(code: str) -> list[Block]:
    return _iter_blocks(code, RE_CONTRACT_WITH_BODY)


# Thân hàm/modifier có kiểm tra người gọi: require(msg.sender == ...), if (msg.sender != ...) revert, owners[msg.sender]...
def _has_sender_check(body: str) -> bool:
    return any(pattern.search(body) for pattern in RE_SENDER_CHECKS)


# Tên các modifier phân quyền: tên kiểu onlyXxx / auth / restricted, HOẶC thân modifier có kiểm tra msg.sender.
def extract_guard_modifiers(code: str) -> set[str]:
    named = {name for name in RE_MODIFIER_NAME.findall(mask_noncode(code)) if RE_GUARD_MODIFIER_NAME.match(name)}
    return named | {block.name for block in _iter_blocks(code, RE_MODIFIER_WITH_BODY) if _has_sender_check(block.body)}


# Hàm đã được bảo vệ quyền truy cập: internal/private, có guard modifier trong header, hoặc thân kiểm tra msg.sender.
def is_function_guarded(header: str, body: str, guard_modifiers: set[str] | None = None) -> bool:
    return bool(RE_INTERNAL_OR_PRIVATE.search(header)
                or any(re_word(guard).search(header) for guard in (guard_modifiers or set()) | _DEFAULT_GUARDS)
                or any(token.startswith("only") and len(token) > 4 for token in RE_WORD_TOKEN.findall(header))
                or _has_sender_check(body))


# Tách danh sách đối số theo dấu phẩy ở mức ngoài cùng (bỏ qua dấu phẩy trong ngoặc lồng nhau / chuỗi).
def _split_top_level_arguments(value: str) -> list[str]:
    arguments, start, depths, quote, escaped = [], 0, {"(": 0, "[": 0, "{": 0}, None, False
    for pos, char in enumerate(value):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in depths:
            depths[char] += 1
        elif char in _CLOSING_TO_OPENING:
            opening = _CLOSING_TO_OPENING[char]
            depths[opening] = max(0, depths[opening] - 1)
        elif char == "," and not any(depths.values()):
            arguments.append(value[start:pos].strip())
            start = pos + 1
    tail = value[start:].strip()
    return arguments + [tail] if tail or arguments else arguments


# Ánh xạ tên biến -> kiểu từ mọi khai báo (biến trạng thái, tham số, biến cục bộ, thành viên struct).
def _collect_symbol_types(code: str) -> dict[str, str]:
    bindings = {}
    for match in RE_SYMBOL_DECLARATION.finditer(mask_noncode(code)):
        type_name = RE_WHITESPACE.sub("", match.group("type"))
        if type_name.lower() not in _IGNORED_TYPES:
            bindings[match.group("name")] = type_name
    return bindings


# Đối số có kiểu động (chuỗi literal, string/bytes, mảng) không - cho kiểm tra va chạm abi.encodePacked.
def _is_dynamic_abi_argument(argument: str, symbol_types: dict[str, str]) -> bool:
    value = argument.strip()
    if value and value[0] in "\"'" and value.endswith(value[0]):
        return True
    identifier = RE_IDENTIFIER.fullmatch(value)
    type_name = symbol_types.get(identifier.group(), "").replace(" ", "") if identifier else ""
    return bool(identifier) and (type_name in {"string", "bytes"} or "[]" in type_name)


# Đối tượng nhận lời gọi có vẻ là token ERC20 (ép kiểu IERC20(...) hoặc biến có kiểu ERC20).
def _looks_like_erc20(receiver: str, symbol_types: dict[str, str]) -> bool:
    compact = RE_WHITESPACE.sub("", receiver)
    cast = RE_CAST_PREFIX.match(compact)
    return "erc20" in (cast.group("type") if cast else symbol_types.get(compact, "")).lower()


# Tham số hàm -> {tên: (kiểu, vị trí dữ liệu memory/storage/calldata)}.
def _parse_parameter_details(params: str) -> dict[str, tuple[str, str]]:
    details = {}
    for parameter in _split_top_level_arguments(params):
        if match := RE_PARAMETER.match(parameter):
            details[match.group("name")] = (RE_WHITESPACE.sub("", match.group("type")), match.group("location") or "")
    return details


# Câu lệnh ";" ở mức ngoài cùng của một contract (khai báo biến trạng thái) -> [(vị trí, câu lệnh gốc)].
def _contract_statements(code: str, contract: Block) -> list[tuple[int, str]]:
    spans = statements_at_depth(mask_noncode(code), contract.body_start, contract.end, 0)
    return [(start, code[start:end]) for start, end in spans]


# Như _contract_statements nhưng cho mọi contract trong file.
def _all_contract_statements(code: str) -> list[tuple[int, str]]:
    return [item for contract in _contracts(code) for item in _contract_statements(code, contract)]


# Biến trạng thái của một contract -> (kiểu, vị trí khai báo, có khởi tạo không).
def _collect_contract_state_declarations(code: str, contract: Block) -> dict[str, tuple[str, int, bool]]:
    result = {}
    for position, statement in _contract_statements(code, contract):
        clean = mask_noncode(statement)
        position += len(clean) - len(clean.lstrip())
        if match := RE_STATE_DECLARATION.match(clean):
            result[match.group("name")] = (RE_WHITESPACE.sub("", match.group("type")), position,
                                           "=" in match.group("tail"))
        elif match := RE_STATE_MAPPING.match(clean):
            result[match.group("name")] = ("mapping", position, False)
    return result


# Biến trạng thái (mọi contract) -> kiểu.
def _collect_state_symbol_types(code: str) -> dict[str, str]:
    matches = (RE_STATE_TYPE_AND_NAME.match(mask_noncode(statement)) for _, statement in _all_contract_statements(code))
    return {match.group("name"): RE_WHITESPACE.sub("", match.group("type")) for match in matches if match}


# Tên mapping -> kiểu giá trị (mapping(address => User) users -> "User").
def _collect_mapping_value_types(code: str) -> dict[str, str]:
    matches = (RE_MAPPING_VALUE_TYPE.match(mask_noncode(statement)) for _, statement in _all_contract_statements(code))
    return {match.group("name"): match.group("value") for match in matches if match}


# Mỗi struct -> (tập kiểu thành viên, có chứa mapping trực tiếp không).
def _collect_struct_members(code: str) -> dict[str, tuple[set[str], bool]]:
    clean, structs = mask_noncode(code), {}
    for match in RE_STRUCT_OPENING.finditer(clean):
        close = matching_close(clean, match.end() - 1)
        if close is not None:
            body = code[match.end():close]
            structs[match.group("name")] = (set(_collect_symbol_types(body).values()),
                                            bool(RE_MAPPING_KEYWORD.search(mask_noncode(body))))
    return structs


# Struct có chứa mapping (trực tiếp hoặc qua struct lồng nhau, chống vòng lặp) không.
def _struct_contains_mapping(name: str, structs: dict, seen: frozenset = frozenset()) -> bool:
    if name not in structs or name in seen:
        return False
    member_types, direct_mapping = structs[name]
    return direct_mapping or any(_struct_contains_mapping(RE_ARRAY_SUFFIX.sub("", member), structs, seen | {name})
                                 for member in member_types)


# Vùng [đầu, cuối) của các vòng lặp for/while trong khoảng [start, end).
def _iter_loop_ranges(code: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    clean = mask_noncode(code)
    for match in RE_LOOP_OPENING.finditer(clean, start, end):
        close_paren = matching_close(clean, clean.find("(", match.start(), match.end()), "(", ")")
        if close_paren is None or close_paren >= end:
            continue
        body_pos = close_paren + 1
        while body_pos < end and clean[body_pos].isspace():
            body_pos += 1
        if body_pos < end and clean[body_pos] == "{":
            close_body = matching_close(clean, body_pos)
            if close_body is not None and close_body <= end:
                yield match.start(), close_body + 1
        elif (semicolon := clean.find(";", body_pos, end)) >= 0:
            yield match.start(), semicolon + 1


# Lời gọi bắt đầu trong [start, end): (vị trí "(", vị trí ")" khớp), hoặc None nếu không đóng ngoặc.
def _call_parens(clean: str, start: int, end: int) -> tuple[int, int] | None:
    open_pos = clean.find("(", start, end)
    close = matching_close(clean, open_pos, "(", ")")
    return None if close is None else (open_pos, close)


# Đóng gói các hit (vị trí, đoạn khớp, trọng số, mô tả) thành MỘT VulnerabilityFinding confirmed (không có hit -> None).
def _build_semantic_finding(code: str, vuln_id: str, vuln_type: str, risk_level: str,
                            hits: list[tuple[int, str, float, str]]) -> VulnerabilityFinding | None:
    if not hits:
        return None
    finding = VulnerabilityFinding(vuln_id, vuln_type, risk_level, confirmed=True)
    for pos, matched_text, weight, description in hits:
        line = line_number(code, pos)
        finding.matches.append(PatternMatch(line, matched_text[:80], _extract_context(code, pos), weight))
        finding.notes.append(f"L{line}: {description}")
    return finding


# [SWC-130] Ký tự Unicode điều khiển hướng chữ (RLO U+202E...) dùng để giấu mã độc.
def _detect_bidirectional_controls(code: str) -> VulnerabilityFinding | None:
    hits = [(pos, f"U+{ord(char):04X}", 0.80, "Ký tự điều khiển hướng văn bản có thể che giấu thứ tự token hiển thị")
            for pos, char in enumerate(code) if char in _BIDI_CONTROLS]
    return _build_semantic_finding(code, "SWC-130", "right_to_left_override", "High", hits)


# Dùng "^" (XOR) với hằng số khi ý định là lũy thừa "**" (vd 2^256); pragma "^0.x" được che trước.
def _detect_incorrect_exponentiation(code: str) -> VulnerabilityFinding | None:
    clean = RE_PRAGMA_STATEMENT.sub(lambda m: " " * len(m.group(0)), mask_noncode(code))
    hits = [(m.start(), code[m.start():m.end()], 0.70, "Toán tử ^ là XOR; nếu muốn lũy thừa phải dùng **")
            for m in RE_XOR_EXPRESSION.finditer(clean)
            if any(RE_DECIMAL_NUMBER.fullmatch(value) for value in (m.group("left"), m.group("right")))]
    return _build_semantic_finding(code, "STATIC-INCORRECT-EXP", "incorrect_exponentiation_operator", "High", hits)


# [SWC-133] abi.encodePacked với >= 2 đối số kiểu động liền nhau -> va chạm hash.
def _detect_encode_packed_collision(code: str) -> VulnerabilityFinding | None:
    clean, symbol_types, hits = mask_noncode(code), _collect_symbol_types(code), []
    for match in RE_ENCODE_PACKED_CALL.finditer(clean):
        if parens := _call_parens(clean, match.start(), match.end()):
            flags = [_is_dynamic_abi_argument(arg, symbol_types)
                     for arg in _split_top_level_arguments(code[parens[0] + 1:parens[1]])]
            if any(left and right for left, right in zip(flags, flags[1:])):
                hits.append((match.start(), code[match.start():parens[1] + 1], 0.70,
                             "abi.encodePacked ghép các đối số kiểu động liền nhau mà không có ranh giới"))
    return _build_semantic_finding(code, "SWC-133", "abi_encode_packed_collision", "High", hits)


# Các lời gọi transfer/transferFrom trên đối tượng ERC20 trong từng hàm -> (hàm, match, method, đối số, vị trí ")").
def _iter_erc20_calls(code: str) -> Iterator[tuple[Block, re.Match, str, list[str], int]]:
    clean, symbol_types = mask_noncode(code), _collect_symbol_types(code)
    for function in _functions(code):
        for match in RE_ERC20_TRANSFER_CALL.finditer(clean, function.body_start, function.end):
            parens = _looks_like_erc20(match.group("receiver"), symbol_types) and _call_parens(
                clean, match.start("method"), match.end())
            if parens and parens[1] <= function.end:
                yield function, match, match.group("method"), _split_top_level_arguments(
                    code[parens[0] + 1:parens[1]]), parens[1]


# transferFrom(from, ...) với "from" là tham số address do caller truyền, không ràng buộc msg.sender.
def _detect_arbitrary_erc20_transfer_from(code: str) -> VulnerabilityFinding | None:
    hits = []
    for function, match, method, arguments, close in _iter_erc20_calls(code):
        source = arguments[0].strip() if arguments else ""
        if method == "transferFrom" and len(arguments) >= 3 \
                and _collect_symbol_types(function.params).get(source, "").lower() == "address":
            hits.append((match.start(), code[match.start():close + 1], 0.65,
                         f"Nguồn token '{source}' do caller cung cấp và không ràng buộc với msg.sender"))
    return _build_semantic_finding(code, "STATIC-ARBITRARY-ERC20", "arbitrary_erc20_transfer_from", "High", hits)


# Gọi token.transfer/transferFrom thành một câu lệnh riêng, bỏ qua giá trị bool trả về.
def _detect_unchecked_erc20_transfer(code: str) -> VulnerabilityFinding | None:
    clean, hits = mask_noncode(code), []
    for function, match, method, _, close in _iter_erc20_calls(code):
        boundary = max(clean.rfind(delimiter, function.body_start, match.start()) for delimiter in ";{}")
        statement_start = boundary if boundary >= function.body_start else function.body_start - 1
        suffix = close + 1
        while suffix < function.end and clean[suffix].isspace():
            suffix += 1
        if not clean[statement_start + 1:match.start()].strip() and suffix < function.end and clean[suffix] == ";":
            hits.append((match.start(), code[match.start():close + 1], 0.60,
                         f"Giá trị bool trả về từ ERC20 {method}() bị bỏ qua"))
    return _build_semantic_finding(code, "STATIC-UNCHECKED-ERC20", "unchecked_erc20_transfer", "High", hits)


# Contract nhận được ETH (hàm payable) nhưng không có đường nào gửi ETH ra -> ETH bị khoá.
def _detect_locked_ether(code: str) -> VulnerabilityFinding | None:
    clean, functions, hits = mask_noncode(code), _functions(code), []
    for contract in _contracts(code):
        abstract = RE_ABSTRACT_AT_END.search(clean[max(0, contract.start - 16):contract.start])
        if abstract or RE_IS_KEYWORD.search(contract.header):
            continue
        payable = [f for f in functions if contract.body_start <= f.start < contract.end and RE_PAYABLE.search(f.header)
                   and (f.name in {"constructor", "receive", "fallback"} or RE_PUBLIC_OR_EXTERNAL.search(f.header))]
        if payable and not RE_OUTGOING_ETHER.search(contract.body):
            entry = payable[0]
            hits.append((entry.start, code[entry.start:entry.body_start].strip(), 0.55,
                         f"Contract '{contract.name}' nhận được Ether nhưng không có đường gửi Ether ra"))
    return _build_semantic_finding(code, "STATIC-LOCKED-ETHER", "locked_ether", "Medium", hits)


# Dùng msg.value trong vòng lặp của hàm payable -> cùng một khoản ETH bị tính nhiều lần.
def _detect_msg_value_in_loop(code: str) -> VulnerabilityFinding | None:
    clean, hits = mask_noncode(code), []
    for function in _functions(code):
        if RE_PAYABLE.search(function.header):
            for loop_start, loop_end in _iter_loop_ranges(code, function.body_start, function.end):
                if value := RE_MSG_VALUE.search(RE_MSG_VALUE_ZERO_CHECK.sub("", clean[loop_start:loop_end])):
                    hits.append((loop_start + value.start(), "msg.value", 0.55,
                                 f"Hàm payable '{function.name}' tái sử dụng msg.value trong vòng lặp"))
    return _build_semantic_finding(code, "STATIC-MSG-VALUE-LOOP", "msg_value_in_loop", "Medium", hits)


# [SWC-112] Hàm payable public gọi delegatecall trong vòng lặp (msg.value bị tái sử dụng).
def _detect_delegatecall_in_loop(code: str) -> VulnerabilityFinding | None:
    clean, hits = mask_noncode(code), []
    for function in _functions(code):
        if RE_PAYABLE.search(function.header) and RE_PUBLIC_OR_EXTERNAL.search(function.header):
            for loop_start, loop_end in _iter_loop_ranges(code, function.body_start, function.end):
                if call := RE_DELEGATECALL.search(clean[loop_start:loop_end]):
                    pos = loop_start + call.start()
                    hits.append((pos, code[pos:pos + len(call.group())], 0.65,
                                 f"Hàm payable '{function.name}' thực hiện delegatecall trong vòng lặp"))
    return _build_semantic_finding(code, "SWC-112", "delegatecall_in_loop", "High", hits)


# Tham số address ghi vào state / làm đích chuyển tiền mà không kiểm tra != address(0) (như Slither missing-zero-check).
def _detect_missing_zero_address(code: str) -> VulnerabilityFinding | None:
    hits = []
    for function in _functions(code):
        parameter_types = _collect_symbol_types(function.params)
        addresses = [name for name, type_name in parameter_types.items() if type_name.lower() == "address"]
        local_names = set(_collect_symbol_types(function.body)) if addresses else set()
        for parameter in addresses:
            body = function.body
            if re_zero_address_require(parameter).search(body) or re_zero_address_revert_if(parameter).search(body):
                continue
            assignment = re_assignment_from(parameter).search(body)
            lhs = RE_IDENTIFIER.match(assignment.group("lhs")) if assignment else None
            lhs_name = lhs.group() if lhs else ""
            if assignment and lhs_name not in local_names and lhs_name not in parameter_types:
                pos = assignment.start()
            elif destination := re_call_target(parameter).search(body):
                pos = destination.start()
            else:
                continue
            hits.append((function.body_start + pos, parameter, 0.40,
                         f"Address '{parameter}' được ghi vào state/dùng làm call target "
                         "mà chưa kiểm tra zero address"))
    return _build_semantic_finding(code, "STATIC-MISSING-ZERO-ADDRESS", "missing_zero_address_validation", "Low", hits)


# Trong assembly (Yul), shl/shr bị đảo tham số (shift là biến, value là hằng).
def _detect_incorrect_shift_in_assembly(code: str) -> VulnerabilityFinding | None:
    clean, hits = mask_noncode(code), []
    for assembly in RE_ASSEMBLY_BLOCK.finditer(clean):
        close = matching_close(clean, assembly.end() - 1)
        if close is not None:
            hits += [(shift.start(), code[shift.start():shift.end()], 0.75,
                      "Tham số shift trong Yul có vẻ bị đảo: shift amount là biến còn value là hằng")
                     for shift in RE_YUL_SHIFT_WITH_CONSTANT_VALUE.finditer(clean, assembly.end(), close)]
    return _build_semantic_finding(code, "STATIC-INCORRECT-SHIFT", "incorrect_shift_in_assembly", "High", hits)


# Gán độ dài mảng storage (arr.length = ...) bằng giá trị phụ thuộc tham số do caller truyền.
def _detect_controlled_array_length(code: str) -> VulnerabilityFinding | None:
    state_arrays = {name for name, type_name in _collect_state_symbol_types(code).items() if "[]" in type_name}
    if not state_arrays:
        return None
    clean, hits = mask_noncode(code), []
    for function in _functions(code):
        parameters = set(_parse_parameter_details(function.params))
        if not RE_PUBLIC_OR_EXTERNAL.search(function.header) or not parameters:
            continue
        for assignment in RE_ARRAY_LENGTH_ASSIGNMENT.finditer(clean, function.body_start, function.end):
            tainted = next((p for p in parameters if re_word(p).search(assignment.group("rhs"))), None)
            if assignment.group("array") in state_arrays and tainted:
                hits.append((assignment.start(), code[assignment.start():assignment.end()], 0.70,
                             f"Độ dài storage array phụ thuộc input '{tainted}' của caller"))
    return _build_semantic_finding(code, "STATIC-CONTROLLED-ARRAY-LENGTH", "controlled_array_length", "High", hits)


# Truyền mảng storage vào tham số "memory" rồi sửa trong hàm được gọi -> chỉ sửa bản sao, state không đổi.
def _detect_array_by_reference(code: str) -> VulnerabilityFinding | None:
    clean, functions, modifying = mask_noncode(code), _functions(code), []
    for function in functions:
        details = _parse_parameter_details(function.params)
        for index, (name, (type_name, location)) in enumerate(details.items()):
            if "[" in type_name and location != "storage" and re_indexed_write(name).search(function.body):
                modifying.append((function, len(details), index, name))
    state_arrays = {name for name, type_name in _collect_state_symbol_types(code).items() if "[" in type_name}
    if not modifying or not state_arrays:
        return None
    hits, seen = [], set()
    for caller in functions:
        storage_arrays = state_arrays | {m.group("name") for m in RE_STORAGE_ARRAY_LOCAL.finditer(caller.body)}
        for callee, arity, index, parameter in modifying:
            for call in re_call(callee.name).finditer(clean, caller.body_start, caller.end):
                parens = _call_parens(clean, call.start(), call.end())
                if parens is None or parens[1] > caller.end:
                    continue
                arguments = _split_top_level_arguments(code[parens[0] + 1:parens[1]])
                argument = arguments[index].strip() if len(arguments) == arity and index < len(arguments) else None
                if argument in storage_arrays and call.start() not in seen:
                    seen.add(call.start())
                    hits.append((call.start(), code[call.start():parens[1] + 1], 0.70,
                                 f"Storage array '{argument}' được truyền vào '{callee.name}' "
                                 f"qua tham số '{parameter}' không phải storage"))
    return _build_semantic_finding(code, "STATIC-ARRAY-BY-REFERENCE", "array_by_reference", "High", hits)


# Contract có cả constructor() lẫn hàm trùng tên contract (kiểu cũ) -> nhập nhằng constructor.
def _detect_multiple_constructors(code: str) -> VulnerabilityFinding | None:
    functions, hits = _functions(code), []
    for contract in _contracts(code):
        declared = [f for f in functions if contract.body_start <= f.start < contract.end]
        modern = [f for f in declared if f.name == "constructor"]
        if modern and any(f.name == contract.name for f in declared):
            hits.append((modern[0].start, code[modern[0].start:modern[0].body_start].strip(), 0.80,
                         f"Contract '{contract.name}' khai báo đồng thời constructor() và function {contract.name}()"))
    return _build_semantic_finding(code, "STATIC-MULTIPLE-CONSTRUCTORS", "multiple_constructor_schemes", "High", hits)


# Mapping public trỏ tới struct chứa struct lồng (getter tự sinh sai trên Solidity 0.4.x).
def _detect_public_mapping_nested(code: str) -> VulnerabilityFinding | None:
    clean = mask_noncode(code)
    if not RE_PRAGMA_0_4.search(clean) or not (structs := _collect_struct_members(code)):
        return None
    hits = [(m.start(), code[m.start():m.end()], 0.75,
             f"Public mapping '{m.group('name')}' trả về struct lồng nhau sai trên Solidity 0.4.x")
            for m in RE_PUBLIC_MAPPING_VALUE_TYPE.finditer(clean)
            if m.group("value") in structs
            and any(RE_ARRAY_SUFFIX.sub("", member) in structs for member in structs[m.group("value")][0])]
    return _build_semantic_finding(code, "STATIC-PUBLIC-MAPPING-NESTED", "public_mapping_with_nested_struct", "High",
                                   hits)


# delete một struct có chứa mapping -> mapping bên trong KHÔNG bị xoá.
def _detect_mapping_deletion(code: str) -> VulnerabilityFinding | None:
    structs = _collect_struct_members(code)
    if not structs:
        return None
    symbol_types, mapping_values, hits = _collect_symbol_types(code), _collect_mapping_value_types(code), []
    for match in RE_DELETE_STATEMENT.finditer(mask_noncode(code)):
        base = match.group("base")
        struct_name = RE_ARRAY_SUFFIX.sub("", mapping_values.get(base) or symbol_types.get(base, ""))
        if _struct_contains_mapping(struct_name, structs):
            hits.append((match.start(), code[match.start():match.end()], 0.65,
                         f"delete không xóa dữ liệu mapping nằm trong struct '{struct_name}'"))
    return _build_semantic_finding(code, "STATIC-MAPPING-DELETION", "mapping_deletion_in_struct", "Medium", hits)


# Khoảng giá trị [min, max] của kiểu uintN / intN (kiểu khác -> None).
def _integer_type_range(type_name: str) -> tuple[int, int] | None:
    match = RE_INTEGER_TYPE.fullmatch(type_name)
    bits = int(match.group("bits") or "256") if match else 0
    if not match or bits < 8 or bits > 256 or bits % 8:
        return None
    return (0, 2 ** bits - 1) if match.group("unsigned") else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)


# So sánh "biến <op> hằng" có luôn đúng / luôn sai theo khoảng giá trị [low, high] của kiểu không.
def _range_makes_comparison_constant(low: int, high: int, operator: str, constant: int) -> bool:
    return {"<": high < constant or low >= constant, ">": low > constant or high <= constant,
            "<=": high <= constant or low > constant, ">=": low >= constant or high < constant}.get(operator, False)


# So sánh luôn đúng / luôn sai do kiểu dữ liệu (vd uint >= 0, uint8 < 300).
def _detect_type_based_tautology(code: str) -> VulnerabilityFinding | None:
    clean, state_types, hits = mask_noncode(code), _collect_state_symbol_types(code), []
    for function in _functions(code):
        parameters = {name: details[0] for name, details in _parse_parameter_details(function.params).items()}
        symbol_types = {**state_types, **parameters, **_collect_symbol_types(code[function.body_start:function.end])}
        for match in RE_ORDER_COMPARISON.finditer(clean, function.body_start, function.end):
            left, right, operator = match.group("left"), match.group("right"), match.group("operator")
            if left in symbol_types and RE_INTEGER_LITERAL.fullmatch(right):
                variable, constant = left, right
            elif right in symbol_types and RE_INTEGER_LITERAL.fullmatch(left):
                variable, constant, operator = right, left, _FLIP[operator]
            else:
                continue
            value_range = _integer_type_range(symbol_types[variable])
            if value_range and _range_makes_comparison_constant(*value_range, operator, int(constant, 0)):
                hits.append((match.start(), code[match.start():match.end()], 0.65,
                             f"So sánh luôn đúng hoặc luôn sai theo miền giá trị của {symbol_types[variable]}"))
    return _build_semantic_finding(code, "STATIC-TYPE-TAUTOLOGY", "type_based_tautology", "Medium", hits)


# Hằng true/false làm điều kiện cố định / dư thừa: if (true), x || true, false && y.
def _detect_boolean_constant_misuse(code: str) -> VulnerabilityFinding | None:
    clean, hits, seen = mask_noncode(code), [], set()
    for pattern in (RE_IF_CONSTANT_CONDITION, RE_BOOLEAN_CONSTANT_OPERAND):
        for match in pattern.finditer(clean):
            if match.start() not in seen:
                seen.add(match.start())
                hits.append((match.start(), code[match.start():match.end()], 0.55,
                             "Boolean constant làm điều kiện trở nên cố định hoặc dư thừa"))
    return _build_semantic_finding(code, "STATIC-BOOLEAN-CONSTANT", "boolean_constant_misuse", "Medium", hits)


# So sánh bằng chặt (==) với this.balance / balanceOf -> kẻ tấn công gửi ETH/token để phá điều kiện.
def _detect_dangerous_balance_equality(code: str) -> VulnerabilityFinding | None:
    clean, hits = mask_noncode(code), []
    for function in _functions(code):
        tainted = {match.group("name") for match in RE_BALANCE_ASSIGNMENT.finditer(function.body)}
        for equality in RE_STRICT_EQUALITY.finditer(function.body):
            absolute = function.body_start + equality.start()
            start = max(clean.rfind(delimiter, function.body_start, absolute) for delimiter in ";{}")
            start = start + 1 if start >= function.body_start else function.body_start
            ends = [pos for pos in (clean.find(delimiter, absolute, function.end) for delimiter in ";{}") if pos >= 0]
            expression = clean[start:min(ends) if ends else function.end]
            if RE_BALANCE_SOURCE.search(expression) or any(re_word(name).search(expression) for name in tainted):
                hits.append((absolute, "==", 0.60,
                             "Strict equality trên Ether/token balance có thể bị phá vỡ "
                             "bởi thay đổi balance ngoài dự kiến"))
    return _build_semantic_finding(code, "STATIC-DANGEROUS-BALANCE-EQUALITY", "dangerous_balance_equality", "Medium",
                                   hits)


# Ép số nguyên sang enum không kiểm tra phạm vi trên solc <= 0.4.4 (chưa revert khi ngoài miền).
def _detect_dangerous_enum_conversion(code: str) -> VulnerabilityFinding | None:
    clean = mask_noncode(code)
    if not RE_PRAGMA_0_4_0_TO_0_4_4.search(clean):
        return None
    enum_names = {match.group("name") for match in RE_ENUM_OPENING.finditer(clean)}
    hits = [(c.start(), code[c.start():c.end()], 0.65,
             f"Ép kiểu sang enum '{enum}' có thể nhận giá trị ngoài miền trên solc <= 0.4.4")
            for function in _functions(code) for enum in enum_names
            for c in re_enum_conversion(enum).finditer(clean, function.body_start, function.end)]
    return _build_semantic_finding(code, "STATIC-ENUM-CONVERSION", "dangerous_enum_conversion", "Medium", hits)


# Vị trí các lệnh Yul return(offset, size) bên trong một hàm.
def _assembly_return_positions(code: str, function: Block) -> list[tuple[int, int]]:
    clean, positions = mask_noncode(code), []
    for assembly in RE_ASSEMBLY_BLOCK.finditer(clean, function.body_start, function.end):
        close = matching_close(clean, assembly.end() - 1)
        if close is not None and close <= function.end:
            positions += [(call.start(), call.end())
                          for call in RE_YUL_RETURN_CALL.finditer(clean, assembly.end() - 1, close)]
    return positions


# Hàm internal/private có 2 giá trị trả về mà dùng Yul return() -> dừng TOÀN BỘ execution (thường ý định là "leave").
def _detect_return_instead_of_leave(code: str) -> VulnerabilityFinding | None:
    hits = []
    for function in _functions(code):
        returns = RE_RETURNS_CLAUSE.search(function.header)
        if RE_INTERNAL_OR_PRIVATE.search(function.header) and returns \
                and len(_split_top_level_arguments(returns.group("values"))) == 2:
            hits += [(start, code[start:end], 0.65, f"Hàm '{function.name}' có hai return value nhưng Yul return() "
                                                    "dừng toàn bộ execution; thường phải dùng leave")
                     for start, end in _assembly_return_positions(code, function)]
    return _build_semantic_finding(code, "STATIC-RETURN-INSTEAD-OF-LEAVE", "return_instead_of_leave", "High", hits)


# Gọi một hàm (trực tiếp hay gián tiếp) có Yul return() -> execution dừng, code phía sau lời gọi không bao giờ chạy.
def _detect_incorrect_assembly_return(code: str) -> VulnerabilityFinding | None:
    clean, functions = mask_noncode(code), _functions(code)
    halting = {function.name for function in functions if _assembly_return_positions(code, function)}
    if not halting:
        return None
    changed = True
    while changed:
        changed = False
        for function in functions:
            if function.name not in halting and any(re_call(name).search(function.body) for name in halting):
                halting.add(function.name)
                changed = True
    hits, seen = [], set()
    for function in functions:
        for target in halting:
            for call in re_call(target).finditer(clean, function.body_start, function.end):
                parens = _call_parens(clean, call.start(), call.end())
                if parens is None or parens[1] > function.end:
                    continue
                if clean[parens[1] + 1:function.end].strip(" \t\r\n;") and call.start() not in seen:
                    seen.add(call.start())
                    hits.append((call.start(), code[call.start():parens[1] + 1], 0.65,
                                 f"Lời gọi '{target}' có thể dừng execution bằng Yul return() "
                                 "trước phần code phía sau"))
    return _build_semantic_finding(code, "STATIC-INCORRECT-ASSEMBLY-RETURN", "incorrect_assembly_return", "High", hits)


# Pragma pin đúng các bản solc dính bug encode storage array năm 2019 (0.4.7-0.4.25, 0.5.0-0.5.9).
def _uses_exact_vulnerable_storage_array_compiler(code: str) -> bool:
    versions = [(m.group("minor"), int(m.group("patch"))) for m in RE_PINNED_PRAGMA_0_4_OR_0_5.finditer(mask_noncode(code))]
    return any((minor == "4" and 7 <= patch <= 25) or (minor == "5" and patch <= 9) for minor, patch in versions)


# ABIEncoderV2 encode sai mảng (struct) trong storage trên các solc bị lỗi.
def _detect_storage_abiencoderv2_array(code: str) -> VulnerabilityFinding | None:
    clean = mask_noncode(code)
    if not (_uses_exact_vulnerable_storage_array_compiler(code) and RE_PRAGMA_ABIENCODERV2.search(clean)):
        return None
    structs = _collect_struct_members(code)
    vulnerable = {name for name, type_name in _collect_state_symbol_types(code).items()
                  if type_name.count("[") >= 2 or ("[" in type_name and RE_ARRAY_SUFFIX.sub("", type_name) in structs)}
    hits = []
    for call in RE_ABI_ENCODE_CALL.finditer(clean):
        if parens := _call_parens(clean, call.start(), call.end()):
            arguments = _split_top_level_arguments(code[parens[0] + 1:parens[1]])
            if affected := next((arg.strip() for arg in arguments if arg.strip() in vulnerable), None):
                hits.append((call.start(), code[call.start():parens[1] + 1], 0.75,
                             f"ABIEncoderV2 có thể encode sai storage array '{affected}' trên compiler đã pin"))
    return _build_semantic_finding(code, "STATIC-ABIENCODERV2-STORAGE-ARRAY", "storage_abiencoderv2_array", "High",
                                   hits)


# Kiểu gốc intN của một kiểu mảng số nguyên có dấu ("int8[]" -> "int8").
def _signed_integer_base(type_name: str) -> str | None:
    match = RE_SIGNED_INT_ARRAY_TYPE.match(type_name)
    return match.group("base") if match else None


# Gán cả mảng số nguyên có dấu trong storage trên solc đã pin dính bug -> ghi sai giá trị.
def _detect_storage_signed_integer_array(code: str) -> VulnerabilityFinding | None:
    if not _uses_exact_vulnerable_storage_array_compiler(code):
        return None
    clean, symbol_types, hits = mask_noncode(code), _collect_symbol_types(code), []
    for name, target_type in _collect_state_symbol_types(code).items():
        if not _signed_integer_base(target_type):
            continue
        for assignment in re_assignment_to(name).finditer(clean):
            rhs = assignment.group("rhs").strip()
            source_base = _signed_integer_base(symbol_types.get(rhs, "")) if RE_IDENTIFIER.fullmatch(rhs) else None
            if rhs.startswith("[") or (source_base and source_base != _signed_integer_base(target_type)):
                hits.append((assignment.start(), code[assignment.start():assignment.end()], 0.65,
                             f"Gán toàn bộ signed storage array '{name}' kích hoạt compiler bug của solc đã pin"))
    return _build_semantic_finding(code, "STATIC-SIGNED-STORAGE-ARRAY", "signed_storage_array_compiler_bug", "High",
                                   hits)


# Các contract cha trực tiếp từ header "contract A is B, C".
def _direct_base_names(header: str) -> list[str]:
    inheritance = RE_INHERITANCE_LIST.search(header)
    bases = _split_top_level_arguments(inheritance.group("bases")) if inheritance else []
    return [match.group(1) for match in map(RE_LEADING_IDENTIFIER.match, bases) if match]


# Mọi biến trạng thái kế thừa từ các contract cha (đệ quy, chống vòng lặp kế thừa).
def _inherited_states(name: str, contracts: dict, declarations: dict, seen: frozenset = frozenset()) -> set[str]:
    if name not in contracts or name in seen:
        return set()
    states = set()
    for base in _direct_base_names(contracts[name].header):
        states |= set(declarations.get(base, {})) | _inherited_states(base, contracts, declarations, seen | {name})
    return states


# Contract con khai báo lại biến trạng thái trùng tên contract cha (che biến, sai logic).
def _detect_state_variable_shadowing(code: str) -> VulnerabilityFinding | None:
    contracts = {contract.name: contract for contract in _contracts(code)}
    declarations = {name: _collect_contract_state_declarations(code, contract) for name, contract in contracts.items()}
    hits = []
    for contract_name, state_variables in declarations.items():
        inherited = _inherited_states(contract_name, contracts, declarations)
        hits += [(position, variable, 0.75,
                  f"State variable '{variable}' trong '{contract_name}' che khuất biến cùng tên từ base contract")
                 for variable, (_, position, _) in state_variables.items()
                 if variable != "__gap" and variable in inherited]
    return _build_semantic_finding(code, "STATIC-STATE-SHADOWING", "state_variable_shadowing", "High", hits)


# Biến address trạng thái không bao giờ được gán mà vẫn được dùng làm đích chuyển ETH / call.
def _detect_uninitialized_state_address(code: str) -> VulnerabilityFinding | None:
    contracts = _contracts(code)
    base_names = {base for contract in contracts for base in _direct_base_names(contract.header)}
    hits = []
    for contract in contracts:
        if contract.name in base_names or _direct_base_names(contract.header):
            continue
        for name, (type_name, position, initialized) in _collect_contract_state_declarations(code, contract).items():
            if type_name == "address" and not initialized and not re_state_write(name).search(contract.body) \
                    and re_ether_destination(name).search(contract.body):
                hits.append((position, name, 0.75,
                             f"State address '{name}' không bao giờ được khởi tạo "
                             "nhưng được dùng làm đích chuyển Ether/call"))
    return _build_semantic_finding(code, "STATIC-UNINITIALIZED-STATE-ADDRESS", "uninitialized_state_address", "High",
                                   hits)


# 25 kiểm tra ngữ nghĩa (cài lại logic giống các detector của Slither, không gọi Slither).
_SEMANTIC_CHECKS = (
    _detect_bidirectional_controls, _detect_incorrect_exponentiation, _detect_encode_packed_collision,
    _detect_arbitrary_erc20_transfer_from, _detect_unchecked_erc20_transfer, _detect_locked_ether,
    _detect_msg_value_in_loop, _detect_delegatecall_in_loop, _detect_missing_zero_address,
    _detect_incorrect_shift_in_assembly, _detect_controlled_array_length, _detect_array_by_reference,
    _detect_multiple_constructors, _detect_public_mapping_nested, _detect_mapping_deletion,
    _detect_type_based_tautology, _detect_boolean_constant_misuse, _detect_dangerous_balance_equality,
    _detect_dangerous_enum_conversion, _detect_return_instead_of_leave, _detect_incorrect_assembly_return,
    _detect_storage_abiencoderv2_array, _detect_storage_signed_integer_array, _detect_state_variable_shadowing,
    _detect_uninitialized_state_address,
)


# Match nằm trong hàm đã được bảo vệ (is_function_guarded) hoặc hàm burn số dư của chính msg.sender -> bỏ qua.
def _is_protected_match(pos: int, clean: str, guards: set[str], functions: list[Block]) -> bool:
    function = next((f for f in functions if f.start <= pos <= f.end), None)
    if function:
        return is_function_guarded(function.header, function.body, guards) \
            or (function.name == "burn" and any(text in function.body for text in _BURN_OWN_BALANCE))
    header = RE_FUNCTION_HEADER_PREFIX.search(clean) if not functions and clean.strip().startswith("function") else None
    if not header:
        return False
    head, body = clean[:header.end()], clean[header.end():]
    return is_function_guarded(head, body, guards) \
        or ("function burn" in head and any(text in body for text in _BURN_OWN_BALANCE))


# Khớp các regex của MỘT luật trên mã đã che (luật SWC-101/105/106 bỏ các match nằm trong hàm đã được bảo vệ).
def _run_pattern_check(entry: dict, clean: str, original: str, guards: set[str],
                       functions: list[Block]) -> VulnerabilityFinding:
    finding = VulnerabilityFinding(entry["vuln_id"], entry["vuln_type"], entry["risk_level"])
    for regex, weight, description in entry["patterns"]:
        for match in re.finditer(regex, clean, re.DOTALL | re.MULTILINE):
            if entry["vuln_id"] in _ACCESS_SWCS and _is_protected_match(match.start(), clean, guards, functions):
                continue
            line = line_number(clean, match.start())
            context = _extract_context(original, match.start())
            finding.matches.append(PatternMatch(line, match.group()[:80], context, weight))
            finding.notes.append(f"L{line}: {description}")
    return finding


# Lượng trừ điểm: mỗi negative pattern khớp trừ theo trọng số; luật SWC-101/105/106 có guard modifier trừ thêm 0.45.
def _negative_reduction(entry: dict, clean: str, raw_score: float, guards: set[str]) -> float:
    reduction = sum((weight for regex, weight, _ in entry["negative_patterns"] if re.search(regex, clean, re.DOTALL)),
                    0.0)
    if guards and entry["vuln_id"] in _ACCESS_SWCS and any(re_word(guard).search(clean) for guard in guards):
        reduction += 0.45
    return min(raw_score, reduction)


# [HÀM CHÍNH PHASE 1A - bộ luật] 27 luật (confirmed khi điểm thô - điểm trừ > 0.25) + 25 kiểm tra ngữ nghĩa.
def run_static_analysis(contract_code: str, guard_modifiers: set[str] | None = None) -> dict:
    clean = mask_noncode(contract_code)
    guards = extract_guard_modifiers(contract_code) if guard_modifiers is None else guard_modifiers
    functions = _functions(clean)
    findings = {}
    for entry in VULNERABILITY_REGISTRY:
        finding = _run_pattern_check(entry, clean, contract_code, guards, functions)
        if finding.matches:
            finding.confirmed = finding.raw_score - _negative_reduction(entry, clean, finding.raw_score, guards) > 0.25
            findings[entry["vuln_type"]] = finding
    for check in _SEMANTIC_CHECKS:
        if finding := check(contract_code):
            findings[finding.vuln_type] = finding
    return {"findings": findings}
