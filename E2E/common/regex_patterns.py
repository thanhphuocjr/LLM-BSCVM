"""Nơi DUY NHẤT chứa regex của pipeline: RE_XXX = regex đã compile, re_xxx(name) = regex dựng từ tên trong mã (re.escape)."""

import re


# ══ DÙNG CHUNG NHIỀU PHASE ══

# Một hoặc nhiều khoảng trắng (bỏ hết: sub("", x), gộp về một dấu cách: sub(" ", x)).
RE_WHITESPACE = re.compile(r"\s+")

# Dòng pragma, group(1) = phiên bản: "pragma solidity ^0.4.24;" -> "^0.4.24".
RE_PRAGMA_SPEC = re.compile(r"pragma\s+solidity\s+([^;]+);")

# Một định danh Solidity ("balances", "_owner"); dùng với .match / .fullmatch.
RE_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")

# Đúng MỘT từ: re_word("owner") khớp "owner" nhưng không khớp "newOwner".
def re_word(name: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(name)}\b")


# Lời gọi hàm theo tên: re_call("withdraw") khớp "withdraw(" và "withdraw (".
def re_call(name: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(name)}\s*\(")


# ══ PHASE 0 - TÁCH CODE UNIT (code_units.py) + NGỮ CẢNH PROMPT (contract_context.py) ══

# Đầu một code unit: "function f(" (group function), "function (" (unnamed_fallback), constructor/receive/fallback (special).
RE_UNIT_DECLARATION = re.compile(
    r"\b(function\s+(?P<function>[A-Za-z_$][\w$]*)\s*\(|"
    r"function\s*(?P<unnamed_fallback>\()\s*|"
    r"(?P<special>constructor|receive|fallback)\s*\()",
    re.MULTILINE,
)

# Khai báo contract/library/interface: group(1) = loại, group(2) = tên.
RE_CONTRACT_DECLARATION = re.compile(r"\b(?:abstract\s+)?(contract|library|interface)\s+([A-Za-z_$][\w$]*)")

# Comment hoặc chuỗi (token sớm nhất thắng: "//" trong "http://" thuộc chuỗi); chưa đóng thì kéo tới hết mã.
RE_NONCODE = re.compile(
    r"//[^\n]*"                               # comment một dòng
    r"|/\*[\s\S]*?(?:\*/|\Z)"                 # comment khối
    r"|\"(?:\\[\s\S]?|[^\"\\])*(?:\"|\Z)"     # "chuỗi" (có ký tự thoát)
    r"|'(?:\\[\s\S]?|[^'\\])*(?:'|\Z)"        # 'chuỗi'
)

# Mọi ký tự trừ xuống dòng (che comment/chuỗi bằng dấu cách mà vẫn giữ số dòng).
RE_NOT_NEWLINE = re.compile(r"[^\n]")

# Từ khoá view / pure / constant (constant = view của Solidity < 0.5).
RE_VIEW_PURE_CONSTANT = re.compile(r"\b(view|pure|constant)\b")

# Visibility đầu tiên trong header hàm: public / external / internal / private.
RE_VISIBILITY = re.compile(r"\b(public|external|internal|private)\b")

# Tham số / biến cục bộ "<kiểu> [memory|storage|calldata] <tên>": group(1) = vị trí dữ liệu, group(2) = tên.
RE_LOCAL_DECLARATION = re.compile(
    r"\b(?:u?int\d*|address(?:\s+payable)?|bool|bytes\d*|string|var|[A-Z][\w$]*)"
    r"(?:\s*\[[^\]]*\])*\s+(?:(memory|storage|calldata)\s+)?([A-Za-z_$][\w$]*)\s*(?=[=;,)])"
)

# Biến bool trong tuple nhận kết quả: "(bool success, ) = to.call(...)" -> success.
RE_BOOL_TUPLE_LOCAL = re.compile(r"\(\s*bool\s+([A-Za-z_$][\w$]*)")

# Phép ghi (=, +=, -=, ...; không tính ==): group(1) = vế trái, group(2) = toán tử.
RE_INVARIANT_STATE_WRITE = re.compile(r"\b([a-zA-Z0-9_]+(?:\s*\[[^\]]+\])*(?:\.[a-zA-Z0-9_]+)*)\s*(\+=|-=|\*=|\/=|%=|=)(?!=)")

# Tên gốc ở đầu vế trái: "balances[msg.sender]" -> "balances".
RE_LEADING_NAME = re.compile(r"[A-Za-z0-9_]+")

# Event được phát: "emit Transfer(" (group 1) hoặc "Transfer(" kiểu Solidity 0.4 (group 2).
RE_INVARIANT_EVENT = re.compile(r"(?:emit\s+([A-Za-z_$][\w$]*)|(?<!function\s)(?<!return\s)(?<!require\()(?<!assert\()(?<!revert\()([A-Z][A-Za-z0-9_$]*))\s*\(")

# Lời gọi chuyển ETH / call ra ngoài, group(1) = "đối tượng.phương thức" (vd "msg.sender.call{value: x}").
RE_INVARIANT_EXTERNAL_CALL = re.compile(r"\b([a-zA-Z0-9_\[\]\.]+\s*\.\s*(?:transfer|send|call\.value|call\{value:[^}]+\}|call))\s*\(")

# Câu lệnh return, group(1) = biểu thức trả về.
RE_INVARIANT_RETURN = re.compile(r"\breturn(?=\s)([^;]+);")

# Modifier tới dấu "{" mở thân, group(1) = tên: "modifier onlyOwner() {".
RE_MODIFIER_OPENING = re.compile(r"\bmodifier\s+([A-Za-z_$][\w$]*)\s*(?:\([^)]*\))?[^{;]*\{")


# Mở đầu contract/library/interface có tên cho trước, tới dấu "{": "contract Bank is Ownable {".
def re_contract_opening(contract_name: str) -> re.Pattern[str]:
    return re.compile(rf"\b(?:contract|library|interface)\s+{re.escape(contract_name)}\b[^{{]*\{{")


# ══ PHASE 1A - PHÂN TÍCH TĨNH (static_analyzer.py) ══

# ── Ranh giới hàm / contract / modifier ──────────────────────────────────────

# Hàm có thân: "function <name>(<params>) <header> {".
RE_FUNCTION_WITH_BODY = re.compile(
    r'\bfunction\s+(?P<name>[A-Za-z_]\w*)\s*'
    r'\((?P<params>[^)]*)\)(?P<header>[^;{}]*)\{',
    re.DOTALL,
)

# constructor / receive / fallback có thân: "<name>(<params>) <header> {".
RE_SPECIAL_FUNCTION_WITH_BODY = re.compile(
    r'\b(?P<name>constructor|receive|fallback)\s*'
    r'\((?P<params>[^)]*)\)(?P<header>[^;{}]*)\{',
    re.DOTALL,
)

# Contract có thân: "contract <name> <header> {" (header = phần "is A, B").
RE_CONTRACT_WITH_BODY = re.compile(r'\bcontract\s+(?P<name>[A-Za-z_]\w*)(?P<header>[^;{}]*)\{', re.DOTALL)

# Modifier có thân: "modifier <name>[(<params>)] <header> {".
RE_MODIFIER_WITH_BODY = re.compile(
    r'\bmodifier\s+(?P<name>[A-Za-z_]\w*)\s*(?:\((?P<params>[^)]*)\))?(?P<header>[^;{}]*)\{',
    re.DOTALL,
)

# Header của hàm đầu tiên tới dấu "{" (khi mã đầu vào chỉ là MỘT hàm).
RE_FUNCTION_HEADER_PREFIX = re.compile(r'function\s+[^{]*\{')

# ── Phân quyền (guard) ───────────────────────────────────────────────────────

# Các kiểu kiểm tra người gọi trong thân hàm/modifier (chỉ cần khớp MỘT).
RE_SENDER_CHECKS = (
    re.compile(r'require\s*\([^;]*msg\.sender\s*=='),                                 # require(msg.sender == owner)
    re.compile(r'require\s*\([^;]*==\s*msg\.sender'),                                 # require(owner == msg.sender)
    re.compile(r'if\s*\([^)]*msg\.sender\s*!=[^)]*\)\s*(?:revert\b|throw\b)'),         # if (msg.sender != owner) revert
    re.compile(r'if\s*\([^)]*!=\s*msg\.sender[^)]*\)\s*(?:revert\b|throw\b)'),         # if (owner != msg.sender) throw
    re.compile(r'require\s*\([^;]*\[msg\.sender\]'),                                  # require(owners[msg.sender])
    re.compile(r'require\s*\([^;]*(?:isOwner|hasRole|authorized)\s*\('),              # require(hasRole(ADMIN, msg.sender))
)

# Tên mọi modifier được khai báo, group(1) = tên.
RE_MODIFIER_NAME = re.compile(r'\bmodifier\s+([A-Za-z0-9_]+)')

# Tên modifier mang nghĩa phân quyền: onlyXxx, auth, authorized, restricted.
RE_GUARD_MODIFIER_NAME = re.compile(r'^(?:only[A-Z]\w*|auth|authorized|restricted)$')

# Visibility internal hoặc private (không gọi được từ bên ngoài).
RE_INTERNAL_OR_PRIVATE = re.compile(r'\b(?:private|internal)\b')

# Visibility public hoặc external.
RE_PUBLIC_OR_EXTERNAL = re.compile(r'\b(?:public|external)\b')

# Từ khoá payable.
RE_PAYABLE = re.compile(r'\bpayable\b')

# Mọi "từ" (chữ/số/_) trong header, group(1) = từ (tìm onlyXxx).
RE_WORD_TOKEN = re.compile(r'\b([A-Za-z0-9_]+)\b')

# ── Khai báo biến / kiểu ─────────────────────────────────────────────────────

# Khai báo "<kiểu>[[]] [từ khoá...] <tên>" bất kỳ đâu: "uint256[] memory ids" -> type, name.
RE_SYMBOL_DECLARATION = re.compile(
    r'\b(?P<type>[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])*)\s+'
    r'(?:(?:public|private|internal|external|constant|immutable|memory|'
    r'calldata|storage|payable|indexed)\s+)*'
    r'(?P<name>[A-Za-z_]\w*)\b'
)

# Một tham số hàm: "uint[] storage arr" -> type="uint[]", location="storage", name="arr".
RE_PARAMETER = re.compile(
    r'^\s*(?P<type>[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])*)\s*'
    r'(?P<location>memory|storage|calldata|payable)?\s*'
    r'(?P<name>[A-Za-z_]\w*)\s*$'
)

# Biến trạng thái tới ";", tail có "=" là đã khởi tạo: "address public owner = msg.sender;".
RE_STATE_DECLARATION = re.compile(
    r'^\s*(?P<type>[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])*)\s+'
    r'(?:(?:public|private|internal|constant|immutable|payable)\s+)*'
    r'(?P<name>[A-Za-z_]\w*)\b(?P<tail>[^;]*);'
)

# Như RE_STATE_DECLARATION nhưng chỉ lấy kiểu + tên.
RE_STATE_TYPE_AND_NAME = re.compile(
    r'^\s*(?P<type>[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])*)\s+'
    r'(?:(?:public|private|internal|constant|immutable|payable)\s+)*'
    r'(?P<name>[A-Za-z_]\w*)\b'
)

# Biến trạng thái mapping, name = tên: "mapping(address => uint) public balances".
RE_STATE_MAPPING = re.compile(r'^\s*mapping\s*\([^;]+\)\s*(?:(?:public|private|internal)\s+)*(?P<name>[A-Za-z_]\w*)\b')

# Mapping trạng thái kèm kiểu giá trị: "mapping(address => User) users" -> value="User".
RE_MAPPING_VALUE_TYPE = re.compile(
    r'^\s*mapping\s*\([^=;]+=>\s*(?P<value>[A-Za-z_]\w*)\s*\)\s*'
    r'(?:(?:public|private|internal)\s+)*(?P<name>[A-Za-z_]\w*)\b'
)

# Mapping PUBLIC kèm kiểu giá trị: "mapping(uint => Order) public orders".
RE_PUBLIC_MAPPING_VALUE_TYPE = re.compile(
    r'\bmapping\s*\([^=;]+=>\s*(?P<value>[A-Za-z_]\w*)\s*\)\s*'
    r'public\s+(?P<name>[A-Za-z_]\w*)\b'
)

# Từ khoá "mapping(" (struct có chứa mapping trực tiếp không).
RE_MAPPING_KEYWORD = re.compile(r'\bmapping\s*\(')

# Mở đầu struct, name = tên: "struct User {".
RE_STRUCT_OPENING = re.compile(r'\bstruct\s+(?P<name>[A-Za-z_]\w*)\s*\{')

# Mở đầu enum, name = tên: "enum State {".
RE_ENUM_OPENING = re.compile(r'\benum\s+(?P<name>[A-Za-z_]\w*)\s*\{')

# Phần mảng cuối tên kiểu: sub("", "User[]") -> "User".
RE_ARRAY_SUFFIX = re.compile(r'\[.*$')

# Kiểu uintN / intN (cả chuỗi): unsigned = "u" hoặc "", bits = N (rỗng = 256).
RE_INTEGER_TYPE = re.compile(r'(?P<unsigned>u?)int(?P<bits>\d*)')

# Mảng số nguyên CÓ DẤU, base = kiểu gốc: "int8[]" -> "int8".
RE_SIGNED_INT_ARRAY_TYPE = re.compile(r'(?P<base>int\d*)\[')

# Ép kiểu ở đầu biểu thức: "IERC20(token)" -> type="IERC20".
RE_CAST_PREFIX = re.compile(r'(?P<type>[A-Za-z_]\w*)\(')

# Biến cục bộ tham chiếu storage tới mảng: "uint[] storage list" -> name="list".
RE_STORAGE_ARRAY_LOCAL = re.compile(r'\b[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])+\s+storage\s+(?P<name>[A-Za-z_]\w*)\b')

# Danh sách contract cha sau "is": " is Ownable, Pausable " -> bases.
RE_INHERITANCE_LIST = re.compile(r'\bis\s+(?P<bases>[\s\S]+)$')

# Tên đứng đầu một mục: " Ownable(msg.sender)" -> "Ownable".
RE_LEADING_IDENTIFIER = re.compile(r'\s*([A-Za-z_]\w*)')

# Từ khoá abstract ở cuối đoạn mã ngay trước "contract".
RE_ABSTRACT_AT_END = re.compile(r'\babstract\s*$')

# Từ khoá "is" (contract có kế thừa).
RE_IS_KEYWORD = re.compile(r'\bis\b')

# ── Pragma ───────────────────────────────────────────────────────────────────

# Cả câu pragma solidity (che đi để "^0.4.24" không bị coi là phép XOR).
RE_PRAGMA_STATEMENT = re.compile(r'pragma\s+solidity\s+[^;]+;')

# Pragma cho phép Solidity 0.4.x.
RE_PRAGMA_0_4 = re.compile(r'\bpragma\s+solidity\s+[^;]*\b0\.4\.')

# Pragma pin cứng 0.4.0 -> 0.4.4 (chưa revert khi ép số sai sang enum).
RE_PRAGMA_0_4_0_TO_0_4_4 = re.compile(r'\bpragma\s+solidity\s+(?:=\s*)?0\.4\.[0-4]\s*;')

# Pragma pin cứng 0.4.x / 0.5.x: "pragma solidity 0.4.24;" -> minor=4, patch=24.
RE_PINNED_PRAGMA_0_4_OR_0_5 = re.compile(r'\bpragma\s+solidity\s+(?:=\s*)?0\.(?P<minor>[45])\.(?P<patch>\d+)\s*;')

# "pragma experimental ABIEncoderV2;".
RE_PRAGMA_ABIENCODERV2 = re.compile(r'\bpragma\s+experimental\s+ABIEncoderV2\s*;')

# ── Biểu thức, lời gọi, vòng lặp ─────────────────────────────────────────────

# Đầu vòng lặp "for (" / "while (".
RE_LOOP_OPENING = re.compile(r'\b(?:for|while)\s*\(')

# Khối assembly (Yul), có thể kèm ("memory-safe"), tới dấu "{".
RE_ASSEMBLY_BLOCK = re.compile(r'\bassembly(?:\s*\([^)]*\))?\s*\{')

# Lệnh Yul return(offset, size).
RE_YUL_RETURN_CALL = re.compile(r'\breturn\s*\([^,]+,[^)]+\)')

# Yul shl/shr/sar(shift, value) với shift là BIẾN, value là HẰNG (dấu hiệu đảo tham số).
RE_YUL_SHIFT_WITH_CONSTANT_VALUE = re.compile(
    r'\b(?:shl|shr|sar)\s*\(\s*'
    r'(?P<shift>[A-Za-z_]\w*)\s*,\s*'
    r'(?P<value>0x[0-9A-Fa-f]+|\d+)\s*\)'
)

# Mệnh đề returns, values = danh sách giá trị trả về: "returns (uint a, bool b)".
RE_RETURNS_CLAUSE = re.compile(r'\breturns\s*\((?P<values>[^)]*)\)')

# Toán hạng của phép "^": số hex, số thập phân hoặc tên.
_XOR_OPERAND = r'(?:0x[0-9A-Fa-f]+|\d+|[A-Za-z_]\w*)'
RE_XOR_EXPRESSION = re.compile(
    rf'(?P<left>{_XOR_OPERAND})\s*\^(?!=)\s*(?P<right>{_XOR_OPERAND})'
)

# Chuỗi chỉ gồm chữ số thập phân (dùng với .fullmatch).
RE_DECIMAL_NUMBER = re.compile(r'\d+')

# Toán hạng của phép so sánh: tên hoặc số nguyên (có thể âm, hex).
_COMPARISON_OPERAND = r'(?:[A-Za-z_]\w*|-?(?:0x[0-9A-Fa-f]+|\d+))'
RE_ORDER_COMPARISON = re.compile(
    rf'(?P<left>{_COMPARISON_OPERAND})\s*(?P<operator><=|>=|<|>)\s*(?P<right>{_COMPARISON_OPERAND})'
)

# Số nguyên (âm, thập phân hoặc hex), dùng với .fullmatch: "-1", "0xff".
RE_INTEGER_LITERAL = re.compile(r'-?(?:0x[0-9A-Fa-f]+|\d+)')

# Điều kiện if là hằng: "if (true)" / "if (false)".
RE_IF_CONSTANT_CONDITION = re.compile(r'\bif\s*\(\s*(?:true|false)\s*\)')

# Hằng true/false cạnh && hoặc ||: "x || true", "false && y".
RE_BOOLEAN_CONSTANT_OPERAND = re.compile(r'(?:\b(?:true|false)\b\s*(?:&&|\|\|)|(?:&&|\|\|)\s*\b(?:true|false)\b)')

# Toán tử so sánh bằng "==".
RE_STRICT_EQUALITY = re.compile(r'==')

# Nguồn số dư: ".balance" hoặc ".balanceOf(".
RE_BALANCE_SOURCE = re.compile(r'\.balance\b|\.balanceOf\s*\(')

# Biến gán từ số dư: "uint bal = address(this).balance;" -> name="bal".
RE_BALANCE_ASSIGNMENT = re.compile(
    r'\b(?:u?int\d*\s+)?(?P<name>[A-Za-z_]\w*)\s*=\s*'
    r'[^;]*(?:\.balance\b|\.balanceOf\s*\()[^;]*;'
)

# Lời gọi abi.encodePacked(.
RE_ENCODE_PACKED_CALL = re.compile(r'\babi\.encodePacked\s*\(')

# Lời gọi abi.encode(.
RE_ABI_ENCODE_CALL = re.compile(r'\babi\.encode\s*\(')

# transfer/transferFrom trên một tên hoặc ép kiểu: "IERC20(t).transferFrom(" -> receiver, method.
RE_ERC20_TRANSFER_CALL = re.compile(
    r'(?P<receiver>(?:[A-Za-z_]\w*|'
    r'[A-Za-z_]\w*\s*\([^()]*\)))\s*\.\s*'
    r'(?P<method>transferFrom|transfer)\s*\('
)

# Mọi đường gửi ETH ra ngoài: call{value:}, send, transfer, selfdestruct/suicide, call trong assembly.
RE_OUTGOING_ETHER = re.compile(
    r'\.call\s*\{[^}]*\bvalue\s*:'
    r'|\.send\s*\('
    r'|\.transfer\s*\('
    r'|\b(?:selfdestruct|suicide)\s*\('
    r'|\bassembly\s*\{[^}]*\bcall\s*\(',
    re.DOTALL,
)

# msg.value so sánh với 0 ("msg.value == 0") - được xoá trước khi tìm msg.value.
RE_MSG_VALUE_ZERO_CHECK = re.compile(r'msg\.value\s*(?:==|!=)\s*0|0\s*(?:==|!=)\s*msg\.value')

# msg.value.
RE_MSG_VALUE = re.compile(r'\bmsg\.value\b')

# Lời gọi .delegatecall(.
RE_DELEGATECALL = re.compile(r'\.delegatecall\s*\(')

# Gán độ dài mảng: "arr.length = <rhs>;".
RE_ARRAY_LENGTH_ASSIGNMENT = re.compile(r'\b(?P<array>[A-Za-z_]\w*)\.length\s*=\s*(?P<rhs>[^;]+);')

# Câu lệnh delete: "delete users[id];" -> target="users[id]", base="users".
RE_DELETE_STATEMENT = re.compile(r'\bdelete\s+(?P<target>(?P<base>[A-Za-z_]\w*)(?:\s*\[[^\]]+\])?)\s*;')

# ── Regex chứa tên lấy từ mã đang phân tích ──────────────────────────────────

# address(0) hoặc address(0x0...), đoạn dùng chung của các regex zero address bên dưới.
_ZERO_ADDRESS = r'address\s*\(\s*(?:0|0x0+)\s*\)'


# require/assert kiểm tra tham số khác address(0): "require(to != address(0))".
def re_zero_address_require(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf'\b(?:require|assert)\s*\([^;]*'
        rf'(?:\b{escaped}\b\s*!=\s*{_ZERO_ADDRESS}|{_ZERO_ADDRESS}\s*!=\s*\b{escaped}\b)'
    )


# if (tham số == address(0)) rồi revert/return.
def re_zero_address_revert_if(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf'\bif\s*\([^;{{}}]*'
        rf'(?:\b{escaped}\b\s*==\s*{_ZERO_ADDRESS}|{_ZERO_ADDRESS}\s*==\s*\b{escaped}\b)'
        rf'[^;{{}}]*\)\s*(?:\{{\s*)?(?:revert|return)\b',
        re.DOTALL,
    )


# Gán tham số vào một biến: "owner = newOwner;" -> lhs="owner".
def re_assignment_from(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf'(?P<lhs>\b[A-Za-z_]\w*(?:\s*\[[^\]]+\]|\.[A-Za-z_]\w*)?)'
        rf'\s*=\s*\b{escaped}\b\s*;'
    )


# Tham số làm đích gọi/chuyển tiền: "to.call", "to.transfer", "payable(to).", "IERC20(to).".
def re_call_target(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf'(?:\b{escaped}\b\s*\.\s*(?:call|send|transfer)\b'
        rf'|payable\s*\(\s*\b{escaped}\b\s*\)\s*\.'
        rf'|[A-Za-z_]\w*\s*\(\s*\b{escaped}\b\s*\)\s*\.)'
    )


# Biến bị ghi (=, +=, -=, ++, --; không tính ==): re_state_write("owner") khớp "owner = x".
def re_state_write(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(rf'\b{escaped}\b\s*(?:=(?!=)|\+=|-=|\+\+|--)')


# Biến address làm đích chuyển ETH: "owner.transfer", "payable(owner).", "selfdestruct(owner".
def re_ether_destination(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf'(?:\b{escaped}\b\s*\.\s*(?:transfer|send|call)\b'
        rf'|payable\s*\(\s*\b{escaped}\b\s*\)\s*\.'
        rf'|(?:selfdestruct|suicide)\s*\(\s*\b{escaped}\b)'
    )


# Ghi vào phần tử mảng: "arr[i] = ...", "arr[i] += ...", "arr[i]++".
def re_indexed_write(name: str) -> re.Pattern[str]:
    return re.compile(
        rf'\b{re.escape(name)}\s*\[[^\]]+\]\s*'
        rf'(?:=|\+=|-=|\*=|/=|%=|\+\+|--)'
    )


# Gán cả biến: "<name> = <rhs>;", rhs = vế phải.
def re_assignment_to(name: str) -> re.Pattern[str]:
    return re.compile(rf'\b{re.escape(name)}\s*=\s*(?P<rhs>[^;]+);')


# Ép kiểu sang enum: "State(x)".
def re_enum_conversion(enum_name: str) -> re.Pattern[str]:
    return re.compile(rf'\b{re.escape(enum_name)}\s*\([^)]*\)')


# ══ PHASE 1B - RAG: 37 SWC AGENT (swc_agents.py) + XÂY CƠ SỞ TRI THỨC (build_knowledge_base.py) ══

# Mã "SWC-xxx" trong một chuỗi bất kỳ.
RE_SWC_ID = re.compile(r"SWC-\d{3}")

# Nội dung các khối ```...``` (có hoặc không tên ngôn ngữ), group(1) = phần mã.
RE_ANY_CODE_FENCE = re.compile(r"```(?:\w+)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

# Như RE_PRAGMA_SPEC nhưng bỏ qua hoa/thường.
RE_PRAGMA_SPEC_ANY_CASE = re.compile(r"pragma\s+solidity\s+([^;]+);", re.IGNORECASE)

# Phiên bản "major.minor[.patch]" đầu tiên: "^0.4.24" -> (0, 4, 24).
RE_VERSION_NUMBER = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")

# Có phép toán số học: ++, --, +=, -=, *=, phép gán có + - *, hoặc add/sub/mul(.
RE_ARITHMETIC_OPERATION = re.compile(
    r"(?:\+\+|--|[+\-*]=|=\s*[^;]*(?:\+|-|\*)[^;]*;|\b(?:add|sub|mul)\s*\()",
    re.IGNORECASE | re.DOTALL,
)

# Lời gọi cấp thấp .call / .delegatecall / .staticcall / .send.
RE_LOW_LEVEL_CALL = re.compile(r"\.(?:call|delegatecall|staticcall|send)\s*[\(\{]", re.IGNORECASE | re.DOTALL)

# Biến bool trong tuple nhận kết quả: "(bool ok, ) = ..." -> ok.
RE_BOOL_TUPLE_RESULT = re.compile(r"\(\s*bool\s+(\w+)\s*,?", re.IGNORECASE)

# Biến bool gán thẳng từ lời gọi cấp thấp: "bool ok = to.send(1);" -> ok.
RE_BOOL_ASSIGNED_CALL_RESULT = re.compile(
    r"\bbool\s+(\w+)\s*=\s*[^;]*\.(?:call|delegatecall|staticcall|send)\s*[\(\{]",
    re.IGNORECASE | re.DOTALL,
)

# Lời gọi cấp thấp nằm ngay trong require/assert: "require(to.send(1))".
RE_CALL_INSIDE_REQUIRE = re.compile(
    r"\b(?:require|assert)\s*\([^;]*\.(?:send|call|delegatecall|staticcall)\s*[\(\{]",
    re.IGNORECASE | re.DOTALL,
)

# Có chống reentrancy: nonReentrant hoặc ReentrancyGuard.
RE_REENTRANCY_GUARD = re.compile(r"\bnonReentrant\b|ReentrancyGuard", re.IGNORECASE)

# Lời gọi gửi ETH: ".call{value: ...}" hoặc ".send(".
RE_VALUE_CALL_OR_SEND = re.compile(r"\.(?:call\s*\{[^}]*value\s*:|send\s*\()", re.IGNORECASE | re.DOTALL)

# Cập nhật state: ghi balances[...], delete x[...], hoặc "a = b" / "a += b" / "a -= b".
RE_STATE_UPDATE = re.compile(
    r"(?:balances?\s*\[[^\]]+\]\s*(?:-=|=|\+=)|\bdelete\s+\w+\s*\[|\b\w+\s*(?:-=|\+=|=)\s*\w+)",
    re.IGNORECASE | re.DOTALL,
)

# Thân vòng lặp for/while (tới dấu "}" đầu tiên), group(1) = thân.
RE_LOOP_BODY = re.compile(r"\b(?:for|while)\s*\([^)]*\)\s*\{(.*?)\}", re.IGNORECASE | re.DOTALL)

# Dấu "=" đứng một mình (phép gán, không phải ==, !=, <=, >=, =>).
RE_SINGLE_EQUALS = re.compile(r"(?<![=!<>])=(?![=>])")

# SWC-100: hàm mà header sau ")" không có visibility.
RE_FUNCTION_WITHOUT_VISIBILITY = re.compile(
    r"\bfunction\s+\w+\s*\([^)]*\)\s*(?![^{};]*(?:public|private|internal|external))",
    re.IGNORECASE | re.DOTALL,
)

# Khối "unchecked {" (tắt kiểm tra tràn số của 0.8+), phân biệt hoa/thường.
RE_UNCHECKED_BLOCK = re.compile(r"\bunchecked\s*\{")

# Có dùng thư viện SafeMath.
RE_SAFEMATH = re.compile(r"\bSafeMath\b")

# Ép kiểu thu hẹp "uint8(" ... "int128(".
RE_NARROW_INT_CAST = re.compile(r"\b(?:u?int)(?:8|16|32|64|128)\s*\(")

# Pragma "trôi" có ^, >=, > hoặc *: "pragma solidity ^0.4.24".
RE_FLOATING_PRAGMA = re.compile(r"pragma\s+solidity\s+[^;]*(?:\^|>=|>|\*)", re.IGNORECASE)

# Lời gọi gửi ETH: ".call{value: ...}", ".send(", ".transfer(".
RE_ETHER_TRANSFER = re.compile(r"\.(?:call\s*\{[^}]*value\s*:|send\s*\(|transfer\s*\()", re.IGNORECASE | re.DOTALL)

# Hàm rút tiền kiểu quản trị: drain / sweep / rescue / withdrawAll / emergencyWithdraw.
RE_ADMIN_WITHDRAW_FUNCTION = re.compile(
    r"\bfunction\s+(?:drain|sweep|rescue|withdrawAll|emergencyWithdraw)\b", re.IGNORECASE
)

# Toàn bộ số dư của contract: "address(this).balance" hoặc "this.balance".
RE_CONTRACT_BALANCE = re.compile(r"address\s*\(\s*this\s*\)\.balance|\bthis\.balance\b", re.IGNORECASE)

# Kiểm tra số dư của chính người gọi: "balances[msg.sender] >= amount".
RE_USER_BALANCE_GUARD = re.compile(r"balances?\s*\[\s*msg\.sender\s*\].{0,80}(?:>=|>)", re.IGNORECASE | re.DOTALL)

# Lời gọi selfdestruct( / suicide(.
RE_SELFDESTRUCT_CALL = re.compile(r"\b(?:selfdestruct|suicide)\s*\(", re.IGNORECASE)

# Lời gọi .delegatecall( / .callcode(.
RE_DELEGATECALL_OR_CALLCODE = re.compile(r"\.\s*(?:delegatecall|callcode)\s*\(", re.IGNORECASE)

# Lời gọi ra ngoài .transfer / .send / .call (trong thân vòng lặp).
RE_EXTERNAL_CALL = re.compile(r"\.(?:transfer|send|call)\s*[\(\{]", re.IGNORECASE | re.DOTALL)

# Hàm tên viết hoa (constructor kiểu cũ trùng tên contract).
RE_CAPITALIZED_FUNCTION = re.compile(r"\bfunction\s+[A-Z]\w+\s*\(")

# Biến lưu giá trị nhạy cảm: "string private password", "bytes32 secretKey"...
RE_SENSITIVE_STATE_VALUE = re.compile(
    r"\b(?:string|bytes\d*|uint\d*|address)\s+(?:private\s+)?\w*(?:password|secret|privateKey|apiKey|seed)\w*",
    re.IGNORECASE,
)

# Có cơ chế phân quyền: onlyOwner, onlyRole, onlyAdmin, AccessControl, Ownable, require(msg.sender == ...).
RE_ACCESS_CONTROL_GUARD = re.compile(
    r"\bonlyOwner\b|onlyRole|onlyAdmin|AccessControl|Ownable|require\s*\(\s*msg\.sender\s*==",
    re.IGNORECASE,
)


# Kết quả bool `name` được kiểm tra: require(name, assert(name, if (!name, if (name == false.
def re_result_check(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf"(?:require\s*\(\s*{escaped}\b|assert\s*\(\s*{escaped}\b|if\s*\(\s*!\s*{escaped}\b|if\s*\(\s*{escaped}\s*==\s*false)",
        re.IGNORECASE,
    )


# Mở đầu điều kiện "<keyword>(" với keyword = "if" hoặc "require".
def re_condition_opening(keyword: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(keyword)}\s*\(", re.IGNORECASE)


# Markdown (build_knowledge_base): làm sạch mô tả SWC thành văn bản thuần.
RE_MARKDOWN_CODE_BLOCK = re.compile(r"```[\s\S]*?```")          # khối ```code```
RE_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")         # [chữ](link) -> group(1) = chữ
RE_MARKDOWN_SYMBOLS = re.compile(r"[`*_>#-]+")                  # ký hiệu định dạng ` * _ > # -
RE_BULLET_PREFIX = re.compile(r"^\s*[-*]\s*")                   # gạch đầu dòng "- " / "* "
RE_SENTENCE_BREAK = re.compile(r"[.;]\s+")                      # chỗ ngắt câu ". " / "; "


# ══ PHASE 2 ADVISOR / PHASE 3 ASSESSOR ══

# Dấu lược "..." / "…" trong guard_quote của LLM (tách thành từng mảnh để so với mã).
RE_QUOTE_ELLIPSIS = re.compile(r"\.\.\.|…")


# ══ PHASE 4 - FIXER (fixer.py, patching.py, behavior_guard.py) ══

# Mọi phiên bản "x.y.z" trong pragma: ">=0.4.22 <0.6.0" -> [(0,4,22), (0,6,0)].
RE_FULL_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")

# JSON (từ "{" đầu tới "}" cuối) có chứa khoá "replacement".
RE_JSON_WITH_REPLACEMENT = re.compile(r"\{[\s\S]*\"replacement\"[\s\S]*\}")

# Khối ```solidity ...``` (hoặc ``` không tên), group(1) = mã.
RE_SOLIDITY_FENCE = re.compile(r"```(?:solidity)?\s*([\s\S]*?)```")

# "function <tên>(" - một khai báo hàm có tên.
RE_NAMED_FUNCTION_START = re.compile(r"\bfunction\s+[A-Za-z_$][\w$]*\s*\(")


# Đầu một trường chuỗi JSON: '"<field>": "'.
def re_json_string_field_start(field: str) -> re.Pattern[str]:
    return re.compile(rf'"{re.escape(field)}"\s*:\s*"')


# Điểm bắt đầu khai báo hàm cần lấy từ câu trả lời của model (fallback/receive, tên cụ thể, hoặc bất kỳ).
def re_declaration_start(target_name: str | None) -> re.Pattern[str]:
    if target_name in ("fallback", "receive"):
        return re.compile(rf"\b((?:function\s+{re.escape(target_name)}\s*\(|function\s*\(|(?:constructor|receive|fallback)\s*\())")
    if target_name:
        return re.compile(rf"\b(function\s+{re.escape(target_name)}\s*\(|(?:constructor|receive|fallback)\s*\()")
    return re.compile(r"\b(function\s+[A-Za-z_$][\w$]*\s*\(|function\s*\(|(?:constructor|receive|fallback)\s*\()")


# Guard hành vi: các kiểu chuyển ETH và dạng an toàn tương đương.
RE_ETH_VALUE_CALL = re.compile(r"(?:\.call\s*\{\s*value\s*:|\.call\.value\s*\()")   # .call{value: x}( / .call.value(x)(
RE_TRANSFER_CALL = re.compile(r"\.transfer\s*\(")                                    # to.transfer(
RE_SAFE_TRANSFER_CALL = re.compile(r"\.safeTransfer\s*\(")                           # token.safeTransfer(
RE_SEND_CALL = re.compile(r"\.send\s*\(")                                            # to.send(
RE_SEND_VALUE_CALL = re.compile(r"\bsendValue\s*\(")                                 # Address.sendValue(
RE_SAFE_TRANSFER_ETH_CALL = re.compile(r"\bsafeTransferETH\s*\(")                    # SafeTransferLib.safeTransferETH(

# Từ khoá return.
RE_RETURN_KEYWORD = re.compile(r"\breturn\b")

# Biến bị ghi (=, +=, ...; không tính ==), group(1) = tên gốc: "balances[to] += v" -> "balances".
RE_STATE_WRITE = re.compile(r"([A-Za-z_$][\w$]*)(?:\s*\[[^\]]+\])*\s*(\+=|-=|\*=|/=|%=|=)(?!=)")

# "emit <Event>(", group(1) = tên event.
RE_EMIT_EVENT = re.compile(r"\bemit\s+([A-Za-z_$][\w$]*)\s*\(")

# Câu lệnh bắt đầu bằng tên viết hoa rồi "(" (event Solidity < 0.4.21), group(1) = tên.
RE_LEGACY_EVENT_STATEMENT = re.compile(r"(?:^|[;{}])\s*([A-Z][\w$]*)\s*\(")


# ══ PHASE 5 - VERIFIER (candidate_check.py, tool_matrix.py) ══

# Số dòng trong vị trí lỗi solc: "--> a.sol:12:5" (>= 0.5) và "a.sol:12:5: Error" (0.4).
RE_SOLC_ERROR_LINE = re.compile(r"\S+?\.sol:(\d+):\d+")

# Đường dẫn tuyệt đối tới file .sol (thư mục tạm -> "Contract.sol" trong feedback).
RE_SOL_FILE_PATH = re.compile(r"(?:/[^\s:]+)+\.sol")

# Vị trí trong thông điệp Slither (.sol#12-15, :12:5, #12, "line 12") - bỏ đi để so finding trước/sau vá.
RE_LOCATION_IN_MESSAGE = re.compile(r"(\S*\.sol(?:#\d+(?:-\d+)?|:\d+(?::\d+)?)?|#\d+(?:-\d+)?|\blines?\s+\d+(?:-\d+)?)")


# ══ PHASE 6 - REPORTER (report.py) ══

# Đoạn ngay trước "{" là tuỳ chọn lời gọi (".call{value: x}", "new Token{salt: s}"), không phải mở khối.
RE_CALL_OPTIONS_PREFIX = re.compile(r"(?:\.(?:call|delegatecall|staticcall)|\bnew\s+[A-Za-z_]\w*)$")


# ══ HẠ TẦNG: common/gpu_client.py, common/solc.py, pipeline/orchestrator.py, webapp/app.py ══

# Khối ```json ...``` (hoặc ``` không tên) trong câu trả lời của LLM, group(1) = nội dung.
RE_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Xuống dòng không có "\" đứng trước (xuống dòng thô trong chuỗi JSON -> "\n").
RE_UNESCAPED_NEWLINE = re.compile(r"(?<!\\)\n")


# Giá trị chuỗi của khoá JSON: '"<key>": "<giá trị>"', group(1) = giá trị.
def re_json_string_value(key: str) -> re.Pattern[str]:
    return re.compile(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"')


# Giá trị số / true / false của khoá JSON, group(1) = giá trị.
def re_json_scalar_value(key: str) -> re.Pattern[str]:
    return re.compile(rf'"{re.escape(key)}"\s*:\s*([0-9.]+|true|false)', re.IGNORECASE)


# Vị trí lỗi trong output solc: "--> file.sol:12:5:" (>= 0.5) hoặc "file.sol:12:5:" (0.4).
RE_SOLC_ERROR_LOCATION = re.compile(r"(-->\s*\S+:\d+:\d+:?|\S+\.sol:\d+:\d+:?)")

# Phiên bản trong tên file solc của solc-select: "solc-0.4.24" -> "0.4.24".
RE_SOLC_BINARY_VERSION = re.compile(r"solc-(\d+\.\d+\.\d+)$")

# Số minor sau "0.": "^0.4.24" -> "4".
RE_MINOR_VERSION = re.compile(r"0\.(\d+)")

# Ký tự không an toàn cho tên file/thư mục (khác chữ, số, _, -) -> thay bằng "-".
RE_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")

# run_id hợp lệ (.fullmatch): chữ/số đầu, rồi chữ/số/_/./-, tối đa 128 ký tự (chống path traversal).
RE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
