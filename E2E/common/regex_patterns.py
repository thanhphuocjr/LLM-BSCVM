"""
Nơi DUY NHẤT chứa các biểu thức regex mà các hàm của pipeline (Phase 0 -> 6) dùng.

Trong hàm chỉ gọi regex theo TÊN, không viết thẳng biểu thức, ví dụ:
    for match in RE_FUNCTION_WITH_BODY.finditer(clean): ...
    if re_word(modifier_name).search(header): ...

Quy ước đặt tên:
  RE_XXX        regex đã compile sẵn (hằng số); tên mô tả thứ nó khớp.
  re_xxx(name)  hàm dựng regex chứa một tên lấy từ mã đang phân tích (tên biến, hàm, enum...);
                tên luôn được re.escape trước khi ghép vào biểu thức.
Mỗi regex có chú thích: khớp cái gì, ví dụ, và hàm nào dùng nó.

KHÔNG nằm ở đây (là bảng dữ liệu của phase, không phải regex viết trong hàm):
  phase1a_static/static_analyzer.py  VULNERABILITY_REGISTRY  - 27 luật: regex + trọng số + mô tả
  phase1b_rag/swc_agents.py          SIGNAL_PATTERNS_BY_SWC  - regex tín hiệu của 37 SWC agent
                                                                (ghi nguyên chuỗi vào swc_registry.json)
"""

from __future__ import annotations

import re


# ══════════════════════════════════════════════════════════════════════════════
# DÙNG CHUNG NHIỀU PHASE
# ══════════════════════════════════════════════════════════════════════════════

# Một hoặc nhiều khoảng trắng (dấu cách, tab, xuống dòng). Dùng để bỏ hết khoảng trắng (sub("", x))
# hoặc gộp về một dấu cách (sub(" ", x)).
# Dùng ở: static_analyzer, contract_context, build_knowledge_base, retriever, swc_agents, patching,
#         tool_matrix, solc.
RE_WHITESPACE = re.compile(r"\s+")

# Dòng pragma, group(1) = phần phiên bản. Vd "pragma solidity ^0.4.24;" -> "^0.4.24".
# Dùng ở: advisor (ghi chú compiler trong prompt), fixer (phiên bản thấp nhất / luật cú pháp cũ),
#         solc.parse_pragma_spec (chọn solc).
RE_PRAGMA_SPEC = re.compile(r"pragma\s+solidity\s+([^;]+);")

# Một định danh Solidity: chữ cái hoặc "_" đứng đầu, sau đó là chữ/số/"_". Vd "balances", "_owner".
# Dùng với .match (đầu chuỗi là tên) hoặc .fullmatch (cả chuỗi là một tên). Dùng ở: static_analyzer.
RE_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")

# Khớp tên ĐÚNG MỘT TỪ (không dính vào tên dài hơn): re_word("owner") khớp "owner" nhưng không khớp "newOwner".
# Dùng ở: static_analyzer (guard modifier trong header, tham số trong biểu thức), contract_context (modifier trong header).
def re_word(name: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(name)}\b")


# Lời gọi tới một hàm theo tên: re_call("withdraw") khớp "withdraw(" và "withdraw (".
# Dùng ở: static_analyzer (_detect_array_by_reference, _detect_incorrect_assembly_return).
def re_call(name: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(name)}\s*\(")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 - TÁCH CODE UNIT (code_units.py) + NGỮ CẢNH PROMPT (contract_context.py)
# ══════════════════════════════════════════════════════════════════════════════

# Điểm BẮT ĐẦU khai báo một code unit, có 3 nhánh (group nào khớp cho biết loại unit):
#   function=<tên>        "function withdraw("
#   unnamed_fallback="("  "function (" - fallback không tên kiểu Solidity < 0.6
#   special=<từ khoá>     "constructor(", "receive(", "fallback("
# Dùng ở: code_units.parse_code_units.
RE_UNIT_DECLARATION = re.compile(
    r"\b(function\s+(?P<function>[A-Za-z_$][\w$]*)\s*\(|"
    r"function\s*(?P<unnamed_fallback>\()\s*|"
    r"(?P<special>constructor|receive|fallback)\s*\()",
    re.MULTILINE,
)

# Khai báo contract/library/interface: group(1) = loại, group(2) = tên. Vd "abstract contract Bank" -> (contract, Bank).
# Dùng ở: code_units._contract_at (unit thuộc contract nào).
RE_CONTRACT_DECLARATION = re.compile(r"\b(?:abstract\s+)?(contract|library|interface)\s+([A-Za-z_$][\w$]*)")

# Comment hoặc chuỗi, theo đúng thứ tự xuất hiện. Regex luôn khớp token SỚM NHẤT trước nên `//` nằm trong
# "http://..." thuộc về chuỗi, còn dấu ' trong `// don't` thuộc về comment. Comment / chuỗi chưa đóng kéo tới
# hết mã. Dùng ở: code_units.mask_noncode (nơi duy nhất che comment/chuỗi của cả pipeline).
RE_NONCODE = re.compile(
    r"//[^\n]*"                               # comment một dòng
    r"|/\*[\s\S]*?(?:\*/|\Z)"                 # comment khối
    r"|\"(?:\\[\s\S]?|[^\"\\])*(?:\"|\Z)"     # "chuỗi" (có ký tự thoát)
    r"|'(?:\\[\s\S]?|[^'\\])*(?:'|\Z)"        # 'chuỗi'
)

# Mọi ký tự trừ xuống dòng: thay bằng dấu cách để che comment/chuỗi mà vẫn giữ số dòng.
# Dùng ở: code_units.mask_noncode.
RE_NOT_NEWLINE = re.compile(r"[^\n]")

# Từ khoá view / pure / constant (constant = view của Solidity < 0.5) trong header hàm.
# Dùng ở: code_units.parse_code_units (is_view_or_pure).
RE_VIEW_PURE_CONSTANT = re.compile(r"\b(view|pure|constant)\b")

# Visibility đầu tiên trong header hàm: group(1) = public/external/internal/private.
# Dùng ở: code_units.parse_code_units.
RE_VISIBILITY = re.compile(r"\b(public|external|internal|private)\b")

# Khai báo tham số / biến cục bộ: "<kiểu>[[]] [memory|storage|calldata] <tên>" đứng trước "=", ";", "," hoặc ")".
# group(1) = vị trí dữ liệu (memory/storage/calldata hoặc None), group(2) = tên. Vd "uint256 amount)" -> amount.
# Dùng ở: contract_context.local_variable_names.
RE_LOCAL_DECLARATION = re.compile(
    r"\b(?:u?int\d*|address(?:\s+payable)?|bool|bytes\d*|string|var|[A-Z][\w$]*)"
    r"(?:\s*\[[^\]]*\])*\s+(?:(memory|storage|calldata)\s+)?([A-Za-z_$][\w$]*)\s*(?=[=;,)])"
)

# Biến bool trong tuple nhận kết quả lời gọi: "(bool success, ) = to.call(...)" -> success.
# Dùng ở: contract_context.local_variable_names.
RE_BOOL_TUPLE_LOCAL = re.compile(r"\(\s*bool\s+([A-Za-z_$][\w$]*)")

# Phép ghi (=, +=, -=, *=, /=, %=; không tính "=="): group(1) = vế trái kể cả [chỉ số] và .thành viên,
# group(2) = toán tử. Vd "balances[msg.sender] -= amount" -> ("balances[msg.sender]", "-=").
# Dùng ở: contract_context.extract_function_invariants (bất biến "state writes").
RE_INVARIANT_STATE_WRITE = re.compile(r"\b([a-zA-Z0-9_]+(?:\s*\[[^\]]+\])*(?:\.[a-zA-Z0-9_]+)*)\s*(\+=|-=|\*=|\/=|%=|=)(?!=)")

# Tên gốc ở đầu vế trái của phép ghi: "balances[msg.sender]" -> "balances".
# Dùng ở: contract_context.extract_function_invariants (bỏ biến cục bộ).
RE_LEADING_NAME = re.compile(r"[A-Za-z0-9_]+")

# Event được phát: "emit Transfer(" (group 1) hoặc "Transfer(" kiểu Solidity 0.4 (group 2, tên viết hoa,
# không đứng sau function/return/require(/assert(/revert().
# Dùng ở: contract_context.extract_function_invariants.
RE_INVARIANT_EVENT = re.compile(r"(?:emit\s+([A-Za-z_$][\w$]*)|(?<!function\s)(?<!return\s)(?<!require\()(?<!assert\()(?<!revert\()([A-Z][A-Za-z0-9_$]*))\s*\(")

# Lời gọi chuyển ETH / call ra ngoài, group(1) = "đối tượng.phương thức".
# Vd "msg.sender.call{value: amount}(" -> "msg.sender.call{value: amount}".
# Dùng ở: contract_context.extract_function_invariants.
RE_INVARIANT_EXTERNAL_CALL = re.compile(r"\b([a-zA-Z0-9_\[\]\.]+\s*\.\s*(?:transfer|send|call\.value|call\{value:[^}]+\}|call))\s*\(")

# Câu lệnh return, group(1) = biểu thức trả về (lấy cả khoảng trắng sau `return`: chuỗi đã bị che thành
# dấu cách, `\s+` sẽ nuốt mất nó). Dùng ở: contract_context.extract_function_invariants.
RE_INVARIANT_RETURN = re.compile(r"\breturn(?=\s)([^;]+);")

# Modifier kèm dấu "{" mở thân (tham số không bắt buộc), group(1) = tên. Vd "modifier onlyOwner() {".
# Dùng ở: contract_context.build_contract_context (đưa modifier kèm thân vào prompt).
RE_MODIFIER_OPENING = re.compile(r"\bmodifier\s+([A-Za-z_$][\w$]*)\s*(?:\([^)]*\))?[^{;]*\{")


# Phần mở đầu của contract/library/interface có tên cho trước, tới dấu "{" mở thân.
# Vd re_contract_opening("Bank") khớp "contract Bank is Ownable {".
# Dùng ở: contract_context.build_contract_context (tìm thân contract chứa hàm).
def re_contract_opening(contract_name: str) -> re.Pattern[str]:
    return re.compile(rf"\b(?:contract|library|interface)\s+{re.escape(contract_name)}\b[^{{]*\{{")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1A - PHÂN TÍCH TĨNH (static_analyzer.py)
# ══════════════════════════════════════════════════════════════════════════════

# ── Ranh giới hàm / contract / modifier ──────────────────────────────────────

# Hàm có thân: "function <name>(<params>) <header> {". Dùng ở: static_analyzer._iter_functions.
RE_FUNCTION_WITH_BODY = re.compile(
    r'\bfunction\s+(?P<name>[A-Za-z_]\w*)\s*'
    r'\((?P<params>[^)]*)\)(?P<header>[^;{}]*)\{',
    re.DOTALL,
)

# constructor / receive / fallback có thân: "<name>(<params>) <header> {". Dùng ở: static_analyzer._iter_functions.
RE_SPECIAL_FUNCTION_WITH_BODY = re.compile(
    r'\b(?P<name>constructor|receive|fallback)\s*'
    r'\((?P<params>[^)]*)\)(?P<header>[^;{}]*)\{',
    re.DOTALL,
)

# Contract có thân: "contract <name> <header> {" (header = phần "is A, B").
# Dùng ở: static_analyzer._iter_contracts.
RE_CONTRACT_WITH_BODY = re.compile(
    r'\bcontract\s+(?P<name>[A-Za-z_]\w*)'
    r'(?P<header>[^;{}]*)\{',
    re.DOTALL,
)

# Modifier có thân: "modifier <name>[(<params>)] <header> {". Dùng ở: static_analyzer._iter_modifiers.
RE_MODIFIER_WITH_BODY = re.compile(
    r'\bmodifier\s+(?P<name>[A-Za-z_]\w*)\s*(?:\((?P<params>[^)]*)\))?(?P<header>[^;{}]*)\{',
    re.DOTALL,
)

# Phần header của hàm đầu tiên, tới dấu "{". Vd "function burn(uint v) public {".
# Dùng ở: StaticAnalyzer._run_pattern_check (khi mã đầu vào chỉ là MỘT hàm).
RE_FUNCTION_HEADER_PREFIX = re.compile(r'function\s+[^{]*\{')

# ── Phân quyền (guard) ───────────────────────────────────────────────────────

# Các kiểu kiểm tra người gọi trong thân hàm/modifier (chỉ cần khớp MỘT):
RE_SENDER_CHECKS = (
    re.compile(r'require\s*\([^;]*msg\.sender\s*=='),                                 # require(msg.sender == owner)
    re.compile(r'require\s*\([^;]*==\s*msg\.sender'),                                 # require(owner == msg.sender)
    re.compile(r'if\s*\([^)]*msg\.sender\s*!=[^)]*\)\s*(?:revert\b|throw\b)'),         # if (msg.sender != owner) revert
    re.compile(r'if\s*\([^)]*!=\s*msg\.sender[^)]*\)\s*(?:revert\b|throw\b)'),         # if (owner != msg.sender) throw
    re.compile(r'require\s*\([^;]*\[msg\.sender\]'),                                  # require(owners[msg.sender])
    re.compile(r'require\s*\([^;]*(?:isOwner|hasRole|authorized)\s*\('),              # require(hasRole(ADMIN, msg.sender))
)
# Dùng ở: static_analyzer._has_sender_check (cho extract_guard_modifiers và is_function_guarded).

# Tên của mọi modifier được khai báo, group(1) = tên. Dùng ở: static_analyzer.extract_guard_modifiers.
RE_MODIFIER_NAME = re.compile(r'\bmodifier\s+([A-Za-z0-9_]+)')

# Tên modifier mang nghĩa phân quyền: onlyXxx, auth, authorized, restricted.
# Dùng ở: static_analyzer.extract_guard_modifiers.
RE_GUARD_MODIFIER_NAME = re.compile(r'^(?:only[A-Z]\w*|auth|authorized|restricted)$')

# Visibility internal hoặc private (hàm không gọi được từ bên ngoài).
# Dùng ở: static_analyzer.is_function_guarded, _detect_return_instead_of_leave.
RE_INTERNAL_OR_PRIVATE = re.compile(r'\b(?:private|internal)\b')

# Visibility public hoặc external. Dùng ở: static_analyzer (_detect_locked_ether, _detect_delegatecall_in_loop,
# _detect_controlled_array_length).
RE_PUBLIC_OR_EXTERNAL = re.compile(r'\b(?:public|external)\b')

# Từ khoá payable trong header hàm. Dùng ở: static_analyzer (_detect_locked_ether, _detect_msg_value_in_loop,
# _detect_delegatecall_in_loop).
RE_PAYABLE = re.compile(r'\bpayable\b')

# Mọi "từ" (chữ/số/_) trong header, group(1) = từ. Dùng ở: static_analyzer.is_function_guarded (tìm onlyXxx).
RE_WORD_TOKEN = re.compile(r'\b([A-Za-z0-9_]+)\b')

# ── Khai báo biến / kiểu ─────────────────────────────────────────────────────

# Khai báo "<kiểu>[[]] [từ khoá...] <tên>" ở bất kỳ đâu (state, tham số, biến cục bộ, thành viên struct).
# Vd "uint256[] memory ids" -> type="uint256[]", name="ids". Dùng ở: static_analyzer._collect_symbol_types.
RE_SYMBOL_DECLARATION = re.compile(
    r'\b(?P<type>[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])*)\s+'
    r'(?:(?:public|private|internal|external|constant|immutable|memory|'
    r'calldata|storage|payable|indexed)\s+)*'
    r'(?P<name>[A-Za-z_]\w*)\b'
)

# Một tham số hàm (cả chuỗi): "<kiểu> [memory|storage|calldata|payable] <tên>".
# Vd "uint[] storage arr" -> type="uint[]", location="storage", name="arr".
# Dùng ở: static_analyzer._parse_parameter_details.
RE_PARAMETER = re.compile(
    r'^\s*(?P<type>[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])*)\s*'
    r'(?P<location>memory|storage|calldata|payable)?\s*'
    r'(?P<name>[A-Za-z_]\w*)\s*$'
)

# Câu lệnh khai báo biến trạng thái tới dấu ";", tail = phần sau tên (có "=" nghĩa là đã khởi tạo).
# Vd "address public owner = msg.sender;" -> type="address", name="owner", tail=" = msg.sender".
# Dùng ở: static_analyzer._collect_contract_state_declarations.
RE_STATE_DECLARATION = re.compile(
    r'^\s*(?P<type>[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])*)\s+'
    r'(?:(?:public|private|internal|constant|immutable|payable)\s+)*'
    r'(?P<name>[A-Za-z_]\w*)\b(?P<tail>[^;]*);'
)

# Như RE_STATE_DECLARATION nhưng chỉ lấy kiểu + tên (không cần tới ";").
# Dùng ở: static_analyzer._collect_state_symbol_types.
RE_STATE_TYPE_AND_NAME = re.compile(
    r'^\s*(?P<type>[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])*)\s+'
    r'(?:(?:public|private|internal|constant|immutable|payable)\s+)*'
    r'(?P<name>[A-Za-z_]\w*)\b'
)

# Biến trạng thái kiểu mapping, name = tên. Vd "mapping(address => uint) public balances" -> balances.
# Dùng ở: static_analyzer._collect_contract_state_declarations.
RE_STATE_MAPPING = re.compile(
    r'^\s*mapping\s*\([^;]+\)\s*'
    r'(?:(?:public|private|internal)\s+)*(?P<name>[A-Za-z_]\w*)\b'
)

# Mapping trạng thái kèm kiểu giá trị. Vd "mapping(address => User) users" -> value="User", name="users".
# Dùng ở: static_analyzer._collect_mapping_value_types.
RE_MAPPING_VALUE_TYPE = re.compile(
    r'^\s*mapping\s*\([^=;]+=>\s*(?P<value>[A-Za-z_]\w*)\s*\)\s*'
    r'(?:(?:public|private|internal)\s+)*(?P<name>[A-Za-z_]\w*)\b'
)

# Mapping PUBLIC (ở bất kỳ đâu trong mã) kèm kiểu giá trị. Vd "mapping(uint => Order) public orders".
# Dùng ở: static_analyzer._detect_public_mapping_nested.
RE_PUBLIC_MAPPING_VALUE_TYPE = re.compile(
    r'\bmapping\s*\([^=;]+=>\s*(?P<value>[A-Za-z_]\w*)\s*\)\s*'
    r'public\s+(?P<name>[A-Za-z_]\w*)\b'
)

# Từ khoá mapping( - struct có chứa mapping trực tiếp không. Dùng ở: static_analyzer._collect_struct_members.
RE_MAPPING_KEYWORD = re.compile(r'\bmapping\s*\(')

# Mở đầu khai báo struct, name = tên. Vd "struct User {". Dùng ở: static_analyzer._collect_struct_members.
RE_STRUCT_OPENING = re.compile(r'\bstruct\s+(?P<name>[A-Za-z_]\w*)\s*\{')

# Mở đầu khai báo enum, name = tên. Vd "enum State {". Dùng ở: static_analyzer._detect_dangerous_enum_conversion.
RE_ENUM_OPENING = re.compile(r'\benum\s+(?P<name>[A-Za-z_]\w*)\s*\{')

# Phần mảng ở cuối tên kiểu (từ "[" đầu tiên): sub("", "User[]") -> "User".
# Dùng ở: static_analyzer (_struct_contains_mapping, _detect_public_mapping_nested, _detect_mapping_deletion,
# _detect_storage_abiencoderv2_array).
RE_ARRAY_SUFFIX = re.compile(r'\[.*$')

# Kiểu số nguyên uintN / intN (cả chuỗi), unsigned = "u" hoặc "", bits = N (rỗng = 256).
# Dùng ở: static_analyzer._integer_type_range.
RE_INTEGER_TYPE = re.compile(r'(?P<unsigned>u?)int(?P<bits>\d*)')

# Mảng số nguyên CÓ DẤU, base = kiểu gốc. Vd "int8[]" -> "int8". Dùng ở: static_analyzer._signed_integer_base.
RE_SIGNED_INT_ARRAY_TYPE = re.compile(r'(?P<base>int\d*)\[')

# Ép kiểu ở đầu biểu thức "<Kiểu>(": "IERC20(token)" -> type="IERC20". Dùng ở: static_analyzer._looks_like_erc20.
RE_CAST_PREFIX = re.compile(r'(?P<type>[A-Za-z_]\w*)\(')

# Biến cục bộ là tham chiếu storage tới mảng. Vd "uint[] storage list" -> name="list".
# Dùng ở: static_analyzer._detect_array_by_reference.
RE_STORAGE_ARRAY_LOCAL = re.compile(
    r'\b[A-Za-z_]\w*(?:\s*\[\s*\d*\s*\])+\s+storage\s+'
    r'(?P<name>[A-Za-z_]\w*)\b'
)

# Danh sách contract cha sau "is" (tới hết header). Vd " is Ownable, Pausable " -> bases="Ownable, Pausable ".
# Dùng ở: static_analyzer._direct_base_names.
RE_INHERITANCE_LIST = re.compile(r'\bis\s+(?P<bases>[\s\S]+)$')

# Tên đứng đầu một mục (cho phép khoảng trắng trước): " Ownable(msg.sender)" -> "Ownable".
# Dùng ở: static_analyzer._direct_base_names.
RE_LEADING_IDENTIFIER = re.compile(r'\s*([A-Za-z_]\w*)')

# Từ khoá abstract ở cuối đoạn mã ngay trước "contract". Dùng ở: static_analyzer._detect_locked_ether.
RE_ABSTRACT_AT_END = re.compile(r'\babstract\s*$')

# Từ khoá "is" (contract có kế thừa). Dùng ở: static_analyzer._detect_locked_ether.
RE_IS_KEYWORD = re.compile(r'\bis\b')

# ── Pragma ───────────────────────────────────────────────────────────────────

# Toàn bộ câu lệnh pragma solidity (để che đi, tránh nhầm "^0.4.24" là phép XOR).
# Dùng ở: static_analyzer._detect_incorrect_exponentiation.
RE_PRAGMA_STATEMENT = re.compile(r'pragma\s+solidity\s+[^;]+;')

# Pragma cho phép Solidity 0.4.x. Dùng ở: static_analyzer._detect_public_mapping_nested.
RE_PRAGMA_0_4 = re.compile(r'\bpragma\s+solidity\s+[^;]*\b0\.4\.')

# Pragma pin cứng 0.4.0 -> 0.4.4 (chưa revert khi ép số sai sang enum).
# Dùng ở: static_analyzer._detect_dangerous_enum_conversion.
RE_PRAGMA_0_4_0_TO_0_4_4 = re.compile(r'\bpragma\s+solidity\s+(?:=\s*)?0\.4\.[0-4]\s*;')

# Pragma pin cứng 0.4.x / 0.5.x, minor = 4|5, patch = số cuối. Vd "pragma solidity 0.4.24;" -> (4, 24).
# Dùng ở: static_analyzer._uses_exact_vulnerable_storage_array_compiler.
RE_PINNED_PRAGMA_0_4_OR_0_5 = re.compile(
    r'\bpragma\s+solidity\s+(?:=\s*)?0\.(?P<minor>[45])\.'
    r'(?P<patch>\d+)\s*;'
)

# "pragma experimental ABIEncoderV2;". Dùng ở: static_analyzer._detect_storage_abiencoderv2_array.
RE_PRAGMA_ABIENCODERV2 = re.compile(r'\bpragma\s+experimental\s+ABIEncoderV2\s*;')

# ── Biểu thức, lời gọi, vòng lặp ─────────────────────────────────────────────

# Đầu vòng lặp "for (" hoặc "while (". Dùng ở: static_analyzer._iter_loop_ranges.
RE_LOOP_OPENING = re.compile(r'\b(?:for|while)\s*\(')

# Khối assembly (Yul), có thể kèm ("memory-safe"), tới dấu "{".
# Dùng ở: static_analyzer._detect_incorrect_shift_in_assembly, _assembly_return_positions.
RE_ASSEMBLY_BLOCK = re.compile(r'\bassembly(?:\s*\([^)]*\))?\s*\{')

# Lệnh Yul return(offset, size). Dùng ở: static_analyzer._assembly_return_positions.
RE_YUL_RETURN_CALL = re.compile(r'\breturn\s*\([^,]+,[^)]+\)')

# Lệnh Yul shl/shr/sar(shift, value) mà shift là BIẾN còn value là HẰNG (dấu hiệu đảo tham số).
# Vd "shl(x, 1)". Dùng ở: static_analyzer._detect_incorrect_shift_in_assembly.
RE_YUL_SHIFT_WITH_CONSTANT_VALUE = re.compile(
    r'\b(?:shl|shr|sar)\s*\(\s*'
    r'(?P<shift>[A-Za-z_]\w*)\s*,\s*'
    r'(?P<value>0x[0-9A-Fa-f]+|\d+)\s*\)'
)

# Mệnh đề returns trong header, values = danh sách giá trị trả về. Vd "returns (uint a, bool b)".
# Dùng ở: static_analyzer._detect_return_instead_of_leave.
RE_RETURNS_CLAUSE = re.compile(r'\breturns\s*\((?P<values>[^)]*)\)')

# Phép "^" (XOR) giữa hai toán hạng (số hex, số thập phân hoặc tên), không tính "^=". Vd "2 ^ 256".
# Dùng ở: static_analyzer._detect_incorrect_exponentiation.
_XOR_OPERAND = r'(?:0x[0-9A-Fa-f]+|\d+|[A-Za-z_]\w*)'
RE_XOR_EXPRESSION = re.compile(
    rf'(?P<left>{_XOR_OPERAND})\s*\^(?!=)\s*(?P<right>{_XOR_OPERAND})'
)

# Chuỗi chỉ gồm chữ số thập phân (dùng với .fullmatch). Dùng ở: static_analyzer._detect_incorrect_exponentiation.
RE_DECIMAL_NUMBER = re.compile(r'\d+')

# Phép so sánh thứ tự (<, >, <=, >=) giữa tên và số nguyên. Vd "x >= 0", "300 > b".
# Dùng ở: static_analyzer._detect_type_based_tautology.
_COMPARISON_OPERAND = r'(?:[A-Za-z_]\w*|-?(?:0x[0-9A-Fa-f]+|\d+))'
RE_ORDER_COMPARISON = re.compile(
    rf'(?P<left>{_COMPARISON_OPERAND})\s*(?P<operator><=|>=|<|>)\s*(?P<right>{_COMPARISON_OPERAND})'
)

# Số nguyên (có thể âm, thập phân hoặc hex) - dùng với .fullmatch. Vd "-1", "0xff".
# Dùng ở: static_analyzer._detect_type_based_tautology.
RE_INTEGER_LITERAL = re.compile(r'-?(?:0x[0-9A-Fa-f]+|\d+)')

# Điều kiện if là hằng: "if (true)" / "if (false)". Dùng ở: static_analyzer._detect_boolean_constant_misuse.
RE_IF_CONSTANT_CONDITION = re.compile(r'\bif\s*\(\s*(?:true|false)\s*\)')

# Hằng true/false đứng cạnh && hoặc ||: "x || true", "false && y". Dùng ở: static_analyzer._detect_boolean_constant_misuse.
RE_BOOLEAN_CONSTANT_OPERAND = re.compile(
    r'(?:\b(?:true|false)\b\s*(?:&&|\|\|)'
    r'|(?:&&|\|\|)\s*\b(?:true|false)\b)'
)

# Toán tử so sánh bằng "==". Dùng ở: static_analyzer._detect_dangerous_balance_equality.
RE_STRICT_EQUALITY = re.compile(r'==')

# Nguồn số dư: ".balance" hoặc ".balanceOf(". Dùng ở: static_analyzer._detect_dangerous_balance_equality.
RE_BALANCE_SOURCE = re.compile(r'\.balance\b|\.balanceOf\s*\(')

# Biến được gán từ số dư, name = tên biến. Vd "uint bal = address(this).balance;" -> bal.
# Dùng ở: static_analyzer._detect_dangerous_balance_equality.
RE_BALANCE_ASSIGNMENT = re.compile(
    r'\b(?:u?int\d*\s+)?(?P<name>[A-Za-z_]\w*)\s*=\s*'
    r'[^;]*(?:\.balance\b|\.balanceOf\s*\()[^;]*;'
)

# Lời gọi abi.encodePacked(. Dùng ở: static_analyzer._detect_encode_packed_collision.
RE_ENCODE_PACKED_CALL = re.compile(r'\babi\.encodePacked\s*\(')

# Lời gọi abi.encode(. Dùng ở: static_analyzer._detect_storage_abiencoderv2_array.
RE_ABI_ENCODE_CALL = re.compile(r'\babi\.encode\s*\(')

# Lời gọi transfer/transferFrom trên một tên hoặc một ép kiểu.
# Vd "token.transfer(" -> receiver="token"; "IERC20(t).transferFrom(" -> receiver="IERC20(t)", method="transferFrom".
# Dùng ở: static_analyzer._iter_erc20_calls.
RE_ERC20_TRANSFER_CALL = re.compile(
    r'(?P<receiver>(?:[A-Za-z_]\w*|'
    r'[A-Za-z_]\w*\s*\([^()]*\)))\s*\.\s*'
    r'(?P<method>transferFrom|transfer)\s*\('
)

# Mọi đường gửi ETH ra khỏi contract: call{value:}, send, transfer, selfdestruct/suicide, call trong assembly.
# Dùng ở: static_analyzer._detect_locked_ether.
RE_OUTGOING_ETHER = re.compile(
    r'\.call\s*\{[^}]*\bvalue\s*:'
    r'|\.send\s*\('
    r'|\.transfer\s*\('
    r'|\b(?:selfdestruct|suicide)\s*\('
    r'|\bassembly\s*\{[^}]*\bcall\s*\(',
    re.DOTALL,
)

# msg.value bị so sánh với 0 ("msg.value == 0", "0 != msg.value") - phần này được xoá trước khi tìm msg.value.
# Dùng ở: static_analyzer._detect_msg_value_in_loop.
RE_MSG_VALUE_ZERO_CHECK = re.compile(
    r'msg\.value\s*(?:==|!=)\s*0|0\s*(?:==|!=)\s*msg\.value'
)

# msg.value. Dùng ở: static_analyzer._detect_msg_value_in_loop.
RE_MSG_VALUE = re.compile(r'\bmsg\.value\b')

# Lời gọi .delegatecall(. Dùng ở: static_analyzer._detect_delegatecall_in_loop.
RE_DELEGATECALL = re.compile(r'\.delegatecall\s*\(')

# Gán độ dài mảng: "arr.length = <rhs>;". Dùng ở: static_analyzer._detect_controlled_array_length.
RE_ARRAY_LENGTH_ASSIGNMENT = re.compile(
    r'\b(?P<array>[A-Za-z_]\w*)\.length\s*=\s*(?P<rhs>[^;]+);'
)

# Câu lệnh delete, base = tên biến gốc. Vd "delete users[id];" -> target="users[id]", base="users".
# Dùng ở: static_analyzer._detect_mapping_deletion.
RE_DELETE_STATEMENT = re.compile(
    r'\bdelete\s+(?P<target>(?P<base>[A-Za-z_]\w*)'
    r'(?:\s*\[[^\]]+\])?)\s*;'
)

# ── Regex chứa tên lấy từ mã đang phân tích ──────────────────────────────────

# address(0) hoặc address(0x0...) - đoạn dùng chung của các regex kiểm tra zero address bên dưới.
_ZERO_ADDRESS = r'address\s*\(\s*(?:0|0x0+)\s*\)'


# require/assert kiểm tra tham số khác address(0). Vd "require(to != address(0))".
# Dùng ở: static_analyzer._missing_zero_address_hits.
def re_zero_address_require(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf'\b(?:require|assert)\s*\([^;]*'
        rf'(?:\b{escaped}\b\s*!=\s*{_ZERO_ADDRESS}|{_ZERO_ADDRESS}\s*!=\s*\b{escaped}\b)'
    )


# if (tham số == address(0)) rồi revert/return. Vd "if (to == address(0)) { revert(); }".
# Dùng ở: static_analyzer._missing_zero_address_hits.
def re_zero_address_revert_if(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf'\bif\s*\([^;{{}}]*'
        rf'(?:\b{escaped}\b\s*==\s*{_ZERO_ADDRESS}|{_ZERO_ADDRESS}\s*==\s*\b{escaped}\b)'
        rf'[^;{{}}]*\)\s*(?:\{{\s*)?(?:revert|return)\b',
        re.DOTALL,
    )


# Gán tham số vào một biến: "<lhs> = <name>;", lhs = vế trái. Vd "owner = newOwner;" -> lhs="owner".
# Dùng ở: static_analyzer._missing_zero_address_hits.
def re_assignment_from(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf'(?P<lhs>\b[A-Za-z_]\w*(?:\s*\[[^\]]+\]|\.[A-Za-z_]\w*)?)'
        rf'\s*=\s*\b{escaped}\b\s*;'
    )


# Tham số được dùng làm đích gọi/chuyển tiền: "to.call", "to.transfer", "payable(to).", "IERC20(to).".
# Dùng ở: static_analyzer._missing_zero_address_hits.
def re_call_target(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf'(?:\b{escaped}\b\s*\.\s*(?:call|send|transfer)\b'
        rf'|payable\s*\(\s*\b{escaped}\b\s*\)\s*\.'
        rf'|[A-Za-z_]\w*\s*\(\s*\b{escaped}\b\s*\)\s*\.)'
    )


# Biến bị ghi (=, +=, -=, ++, --; không tính "=="). Vd re_state_write("owner") khớp "owner = x".
# Dùng ở: static_analyzer._detect_uninitialized_state_address.
def re_state_write(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(rf'\b{escaped}\b\s*(?:=(?!=)|\+=|-=|\+\+|--)')


# Biến address được dùng làm đích chuyển ETH: "owner.transfer", "payable(owner).", "selfdestruct(owner".
# Dùng ở: static_analyzer._detect_uninitialized_state_address.
def re_ether_destination(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf'(?:\b{escaped}\b\s*\.\s*(?:transfer|send|call)\b'
        rf'|payable\s*\(\s*\b{escaped}\b\s*\)\s*\.'
        rf'|(?:selfdestruct|suicide)\s*\(\s*\b{escaped}\b)'
    )


# Ghi vào một phần tử của mảng: "arr[i] = ...", "arr[i] += ...", "arr[i]++".
# Dùng ở: static_analyzer._detect_array_by_reference.
def re_indexed_write(name: str) -> re.Pattern[str]:
    return re.compile(
        rf'\b{re.escape(name)}\s*\[[^\]]+\]\s*'
        rf'(?:=|\+=|-=|\*=|/=|%=|\+\+|--)'
    )


# Gán cả biến: "<name> = <rhs>;", rhs = vế phải. Dùng ở: static_analyzer._detect_storage_signed_integer_array.
def re_assignment_to(name: str) -> re.Pattern[str]:
    return re.compile(rf'\b{re.escape(name)}\s*=\s*(?P<rhs>[^;]+);')


# Ép kiểu sang enum: "State(x)". Dùng ở: static_analyzer._detect_dangerous_enum_conversion.
def re_enum_conversion(enum_name: str) -> re.Pattern[str]:
    return re.compile(rf'\b{re.escape(enum_name)}\s*\([^)]*\)')


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1B - RAG: 37 SWC AGENT (swc_agents.py) + XÂY CƠ SỞ TRI THỨC (build_knowledge_base.py)
# Các regex của swc_agents bỏ qua hoa/thường (IGNORECASE) như bản gốc, trừ khi ghi khác.
# ══════════════════════════════════════════════════════════════════════════════

# Mã SWC "SWC-xxx" trong một chuỗi bất kỳ. Dùng ở: swc_agents.normalize_swc_id.
RE_SWC_ID = re.compile(r"SWC-\d{3}")

# Nội dung các khối ```...``` (có hoặc không tên ngôn ngữ), group(1) = phần mã.
# Dùng ở: swc_agents.extract_signal_code.
RE_ANY_CODE_FENCE = re.compile(r"```(?:\w+)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

# Như RE_PRAGMA_SPEC nhưng bỏ qua hoa/thường. Dùng ở: swc_agents.pragma_version.
RE_PRAGMA_SPEC_ANY_CASE = re.compile(r"pragma\s+solidity\s+([^;]+);", re.IGNORECASE)

# Số phiên bản "major.minor[.patch]" đầu tiên. Vd "^0.4.24" -> (0, 4, 24). Dùng ở: swc_agents.pragma_version.
RE_VERSION_NUMBER = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")

# Có phép toán số học: ++, --, +=, -=, *=, phép gán có + - *, hoặc add/sub/mul(.
# Dùng ở: swc_agents.has_arithmetic_operation.
RE_ARITHMETIC_OPERATION = re.compile(
    r"(?:\+\+|--|[+\-*]=|=\s*[^;]*(?:\+|-|\*)[^;]*;|\b(?:add|sub|mul)\s*\()",
    re.IGNORECASE | re.DOTALL,
)

# Lời gọi cấp thấp .call / .delegatecall / .staticcall / .send. Dùng ở: swc_agents.low_level_call_statements.
RE_LOW_LEVEL_CALL = re.compile(r"\.(?:call|delegatecall|staticcall|send)\s*[\(\{]", re.IGNORECASE | re.DOTALL)

# Tên biến bool trong tuple nhận kết quả: "(bool ok, ) = ..." -> ok. Dùng ở: swc_agents.statement_result_names.
RE_BOOL_TUPLE_RESULT = re.compile(r"\(\s*bool\s+(\w+)\s*,?", re.IGNORECASE)

# Biến bool gán thẳng từ lời gọi cấp thấp: "bool ok = to.send(1);" -> ok. Dùng ở: swc_agents.statement_result_names.
RE_BOOL_ASSIGNED_CALL_RESULT = re.compile(
    r"\bbool\s+(\w+)\s*=\s*[^;]*\.(?:call|delegatecall|staticcall|send)\s*[\(\{]",
    re.IGNORECASE | re.DOTALL,
)

# Lời gọi cấp thấp nằm ngay trong require/assert: "require(to.send(1))". Dùng ở: swc_agents.is_result_checked.
RE_CALL_INSIDE_REQUIRE = re.compile(
    r"\b(?:require|assert)\s*\([^;]*\.(?:send|call|delegatecall|staticcall)\s*[\(\{]",
    re.IGNORECASE | re.DOTALL,
)

# Có chống reentrancy: modifier nonReentrant hoặc kế thừa ReentrancyGuard.
# Dùng ở: swc_agents.has_state_update_after_external_value_call.
RE_REENTRANCY_GUARD = re.compile(r"\bnonReentrant\b|ReentrancyGuard", re.IGNORECASE)

# Lời gọi gửi ETH: ".call{value: ...}" hoặc ".send(". Dùng ở: swc_agents.has_state_update_after_external_value_call.
RE_VALUE_CALL_OR_SEND = re.compile(r"\.(?:call\s*\{[^}]*value\s*:|send\s*\()", re.IGNORECASE | re.DOTALL)

# Cập nhật state: ghi balances[...], delete x[...], hoặc "a = b" / "a += b" / "a -= b".
# Dùng ở: swc_agents.has_state_update_after_external_value_call (tìm SAU lời gọi gửi ETH).
RE_STATE_UPDATE = re.compile(
    r"(?:balances?\s*\[[^\]]+\]\s*(?:-=|=|\+=)|\bdelete\s+\w+\s*\[|\b\w+\s*(?:-=|\+=|=)\s*\w+)",
    re.IGNORECASE | re.DOTALL,
)

# Thân vòng lặp for/while (tới dấu "}" đầu tiên), group(1) = thân. Dùng ở: swc_agents.loop_bodies.
RE_LOOP_BODY = re.compile(r"\b(?:for|while)\s*\([^)]*\)\s*\{(.*?)\}", re.IGNORECASE | re.DOTALL)

# Dấu "=" đứng một mình (phép gán, không phải ==, !=, <=, >=, =>). Dùng ở: swc_agents.find_assignment_in_condition.
RE_SINGLE_EQUALS = re.compile(r"(?<![=!<>])=(?![=>])")

# SWC-100: hàm mà phần header sau ")" không có visibility. Dùng ở: swc_agents.find_custom_signal_hits.
RE_FUNCTION_WITHOUT_VISIBILITY = re.compile(
    r"\bfunction\s+\w+\s*\([^)]*\)\s*(?![^{};]*(?:public|private|internal|external))",
    re.IGNORECASE | re.DOTALL,
)

# Khối "unchecked {" (tắt kiểm tra tràn số của 0.8+). Phân biệt hoa/thường. Dùng ở: swc_agents (SWC-101).
RE_UNCHECKED_BLOCK = re.compile(r"\bunchecked\s*\{")

# Có dùng thư viện SafeMath. Phân biệt hoa/thường. Dùng ở: swc_agents (SWC-101).
RE_SAFEMATH = re.compile(r"\bSafeMath\b")

# Ép kiểu thu hẹp "uint8(" ... "int128(". Phân biệt hoa/thường. Dùng ở: swc_agents (SWC-101).
RE_NARROW_INT_CAST = re.compile(r"\b(?:u?int)(?:8|16|32|64|128)\s*\(")

# Pragma "trôi": có ^, >=, > hoặc *. Vd "pragma solidity ^0.4.24". Dùng ở: swc_agents (SWC-103).
RE_FLOATING_PRAGMA = re.compile(r"pragma\s+solidity\s+[^;]*(?:\^|>=|>|\*)", re.IGNORECASE)

# Lời gọi gửi ETH: ".call{value: ...}", ".send(", ".transfer(". Dùng ở: swc_agents (SWC-105).
RE_ETHER_TRANSFER = re.compile(r"\.(?:call\s*\{[^}]*value\s*:|send\s*\(|transfer\s*\()", re.IGNORECASE | re.DOTALL)

# Hàm rút tiền kiểu quản trị: drain / sweep / rescue / withdrawAll / emergencyWithdraw. Dùng ở: swc_agents (SWC-105).
RE_ADMIN_WITHDRAW_FUNCTION = re.compile(
    r"\bfunction\s+(?:drain|sweep|rescue|withdrawAll|emergencyWithdraw)\b", re.IGNORECASE
)

# Toàn bộ số dư của contract: "address(this).balance" hoặc "this.balance". Dùng ở: swc_agents (SWC-105).
RE_CONTRACT_BALANCE = re.compile(r"address\s*\(\s*this\s*\)\.balance|\bthis\.balance\b", re.IGNORECASE)

# Kiểm tra số dư của chính người gọi: "balances[msg.sender] >= amount". Dùng ở: swc_agents (SWC-105).
RE_USER_BALANCE_GUARD = re.compile(r"balances?\s*\[\s*msg\.sender\s*\].{0,80}(?:>=|>)", re.IGNORECASE | re.DOTALL)

# Lời gọi selfdestruct( / suicide(. Dùng ở: swc_agents (SWC-106).
RE_SELFDESTRUCT_CALL = re.compile(r"\b(?:selfdestruct|suicide)\s*\(", re.IGNORECASE)

# Lời gọi .delegatecall( / .callcode(. Dùng ở: swc_agents (SWC-112).
RE_DELEGATECALL_OR_CALLCODE = re.compile(r"\.\s*(?:delegatecall|callcode)\s*\(", re.IGNORECASE)

# Lời gọi ra ngoài .transfer / .send / .call (trong thân vòng lặp). Dùng ở: swc_agents (SWC-113).
RE_EXTERNAL_CALL = re.compile(r"\.(?:transfer|send|call)\s*[\(\{]", re.IGNORECASE | re.DOTALL)

# Hàm tên viết hoa (constructor kiểu cũ trùng tên contract). Phân biệt hoa/thường. Dùng ở: swc_agents (SWC-118).
RE_CAPITALIZED_FUNCTION = re.compile(r"\bfunction\s+[A-Z]\w+\s*\(")

# Biến lưu giá trị nhạy cảm: "string private password", "bytes32 secretKey"... Dùng ở: swc_agents (SWC-136).
RE_SENSITIVE_STATE_VALUE = re.compile(
    r"\b(?:string|bytes\d*|uint\d*|address)\s+(?:private\s+)?\w*(?:password|secret|privateKey|apiKey|seed)\w*",
    re.IGNORECASE,
)

# Có cơ chế phân quyền: onlyOwner, onlyRole, onlyAdmin, AccessControl, Ownable, require(msg.sender == ...).
# Dùng ở: swc_agents.has_access_control_guard.
RE_ACCESS_CONTROL_GUARD = re.compile(
    r"\bonlyOwner\b|onlyRole|onlyAdmin|AccessControl|Ownable|require\s*\(\s*msg\.sender\s*==",
    re.IGNORECASE,
)


# Kết quả bool tên `name` có được kiểm tra: require(name, assert(name, if (!name, if (name == false.
# Dùng ở: swc_agents.is_result_checked.
def re_result_check(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf"(?:require\s*\(\s*{escaped}\b|assert\s*\(\s*{escaped}\b|if\s*\(\s*!\s*{escaped}\b|if\s*\(\s*{escaped}\s*==\s*false)",
        re.IGNORECASE,
    )


# Mở đầu điều kiện "<keyword>(" với keyword = "if" hoặc "require". Dùng ở: swc_agents.condition_blocks.
def re_condition_opening(keyword: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(keyword)}\s*\(", re.IGNORECASE)


# Markdown (build_knowledge_base.clean_markdown / checklist): làm sạch mô tả SWC thành văn bản thuần.
RE_MARKDOWN_CODE_BLOCK = re.compile(r"```[\s\S]*?```")          # khối ```code```
RE_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")         # [chữ](link) -> group(1) = chữ
RE_MARKDOWN_SYMBOLS = re.compile(r"[`*_>#-]+")                  # ký hiệu định dạng ` * _ > # -
RE_BULLET_PREFIX = re.compile(r"^\s*[-*]\s*")                   # gạch đầu dòng "- " / "* "
RE_SENTENCE_BREAK = re.compile(r"[.;]\s+")                      # chỗ ngắt câu ". " / "; "


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 ADVISOR / PHASE 3 ASSESSOR
# Advisor chỉ dùng RE_PRAGMA_SPEC (mục dùng chung); Assessor không dùng regex.
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 - FIXER (fixer.py, patching.py, behavior_guard.py)
# ══════════════════════════════════════════════════════════════════════════════

# Mọi phiên bản đầy đủ "x.y.z" trong pragma, mỗi kết quả = (major, minor, patch).
# Vd ">=0.4.22 <0.6.0" -> [("0","4","22"), ("0","6","0")]. Dùng ở: fixer._min_pragma_version.
RE_FULL_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")

# JSON (từ "{" đầu tới "}" cuối) có chứa khoá "replacement". Dùng ở: patching.extract_declaration_body.
RE_JSON_WITH_REPLACEMENT = re.compile(r"\{[\s\S]*\"replacement\"[\s\S]*\}")

# Khối ```solidity ...``` (hoặc ``` không tên), group(1) = mã. Dùng ở: patching.extract_declaration_body.
RE_SOLIDITY_FENCE = re.compile(r"```(?:solidity)?\s*([\s\S]*?)```")

# "function <tên>(" - một khai báo hàm có tên. Dùng ở: patching.extract_declaration_body (cắt hàm thừa phía sau).
RE_NAMED_FUNCTION_START = re.compile(r"\bfunction\s+[A-Za-z_$][\w$]*\s*\(")


# Đầu một trường chuỗi JSON: '"<field>": "'. Dùng ở: patching.lenient_json_string_field.
def re_json_string_field_start(field: str) -> re.Pattern[str]:
    return re.compile(rf'"{re.escape(field)}"\s*:\s*"')


# Điểm bắt đầu khai báo của hàm cần lấy từ câu trả lời của model (group(1) = phần đầu khai báo):
#   target là fallback/receive -> "function <tên>(", "function (" (fallback cũ), constructor/receive/fallback(
#   target là tên khác         -> "function <tên>(", constructor/receive/fallback(
#   không có target            -> bất kỳ "function <tên>(", "function (", constructor/receive/fallback(
# Dùng ở: patching.extract_declaration_body.
def re_declaration_start(target_name: str | None) -> re.Pattern[str]:
    if target_name in ("fallback", "receive"):
        return re.compile(rf"\b((?:function\s+{re.escape(target_name)}\s*\(|function\s*\(|(?:constructor|receive|fallback)\s*\())")
    if target_name:
        return re.compile(rf"\b(function\s+{re.escape(target_name)}\s*\(|(?:constructor|receive|fallback)\s*\()")
    return re.compile(r"\b(function\s+[A-Za-z_$][\w$]*\s*\(|function\s*\(|(?:constructor|receive|fallback)\s*\()")


# Guard hành vi (behavior_guard.validate_observable_behavior): các kiểu chuyển ETH và dạng an toàn tương đương.
RE_ETH_VALUE_CALL = re.compile(r"(?:\.call\s*\{\s*value\s*:|\.call\.value\s*\()")   # .call{value: x}( / .call.value(x)(
RE_TRANSFER_CALL = re.compile(r"\.transfer\s*\(")                                    # to.transfer(
RE_SAFE_TRANSFER_CALL = re.compile(r"\.safeTransfer\s*\(")                           # token.safeTransfer(
RE_SEND_CALL = re.compile(r"\.send\s*\(")                                            # to.send(
RE_SEND_VALUE_CALL = re.compile(r"\bsendValue\s*\(")                                 # Address.sendValue(
RE_SAFE_TRANSFER_ETH_CALL = re.compile(r"\bsafeTransferETH\s*\(")                    # SafeTransferLib.safeTransferETH(

# Từ khoá return. Dùng ở: behavior_guard.validate_observable_behavior (bản vá không được bỏ return).
RE_RETURN_KEYWORD = re.compile(r"\breturn\b")

# Biến bị ghi (=, +=, -=, *=, /=, %=; không tính "=="), group(1) = tên gốc (bỏ [chỉ số]).
# Vd "balances[to] += v" -> "balances". Dùng ở: behavior_guard._state_writes.
RE_STATE_WRITE = re.compile(r"([A-Za-z_$][\w$]*)(?:\s*\[[^\]]+\])*\s*(\+=|-=|\*=|/=|%=|=)(?!=)")

# "emit <Event>(", group(1) = tên event. Dùng ở: behavior_guard._emitted_events.
RE_EMIT_EVENT = re.compile(r"\bemit\s+([A-Za-z_$][\w$]*)\s*\(")

# Câu lệnh bắt đầu bằng tên viết hoa rồi "(" (event kiểu Solidity < 0.4.21, vd "Transfer(a, b, v);"),
# group(1) = tên. Dùng ở: behavior_guard._emitted_events.
RE_LEGACY_EVENT_STATEMENT = re.compile(r"(?:^|[;{}])\s*([A-Z][\w$]*)\s*\(")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 - VERIFIER (candidate_check.py, tool_matrix.py)
# ══════════════════════════════════════════════════════════════════════════════

# Số dòng trong vị trí lỗi solc, group(1) = dòng: "--> a.sol:12:5" (>= 0.5) và "a.sol:12:5: Error" (0.4).
# Dùng ở: candidate_check.units_with_compiler_errors.
RE_SOLC_ERROR_LINE = re.compile(r"\S+?\.sol:(\d+):\d+")

# Đường dẫn tuyệt đối tới file .sol (thư mục tạm, thay bằng "Contract.sol" trong feedback).
# Dùng ở: candidate_check._compiler_feedback.
RE_SOL_FILE_PATH = re.compile(r"(?:/[^\s:]+)+\.sol")

# Vị trí trong thông điệp của Slither (đường dẫn .sol#12-15, :12:5, #12, "line 12", "lines 3-4") - bỏ đi để
# so finding trước/sau vá mà không phụ thuộc số dòng. Dùng ở: tool_matrix._finding_signature.
RE_LOCATION_IN_MESSAGE = re.compile(r"(\S*\.sol(?:#\d+(?:-\d+)?|:\d+(?::\d+)?)?|#\d+(?:-\d+)?|\blines?\s+\d+(?:-\d+)?)")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 - REPORTER (report.py)
# ══════════════════════════════════════════════════════════════════════════════

# Đoạn mã ngay trước "{" là lời gọi kèm tuỳ chọn (".call{value: x}", "new Token{salt: s}"), không phải mở khối.
# Dùng ở: report (định dạng lại mã Solidity khi hiển thị).
RE_CALL_OPTIONS_PREFIX = re.compile(r"(?:\.(?:call|delegatecall|staticcall)|\bnew\s+[A-Za-z_]\w*)$")


# ══════════════════════════════════════════════════════════════════════════════
# HẠ TẦNG: common/gpu_client.py, common/solc.py, pipeline/orchestrator.py, webapp/app.py
# ══════════════════════════════════════════════════════════════════════════════

# Khối ```json ...``` (hoặc ``` không tên) trong câu trả lời của LLM, group(1) = nội dung.
# Dùng ở: gpu_client.extract_json.
RE_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Ký tự xuống dòng không có "\" đứng trước (xuống dòng thô trong chuỗi JSON -> đổi thành "\n").
# Dùng ở: gpu_client.extract_json.
RE_UNESCAPED_NEWLINE = re.compile(r"(?<!\\)\n")


# Giá trị chuỗi của khoá JSON: '"<key>": "<giá trị>"', group(1) = giá trị (còn ký tự thoát).
# Dùng ở: gpu_client.extract_json (vớt khoá từ JSON hỏng).
def re_json_string_value(key: str) -> re.Pattern[str]:
    return re.compile(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"')


# Giá trị số / true / false của khoá JSON, group(1) = giá trị. Dùng ở: gpu_client.extract_json.
def re_json_scalar_value(key: str) -> re.Pattern[str]:
    return re.compile(rf'"{re.escape(key)}"\s*:\s*([0-9.]+|true|false)', re.IGNORECASE)


# Vị trí lỗi trong output của solc: `--> file.sol:12:5:` (>= 0.5) hoặc `file.sol:12:5:` (0.4).
# Dùng ở: solc.compiler_error_signatures (bỏ vị trí để so lỗi gốc với lỗi của bản vá).
RE_SOLC_ERROR_LOCATION = re.compile(r"(-->\s*\S+:\d+:\d+:?|\S+\.sol:\d+:\d+:?)")

# Phiên bản trong tên file solc của solc-select, group(1) = "x.y.z". Vd "solc-0.4.24" -> "0.4.24".
# Dùng ở: solc.get_installed_solc_versions.
RE_SOLC_BINARY_VERSION = re.compile(r"solc-(\d+\.\d+\.\d+)$")

# Số minor sau "0.", group(1) = minor. Vd "^0.4.24" -> "4". Dùng ở: solc.select_best_solc.
RE_MINOR_VERSION = re.compile(r"0\.(\d+)")

# Ký tự không an toàn cho tên file/thư mục (khác chữ, số, "_", "-") -> thay bằng "-".
# Dùng ở: orchestrator.make_run_id, webapp safe_contract_file_name.
RE_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")

# run_id hợp lệ (dùng với .fullmatch): chữ/số đứng đầu, sau đó chữ/số/"_"/"."/"-", tối đa 128 ký tự.
# Dùng ở: orchestrator.validated_run_id, webapp run_dir_for (chống path traversal).
RE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
