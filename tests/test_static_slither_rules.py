from __future__ import annotations

import unittest

from E2E.phase1a_static.static_analyzer import run_static_analysis
from E2E.phase1a_static.static_runner import run_static


# Tập loại lỗ hổng (vuln_type) mà run_static_analysis tìm thấy trong đoạn mã.
def finding_types(code: str) -> set[str]:
    return set(run_static_analysis(code)["findings"])


# Kiểm tra 25 kiểm tra ngữ nghĩa kiểu Slither trong static_analyzer: có lỗi thì báo, ca an toàn tương tự thì không.
class SlitherDerivedStaticRuleTests(unittest.TestCase):
    # Ký tự Unicode đảo chiều (U+202E) bị phát hiện kể cả nằm trong comment.
    def test_bidirectional_control_is_detected_even_inside_comment(self) -> None:
        code = "// hidden direction: \u202e\ncontract C {}"
        result = run_static_analysis(code)

        self.assertIn("right_to_left_override", result["findings"])
        finding = result["findings"]["right_to_left_override"]
        self.assertEqual(finding.vuln_id, "SWC-130")
        self.assertEqual(finding.matches[0].line, 1)

    # "^" chỉ bị báo khi đi với hằng số thập phân (không báo pragma ^0.x).
    def test_incorrect_exponentiation_requires_decimal_constant(self) -> None:
        vulnerable = """
        contract C {
            function square(uint256 value) external pure returns (uint256) {
                return value ^ 2;
            }
        }
        """
        xor_mask = """
        contract C {
            function xor(uint256 value, uint256 mask) external pure returns (uint256) {
                return value ^ mask;
            }
        }
        """

        self.assertIn(
            "incorrect_exponentiation_operator", finding_types(vulnerable)
        )
        self.assertNotIn(
            "incorrect_exponentiation_operator", finding_types(xor_mask)
        )

    # abi.encodePacked chỉ bị báo khi có 2 đối số động liền nhau.
    def test_encode_packed_requires_adjacent_dynamic_arguments(self) -> None:
        vulnerable = """
        contract C {
            function digest(string memory first, bytes memory second)
                external pure returns (bytes32)
            {
                return keccak256(abi.encodePacked(first, second));
            }
        }
        """
        separated = """
        contract C {
            function digest(string memory first, uint256 separator, bytes memory second)
                external pure returns (bytes32)
            {
                return keccak256(abi.encodePacked(first, separator, second));
            }
        }
        """

        self.assertIn("abi_encode_packed_collision", finding_types(vulnerable))
        self.assertNotIn("abi_encode_packed_collision", finding_types(separated))

    # transferFrom với "from" tuỳ ý và transfer bỏ qua kết quả bị báo; dạng an toàn thì không.
    def test_arbitrary_and_unchecked_erc20_transfer_from(self) -> None:
        vulnerable = """
        interface IERC20 {
            function transferFrom(address, address, uint256) external returns (bool);
        }
        contract Puller {
            IERC20 public token;
            function pull(address from, uint256 amount) external {
                token.transferFrom(from, msg.sender, amount);
            }
        }
        """
        checked = """
        interface IERC20 {
            function transferFrom(address, address, uint256) external returns (bool);
        }
        contract Puller {
            IERC20 public token;
            function pull(uint256 amount) external {
                require(token.transferFrom(msg.sender, address(this), amount));
            }
        }
        """

        vulnerable_types = finding_types(vulnerable)
        self.assertIn("arbitrary_erc20_transfer_from", vulnerable_types)
        self.assertIn("unchecked_erc20_transfer", vulnerable_types)
        checked_types = finding_types(checked)
        self.assertNotIn("arbitrary_erc20_transfer_from", checked_types)
        self.assertNotIn("unchecked_erc20_transfer", checked_types)

    # Locked ether: nhận ETH mà không có đường rút -> báo; có đường rút -> không.
    def test_ether_must_have_an_outgoing_path(self) -> None:
        locked = """
        contract Vault {
            function deposit() external payable {}
        }
        """
        withdrawable = """
        contract Vault {
            function deposit() external payable {}
            function withdraw(address payable recipient, uint256 amount) external {
                (bool ok,) = recipient.call{value: amount}("");
                require(ok);
            }
        }
        """

        self.assertIn("locked_ether", finding_types(locked))
        self.assertNotIn("locked_ether", finding_types(withdrawable))

    # msg.value / delegatecall trong vòng lặp của hàm payable bị báo.
    def test_msg_value_and_delegatecall_in_payable_loops(self) -> None:
        vulnerable = """
        contract Batch {
            function execute(address target, bytes[] calldata calls)
                external payable
            {
                for (uint256 i = 0; i < calls.length; i++) {
                    (bool ok,) = target.delegatecall(calls[i]);
                    require(ok);
                    uint256 share = msg.value / calls.length;
                }
            }
        }
        """
        controls = """
        contract Batch {
            function inspect(uint256 count) external payable {
                for (uint256 i = 0; i < count; i++) {
                    require(msg.value != 0);
                }
            }
            function execute(address target, bytes[] calldata calls) external {
                for (uint256 i = 0; i < calls.length; i++) {
                    target.delegatecall(calls[i]);
                }
            }
        }
        """

        vulnerable_types = finding_types(vulnerable)
        self.assertIn("msg_value_in_loop", vulnerable_types)
        self.assertIn("delegatecall_in_loop", vulnerable_types)
        control_types = finding_types(controls)
        self.assertNotIn("msg_value_in_loop", control_types)
        self.assertNotIn("delegatecall_in_loop", control_types)

    # Thiếu kiểm tra zero-address chỉ báo khi tham số thực sự được ghi vào state / dùng làm đích.
    def test_zero_address_check_is_usage_based(self) -> None:
        vulnerable = """
        contract Owned {
            address public owner;
            function setOwner(address newOwner) external {
                owner = newOwner;
            }
        }
        """
        checked = """
        contract Owned {
            address public owner;
            function setOwner(address newOwner) external {
                require(newOwner != address(0));
                owner = newOwner;
            }
            function balanceOf(address account) external view returns (uint256) {
                return account.balance;
            }
        }
        """

        self.assertIn(
            "missing_zero_address_validation", finding_types(vulnerable)
        )
        self.assertNotIn(
            "missing_zero_address_validation", finding_types(checked)
        )

        custom_error_check = """
        contract Owned {
            address public owner;
            error ZeroAddress();
            function setOwner(address newOwner) external {
                if (newOwner == address(0)) { revert ZeroAddress(); }
                owner = newOwner;
            }
        }
        """
        self.assertNotIn(
            "missing_zero_address_validation", finding_types(custom_error_check)
        )

    # Lỗi gõ "=+", so sánh với chính nó, cú pháp lỗi thời (throw, sha3, msg.gas...) bị báo.
    def test_logic_typo_and_deprecated_usage_rules(self) -> None:
        code = """
        pragma solidity 0.4.26;
        contract C {
            function update(uint256 value) public {
                value =+ 1;
                require(value == value);
                bytes32 digest = sha3(value);
            }
        }
        """

        types = finding_types(code)
        self.assertIn("incorrect_unary_expression", types)
        self.assertIn("tautological_comparison", types)
        self.assertIn("deprecated_solidity_usage", types)

        legitimate_negative = """
        contract C {
            function negate(int256 value) external pure returns (int256 result) {
                result = -value;
            }
        }
        """
        self.assertNotIn(
            "incorrect_unary_expression", finding_types(legitimate_negative)
        )

    # Chữ ký transfer chuẩn không bị báo nhầm là short-address.
    def test_standard_transfer_signature_is_not_short_address_finding(self) -> None:
        code = """
        pragma solidity 0.8.20;
        interface IERC20 {
            function transfer(address recipient, uint256 amount)
                external returns (bool);
        }
        """

        self.assertNotIn("short_address_attack", finding_types(code))

    # Che chuỗi/comment không làm lệch số dòng của finding.
    def test_semantic_match_keeps_original_line_after_string_masking(self) -> None:
        code = """contract C {
            string constant TEXT = "a long string that changes columns";
            function square(uint256 value) external pure returns (uint256) {
                return value ^ 2;
            }
        }"""
        result = run_static_analysis(code)
        match = result["findings"]["incorrect_exponentiation_operator"].matches[0]

        self.assertEqual(match.line, 4)

    # Luật ngữ nghĩa mới đi qua run_static thành finding confirmed.
    def test_new_rule_flows_through_static_runner_as_confirmed_finding(self) -> None:
        code = """
        contract C {
            function digest(string memory first, string memory second)
                external pure returns (bytes32)
            {
                return keccak256(abi.encodePacked(first, second));
            }
        }
        """
        result = run_static(code)

        self.assertEqual(result["status"], "ok")
        self.assertTrue(
            any(item["swc"] == "SWC-133" for item in result["findings"])
        )

    # shl/shr đảo tham số chỉ bị báo trong assembly.
    def test_incorrect_shift_only_inside_assembly_with_reversed_shape(self) -> None:
        vulnerable = """
        contract C {
            function shift(uint256 value) external pure returns (uint256 result) {
                assembly { result := shr(result, 8) }
            }
        }
        """
        correct = """
        contract C {
            function shift(uint256 value) external pure returns (uint256 result) {
                assembly { result := shr(8, value) }
            }
        }
        """

        self.assertIn("incorrect_shift_in_assembly", finding_types(vulnerable))
        self.assertNotIn("incorrect_shift_in_assembly", finding_types(correct))

    # Độ dài mảng chỉ bị báo khi là mảng state và giá trị từ caller.
    def test_controlled_array_length_requires_state_array_and_caller_input(self) -> None:
        vulnerable = """
        pragma solidity 0.5.16;
        contract C {
            uint256[] private values;
            function resize(uint256 newLength) external {
                values.length = newLength;
            }
        }
        """
        constant_resize = """
        pragma solidity 0.5.16;
        contract C {
            uint256[] private values;
            function reset() external {
                values.length = 7;
            }
        }
        """

        self.assertIn("controlled_array_length", finding_types(vulnerable))
        self.assertNotIn("controlled_array_length", finding_types(constant_resize))

    # Truyền mảng storage vào tham số memory bị báo; tham số storage thì không.
    def test_array_by_reference_distinguishes_memory_and_storage(self) -> None:
        vulnerable = """
        contract C {
            uint256[1] private values;
            function execute() external { mutate(values); }
            function mutate(uint256[1] memory input) internal {
                input[0] = 7;
            }
        }
        """
        correct = """
        contract C {
            uint256[1] private values;
            function execute() external { mutate(values); }
            function mutate(uint256[1] storage input) internal {
                input[0] = 7;
            }
        }
        """

        self.assertIn("array_by_reference", finding_types(vulnerable))
        self.assertNotIn("array_by_reference", finding_types(correct))

    # Chỉ báo khi có CẢ constructor() lẫn hàm trùng tên contract.
    def test_multiple_constructor_schemes_require_both_forms(self) -> None:
        vulnerable = """
        pragma solidity 0.4.22;
        contract Wallet {
            constructor() public {}
            function Wallet() public {}
        }
        """
        one_constructor = """
        pragma solidity 0.4.22;
        contract Wallet {
            constructor() public {}
        }
        """

        self.assertIn("multiple_constructor_schemes", finding_types(vulnerable))
        self.assertNotIn(
            "multiple_constructor_schemes", finding_types(one_constructor)
        )

    # Mapping public trỏ struct lồng chỉ bị báo với Solidity 0.4.
    def test_public_nested_mapping_is_limited_to_solidity_04(self) -> None:
        vulnerable = """
        pragma solidity 0.4.25;
        contract Registry {
            struct Inner { uint256 value; }
            struct Outer { Inner inner; }
            mapping(uint256 => Outer) public records;
        }
        """
        modern = """
        pragma solidity 0.8.20;
        contract Registry {
            struct Inner { uint256 value; }
            struct Outer { Inner inner; }
            mapping(uint256 => Outer) public records;
        }
        """

        self.assertIn(
            "public_mapping_with_nested_struct", finding_types(vulnerable)
        )
        self.assertNotIn(
            "public_mapping_with_nested_struct", finding_types(modern)
        )

    # delete struct chứa mapping (kể cả lồng nhiều lớp) bị báo.
    def test_delete_struct_with_mapping_is_detected_recursively(self) -> None:
        vulnerable = """
        contract Registry {
            struct Balances { mapping(address => uint256) value; }
            struct Account { Balances balances; address owner; }
            mapping(uint256 => Account) private accounts;
            function remove(uint256 id) external { delete accounts[id]; }
        }
        """
        simple_struct = """
        contract Registry {
            struct Account { uint256 balance; address owner; }
            mapping(uint256 => Account) private accounts;
            function remove(uint256 id) external { delete accounts[id]; }
        }
        """

        self.assertIn("mapping_deletion_in_struct", finding_types(vulnerable))
        self.assertNotIn("mapping_deletion_in_struct", finding_types(simple_struct))

    # So sánh luôn đúng/sai theo miền giá trị kiểu số bị báo.
    def test_type_based_tautology_uses_integer_range(self) -> None:
        vulnerable = """
        contract C {
            function check(uint256 amount, uint8 small) external pure returns (bool) {
                return amount >= 0 && small < 512;
            }
        }
        """
        meaningful = """
        contract C {
            function check(uint8 amount) external pure returns (bool) {
                return amount < 200;
            }
        }
        """

        self.assertIn("type_based_tautology", finding_types(vulnerable))
        self.assertNotIn("type_based_tautology", finding_types(meaningful))

        shadowed_names = """
        contract C {
            function wide(uint256 amount) external pure returns (bool) {
                return amount < 512;
            }
            function narrow(uint8 amount) external pure returns (bool) {
                return amount < 200;
            }
        }
        """
        self.assertNotIn("type_based_tautology", finding_types(shadowed_names))

    # Dùng true/false sai trong điều kiện bị báo; return true hợp lệ thì không.
    def test_boolean_constant_misuse_keeps_legitimate_bare_constants(self) -> None:
        vulnerable = """
        contract C {
            function check(bool enabled) external pure returns (bool) {
                if (false) { return enabled; }
                return enabled || true;
            }
        }
        """
        legitimate = """
        contract C {
            function run(bool enabled) external pure returns (bool) {
                while (true) { return enabled == true; }
                return true;
            }
        }
        """

        self.assertIn("boolean_constant_misuse", finding_types(vulnerable))
        self.assertNotIn("boolean_constant_misuse", finding_types(legitimate))

    # So sánh == với balance bị báo; so sánh >= (ngưỡng) thì không.
    def test_balance_equality_flags_exact_target_not_threshold(self) -> None:
        vulnerable = """
        contract Vault {
            function reached() external view returns (bool) {
                uint256 current = address(this).balance;
                return current == 10 ether;
            }
        }
        """
        threshold = """
        contract Vault {
            function reached() external view returns (bool) {
                return address(this).balance >= 10 ether;
            }
        }
        """

        self.assertIn("dangerous_balance_equality", finding_types(vulnerable))
        self.assertNotIn("dangerous_balance_equality", finding_types(threshold))

    # Ép kiểu enum chỉ bị báo với solc bị lỗi (<= 0.4.4).
    def test_enum_conversion_is_limited_to_vulnerable_exact_compiler(self) -> None:
        vulnerable = """
        pragma solidity 0.4.2;
        contract C {
            enum Status { Pending, Complete }
            function convert(uint256 value) public returns (Status) {
                return Status(value);
            }
        }
        """
        fixed_compiler = vulnerable.replace("0.4.2", "0.4.5")

        self.assertIn("dangerous_enum_conversion", finding_types(vulnerable))
        self.assertNotIn("dangerous_enum_conversion", finding_types(fixed_compiler))

    # Hàm internal 2 giá trị trả về dùng Yul return() bị báo (nên dùng leave).
    def test_internal_two_value_assembly_return_should_use_leave(self) -> None:
        vulnerable = """
        contract C {
            function pair() internal pure returns (uint256 first, uint256 second) {
                assembly { return(5, 6) }
            }
        }
        """
        external_function = vulnerable.replace("internal", "external")

        self.assertIn("return_instead_of_leave", finding_types(vulnerable))
        self.assertNotIn(
            "return_instead_of_leave", finding_types(external_function)
        )

    # Gọi hàm có Yul return() mà phía sau còn code -> bị báo.
    def test_assembly_return_with_following_caller_code_is_detected(self) -> None:
        vulnerable = """
        contract C {
            function halt() internal pure { assembly { return(0, 0) } }
            function execute() external pure returns (bool) {
                halt();
                return true;
            }
        }
        """
        terminal_call = """
        contract C {
            function halt() internal pure { assembly { return(0, 0) } }
            function execute() external pure { halt(); }
        }
        """

        self.assertIn("incorrect_assembly_return", finding_types(vulnerable))
        self.assertNotIn("incorrect_assembly_return", finding_types(terminal_call))

    # Bug ABIEncoderV2 chỉ báo khi pin đúng solc bị lỗi.
    def test_abiencoderv2_storage_array_bug_requires_pinned_bad_compiler(self) -> None:
        vulnerable = """
        pragma solidity 0.5.9;
        pragma experimental ABIEncoderV2;
        contract C {
            uint256[2][2] private matrix;
            function encode() external view returns (bytes memory) {
                return abi.encode(matrix);
            }
        }
        """
        fixed_compiler = vulnerable.replace("0.5.9", "0.5.10")

        self.assertIn("storage_abiencoderv2_array", finding_types(vulnerable))
        self.assertNotIn(
            "storage_abiencoderv2_array", finding_types(fixed_compiler)
        )

    # Bug mảng int storage chỉ báo khi gán cả mảng, không báo khi gán từng phần tử.
    def test_signed_storage_array_bug_ignores_element_assignment(self) -> None:
        vulnerable = """
        pragma solidity 0.5.9;
        contract C {
            int256[3] private values;
            function reset() external { values = [-1, -2, -3]; }
        }
        """
        element_only = """
        pragma solidity 0.5.9;
        contract C {
            int256[3] private values;
            function reset() external { values[0] = -1; }
        }
        """
        fixed_compiler = vulnerable.replace("0.5.9", "0.5.10")

        self.assertIn("signed_storage_array_compiler_bug", finding_types(vulnerable))
        self.assertNotIn(
            "signed_storage_array_compiler_bug", finding_types(element_only)
        )
        self.assertNotIn(
            "signed_storage_array_compiler_bug", finding_types(fixed_compiler)
        )

    # Che biến state chỉ báo khi contract cha (đã biết) có biến cùng tên.
    def test_state_shadowing_requires_same_name_in_known_base(self) -> None:
        vulnerable = """
        contract Base { address internal owner; }
        contract Child is Base { address internal owner; }
        """
        distinct = """
        contract Base { address internal owner; }
        contract Child is Base { address internal childOwner; }
        """
        upgrade_gap = """
        contract Base { uint256[50] private __gap; }
        contract Child is Base { uint256[50] private __gap; }
        """

        self.assertIn("state_variable_shadowing", finding_types(vulnerable))
        self.assertNotIn("state_variable_shadowing", finding_types(distinct))
        self.assertNotIn("state_variable_shadowing", finding_types(upgrade_gap))

    # Biến address chưa khởi tạo chỉ báo khi được dùng làm đích chuyển ETH/call.
    def test_uninitialized_state_address_requires_dangerous_read(self) -> None:
        vulnerable = """
        contract Payout {
            address payable private destination;
            function pay() external payable { destination.transfer(msg.value); }
        }
        """
        initialized = """
        contract Payout {
            address payable private destination;
            constructor(address payable target) { destination = target; }
            function pay() external payable { destination.transfer(msg.value); }
        }
        """
        harmless_default = """
        contract Registry {
            address private optionalAddress;
            function empty() external view returns (bool) {
                return optionalAddress == address(0);
            }
        }
        """

        self.assertIn("uninitialized_state_address", finding_types(vulnerable))
        self.assertNotIn("uninitialized_state_address", finding_types(initialized))
        self.assertNotIn(
            "uninitialized_state_address", finding_types(harmless_default)
        )


if __name__ == "__main__":
    unittest.main()
