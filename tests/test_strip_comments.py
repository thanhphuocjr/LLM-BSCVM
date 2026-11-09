import unittest

from E2E.data_tools.strip_comments import strip_solidity_comments


# Kiểm tra data_tools/strip_comments.py: bỏ comment nhưng không phá chuỗi/cú pháp.
class TestStripComments(unittest.TestCase):
    # URL "http://..." trong chuỗi không bị coi là comment "//".
    def test_url_in_double_quote(self):
        code = 'string a = "http://test.com//hello/*world*/";'
        self.assertEqual(strip_solidity_comments(code), code + '\n')

    # Dấu \" trong chuỗi không làm kết thúc chuỗi sớm.
    def test_escaped_quote_in_string(self):
        code = r'string b = "He said \"// not a comment\"";'
        self.assertEqual(strip_solidity_comments(code), code + '\n')

    # Chuỗi '...' chứa // hoặc /* được giữ nguyên.
    def test_single_quoted_string_with_comment_markers(self):
        code = r"string c = 'It\'s a string with // inside';"
        self.assertEqual(strip_solidity_comments(code), code + '\n')

    # Comment // có chứa dấu nháy vẫn bị xoá đúng.
    def test_single_line_comment_with_quotes(self):
        code = '// comment with "quotes" and \'single\' and /* nested */\ncontract A {}'
        self.assertEqual(strip_solidity_comments(code), 'contract A {}\n')

    # Comment /* */ nhiều dòng có dấu nháy vẫn bị xoá đúng.
    def test_multi_line_comment_with_quotes(self):
        code = '/* multi \n "quotes" \n // nested */\ncontract B {}'
        self.assertEqual(strip_solidity_comments(code), 'contract B {}\n')

    # Phép chia "a / b" không bị nhầm là comment.
    def test_division_operator(self):
        code = 'a = b / /* divide */c;'
        self.assertEqual(strip_solidity_comments(code), 'a = b / c;\n')

    # Xoá /* */ giữa hai token không làm dính hai token vào nhau.
    def test_inline_block_comment_preserves_token_separation(self):
        code = 'uint/*foo*/a;'
        self.assertEqual(strip_solidity_comments(code), 'uint a;\n')

    # Comment cuối dòng bị xoá, mã phía trước giữ nguyên.
    def test_trailing_comment(self):
        code = 'uint a; // comment'
        self.assertEqual(strip_solidity_comments(code), 'uint a;\n')

    # Comment NatSpec /// và /** */ bị xoá.
    def test_natspec_docstrings(self):
        code = '''/// @notice Transfer tokens
/// @param to Recipient
function transfer(address to) public {}'''
        self.assertEqual(strip_solidity_comments(code), 'function transfer(address to) public {}\n')

    # Phần đầu file sau khi bỏ comment được dọn sạch (không thừa dòng trống).
    def test_header_cleanup(self):
        code = '''/*
 * Header
 */

// Second header

pragma solidity ^0.4.24;



contract C {}'''
        expected = '''pragma solidity ^0.4.24;

contract C {}\n'''
        self.assertEqual(strip_solidity_comments(code), expected)


if __name__ == '__main__':
    unittest.main()
