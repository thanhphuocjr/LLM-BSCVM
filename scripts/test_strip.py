import re
import unittest

def strip_solidity_comments(source: str) -> str:
    """
    Safely remove all Solidity comments (single-line //, multi-line /* */, NatSpec)
    while strictly preserving string literals, assembly blocks, and syntax tokens.
    Cleans up trailing whitespaces and redundant consecutive blank lines.
    """
    result = []
    i = 0
    n = len(source)

    while i < n:
        char = source[i]

        # Double-quoted string literal
        if char == '"':
            start = i
            i += 1
            while i < n:
                if source[i] == '\\':
                    i += 2
                    continue
                if source[i] == '"':
                    i += 1
                    break
                i += 1
            result.append(source[start:i])

        # Single-quoted string literal
        elif char == "'":
            start = i
            i += 1
            while i < n:
                if source[i] == '\\':
                    i += 2
                    continue
                if source[i] == "'":
                    i += 1
                    break
                i += 1
            result.append(source[start:i])

        # Single-line comment: // ...
        elif char == '/' and i + 1 < n and source[i + 1] == '/':
            i += 2
            while i < n and source[i] != '\n':
                i += 1
            # We do NOT consume the newline here, it stays in the stream

        # Multi-line comment: /* ... */
        elif char == '/' and i + 1 < n and source[i + 1] == '*':
            i += 2
            has_newline = False
            while i + 1 < n and not (source[i] == '*' and source[i + 1] == '/'):
                if source[i] == '\n':
                    has_newline = True
                i += 1
            if i + 1 < n:
                i += 2  # skip '*/'
            else:
                i = n   # unclosed comment at EOF

            if has_newline:
                result.append('\n')
            else:
                # Only insert a space if neither the preceding character nor the next character is whitespace
                prev_is_space = bool(result and result[-1].isspace())
                next_is_space = bool(i < n and source[i].isspace())
                if not prev_is_space and not next_is_space:
                    result.append(' ')

        else:
            result.append(char)
            i += 1

    raw = ''.join(result)

    # Post-processing: clean whitespace and blank lines
    lines = [line.rstrip() for line in raw.split('\n')]
    text = '\n'.join(lines)

    # Collapse 3 or more consecutive newlines into 2 (i.e. at most one blank line between elements)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing blank lines, ensure exactly one trailing newline if not empty
    text = text.strip()
    return (text + '\n') if text else ''


class TestStripComments(unittest.TestCase):
    def test_url_in_double_quote(self):
        code = 'string a = "http://test.com//hello/*world*/";'
        self.assertEqual(strip_solidity_comments(code), code + '\n')

    def test_escaped_quote_in_string(self):
        code = r'string b = "He said \"// not a comment\"";'
        self.assertEqual(strip_solidity_comments(code), code + '\n')

    def test_single_quoted_string_with_comment_markers(self):
        code = r"string c = 'It\'s a string with // inside';"
        self.assertEqual(strip_solidity_comments(code), code + '\n')

    def test_single_line_comment_with_quotes(self):
        code = '// comment with "quotes" and \'single\' and /* nested */\ncontract A {}'
        self.assertEqual(strip_solidity_comments(code), 'contract A {}\n')

    def test_multi_line_comment_with_quotes(self):
        code = '/* multi \n "quotes" \n // nested */\ncontract B {}'
        self.assertEqual(strip_solidity_comments(code), 'contract B {}\n')

    def test_division_operator(self):
        code = 'a = b / /* divide */c;'
        self.assertEqual(strip_solidity_comments(code), 'a = b / c;\n')

    def test_inline_block_comment_preserves_token_separation(self):
        code = 'uint/*foo*/a;'
        self.assertEqual(strip_solidity_comments(code), 'uint a;\n')

    def test_trailing_comment(self):
        code = 'uint a; // comment'
        self.assertEqual(strip_solidity_comments(code), 'uint a;\n')

    def test_natspec_docstrings(self):
        code = '''/// @notice Transfer tokens
/// @param to Recipient
function transfer(address to) public {}'''
        self.assertEqual(strip_solidity_comments(code), 'function transfer(address to) public {}\n')

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
