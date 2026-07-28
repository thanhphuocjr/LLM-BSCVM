import json
import re
import os


def remove_comments_and_docstrings(source):
    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split("\n"):
        if x.strip() != "":
            temp.append(x)
    return "\n".join(temp)


def find_code_function_remove_space_comments(text):
    text = remove_comments_and_docstrings(text)
    matches = re.findall(r"```Solidiy\n(.*?)\n```", text, re.DOTALL)

    # Remove new lines in the found code
    cleaned_matches = [match.replace("\n", " ") for match in matches]

    # Replace the original code with the cleaned code in the text
    for original, cleaned in zip(matches, cleaned_matches):
        text = text.replace(original, cleaned)
    return text
