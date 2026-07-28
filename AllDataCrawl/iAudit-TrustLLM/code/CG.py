# %%
import json
import subprocess
import tqdm
import re
import re
from io import StringIO
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


def remove_inherited_contracts(solidity_code):
    # This regex matches 'is <InheritanceList>' patterns
    # We capture the contract name and the 'is' in separate groups
    pattern = r"(contract\s+\w+\s+)is\s+[^{]+"

    # Substitute matches in the Solidity code with just the first group (contract declaration)
    modified_code = re.sub(pattern, r"\1", solidity_code)

    return modified_code


def generate_call_graph(source_code):
    # The command you want to run
    command = os.environ["SURYA_BIN"]

    # If 'surya' requires additional arguments, add them to the list
    # For example: ["surya", "arg1", "arg2"]
    source_code = remove_comments_and_docstrings(source_code)
    source_code = remove_inherited_contracts(source_code)
    os.makedirs("./tmp", exist_ok=True)
    with open("./tmp/tmp_solodit.sol", "w") as f:
        f.write(source_code)
    args = [command, "graph", "--modifiers", "--libraries", "./tmp/tmp_solodit.sol"]

    # Run the command and capture the output
    # print(args)
    result = subprocess.run(args, capture_output=True, text=True)

    # The stdout attribute contains the output of the command
    output = result.stdout
    return output
