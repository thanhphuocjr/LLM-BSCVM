from antlr4 import CommonTokenStream, InputStream
from .SolidityLexer import SolidityLexer
from .SolidityParser import parseString
import os
import collections


def parse(source: str):
    lexer = SolidityLexer(InputStream(source))
    stream = CommonTokenStream(lexer)
    parser = SolidityParser(stream)
    return parser.sourceUnit()


def get_tokens(source: str):
    lexer = SolidityLexer(InputStream(source))
    return lexer.getAllTokens()


class ParseStream:
    def __init__(self, source_code) -> None:
        self.file_contract_function_map = {}
        self.source_code = source_code
        self.res = self.__parse_source_code(source_code)

    def __parse_source_code(self, source_code):
        # with open(file) as f:
        file = "0"
        res = parseString(source_code)
        if file not in self.file_contract_function_map:
            self.file_contract_function_map[file] = {}
        for contract_data in res["subcontracts"]:
            if contract_data["name"] not in self.file_contract_function_map[file]:
                self.file_contract_function_map[file][contract_data["name"]] = {}
            for function_data in contract_data["functions"]:
                if (
                    function_data["name"]
                    not in self.file_contract_function_map[file][contract_data["name"]]
                ):
                    self.file_contract_function_map[file][contract_data["name"]][
                        function_data["name"]
                    ] = {}
                self.file_contract_function_map[file][contract_data["name"]][
                    function_data["name"]
                ] = {
                    "start": int(function_data["loc"]["start"].split(":")[0]) - 1,
                    "end": int(function_data["loc"]["end"].split(":")[0]),
                }
        return res

    def get_all_functions(self):
        file = "0"
        items = collections.defaultdict(list)
        for contract in self.file_contract_function_map[file]:
            for function in self.file_contract_function_map[file][contract]:
                items[contract].append(
                    (
                        contract,
                        function,
                        self.get_function_lines(contract, function),
                        self.get_function_context(contract, function),
                    )
                )
        return items

    def get_function_lines(self, contract, function):
        file = "0"
        if file not in self.file_contract_function_map:
            return None
        if contract not in self.file_contract_function_map[file]:
            return None
        if function not in self.file_contract_function_map[file][contract]:
            return None
        data = self.file_contract_function_map[file][contract][function]
        return {"start": data["start"], "end": data["end"]}

    def get_function_context(self, contract, function):
        file = "0"
        if file not in self.file_contract_function_map:
            return None
        if contract not in self.file_contract_function_map[file]:
            return None
        if function not in self.file_contract_function_map[file][contract]:
            return None
        data = self.file_contract_function_map[file][contract][function]
        return "\n".join(self.source_code.splitlines()[data["start"] : data["end"]])

    def get_all_function_except_line_number(self, line_number_list):
        result = []
        for file in self.file_contract_function_map:
            for contract in self.file_contract_function_map[file]:
                for function in self.file_contract_function_map[file][contract]:
                    data = self.file_contract_function_map[file][contract][function]
                    for line_number in line_number_list:
                        if data["start"] <= line_number <= data["end"]:
                            continue
                    context = self.get_function_context(contract, function)
                    result.append((contract, function, context))
        return result

    def get_function_from_loc(self, line_number):
        for file in self.file_contract_function_map:
            for contract in self.file_contract_function_map[file]:
                for function in self.file_contract_function_map[file][contract]:
                    data = self.file_contract_function_map[file][contract][function]
                    if data["start"] <= line_number <= data["end"]:
                        context = self.get_function_context(contract, function)
                        return contract, function, context

    def get_locs_code(self, line_number):
        contract = None
        for contract_data in self.res["subcontracts"]:
            start = int(contract_data["loc"]["start"].split(":")[0])
            end = int(contract_data["loc"]["end"].split(":")[0])
            if line_number >= start and line_number <= end:
                contract = contract_data["name"]
        l = len(self.source_code.splitlines())
        return contract, None, self.source_code.splitlines()[min(line_number, l - 1)]


class ParsedDir:
    def __init__(self, path) -> None:
        self.file_contract_function_map = {}
        self.__parse_all(path)

    def __parse_file(self, file):
        with open(file) as f:
            res = parseString(f.read())
            if file not in self.file_contract_function_map:
                self.file_contract_function_map[file] = {}
            for contract_data in res["subcontracts"]:
                if contract_data["name"] not in self.file_contract_function_map[file]:
                    self.file_contract_function_map[file][contract_data["name"]] = {}
                for function_data in contract_data["functions"]:
                    if (
                        function_data["name"]
                        not in self.file_contract_function_map[file][
                            contract_data["name"]
                        ]
                    ):
                        self.file_contract_function_map[file][contract_data["name"]][
                            function_data["name"]
                        ] = {}
                    self.file_contract_function_map[file][contract_data["name"]][
                        function_data["name"]
                    ] = {
                        "start": int(function_data["loc"]["start"].split(":")[0]) - 1,
                        "end": int(function_data["loc"]["end"].split(":")[0]),
                    }

    def __parse_all(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                self.__parse_file(os.path.join(root, file))

    def get_function_context(self, file, contract, function):
        if file not in self.file_contract_function_map:
            return None
        if contract not in self.file_contract_function_map[file]:
            return None
        if function not in self.file_contract_function_map[file][contract]:
            return None
        data = self.file_contract_function_map[file][contract][function]
        return "\n".join(open(file).read().splitlines()[data["start"] : data["end"]])

    def get_function_from_loc(self, line_number):
        for file in self.file_contract_function_map:
            for contract in self.file_contract_function_map[file]:
                for function in self.file_contract_function_map[file][contract]:
                    data = self.file_contract_function_map[file][contract][function]
                    if data["start"] <= line_number <= data["end"]:
                        context = self.get_function_context(file, contract, function)
                        return file, contract, function, context
