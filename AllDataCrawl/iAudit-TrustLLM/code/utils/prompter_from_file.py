from .normalization import remove_comments_and_docstrings
import os
from dataclasses import dataclass, field


def split_file_by_empty_line(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        parts = content.split("\n\n")
        return parts


@dataclass
class PrompterConfig:
    header: str = ""
    instruction: str = ""
    input: str = ""
    caller: str = ""
    callee: str = ""
    tailer: str = ""


class PrompterClass:
    contract_line_template = """
The below code from the contract {contract} 
```Solidiy
{context}
```
"""

    contract_line_code_template = """
The code piece from the contract {contract} 
```Solidity
{lines_code}
```
"""

    def __init__(self, single_prompt=True, prompt_folder="./prompts"):
        self.prompt_folder = prompt_folder
        self.PROMPT_TEMPLATE_class_without_call = []
        self.PROMPT_TEMPLATE_with_calling = []
        for fname in os.listdir(os.path.join(self.prompt_folder, "class")):
            if "0" not in fname and single_prompt:
                continue
            prompt_tuple = split_file_by_empty_line(
                os.path.join(self.prompt_folder, "class", fname)
            )
            self.PROMPT_TEMPLATE_class_without_call.append(
                PrompterConfig(
                    header=prompt_tuple[0],
                    instruction=prompt_tuple[1],
                    input=prompt_tuple[2],
                    tailer=prompt_tuple[3],
                )
            )

        if os.path.isdir(os.path.join(self.prompt_folder, "call_class")):
            for fname in os.listdir(os.path.join(self.prompt_folder, "call_class")):
                if "0" not in fname and single_prompt:
                    continue
                prompt_tuple = split_file_by_empty_line(
                    os.path.join(self.prompt_folder, "call_class", fname)
                )
                self.PROMPT_TEMPLATE_with_calling.append(
                    PrompterConfig(
                        header=prompt_tuple[0],
                        instruction=prompt_tuple[1],
                        input=prompt_tuple[2],
                        tailer=prompt_tuple[3],
                    )
                )

    def calls_seq(self, call_list):
        codes = []
        for c in call_list:
            if c["code"] is not None and c["code"] != "":
                codes.append(self.normazlie(c["code"].strip()))
            else:
                codes.append(c["contract"] + "." + c["fn_name"])
        return "\n".join(codes)

    def normazlie(self, code):
        code = remove_comments_and_docstrings(code)
        code = code.replace("\n", " ")
        code = " ".join(code.split())
        return code

    def generate_prompt_forFunction(self, in_calls, out_calls):
        in_call_seq = ""
        out_call_seq = ""
        if len(in_calls) > 0:
            in_call_seq = self.calls_seq(in_calls).strip()
        if len(out_calls) > 0:
            out_call_seq = self.calls_seq(out_calls).strip()
            # context = self.normazlie(context).strip()
        return in_call_seq, out_call_seq

    def generate_prompt_forLineCode(self, contract, context):
        context = self.normazlie(context).strip()
        fn_under_input = self.contract_line_template.format(
            contract=contract, context=context
        ).strip()
        return fn_under_input

    def generate_inputClassWithOutCall(self, code_under_checking):
        input4model = []
        for p in self.PROMPT_TEMPLATE_class_without_call:
            inputs_list = []
            for egg in code_under_checking:
                # print(f"Egg {len(egg)}")
                if len(egg) == 3:
                    (contract, function, context) = egg
                    context = self.normazlie(context).strip()
                    input = p.input.format(
                        fn_name=function, contract=contract, context=context
                    ).strip()
                    inputs_list.append(input)
                elif len(egg) == 2:
                    (contract, context) = egg
                    context = self.normazlie(context)
                    code_lines = self.contract_line_template.format(
                        contract=contract, context=context
                    ).strip()
                    inputs_list.append(code_lines)
                else:
                    assert False, f"Invalid input: {egg}"

            input_code = "\n".join(inputs_list).strip()
            t = [p.header, p.instruction, input_code, p.tailer]
            t = [x for x in t if x != ""]
            input4model.append("\n".join(t).strip())
        return input4model

    def generate_inputWithCall(self, code_under_checking, label):
        input4model = []
        for p in self.PROMPT_TEMPLATE_with_calling:
            inputs_list = []
            callee_list = []
            caller_list = []
            for egg in code_under_checking:
                if len(egg) == 5:
                    (contract, function, context, callee_seq, caller_seq) = egg
                    context = self.normazlie(context).strip()
                    input = p.input.format(
                        fn_name=function, contract=contract, context=context
                    ).strip()
                    inputs_list.append(input)
                    if len(caller_seq) > 0:
                        caller_list.append(
                            p.caller.format(fn_name=function, out_calls=caller_seq)
                        )
                    if len(callee_seq) > 0:
                        callee_list.append(
                            p.callee.format(fn_name=function, in_calls=callee_seq)
                        )
                if len(egg) == 2:
                    (contract, context) = egg
                    context = self.normazlie(context)
                    code_lines = self.contract_line_template.format(
                        contract=contract, context=context
                    ).strip()
                    inputs_list.append(code_lines)

            input_code = "\n".join(inputs_list).strip()
            callee_code = "\n".join(callee_list).strip() if len(callee_list) > 0 else ""
            caller_code = "\n".join(caller_list).strip() if len(caller_list) > 0 else ""
            tailer_code = p.tailer.format(label_name=label)
            t = [
                p.header,
                p.instruction,
                input_code,
                callee_code,
                caller_code,
                tailer_code,
            ]
            t = [x for x in t if x != ""]
            input4model.append("\n".join(t).strip())
        return input4model

    def generate_raw_input_without_prompt(
        self, contract, fn_name, context, in_calls, out_calls, using_calling
    ):
        if using_calling:
            in_call_seq = ""
            out_call_seq = ""
            if len(in_calls) > 0:
                in_call_seq = self.calls_seq(in_calls)
            if len(out_calls) > 0:
                out_call_seq = self.calls_seq(out_calls)
            fn_under_input = self.normazlie(context).strip()
            return fn_under_input, in_call_seq, out_call_seq
        else:
            fn_under_input = self.normazlie(context).strip()
            # fn_under_input = self.contract_fn_template.format(fn_name=fn_name, contract=contract, context=context).strip()
            return fn_under_input, "", ""


class PrompterReason:
    contract_line_template = """
The below code from the contract {contract} 
```Solidiy
{context}
```
"""

    contract_line_code_template = """
The code piece from the contract {contract} 
```Solidity
{lines_code}
```
"""

    def __init__(self, single_prompt=True, prompt_folder="./prompts_reasoning"):
        self.prompt_folder = prompt_folder
        self.PROMPT_TEMPLATE_Reason_with_calling = []
        self.PROMPT_TEMPLATE_Reason_without_calling = []
        for fname in os.listdir(os.path.join(self.prompt_folder, "call")):
            if "0" not in fname and single_prompt:
                continue
            prompt_tuple = split_file_by_empty_line(
                os.path.join(self.prompt_folder, "call", fname)
            )
            self.PROMPT_TEMPLATE_Reason_with_calling.append(
                PrompterConfig(
                    header=prompt_tuple[0],
                    instruction=prompt_tuple[1],
                    input=prompt_tuple[2],
                    caller=prompt_tuple[3],
                    callee=prompt_tuple[4],
                    tailer=prompt_tuple[5],
                )
            )

        for fname in os.listdir(os.path.join(self.prompt_folder, "no_call")):
            if "0" not in fname and single_prompt:
                continue
            prompt_tuple = split_file_by_empty_line(
                os.path.join(self.prompt_folder, "no_call", fname)
            )
            self.PROMPT_TEMPLATE_Reason_without_calling.append(
                PrompterConfig(
                    header=prompt_tuple[0],
                    instruction=prompt_tuple[1],
                    input=prompt_tuple[2],
                    tailer=prompt_tuple[3],
                )
            )

    def calls_seq(self, call_list):
        codes = []
        for c in call_list:
            if c["code"] is not None and c["code"] != "":
                codes.append(self.normazlie(c["code"].strip()))
            else:
                codes.append(c["contract"] + "." + c["fn_name"])
        return "\n".join(codes)

    def normazlie(self, code):
        code = remove_comments_and_docstrings(code)
        code = code.replace("\n", " ")
        code = " ".join(code.split())
        return code

    def generate_prompt_forFunction(self, in_calls, out_calls):
        in_call_seq = ""
        out_call_seq = ""
        if len(in_calls) > 0:
            in_call_seq = self.calls_seq(in_calls).strip()
        if len(out_calls) > 0:
            out_call_seq = self.calls_seq(out_calls).strip()
            # context = self.normazlie(context).strip()
        return in_call_seq, out_call_seq

    def generate_prompt_forLineCode(self, contract, context):
        context = self.normazlie(context).strip()
        fn_under_input = self.contract_line_template.format(
            contract=contract, context=context
        ).strip()
        return fn_under_input

    def generate_inputWithCallReasoning(self, code_under_checking, label):
        input4model = []
        # print(f"Code under checking: {code_under_checking}")
        for p in self.PROMPT_TEMPLATE_Reason_with_calling:
            inputs_list = []
            callee_list = []
            caller_list = []
            for egg in code_under_checking:
                # print(f"Egg {len(egg)}")
                if len(egg) == 5:
                    (contract, function, context, callee_seq, caller_seq) = egg
                    context = self.normazlie(context).strip()
                    input = p.input.format(
                        fn_name=function, contract=contract, context=context
                    ).strip()
                    inputs_list.append(input)
                    if len(caller_seq) > 0:
                        caller_list.append(
                            p.caller.format(fn_name=function, out_calls=caller_seq)
                        )
                    if len(callee_seq) > 0:
                        callee_list.append(
                            p.callee.format(fn_name=function, in_calls=callee_seq)
                        )
                elif len(egg) == 2:
                    (contract, context) = egg
                    context = self.normazlie(context)
                    code_lines = self.contract_line_template.format(
                        contract=contract, context=context
                    ).strip()
                    inputs_list.append(code_lines)
                else:
                    assert False, f"Invalid input: {egg}"

            input_code = "\n".join(inputs_list).strip()
            callee_code = "\n".join(callee_list).strip() if len(callee_list) > 0 else ""
            caller_code = "\n".join(caller_list).strip() if len(caller_list) > 0 else ""
            tailer_code = p.tailer.format(label_name=label)
            t = [
                p.header,
                p.instruction,
                input_code,
                callee_code,
                caller_code,
                tailer_code,
            ]
            t = [x for x in t if x != ""]
            input4model.append("\n".join(t).strip())
        return input4model

    def generate_inputWithReasoningOutCall(self, code_under_checking, label):
        input4model = []
        for p in self.PROMPT_TEMPLATE_Reason_without_calling:
            inputs_list = []
            for egg in code_under_checking:
                print(f"Egg {len(egg)}")
                if len(egg) == 5:
                    (contract, function, context, callee_seq, caller_seq) = egg
                    context = self.normazlie(context).strip()
                    input = p.input.format(
                        fn_name=function, contract=contract, context=context
                    ).strip()
                    inputs_list.append(input)
                elif len(egg) == 2:
                    (contract, context) = egg
                    context = self.normazlie(context)
                    code_lines = self.contract_line_template.format(
                        contract=contract, context=context
                    ).strip()
                    inputs_list.append(code_lines)
                else:
                    assert False, print(f"Invalid input: {egg}")

            input_code = "\n".join(inputs_list).strip()
            tailer_code = p.tailer.format(label_name=label)
            t = [p.header, p.instruction, input_code, tailer_code]
            t = [x for x in t if x != ""]
            input4model.append("\n".join(t).strip())
        return input4model

    def generate_raw_input_without_prompt(
        self, contract, fn_name, context, in_calls, out_calls, using_calling
    ):
        if using_calling:
            in_call_seq = ""
            out_call_seq = ""
            if len(in_calls) > 0:
                in_call_seq = self.calls_seq(in_calls)
            if len(out_calls) > 0:
                out_call_seq = self.calls_seq(out_calls)
            fn_under_input = self.normazlie(context).strip()
            return fn_under_input, in_call_seq, out_call_seq
        else:
            fn_under_input = self.normazlie(context).strip()
            # fn_under_input = self.contract_fn_template.format(fn_name=fn_name, contract=contract, context=context).strip()
            return fn_under_input, "", ""
