from SolidityParser import ParseStream
import os
import collections
import glob
from CG import generate_call_graph
from parse_graph import get_calling_relationships, parse_calling_relationship
from .prompter_from_file import PrompterReason, PrompterClass, PrompterConfig
import hashlib


def get_hash(string: str):
    return hashlib.sha256(string.encode("utf-8")).hexdigest()


def generate_reasoning_prompt_data(eval_dataset, args):
    prompter = PrompterReason(
        single_prompt=args.single_prompt, prompt_folder="./prompts_reasoning"
    )
    for audit_file in eval_dataset:
        for hash_id in eval_dataset[audit_file]:
            prompts_item = eval_dataset[audit_file][hash_id]
            meta_info = prompts_item["meta"]
            contract, function, locations, context = (
                meta_info["contract"],
                meta_info["fn_name"],
                meta_info["locations"],
                meta_info["context"],
            )
            in_call_sequence, out_call_sequence = prompter.generate_prompt_forFunction(
                meta_info["in_calls"], meta_info["out_calls"]
            )
            prompts_item["meta"]["in_call_sequence"] = in_call_sequence
            prompts_item["meta"]["out_call_sequence"] = out_call_sequence
            pre_label = prompts_item["pre_label"]
            check_fn_list = [
                (contract, function, context, in_call_sequence, out_call_sequence)
            ]
            promt_with_calling = prompter.generate_inputWithCallReasoning(
                check_fn_list, pre_label
            )
            promt_without_calling = prompter.generate_inputWithReasoningOutCall(
                check_fn_list, pre_label
            )
            prompts_item["reasoning_prompt_withcall"] = promt_with_calling
            prompts_item["reasoning_prompt_withoutcall"] = promt_without_calling
    return eval_dataset


def parse_class_dir(code_dir, scope__file, single_prompt):
    prompter = PrompterClass(
        single_prompt=single_prompt, prompt_folder="./prompts_class"
    )
    results = collections.defaultdict(list)
    scope_list = []
    scope_files = []
    with open(scope__file) as f:
        lines = f.readlines()
        for l in lines:
            info = l.split()
            scope_list.append((info[0], info[1])) if len(info) == 2 else info[0]
            scope_files.append(info[0])

    files = []
    for f in glob.glob(f"{code_dir}/**/*.sol", recursive=True):
        p = f.replace(f"{code_dir}/", "")
        if p in scope_files:
            files.append(f)

    assert len(files) == len(scope_files)
    for audit_file in files:
        results[audit_file] = {}
        with open(audit_file) as f:
            source_code = f.read()
            parser = ParseStream(source_code)
            items = parser.get_all_functions()
            CG = generate_call_graph(source_code)
            print("===============")
            print(CG)
            for contract in items:
                for contract, fn_name, locations, context in items[contract]:
                    if not (
                        context.strip().startswith("function")
                        and ("{" in context and "}" in context)
                    ):
                        continue
                    check_fn_list = [(contract, fn_name, context)]
                    data = prompter.generate_inputClassWithOutCall(
                        check_fn_list
                    )  # multiple prompts as the input, it returns the list
                    call_relationships = get_calling_relationships(
                        CG, fn_name, contract
                    )
                    in_calls, out_calls = parse_calling_relationship(
                        call_relationships, parser, f"{contract}.{fn_name}"
                    )
                    # for pidx, prompt_input_txt in enumerate(data):
                    if (
                        get_hash(f"{contract}{fn_name}{context}")
                        not in results[audit_file]
                    ):
                        results[audit_file][
                            get_hash(f"{contract}{fn_name}{context}")
                        ] = {
                            "meta": {
                                "contract": contract,
                                "fn_name": fn_name,
                                "locations": locations,
                                "context": context,
                                "in_calls": in_calls,
                                "out_calls": out_calls,
                            }
                        }
                    results[audit_file][get_hash(f"{contract}{fn_name}{context}")][
                        "class_prompt"
                    ] = data
    return results
