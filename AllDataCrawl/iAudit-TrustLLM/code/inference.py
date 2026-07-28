import torch
import argparse
from ConfigArgs import ConfigArguments
import collections
import json
from transformers import HfArgumentParser
import os
from classifier import classification
from reasoner import reasoner
from utils.helper import parse_class_dir
from post_process import (
    selection,
)  # (output_folder, eval_dataset, model_name_or_path, quantization=False)
import json
import random


def main():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser((ConfigArguments))
    (args,) = parser.parse_args_into_dataclasses()
    print(args)

    eval_dataset = parse_class_dir(args.code_dir, args.scope_file, args.single_prompt)
    # json.dump(eval_dataset, open(args.code_dir_cache, "w"), indent=4)

    eval_dataset = classification(args, eval_dataset)
    json.dump(eval_dataset, open(f"{args.output_file}", "w"), indent=4)

    eval_dataset = json.load(open(args.output_file, "r"))
    # # Randomly select 2 items
    # # eval_dataset = random.sample(list(eval_dataset.items()), 2)
    # # eval_dataset = dict(eval_dataset)
    eval_dataset = reasoner(args, eval_dataset)
    json.dump(eval_dataset, open(f"{args.output_file}_reason", "w"), indent=4)

    eval_dataset = json.load(open(f"{args.output_file}_reason", "r"))
    eval_dataset = selection(eval_dataset, quantization=True)
    json.dump(eval_dataset, open(f"{args.output_file}_selection", "w"), indent=4)


import re


def extract_label(text):
    match = re.search(r"(?i)the label is (safe|vulnerable)", text)
    if match:
        return match.group(0)
    return None


if __name__ == "__main__":
    main()
