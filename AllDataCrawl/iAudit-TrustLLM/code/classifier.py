# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
from utils.helper import parse_class_dir
import collections
import json
import os
# from convert2md import convert2md
from utils.ensambler_strategy import Majority

voting_policy = {"majority": Majority()}


def classification(args, eval_dataset):
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    class_lora_model_name_or_path = args.class_lora_model_name_or_path
    class_lora_tokenizer_name_or_path = args.class_lora_tokenizer_name_or_path
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )

    if class_lora_model_name_or_path is not None:
        config = PeftConfig.from_pretrained(class_lora_model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(class_lora_tokenizer_name_or_path)
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            assert tokenzier_vocab_size > model_vocab_size
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

        model = PeftModel.from_pretrained(
            base_model,
            class_lora_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        model.config.inference_mode = True
    elif args.class_merged_peft_model_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.class_merged_peft_model_name_or_path, token=os.environ["Access_Token"]
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.class_merged_peft_model_name_or_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            token=os.environ["Access_Token"],
        )
        model.config.inference_mode = True
    else:
        raise ValueError(
            "No model is provided, please indicate merged_peft_model_name_or_path or class_lora_model_name_or_path"
        )
    temparature = 0.6
    generation_config = dict(
        temperature=temparature,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        max_new_tokens=512,
    )

    model.eval()
    # results = collections.defaultdict(dict)
    with torch.no_grad():
        for audit_file in eval_dataset:
            audit_file_dataset = eval_dataset[audit_file]
            for i, (hash_id, prompts_item) in enumerate(audit_file_dataset.items()):
                print(prompts_item)
                input_prompt_list = prompts_item["class_prompt"]
                prompts_item["response_list"] = []
                for input_prompt in input_prompt_list:
                    inputs = tokenizer(
                        input_prompt, return_tensors="pt"
                    )  # add_special_tokens=False ?
                    print(f"Length: {len(inputs['input_ids'])}\n")
                    generation_output = model.generate(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        **generation_config,
                    )
                    s = generation_output[0]
                    output = tokenizer.decode(s, skip_special_tokens=True)
                    response = output.split("### Response:")[1].strip()
                    print(f"==============")
                    print(f"Input: {input_prompt}\n")
                    print(f"response: {response}\n")
                    print(f"Output: {output}\n")
                    prompts_item["response_list"].append(response)
    json.dump(eval_dataset, open(f"{args.output_file}", "w"), indent=4)
    post_process_res = voting_policy["majority"].evaluate(eval_dataset)
    json.dump(eval_dataset, open(f"{args.output_file}", "w"), indent=4)
    # convert2md(post_process_res, args.markdown_file_path)
    return post_process_res


import re


def extract_label(text):
    match = re.search(r"(?i)the label is (safe|vulnerable)", text)
    if match:
        return match.group(0)
    return None
