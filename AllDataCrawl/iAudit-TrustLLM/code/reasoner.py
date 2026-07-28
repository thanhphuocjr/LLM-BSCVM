# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
from utils.helper import generate_reasoning_prompt_data
import collections
import json
import os
# from convert2md import convert2md
from utils.ensambler_strategy import Majority
from transformers import LlamaForCausalLM, LlamaTokenizer

voting_policy = {"majority": Majority()}


def reasoner(args, eval_dataset):
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    reason_lora_model_name_or_path = args.reason_lora_model_name_or_path
    reason_lora_tokenizer_name_or_path = args.reason_lora_tokenizer_name_or_path
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )

    eval_dataset = generate_reasoning_prompt_data(eval_dataset, args)

    if reason_lora_model_name_or_path is not None:
        config = PeftConfig.from_pretrained(reason_lora_model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(reason_lora_tokenizer_name_or_path)

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
            reason_lora_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.config.inference_mode = True
    elif args.reason_merged_peft_model_name_or_path is not None:
        # tokenizer = AutoTokenizer.from_pretrained(
        #     args.reason_merged_peft_model_name_or_path, token=os.environ["Access_Token"]
        # )
        tokenizer = LlamaTokenizer.from_pretrained(args.reason_merged_peft_model_name_or_path, token=os.environ["Access_Token"])
        # Add special tokens if they are missing
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({
                'eos_token': '</s>',
                'bos_token': '<s>',
                'unk_token': '<unk>',
                'pad_token': '<pad>'
            })
        # Load the model with the language modeling head
        model = LlamaForCausalLM.from_pretrained(args.reason_merged_peft_model_name_or_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            token=os.environ["Access_Token"] )
        model.resize_token_embeddings(len(tokenizer))
        # model = AutoModelForCausalLM.from_pretrained(
        #     args.reason_merged_peft_model_name_or_path,
        #     device_map="auto",
        #     torch_dtype=torch_dtype,
        #     low_cpu_mem_usage=True,
        #     token=os.environ["Access_Token"],
        # )
        model.config.inference_mode = True
    else:
        raise ValueError(
            "No model is provided, please indicate reason_merged_model_name_or_path or reason_lora_model_name_or_path"
        )
    temparature = 0.2
    generation_config = dict(
        temperature=temparature,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        max_new_tokens=512 * 2,
    )

    model.eval()
    with torch.no_grad():
        for audit_file in eval_dataset:
            audit_file_dataset = eval_dataset[audit_file]
            for i_, (hash_id, prompts_item) in enumerate(audit_file_dataset.items()):
                input_prompt_list = (
                    prompts_item["reasoning_prompt_withcall"]
                    + prompts_item["reasoning_prompt_withoutcall"]
                )
                prompts_item["response_reason_list"] = []
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
                    prompts_item["response_reason_list"].append(response)
                    # predict_cur_item.append( {  "contract":contract, "function":function, \
                    #                             "locations":locations, "context":context, \
                    #                             "input_prompt":input_prompt, \
                    #                             "in_call_sequence":in_call_sequence, \
                    #                             "out_call_sequence":out_call_sequence,\
                    #                             "response":response} )
                # results[audit_file][hash_id] = predict_cur_item
    return eval_dataset


import re


def extract_label(text):
    match = re.search(r"(?i)the label is (safe|vulnerable)", text)
    if match:
        return match.group(0)
    return None
