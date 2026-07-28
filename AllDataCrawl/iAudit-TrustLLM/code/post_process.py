import json

# import the relavant libraries for loggin in
from huggingface_hub import HfFolder
import glob
import os

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
import random
from selection.Ranker import Ranker
from selection.Criticor import Critic
import re

print(torch.__version__)
print(transformers.__version__)
import random


def selection_reasons(eval_dataset_files, model, tokenizer, maximum_round=5):
    ranker = Ranker(model, tokenizer)
    critic = Critic(model, tokenizer)
    failed_ids = {}
    for _, (audit_file, _) in enumerate(eval_dataset_files.items()):
        print(f"Start for {audit_file}")
        for i, (hash_id, prompts_item) in enumerate(
            eval_dataset_files[audit_file].items()
        ):
            print(f"Start for {hash_id}, {i}/{len(eval_dataset_files[audit_file])}")
            print(f"prompts_item: {prompts_item.keys()}")
            reponse_prompt_list = prompts_item["response_reason_list"]
            meta_info = prompts_item["meta"]
            contract, function, context = (
                meta_info["contract"],
                meta_info["fn_name"],
                meta_info["context"],
            )
            pre_label = prompts_item["pre_label"]
            feedback = None
            trial = 0
            while True:
                if feedback is None or "merge" not in feedback["feedback"].lower():
                    res, response = ranker.rank(
                        {"code": context, "reason_pred": reponse_prompt_list},
                        pre_label,
                        feedback=feedback,
                    )
                else:
                    res, response = ranker.merge(
                        {"code": context, "reason_pred": reponse_prompt_list},
                        pre_label,
                        feedback=feedback,
                    )
                print("Ranker says: ", response)
                crk = critic.critic(res, response)
                if "yes" in crk:
                    print(f"Criticor says yes, {crk}")
                    break
                else:
                    print("Criticor says no, retrying...")
                    print("Feedback: ", crk)
                    feedback = {"pre_answer": response, "feedback": crk}
                if trial < maximum_round:
                    trial += 1
                else:
                    print(f"Retry Reach Maximum Round {maximum_round}")
                    break
            print(f"Finish for {hash_id}, {i}/{len(eval_dataset_files[audit_file])}")
            try:
                json_response = extract_json_from_text(response)
            except:
                failed_ids[hash_id] = {
                    "id": hash_id,
                    "prompts_item": prompts_item,
                    "response": response,
                }
                json.dump(failed_ids, open("failed_ids.json", "w"), indent=4)
                # assert False, f"Error in response {response}"
            try:
                print(f"json_response: {json_response}")
                r = json.loads(json_response)
                if "id" in r:
                    print(f"Response: {r['id']}")
                    reason_id = r["id"]
                    analysis = r["analysis"]
                    prompts_item["reason_id"] = reason_id
                    prompts_item["analysis"] = analysis
                    id_list = re.findall(r"\d+", reason_id)
                    id_list = [int(id) for id in id_list]
                    prompts_item["selected_reason"] = [
                        reponse_prompt_list[id] for id in id_list
                    ]
                else:
                    print(f"Response: {r['merged id']}")
                    reason_id = r["merged id"]
                    merged_reason = r["merged reason"]
                    analysis = r["analysis"]
                    prompts_item["reason_id"] = reason_id
                    prompts_item["analysis"] = analysis
                    id_list = re.findall(r"\d+", reason_id)
                    id_list = [int(id) for id in id_list]
                    prompts_item["selected_reason"] = [
                        reponse_prompt_list[id] for id in id_list
                    ]
                    prompts_item["merged_reason"] = merged_reason
            except:
                failed_ids[hash_id] = {
                    "id": hash_id,
                    "prompts_item": prompts_item,
                    "response": response,
                }
                json.dump(failed_ids, open("failed_ids.json", "w"), indent=4)
                # assert False, f"Error in response {response}"

            if i % 10 == 0:
                json.dump(
                    eval_dataset_files, open("eval_dataset_tmp.json", "w"), indent=4
                )

    return eval_dataset_files


import re


def extract_json_from_text(input_text):
    """
    Extracts a JSON string from the provided text that is enclosed between ```json and ```.

    Parameters:
    - input_text (str): The text containing the JSON string to be extracted.

    Returns:
    - str: The extracted JSON string, or None if no JSON string is found.
    """
    match = re.search(r"\{(.+?)\}", input_text, re.DOTALL).group()
    return match


def selection(
    eval_dataset,
    model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
    quantization=False,
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        quantization_config=bnb_config if quantization else False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print("Model Loaded")
    # os.makedirs(output_folder, exist_ok=True)
    eval_dataset = selection_reasons(eval_dataset, model, tokenizer)
    return eval_dataset
