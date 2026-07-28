class Critic:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def critic(self, item, report):
        prompt_check = """   
You are a helpful and honest code reviewer and work with your partner. Your responsibility is to check if the report from your partner is reasonable and accurate. I will provide all the information about the code and the report from your partner. Your task is to check if the report is reasonable and accurate. You should consider all constraints and facts provided in the code. If you think the report is reasonable and accurate, please answer yes. If you think the report is not reasonable and accurate, please answer no. If you are not sure, please answer no. 
After reviewing the report, you should advise the next action for your partner. The actions have three types,
 1. Agree if you think it is reasonbale.
 2. Rerank if you think your partner should choose another one.
 3. Merge if you think your partner should merge some reaons into one final reason.

I will provide the code, some unchecked reasons for this decision, and thre report from your partner. 

# Code
```
{code}
```

# Reasons 
{reason} 

# Report from the partner
{report}

Your response has to be in this JSON format, 
```json
{{ 
    "response": "yes or no",  
    "analysis":"your thought", 
    "action": "action type"
}}
```
Let's get started. The code, the reason candidates and the report from your partner are as follows.
# Response
"""
        reason_pred_list = [
            f"## Reason {i+1}\n  {r.strip()}" for i, r in enumerate(item["reason_pred"])
        ]
        prompt = prompt_check.format(
            code=item["code"], reason=reason_pred_list, report=report
        )
        generation_config = dict(
            temperature=0.6,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            # eos_token_id=tokenizer.eos_token_id,
            eos_token_id=2,
            pad_token_id=2,
            max_length=4096 * 4 * 2,
        )
        messages = ({"role": "user", "content": prompt},)
        input_txt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )  # (prompt_question, return_tensors="pt")
        inputs = self.tokenizer(input_txt, return_tensors="pt")
        # print("Model Generation")
        if "gemma" in self.model.config.model_type.lower():
            generation_output = self.model.generate(
                inputs["input_ids"].to(self.device), max_length=4096 * 4
            )
        if "mixtral" in self.model.config.model_type.lower():
            generation_output = self.model.generate(
                inputs["input_ids"].to(self.device), **generation_config
            )
        s = generation_output[0]
        output = self.tokenizer.decode(s, skip_special_tokens=True)
        output = output.replace("[/INST]", "")
        output = output.replace("<start_of_turn>model", "")
        response = output.split("# Response")[-1].strip()
        return response
