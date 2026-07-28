class Ranker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def rank(self, item, pre_label, feedback=None):
        prompt_check = """
You are a helpful and honest code reviewer, who is responsible for checking the quality of the code and some unchecked comments about the code. The code has been labelled with safe or vulnerable. You have to select the best comment about the code label. Below, I define your abilities, your responsibilities, and your constraints. You should consider all constraints and facts provided in the code.

# Abilities
1. You are an excellent code reviewer for Smart Contract.
2. You are familiar with the vunerabilities in Smart Contract.
3. You can refine your result according to the given feedback.

# Responsibilities
1. You have to select the best comment that supports the decision.
2. You have to consider all constraints and facts provided in the code.
3. You have to select the most relevant, reasonable and accurate comment.
      
# Constraints
1. If one reason describes code that does not exist in the provided input, it is not valid.
2. If one reason is not related to the code, the reason is not valid.
3. If this code violates the facts, the reason is unreasonable.
4. If one reason is not related to the decision, the reason is not valid.
5. If one reason assume any information that is not provided, the reason is not valid.
6. If the code is safe and one reason supports the decision, please check if the code has other potential vulnerabilities. If the code has other potential vulnerabilities, the reason is not valid.
7. The selected reason should be the most relevant to the decision.
8. The selected reason must be the most reasonable and accurate one.
9. The selected reason must be factual, logical and convicing.
10. Do not make any assumption out of the given code.
11. If the reason breaks any above constraint, the reason does not supoort the decision.

Please select the best reason and give its id. As well, please provide the confidence score and the analysis about your answer. The confidence score is 1 to 10. High score means more confident and if you are not sure, please give the low score. 
If the input contains the previous answer and the feedback from the coorperator, please consider these information and make more better choice. The feedback can contain three types of actions, agree, rerank and merge. 
1. If the feedback action is agree, you should consider the previous answer and the feedback and make the final decision;
2. If the feedback action is rerank, you should re-make your choice based on the feedback;
3. If the feedback action is merge, you should consider merging some reasons according to the feedback and make the final decision. 

The input is following:

# Code
```
{code}
```

# Reasons 
{reason} 
"""

        prompt_feedback = """
# Previous Answer
{pre_answer}

# Feedback
{feedback}
"""
        prompt_response = """
Your answer has to alway respond in this JSON format, 
```json
{{ 
 "id": "reason id", 
 "score":"value",  
 "analysis": "your thoughts" 
 }}
```

Let's get started. The following code is decided to be {label}. I will provide these unchecked reasons for this decision. 
# Response
"""

        code_fn = item["code"]
        reason_pred_list = [
            f"## Reason {i+1}\n  {r.strip()}" for i, r in enumerate(item["reason_pred"])
        ]
        reason_pred = "\n\n".join(reason_pred_list)
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
        res = {
            "code": code_fn,
            "reason_pred": reason_pred_list,
            "pre_label": pre_label,
            "check_result": {},
            "raw": item,
        }
        prompt_question = prompt_check.format(reason=reason_pred, code=code_fn)
        if feedback is not None:
            prompt_question = (
                prompt_question
                + "\n"
                + prompt_feedback.format(
                    pre_answer=feedback["pre_answer"], feedback=feedback["feedback"]
                )
            )
        prompt_question = (
            prompt_question + "\n" + prompt_response.format(label=pre_label)
        )
        print(f"Prompt Question: {prompt_question}")
        messages = ({"role": "user", "content": prompt_question},)
        input_txt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )  # (prompt_question, return_tensors="pt")
        inputs = self.tokenizer(input_txt, return_tensors="pt")
        print(f"Model Generation {self.model.config.model_type.lower()}")
        if "gemma" in self.model.config.model_type.lower():
            generation_output = self.model.generate(
                inputs["input_ids"].to(self.device), max_length=4096 * 4
            )
        if "mixtral" in self.model.config.model_type.lower():
            generation_output = self.model.generate(
                inputs["input_ids"].to(self.device), **generation_config
            )
        s = generation_output[0]
        print("Model Generation Done")
        output = self.tokenizer.decode(s, sip_special_tokens=True)  #
        output = output.replace("[/INST]", "")
        output = output.replace("<start_of_turn>model", "")
        response = output.split("# Response")[-1].strip()
        res["check_result"] = response
        return res, response

    def merge(self, item, pre_label, feedback=None):
        prompt_check = """
You are a helpful and honest code reviewer, who is responsible for checking the quality of the code and some unchecked comments about the code. The code has been labelled with safe or vulnerable. You have to select the best comment about the code label. Below, I define your abilities, your responsibilities, and your constraints. You should consider all constraints and facts provided in the code.

# Abilities
1. You are an excellent code reviewer for Smart Contract.
2. You are familiar with the vunerabilities in Smart Contract.
3. You can refine your result according to the given feedback.

# Responsibilities
1. You have to select the best comment that supports the decision.
2. You have to consider all constraints and facts provided in the code.
3. You have to select the most relevant, reasonable and accurate comment.
      
# Constraints
1. If one reason describes code that does not exist in the provided input, it is not valid.
2. If one reason is not related to the code, the reason is not valid.
3. If this code violates the facts, the reason is unreasonable.
4. If one reason is not related to the decision, the reason is not valid.
5. If one reason assume any information that is not provided, the reason is not valid.
6. If the code is safe and one reason supports the decision, please check if the code has other potential vulnerabilities. If the code has other potential vulnerabilities, the reason is not valid.
7. The selected reason should be the most relevant to the decision.
8. The selected reason must be the most reasonable and accurate one.
9. The selected reason must be factual, logical and convicing.
10. Do not make any assumption out of the given code.
11. If the reason breaks any above constraint, the reason does not supoort the decision.

The input is following:

# Code
```
{code}
```

# Reasons 
{reason} 
"""

        prompt_feedback = """
# Previous Answer
{pre_answer}

# Feedback
{feedback}
"""
        prompt_response = """
Your answer has to alway respond in this JSON format, 
```json
{{ 
 "merged id": "reason id list", 
 "merged reason": "please input the merged reason here",
 "score":"value",  
 "analysis": "your thoughts" 
 }}
```
According to the feedback, you need to merge some reasons but the maximum merged number is 3. You need to provide the merged id list and the merged reason. The merged id list is the list of the selected reason id. The merged reason is the merged reason based on the selected reasons.
Let's get started. The following code is decided to be {label}. I will provide these unchecked reasons for this decision. 
# Response
"""

        code_fn = item["code"]
        reason_pred_list = [
            f"## Reason {i+1}\n  {r.strip()}" for i, r in enumerate(item["reason_pred"])
        ]
        reason_pred = "\n\n".join(reason_pred_list)
        generation_config = dict(
            temperature=0.6,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            # eos_token_id=tokenizer.eos_token_id,
            eos_token_id=2,
            pad_token_id=2,
            max_length=4096 * 4,
        )
        res = {
            "code": code_fn,
            "reason_pred": reason_pred_list,
            "pre_label": pre_label,
            "check_result": {},
            "raw": item,
        }
        prompt_question = prompt_check.format(reason=reason_pred, code=code_fn)
        if feedback is not None:
            prompt_question = (
                prompt_question
                + "\n"
                + prompt_feedback.format(
                    pre_answer=feedback["pre_answer"], feedback=feedback["feedback"]
                )
            )
        prompt_question = (
            prompt_question + "\n" + prompt_response.format(label=pre_label)
        )
        print(f"Prompt Question: {prompt_question}")
        messages = ({"role": "user", "content": prompt_question},)
        input_txt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )  # (prompt_question, return_tensors="pt")
        inputs = self.tokenizer(input_txt, return_tensors="pt")
        print(f"Model Generation {self.model.config.model_type.lower()}")
        if "gemma" in self.model.config.model_type.lower():
            generation_output = self.model.generate(
                inputs["input_ids"].to(self.device), max_length=4096 * 4
            )
        if "mixtral" in self.model.config.model_type.lower():
            generation_output = self.model.generate(
                inputs["input_ids"].to(self.device), **generation_config
            )
        s = generation_output[0]
        print("Model Generation Done")
        output = self.tokenizer.decode(s, sip_special_tokens=True)  #
        output = output.replace("[/INST]", "")
        output = output.replace("<start_of_turn>model", "")
        response = output.split("# Response")[-1].strip()
        res["check_result"] = response
        return res, response
