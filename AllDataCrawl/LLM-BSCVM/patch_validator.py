import openai
import os
import requests
from typing import Dict, Any, Optional
from modules.base_analyzer import BaseAnalyzer
from config.config import Config

class PatchValidator(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.api_key = 'XXXXXXXXXXXX'
        # 为API调用设置固定温度值
        self.temperature = 0.1
        # 为每次调用设置固定的种子值
        self.seed = 42
        self.api_url = "XXXXXXXXXXX"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _create_analysis_prompt(self, original_code: str, fixed_code: str) -> str:
        return f"""Please analyze this smart contract patch and provide a structured response with the following format:

1. Confirm that the patch effectively mitigates the vulnerability? [Yes/No]
   Explanation: [Brief explanation]

Original Contract:
{original_code}

Patched Contract:
{fixed_code}

Please provide your analysis in the exact format above, with a clear Yes/No answer followed by a brief explanation."""

    def _parse_validation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the validation response to determine if the vulnerability is effectively mitigated"""
        result = {
            "security_impact": False,
            "validation_passed": False,
            "details": response_text
        }

        try:
            # Look for Yes/No answer in the response regarding vulnerability mitigation
            if "[Yes]" in response_text and "mitigate" in response_text:
                result["security_impact"] = True

            # Only pass validation if vulnerability is effectively mitigated
            result["validation_passed"] = result["security_impact"]

        except Exception:
            pass

        return result

    def analyze(self, contract_code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if not context or not contract_code:
            return {
                "patch_validation": {
                    "validation_result": "Missing input",
                    "status": "error",
                    "is_successful": False
                }
            }

        fixed_code = context.get("fixed_code", "")
        if not fixed_code:
            return {
                "patch_validation": {
                    "validation_result": "No patch found",
                    "status": "error",
                    "is_successful": False
                }
            }

        try:
            prompt = self._create_analysis_prompt(contract_code, fixed_code)
            payload = {
                "model": "gpt-4-1106-preview",
                "messages": [
                    {"role": "system", "content": "You are a smart contract security expert. Analyze the provided patch and give concrete, specific feedback about its effectiveness in mitigating identified vulnerabilities."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": self.temperature,
                "seed": self.seed
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            validation_text = result['choices'][0]['message']['content']
            validation_result = self._parse_validation_response(validation_text)
            
            return {
                "patch_validation": {
                    "validation_result": validation_text,
                    "parsed_results": validation_result,
                    "status": "success",
                    "is_successful": validation_result["validation_passed"],
                    "model_used": "gpt-4-1106-preview"
                }
            }

        except Exception as e:
            return {
                "patch_validation": {
                    "validation_result": f"Error during analysis: {str(e)}",
                    "status": "error",
                    "is_successful": False
                }
            }

    def validate_input(self, contract_code: str, context: Optional[Dict[str, Any]] = None) -> bool:
        if not contract_code or not context:
            return False
        if not context.get("fixed_code"):
            return False
        return True

