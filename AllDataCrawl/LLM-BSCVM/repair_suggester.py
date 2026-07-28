
from typing import Dict, Any
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.base_analyzer import BaseAnalyzer
import re
from typing import Dict, Any, List, Optional
import json
from difflib import SequenceMatcher
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RepairSuggester(BaseAnalyzer):
    def __init__(self, model_path: str = None, knowledge_base_path: str = None):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.knowledge_base = []
        
        if model_path:
            self.initialize(model_path)
            
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
            
    @staticmethod
    def similar(a: str, b: str) -> float:
        """Calculate string similarity ratio using SequenceMatcher"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def load_knowledge_base(self, knowledge_base_path: str):
        try:
            with open(knowledge_base_path, 'r') as f:
                self.knowledge_base = json.load(f)
            logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} entries")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            raise

    def initialize(self, model_path: str):
        try:
            logger.info("Loading repair suggestion model...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map='auto'
            )
            self.model.eval()
            logger.info("Model initialization completed successfully.")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def _generate_repair_prompt(self, code: str) -> str:
        prompt = """[INST] <<SYS>> You are an expert Solidity smart contract security auditor. Your task is to analyze the given code and provide detailed repair suggestions in a specific format.
<</SYS>>

Analyze this Solidity code for security vulnerabilities and provide repair suggestions:
```solidity
{code}
```

For each vulnerability found, use exactly this format:

Vulnerability [number]:

Name: [Name of the vulnerability]

Description: [Clear explanation of what's wrong and why it's dangerous]

Impact: [Detailed explanation of potential consequences]

Fix: [Clear step-by-step instructions to fix the issue]

Prevention: [Best practices and recommendations to prevent this type of issue]

Make sure to separate each vulnerability with a blank line, and number them consecutively.
Do not include any introductory text or conclusion - start directly with "Vulnerability 1:".

[/INST]"""
        return prompt.format(code=code.strip())

    def _format_repair_suggestion(self, suggestion: str) -> Dict:
        # Remove the instruction wrapper if present
        suggestion = re.sub(r'\[INST\].*?\[/INST\]', '', suggestion, flags=re.DOTALL)
        suggestion = suggestion.strip()
        
        if not suggestion:
            return {"repair_suggestions": "No vulnerabilities found."}
        
        # Clean up formatting
        suggestion = re.sub(r'\n{3,}', '\n\n', suggestion)
        
        # Extract individual vulnerabilities
        vulnerabilities = re.split(r'Vulnerability\s+\d+:', suggestion)[1:]  # Split by vulnerability header
        if not vulnerabilities:
            return {"repair_suggestions": "No clear vulnerabilities identified."}
        
        # Helper function to extract key information from vulnerability
        def extract_info(vuln_text):
            name_match = re.search(r'Name:\s*(.+?)\n', vuln_text)
            desc_match = re.search(r'Description:\s*(.+?)\n(?:Impact|Fix|Prevention):', vuln_text, re.DOTALL)
            impact_match = re.search(r'Impact:\s*(.+?)\n(?:Fix|Prevention):', vuln_text, re.DOTALL)
            
            return {
                'name': name_match.group(1).strip() if name_match else '',
                'description': desc_match.group(1).strip() if desc_match else '',
                'impact': impact_match.group(1).strip() if impact_match else ''
            }
        
        # Process and deduplicate vulnerabilities
        processed_vulns = []
        seen_combinations = set()
        
        for vuln in vulnerabilities:
            info = extract_info(vuln)
            
            # Create a signature for similarity checking
            # Combine name and key terms from description for comparison
            desc_terms = set(re.findall(r'\b\w+\b', info['description'].lower()))
            signature = (info['name'], frozenset(desc_terms))
            
            # Check similarity with existing vulnerabilities
            is_duplicate = False
            for seen_sig in seen_combinations:
                # Compare names and description overlap
                name_similarity = self.similar(seen_sig[0], signature[0])
                desc_overlap = len(signature[1].intersection(seen_sig[1])) / len(signature[1].union(seen_sig[1]))
                
                if name_similarity > 0.8 or desc_overlap > 0.7:  # Adjustable thresholds
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_combinations.add(signature)
                processed_vulns.append(vuln)
        
        # Reconstruct the suggestion with deduplicated vulnerabilities
        final_suggestion = ""
        for i, vuln in enumerate(processed_vulns, 1):
            final_suggestion += f"Vulnerability {i}:\n{vuln.strip()}\n\n"
        
        return {"repair_suggestions": final_suggestion.strip()}


    
    def _merge_similar_descriptions(self, vulns: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Merge descriptions and fixes for similar vulnerabilities
        
        Args:
            vulns: List of vulnerability dictionaries with 'name', 'description', 'impact' keys
            
        Returns:
            Dict with merged vulnerability information
        """
        if not vulns:
            return {}
        
        # Use the most general name
        names = [v['name'] for v in vulns]
        merged_name = min(names, key=len)  # Choose shortest name as likely most general
        
        # Combine unique aspects of descriptions
        descriptions = [v['description'] for v in vulns]
        desc_sentences = set()
        for desc in descriptions:
            desc_sentences.update(sent.strip() for sent in re.split(r'[.!?]+', desc) if sent.strip())
        merged_description = '. '.join(sorted(desc_sentences)) + '.'
        
        # Combine unique impacts
        impacts = [v['impact'] for v in vulns]
        impact_sentences = set()
        for impact in impacts:
            impact_sentences.update(sent.strip() for sent in re.split(r'[.!?]+', impact) if sent.strip())
        merged_impact = '. '.join(sorted(impact_sentences)) + '.'
        
        return {
            'name': merged_name,
            'description': merged_description,
            'impact': merged_impact
        }

    def _generate_suggestion(self, prompt: str, max_length: int = 800) -> str:
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    num_beams=3,
                    temperature=0.7,
                    no_repeat_ngram_size=0
                )
                
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"Error generating repair suggestion: {str(e)}")
            return ""



    def analyze(self, contract_code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized. Please call initialize() first.")

        repair_prompt = self._generate_repair_prompt(contract_code)
        raw_suggestion = self._generate_suggestion(repair_prompt)
        formatted_suggestion = self._format_repair_suggestion(raw_suggestion)

        return formatted_suggestion

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

