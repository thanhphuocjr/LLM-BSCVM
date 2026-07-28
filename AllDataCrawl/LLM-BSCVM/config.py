import os
from pathlib import Path

class Config:
    """Configuration settings for the smart contract analyzer"""

    
    # 基础路径配置
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    MODEL_DIR = OUTPUT_DIR / "finetuned_models"

    # GPU配置
    CUDA_VISIBLE_DEVICES = 'XXXXXXX'  # 
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES


    DEVICE_MAP = {
        'vulnerability_detector': 0,  #
        'repair_suggester': 0,       #
        'vulnerability_fixer': 1     # 
    }

    
    MODEL_PATH = "XXXXXXXXXX"  # 
    FINETUNED_MODEL_PATH = "XXXXXXXXXX"
    KNOWLEDGE_BASE_PATH = str(DATA_DIR / "XXXXXXXXXX")
    VALID_DATA_PATH = str(DATA_DIR / "/XXXXXXXXXX")
    
    MODEL_CONFIG = {
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "max_new_tokens": 512,
        "context_length": 4096  # 
    }
    
    # OpenAI配置
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY','XXXXXXXXXX')
    
    # 风险等级定义
    RISK_LEVELS = {
        "Critical": ["Reentrancy", "Access control issues", "Unchecked external calls"],
        "High": [
            "Integer overflow/underflow", 
            "Denial of Service - DoS",
            "Flash loan vulnerabilities",
            "Front-running vulnerabilities"
        ],
        "Medium": [
            "Timestamp dependence",
            "Block information dependence",
            "Gas limit issues",
            "Transaction Ordering Dependency",
            "Unsafe Type Casting"
        ],
        "Low": [
            "Outdated compiler version",
            "Naming conventions",
            "Redundant code"
        ]
    }