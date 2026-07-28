from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConfigArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    class_merged_peft_model_name_or_path: Optional[str] = field(
        default="XXXX-2/13b_classify",
        metadata={
            "help": (
                "The merged model based on Lora model and BASE model. This model has interated Lora adapter and the base model. Use this model to classify the contract."
            )
        },
    )
    class_lora_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The class lora model name path to lora adapter. If you do not specify class_merged_peft_model_name_or_path, provide this parameter."
            )
        },
    )

    class_lora_tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The class lora model name path to lora tokenizer. If you do not specify class_merged_peft_model_name_or_path, provide this parameter."
            )
        },
    )

    reason_merged_peft_model_name_or_path: Optional[str] = field(
        default="XXXX-2/13b_reasoner",
        metadata={
            "help": (
                "The merged reason model based on Lora model and BASE model. This model has interated Lora adapter and the base model. Use this model to reason the contract."
            )
        },
    )
    reason_lora_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The reason lora model name path to lora adapter. If you do not specify reason_merged_peft_model_name_or_path, provide this parameter."
            )
        },
    )

    reason_lora_tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The reason lora model name path to lora tokenizer. If you do not specify reason_merged_peft_model_name_or_path, provide this parameter."
            )
        },
    )

    code_dir: Optional[str] = field(
        default="contracts",
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )

    code_dir_cache: Optional[str] = field(
        default="zetachain_cache_poc_pipeline.json",
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )

    scope_file: Optional[str] = field(
        default="./scope.txt", metadata={"help": "Auditing scope file list"}
    )

    max_seq_length: Optional[str] = field(
        default=4096 * 4,
        metadata={
            "help": "max sequence length. this parameter may not be used. Please check generation config in the code."
        },
    )

    output_file: Optional[str] = field(
        default="zetachain_result_poc_pipeline.json",
        metadata={"help": "output file name prefix"},
    )

    single_prompt: bool = field(
        default=False,
        metadata={
            "help": "If only consider singple prompt. It is for the research experiment purpose. The default value is False."
        },
    )

    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
