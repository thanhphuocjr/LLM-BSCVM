import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# USER CONFIG (EDIT HERE)
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY = "Check Vulnerable safeMath in this code"
CODE = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract VulnerableBank {
    mapping(address => uint256) public balances;

    // Người dùng gửi ETH vào bank
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    // Hàm này bị lỗi Reentrancy
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Not enough balance");

        // LỖI: gửi ETH trước khi cập nhật số dư
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        // Cập nhật balance sau khi gửi tiền
        // Attacker có thể gọi lại withdraw() trước khi dòng này chạy
        balances[msg.sender] -= amount;
    }

    // Xem tổng ETH trong contract
    function getContractBalance() public view returns (uint256) {
        return address(this).balance;
    }
}
"""
KNOWLEDGE_DIR = str(PROJECT_ROOT / "rag" / "knowledge_store")
SWC_REGISTRY_PATH = str(PROJECT_ROOT / "dataset" / "swc_registry.json")
MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cpu"  # auto | cpu | cuda | mps
MAX_SEQ_LENGTH = 512
MAX_INPUT_CHARS = 15000
TOP_K = 3


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", " ", text.lower()).strip()


def compact_phrase(text: str) -> str:
    return normalize_for_match(text).replace(" ", "")


def phrase_match(input_norm: str, input_compact: str, phrase: str) -> bool:
    phrase_norm = normalize_for_match(phrase)
    if not phrase_norm:
        return False
    if phrase_norm in input_norm:
        return True
    phrase_compact = phrase_norm.replace(" ", "")
    return len(phrase_compact) >= 5 and phrase_compact in input_compact


def lexical_boost(input_text: str, record: Dict) -> float:
    input_norm = normalize_for_match(input_text)
    input_compact = compact_phrase(input_text)
    boost = 0.0

    swc = str(record.get("Swc") or "")
    if swc and phrase_match(input_norm, input_compact, swc):
        boost += 0.18

    alias_matched = False
    for alias in record.get("WeaknessAliases", []) or []:
        if phrase_match(input_norm, input_compact, str(alias)):
            boost += 0.14
            alias_matched = True
            break

    for pattern in record.get("CodePatterns", []) or []:
        if phrase_match(input_norm, input_compact, str(pattern)):
            boost += 0.08
            break

    title = str(record.get("Title") or "")
    if phrase_match(input_norm, input_compact, title):
        boost += 0.06

    source = str(record.get("Source") or "")
    if alias_matched and source in {"swc_template", "query_alias_augmentation", "code_pattern_augmentation"}:
        boost += 0.04

    return min(boost, 0.30)



def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_input_text(query: str, code: str, max_chars: int) -> str:
    text = "\n".join(
        [
            f"Query: {query.strip()}",
            f"Code: {code.strip()}",
        ]
    )
    return text[:max_chars]


def map_result(record: Dict, score: float, semantic_score: float, boost: float) -> Dict:
    return {
        "Swc": record.get("Swc"),
        "Title": record.get("Title"),
        "Vulnerable": record.get("Description"),
        "BestPractice": record.get("Remediation"),
        "CodePatterns": record.get("CodePatterns", []),
        "BestPracticeChecklist": record.get("BestPracticeChecklist", []),
        "Source": record.get("Source"),
        "AugmentationType": record.get("AugmentationType"),
        "OriginalSwc": record.get("OriginalSwc"),
        "InferredSwc": record.get("InferredSwc"),
        "SwcCorrected": record.get("SwcCorrected", False),
        "SemanticScore": float(semantic_score),
        "LexicalBoost": float(boost),
        "SimilarityScore": float(score),
    }


class KnowledgeBaseRetriever:
    def __init__(
        self,
        knowledge_dir: str = KNOWLEDGE_DIR,
        model_name: str = MODEL_NAME,
        device_arg: str = DEVICE,
        max_seq_length: int = MAX_SEQ_LENGTH,
        max_input_chars: int = MAX_INPUT_CHARS,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.model_name = model_name
        self.device = resolve_device(device_arg)
        self.max_seq_length = max_seq_length
        self.max_input_chars = max_input_chars
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        self.vectors = self._load_vectors()
        self.metadata = self._load_metadata()
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.max_seq_length = max_seq_length

    def _load_vectors(self) -> np.ndarray:
        vectors_path = self.knowledge_dir / "sample_embeddings_local_l2.npy"
        if not vectors_path.exists():
            raise FileNotFoundError(
                f"Embedding vectors not found: {vectors_path}. "
                "Run build_knowledge_store.py first."
            )
        vectors = np.load(vectors_path, mmap_mode="r")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"Invalid vectors shape: {vectors.shape}")
        return vectors

    def _load_metadata(self) -> List[Dict]:
        metadata_path = self.knowledge_dir / "sample_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}. "
                "Run build_knowledge_store.py first."
            )
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata: List[Dict] = json.load(f)
        if len(metadata) != self.vectors.shape[0]:
            raise ValueError(
                f"Metadata size ({len(metadata)}) does not match vector rows ({self.vectors.shape[0]})."
            )
        return metadata

    def search(self, query: str, code: str, top_k: int = TOP_K) -> Dict:
        return self.search_many([query], code, top_k=top_k)[0]

    def search_many(self, queries: List[str], code: str, top_k: int = TOP_K) -> List[Dict]:
        return [
            self.search_single_old(query=query, code=code, top_k=top_k)
            for query in queries
        ]

    def search_single_old(self, query: str, code: str, top_k: int = TOP_K) -> Dict:
        input_text = build_input_text(query, code, self.max_input_chars)
        query_vec = self.model.encode(
            [input_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        query_vec = l2_normalize_rows(query_vec)[0]

        semantic_scores = self.vectors @ query_vec
        boosts = np.array(
            [lexical_boost(input_text, record) for record in self.metadata],
            dtype=np.float32,
        )
        scores = semantic_scores + boosts

        top_k = max(1, min(top_k, len(scores)))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = [
            map_result(
                self.metadata[idx],
                float(scores[idx]),
                float(semantic_scores[idx]),
                float(boosts[idx]),
            )
            for idx in top_indices
        ]
        return {"query": query, "top_k": top_k, "results": results}


def retrieve(
    query: str,
    code: str,
    knowledge_dir: str,
    model_name: str,
    device_arg: str,
    max_seq_length: int,
    max_input_chars: int,
    top_k_arg: int,
) -> Dict:
    retriever = KnowledgeBaseRetriever(
        knowledge_dir=knowledge_dir,
        model_name=model_name,
        device_arg=device_arg,
        max_seq_length=max_seq_length,
        max_input_chars=max_input_chars,
    )
    return retriever.search(query=query, code=code, top_k=top_k_arg)


def retrieve_multi_agent(
    code: str,
    knowledge_dir: str = KNOWLEDGE_DIR,
    model_name: str = MODEL_NAME,
    device_arg: str = DEVICE,
    max_seq_length: int = MAX_SEQ_LENGTH,
    max_input_chars: int = MAX_INPUT_CHARS,
    top_k_arg: int = 1,
    swc_registry_path: str = SWC_REGISTRY_PATH,
    max_findings: int = 3,
    include_all_agents: bool = False,
) -> Dict:
    try:
        from .multi_agent_vuln_detector import run_multi_agent_analysis
    except ImportError:
        from multi_agent_vuln_detector import run_multi_agent_analysis

    return run_multi_agent_analysis(
        code=code,
        knowledge_dir=knowledge_dir,
        model_name=model_name,
        device=device_arg,
        max_seq_length=max_seq_length,
        max_input_chars=max_input_chars,
        top_k=top_k_arg,
        swc_registry_path=swc_registry_path,
        max_findings=max_findings,
        include_all_agents=include_all_agents,
    )


def resolve_code_input(code: str | None, code_file: str | None) -> str:
    if code_file:
        return Path(code_file).read_text(encoding="utf-8")
    return code if code is not None else CODE


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-agent SWC retrieval by default. Use --single-query "
            "to run the old one-query retriever."
        )
    )
    parser.add_argument("--single-query", action="store_true")
    parser.add_argument("--query", default=QUERY)
    parser.add_argument("--code", default=None)
    parser.add_argument("--code-file", default=None)
    parser.add_argument("--swc-registry", default=SWC_REGISTRY_PATH)
    parser.add_argument("--knowledge-dir", default=KNOWLEDGE_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--max-input-chars", type=int, default=MAX_INPUT_CHARS)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--max-findings",
        type=int,
        default=3,
        help="Maximum number of final multi-agent findings to return.",
    )
    parser.add_argument(
        "--include-all-agents",
        action="store_true",
        help="Include raw results for all SWC agents in the JSON output.",
    )
    args = parser.parse_args()

    code = resolve_code_input(args.code, args.code_file)
    if args.single_query:
        output = retrieve(
            query=args.query,
            code=code,
            knowledge_dir=args.knowledge_dir,
            model_name=args.model_name,
            device_arg=args.device,
            max_seq_length=args.max_seq_length,
            max_input_chars=args.max_input_chars,
            top_k_arg=args.top_k or TOP_K,
        )
    else:
        output = retrieve_multi_agent(
            code=code,
            knowledge_dir=args.knowledge_dir,
            model_name=args.model_name,
            device_arg=args.device,
            max_seq_length=args.max_seq_length,
            max_input_chars=args.max_input_chars,
            top_k_arg=args.top_k or 1,
            swc_registry_path=args.swc_registry,
            max_findings=args.max_findings,
            include_all_agents=args.include_all_agents,
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
