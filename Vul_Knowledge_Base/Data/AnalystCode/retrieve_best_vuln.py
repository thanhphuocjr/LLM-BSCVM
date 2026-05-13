import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# USER CONFIG (EDIT HERE)
# =========================
QUERY = "Check Vulnerable safeMath in this code"
CODE = """
pragma solidity ^0.8.20;
The function FuseTokenAdapterV1 from the contract wrap \n```Solidiy\nfunction wrap( uint256 amount, address recipient ) external onlyAlchemist returns (uint256) { SafeERC20.safeTransferFrom(underlyingToken, msg.sender, address(this), amount); SafeERC20.safeApprove(underlyingToken, token, amount); uint256 startingBalance = IERC20(token).balanceOf(address(this)); uint256 error; if ((error = ICERC20(token).mint(amount)) != NO_ERROR) { revert FuseError(error); } uint256 endingBalance = IERC20(token).balanceOf(address(this)); uint256 mintedAmount = endingBalance - startingBalance; SafeERC20.safeTransfer(token, recipient, mintedAmount); return mintedAmount; }\n```\n### As a Caller:\nFuseTokenAdapterV1 calls these functions:\n```\nSafeERC20.safeTransferFrom\nFuseTokenAdapterV1.address\nSafeERC20.safeApprove\nFuseTokenAdapterV1.IERC20\nFuseTokenAdapterV1.ICERC20\nFuseTokenAdapterV1.FuseError\nSafeERC20.safeTransfer\nmodifier onlyAlchemist() { if (msg.sender != alchemist) { revert Unauthorized(\"Not alchemist\"); } _; }\n
"""
KNOWLEDGE_DIR = "/Users/phuocthanh/Documents/RAG/Vul_Knowledge_Base/Data/Knowledge_Store"
MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cpu"  # auto | cpu | cuda | mps
MAX_SEQ_LENGTH = 1024
MAX_INPUT_CHARS = 15000
TOP_K = 1



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


def map_result(record: Dict, score: float) -> Dict:
    return {
        "Swc": record.get("Swc"),
        "Title": record.get("Title"),
        "Vulnerable": record.get("Description"),
        "BestPractice": record.get("Remediation"),
        "SimilarityScore": float(score),
    }


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
    knowledge_dir = Path(knowledge_dir)
    vectors_path = knowledge_dir / "sample_embeddings_local_l2.npy"
    metadata_path = knowledge_dir / "sample_metadata.json"

    if not vectors_path.exists():
        raise FileNotFoundError(
            f"Embedding vectors not found: {vectors_path}. "
            "Run build_knowledge_store.py first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}. "
            "Run build_knowledge_store.py first."
        )

    vectors = np.load(vectors_path).astype(np.float32)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata: List[Dict] = json.load(f)

    if vectors.ndim != 2:
        raise ValueError(f"Invalid vectors shape: {vectors.shape}")
    if len(metadata) != vectors.shape[0]:
        raise ValueError(
            f"Metadata size ({len(metadata)}) does not match vector rows ({vectors.shape[0]})."
        )

    # Ensure store vectors are normalized for cosine via dot-product.
    vectors = l2_normalize_rows(vectors)

    device = resolve_device(device_arg)
    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
    )
    model.max_seq_length = max_seq_length

    input_text = build_input_text(query, code, max_input_chars)
    query_vec = model.encode(
        [input_text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    query_vec = l2_normalize_rows(query_vec)[0]

    scores = vectors @ query_vec

    top_k = max(1, min(top_k_arg, len(scores)))
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results = [map_result(metadata[idx], float(scores[idx])) for idx in top_indices]
    return {"query": query, "top_k": top_k, "results": results}


def main() -> None:
    # No CLI arguments: use editable config block above.
    if len(sys.argv) == 1:
        output = retrieve(
            query=QUERY,
            code=CODE,
            knowledge_dir=KNOWLEDGE_DIR,
            model_name=MODEL_NAME,
            device_arg=DEVICE,
            max_seq_length=MAX_SEQ_LENGTH,
            max_input_chars=MAX_INPUT_CHARS,
            top_k_arg=TOP_K,
        )
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # Optional CLI mode for advanced usage.
    parser = argparse.ArgumentParser(
        description=(
            "Embed query+code and retrieve the most similar vulnerability "
            "records from local knowledge store."
        )
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--knowledge-dir", default=KNOWLEDGE_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--max-input-chars", type=int, default=MAX_INPUT_CHARS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    output = retrieve(
        query=args.query,
        code=args.code,
        knowledge_dir=args.knowledge_dir,
        model_name=args.model_name,
        device_arg=args.device,
        max_seq_length=args.max_seq_length,
        max_input_chars=args.max_input_chars,
        top_k_arg=args.top_k,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
