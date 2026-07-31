import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "processed" / "augmented_solodit_reason.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "rag" / "knowledge_store"


def format_list_field(label: str, value) -> str:
    if isinstance(value, list):
        if not value:
            return f"{label}:"
        if all(isinstance(item, dict) for item in value):
            parts = []
            for item in value:
                vulnerable = str(item.get("vulnerable", "") or "").strip()
                fixed = str(item.get("fixed", "") or "").strip()
                parts.append(f"vulnerable: {vulnerable} fixed: {fixed}".strip())
            return f"{label}: " + " | ".join(parts)
        return f"{label}: " + " | ".join(str(item).strip() for item in value if str(item).strip())
    return f"{label}: {str(value or '').strip()}"


def build_document(sample: Dict, max_chars: int) -> str:
    swc = str(sample.get("Swc", "") or "").strip()
    title = str(sample.get("Title", "") or "").strip()
    description = str(sample.get("Description", "") or "").strip()
    remediation = str(sample.get("Remediation", "") or "").strip()

    doc = "\n".join(
        [
            f"SWC: {swc}",
            f"Title: {title}",
            format_list_field("Weakness aliases", sample.get("WeaknessAliases")),
            format_list_field("Code patterns", sample.get("CodePatterns")),
            format_list_field("Query patterns", sample.get("QueryPatterns")),
            format_list_field("Detection hints", sample.get("DetectionHints")),
            f"Description: {description}",
            f"Remediation: {remediation}",
            format_list_field("Best practice checklist", sample.get("BestPracticeChecklist")),
            format_list_field("Code examples", sample.get("CodeExamples")),
            f"Source: {sample.get('Source', '')}",
            f"Augmentation type: {sample.get('AugmentationType', '')}",
            f"Original SWC: {sample.get('OriginalSwc', '')}",
            f"Original title: {sample.get('OriginalTitle', '')}",
            f"Inferred SWC: {sample.get('InferredSwc', '')}",
        ]
    )
    return doc[:max_chars]


def normalize_sample(sample: Dict, idx: int) -> Dict:
    return {
        "doc_id": sample.get("DocId", idx),
        "Swc": sample.get("Swc"),
        "Title": sample.get("Title"),
        "Description": sample.get("Description"),
        "Remediation": sample.get("Remediation"),
        "WeaknessAliases": sample.get("WeaknessAliases", []),
        "CodePatterns": sample.get("CodePatterns", []),
        "DetectionHints": sample.get("DetectionHints", []),
        "BestPracticeChecklist": sample.get("BestPracticeChecklist", []),
        "CodeExamples": sample.get("CodeExamples", []),
        "QueryPatterns": sample.get("QueryPatterns", []),
        "Source": sample.get("Source", ""),
        "AugmentationType": sample.get("AugmentationType", ""),
        "OriginalSwc": sample.get("OriginalSwc", ""),
        "OriginalTitle": sample.get("OriginalTitle", ""),
        "InferredSwc": sample.get("InferredSwc", ""),
        "SwcCorrected": sample.get("SwcCorrected", False),
    }


def load_samples(input_path: Path) -> List[Dict]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Input JSON must be a list, got: {type(data).__name__}")

    cleaned = [item for item in data if isinstance(item, dict)]
    if not cleaned:
        raise ValueError("Input JSON has no valid dictionary samples.")

    return cleaned


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local embedding-based knowledge store from Solodit reason dataset."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to input JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save knowledge store artifacts.",
    )
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-m3",
        help="Local Hugging Face embedding model name.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory for downloaded model files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for local embedding inference.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference: auto | cpu | cuda | mps.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=15000,
        help="Max characters per sample before embedding.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Max input sequence length used by the embedding model.",
    )
    parser.add_argument(
        "--use-safetensors",
        action="store_true",
        help="Force loading model weights from safetensors if available.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    samples = load_samples(input_path)
    metadata = [normalize_sample(sample, idx) for idx, sample in enumerate(samples)]
    documents = [build_document(sample, max_chars=args.max_chars) for sample in samples]

    device = resolve_device(args.device)
    model = SentenceTransformer(
        args.model_name,
        device=device,
        cache_folder=args.cache_dir,
        trust_remote_code=True,
        model_kwargs={"use_safetensors": args.use_safetensors},
    )
    model.max_seq_length = args.max_seq_length

    embeddings = model.encode(
        documents,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    embeddings_norm = l2_normalize_rows(embeddings)

    vectors_path = output_dir / "sample_embeddings_local.npy"
    vectors_norm_path = output_dir / "sample_embeddings_local_l2.npy"
    metadata_path = output_dir / "sample_metadata.json"
    corpus_path = output_dir / "sample_corpus.json"
    stats_path = output_dir / "store_stats.json"

    np.save(vectors_path, embeddings)
    np.save(vectors_norm_path, embeddings_norm)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with corpus_path.open("w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    stats = {
        "input_file": str(input_path),
        "num_samples": len(samples),
        "embedding_model": args.model_name,
        "device": device,
        "max_seq_length": args.max_seq_length,
        "embedding_dim": int(embeddings.shape[1]),
        "matrix_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "artifacts": {
            "vectors": str(vectors_path),
            "vectors_l2": str(vectors_norm_path),
            "metadata": str(metadata_path),
            "corpus": str(corpus_path),
        },
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Knowledge store created successfully (Local embedding vectors).")
    print(f"- Samples: {len(samples)}")
    print(f"- Model: {args.model_name}")
    print(f"- Device: {device}")
    print(f"- Embedding dim: {embeddings.shape[1]}")
    print(f"- Vectors: {vectors_path}")
    print(f"- L2 vectors: {vectors_norm_path}")
    print(f"- Metadata: {metadata_path}")
    print(f"- Corpus: {corpus_path}")
    print(f"- Stats: {stats_path}")


if __name__ == "__main__":
    main()
