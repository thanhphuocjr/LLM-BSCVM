"""
build_tfidf_corpus.py
=====================
Bước 1 của RAG Pipeline: Tiền xử lý + Vector hóa TF-IDF Smart Contract Corpus

Pipeline:
  JSON corpus → [Chunking theo function] → [TF-IDF fit] → Lưu model + matrix + metadata

Cách chạy (dùng path mặc định):
    python3 build_tfidf_corpus.py

Override path:
    python3 build_tfidf_corpus.py \
        --input  /path/to/smart_contract_corpus.json \
        --outdir /path/to/output_dir

Output files:
    tfidf_vectorizer.joblib   – TfidfVectorizer đã fit (dùng lại để transform query)
    tfidf_matrix.npz          – Sparse matrix TF-IDF của toàn bộ corpus (scipy)
    corpus_metadata.json      – Metadata của từng snippet (label, source_index, snippet_index)
"""

import json
import re
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# ══════════════════════════════════════════════════════════════════
# ★ CẤU HÌNH PATH MẶC ĐỊNH
# ══════════════════════════════════════════════════════════════════
DEFAULT_INPUT  = "/Users/phuocthanh/Documents/RAG/RAG_Dataset/smart_contract_corpus.json"
DEFAULT_OUTDIR = "/Users/phuocthanh/Documents/RAG/RAG_Dataset/tfidf_output"

# ══════════════════════════════════════════════════════════════════
# Cấu hình TF-IDF
# ══════════════════════════════════════════════════════════════════
TFIDF_CONFIG = {
    "max_features": 8000,       # ~8k features: cân bằng giữa chất lượng và RAM
    "stop_words":   "english",  # loại bỏ stop words tiếng Anh
    "ngram_range":  (1, 2),     # unigram + bigram để bắt context tốt hơn
    "min_df":       2,          # bỏ token chỉ xuất hiện 1 lần (noise)
    "max_df":       0.95,       # bỏ token xuất hiện >95% document (quá phổ biến)
    "sublinear_tf": True,       # log(1+tf) thay vì tf thô → giảm bias token lặp nhiều
    "analyzer":     "word",
    "token_pattern": r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]{1,}\b",  # match Solidity identifiers
}

# Snippet ngắn hơn ngưỡng này sẽ bị loại
MIN_SNIPPET_LENGTH = 10


# ══════════════════════════════════════════════════════════════════
# 1. Argument parser
# ══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Build TF-IDF corpus for smart contracts.")
    p.add_argument("--input",  default=DEFAULT_INPUT,
                   help=f"Path to smart_contract_corpus.json (default: {DEFAULT_INPUT})")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR,
                   help=f"Output directory (default: {DEFAULT_OUTDIR})")
    p.add_argument("--max-features", type=int, default=TFIDF_CONFIG["max_features"],
                   help="TF-IDF max_features (default: 8000)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# 2. Load JSON
# ══════════════════════════════════════════════════════════════════
def load_corpus(path: str) -> list:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Loading corpus: {path}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("[ERROR] JSON phải là một array.", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Loaded {len(data):,} records.")
    return data


# ══════════════════════════════════════════════════════════════════
# 3. Trích xuất code thuần từ markdown block
#    Corpus có format: "The function X ... ```Solidiy\n<code>\n```"
#    → Lấy phần code bên trong backtick block, fallback về full text
# ══════════════════════════════════════════════════════════════════
def extract_raw_code(text: str) -> str:
    """
    Tách code Solidity ra khỏi markdown wrapper.
    Hỗ trợ: ```Solidiy, ```Solidity, ```solidity, ``` (no language)
    """
    # Pattern bắt nội dung bên trong ```...``` (non-greedy)
    pattern = r"```(?:solidiy|solidity|sol|)?\s*\n?(.*?)```"
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    if matches:
        # Ghép tất cả block code lại (một số record có nhiều block)
        return "\n\n".join(m.strip() for m in matches if m.strip())
    # Fallback: dùng toàn bộ text nếu không có backtick block
    return text.strip()


# ══════════════════════════════════════════════════════════════════
# 4. Chunking: tách code thành các function snippet riêng lẻ
# ══════════════════════════════════════════════════════════════════

# Regex nhận diện phần đầu của một hàm Solidity
# Bắt: function <name>(...) [visibility] [modifiers] { ... }
_FUNC_HEADER = re.compile(
    r"""
    (?:^|\n)                         # bắt đầu dòng
    [ \t]*                           # indent tùy ý
    (?:                              # optional: visibility/modifier trước từ khoá function
        (?:public|private|internal|external|pure|view|payable|virtual|override|
           constructor|receive|fallback|abstract)
        \s+
    )*
    function\s+\w+\s*\(              # "function tên("
    """,
    re.VERBOSE | re.IGNORECASE,
)


def split_into_functions(code: str) -> list[str]:
    """
    Chia code thành các function snippet dựa vào dấu ngoặc nhọn.
    Trả về list các string, mỗi string là 1 function.
    """
    snippets = []

    # Tìm tất cả vị trí bắt đầu của function header
    headers = [m.start() for m in _FUNC_HEADER.finditer(code)]

    if not headers:
        # Không tìm thấy function nào → trả nguyên code
        return [code] if len(code) >= MIN_SNIPPET_LENGTH else []

    for i, start in enumerate(headers):
        # Xác định đoạn code của function này: từ header đến header tiếp theo
        end = headers[i + 1] if i + 1 < len(headers) else len(code)
        candidate = code[start:end].strip()

        # Trích phần thân hàm bằng cách đếm ngoặc nhọn
        snippet = extract_function_body(candidate)
        if snippet and len(snippet) >= MIN_SNIPPET_LENGTH:
            snippets.append(snippet)

    return snippets


def extract_function_body(code_segment: str) -> str:
    """
    Trích hàm hoàn chỉnh (từ chữ 'function' đến dấu '}' đóng tương ứng).
    Dùng thuật toán đếm ngoặc nhọn để xác định phần thân.
    """
    # Tìm vị trí '{' đầu tiên
    brace_start = code_segment.find("{")
    if brace_start == -1:
        # Hàm không có thân (abstract / interface) → giữ nguyên
        return code_segment.strip()

    depth  = 0
    in_str = False
    str_ch = None
    i      = brace_start

    while i < len(code_segment):
        ch = code_segment[i]

        # Xử lý string literals (bỏ qua ngoặc bên trong string)
        if in_str:
            if ch == "\\" :          # escape character
                i += 2
                continue
            if ch == str_ch:
                in_str = False
        else:
            if ch in ('"', "'", "`"):
                in_str = True
                str_ch = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # Đã tìm được dấu đóng ngoặc khớp → kết thúc hàm
                    return code_segment[: i + 1].strip()
        i += 1

    # Nếu không đóng ngoặc hoàn chỉnh, trả về phần đã có
    return code_segment.strip()


# ══════════════════════════════════════════════════════════════════
# 5. Main chunking loop
# ══════════════════════════════════════════════════════════════════
def build_snippets(corpus: list) -> tuple[list[str], list[dict]]:
    """
    Duyệt toàn bộ corpus, tạo snippet và metadata tương ứng.

    Returns:
        snippets  – list[str]  : nội dung từng snippet
        metadata  – list[dict] : {source_index, snippet_index, label, original_length}
    """
    snippets: list[str]  = []
    metadata: list[dict] = []

    skipped_short = 0
    skipped_empty = 0

    for src_idx, record in enumerate(corpus):
        code  = record.get("Code", "") or ""
        label = record.get("Label", "Unknown")

        # Bước 2a: tách code khỏi markdown
        raw_code = extract_raw_code(code)

        if not raw_code.strip():
            skipped_empty += 1
            continue

        # Bước 2b: chia thành function snippets
        func_snippets = split_into_functions(raw_code)

        if not func_snippets:
            # Không tìm được function → dùng nguyên raw_code nếu đủ dài
            if len(raw_code) >= MIN_SNIPPET_LENGTH:
                func_snippets = [raw_code]
            else:
                skipped_short += 1
                continue

        for snip_idx, snippet in enumerate(func_snippets):
            snippets.append(snippet)
            metadata.append({
                "source_index":   src_idx,
                "snippet_index":  snip_idx,
                "label":          label,
                "original_length": len(raw_code),
                "snippet_length":  len(snippet),
            })

    print(f"[INFO] Chunking xong:")
    print(f"       Tổng snippets      : {len(snippets):,}")
    print(f"       Records bị bỏ qua :")
    print(f"         - Code rỗng      : {skipped_empty}")
    print(f"         - Snippet quá ngắn: {skipped_short}")

    # Thống kê label
    label_count: dict = {}
    for m in metadata:
        label_count[m["label"]] = label_count.get(m["label"], 0) + 1
    print(f"       Label distribution: {label_count}")

    return snippets, metadata


# ══════════════════════════════════════════════════════════════════
# 6. TF-IDF Vectorization
# ══════════════════════════════════════════════════════════════════
def vectorize(snippets: list[str], max_features: int):
    """
    Fit TfidfVectorizer trên toàn bộ snippets.

    Returns:
        vectorizer   – TfidfVectorizer đã fit
        tfidf_matrix – scipy sparse matrix (n_snippets × n_features)
    """
    print(f"\n[INFO] Bắt đầu TF-IDF vectorization ...")
    print(f"       Số snippets  : {len(snippets):,}")
    print(f"       max_features : {max_features:,}")

    config = {**TFIDF_CONFIG, "max_features": max_features}
    vectorizer = TfidfVectorizer(**config)

    t0 = time.time()
    tfidf_matrix = vectorizer.fit_transform(snippets)
    elapsed = time.time() - t0

    print(f"[INFO] Vectorization hoàn tất trong {elapsed:.1f}s")
    print(f"       Matrix shape : {tfidf_matrix.shape}  "
          f"(snippets × features)")
    print(f"       Vocabulary size : {len(vectorizer.vocabulary_):,}")

    # Ước tính bộ nhớ
    mem_mb = (tfidf_matrix.data.nbytes +
               tfidf_matrix.indices.nbytes +
               tfidf_matrix.indptr.nbytes) / 1024 / 1024
    print(f"       Matrix size in RAM : {mem_mb:.1f} MB (sparse)")

    return vectorizer, tfidf_matrix


# ══════════════════════════════════════════════════════════════════
# 7. Lưu kết quả
# ══════════════════════════════════════════════════════════════════
def save_artifacts(
    outdir: Path,
    vectorizer,
    tfidf_matrix,
    snippets: list[str],
    metadata: list[dict],
):
    outdir.mkdir(parents=True, exist_ok=True)

    # 7a. Lưu TF-IDF model
    vec_path = outdir / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vec_path, compress=3)
    print(f"\n[SAVE] Vectorizer    → {vec_path}  ({vec_path.stat().st_size/1024:.0f} KB)")

    # 7b. Lưu sparse matrix
    mat_path = outdir / "tfidf_matrix.npz"
    sp.save_npz(str(mat_path), tfidf_matrix)
    print(f"[SAVE] TF-IDF matrix → {mat_path}  ({mat_path.stat().st_size/1024:.0f} KB)")

    # 7c. Lưu metadata (kèm snippet text để tra cứu nhanh)
    meta_path = outdir / "corpus_metadata.json"
    meta_with_text = [
        {**m, "snippet_text": snippets[i]}
        for i, m in enumerate(metadata)
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_with_text, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Metadata      → {meta_path}  ({meta_path.stat().st_size/1024:.0f} KB)")

    # 7d. Lưu config summary
    summary = {
        "total_records":   len(set(m["source_index"] for m in metadata)),
        "total_snippets":  len(snippets),
        "matrix_shape":    list(tfidf_matrix.shape),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "tfidf_config":    {k: str(v) for k, v in TFIDF_CONFIG.items()},
        "label_distribution": {
            label: sum(1 for m in metadata if m["label"] == label)
            for label in set(m["label"] for m in metadata)
        },
        "files": {
            "vectorizer":  str(vec_path),
            "matrix":      str(mat_path),
            "metadata":    str(meta_path),
        },
    }
    sum_path = outdir / "build_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Summary       → {sum_path}")

    return summary


# ══════════════════════════════════════════════════════════════════
# 8. Demo: kiểm tra retrieval nhanh (cosine similarity)
# ══════════════════════════════════════════════════════════════════
def demo_retrieval(vectorizer, tfidf_matrix, metadata: list[dict], top_k: int = 3):
    """
    Test nhanh: lấy snippet đầu tiên làm query, tìm top-k tương đồng nhất.
    Dùng để xác nhận pipeline hoạt động đúng trước khi deploy.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    print("\n" + "═" * 60)
    print("  DEMO: Cosine Similarity Retrieval (top-k=3)")
    print("═" * 60)

    # Dùng snippet index 0 làm query mẫu
    query_vec = tfidf_matrix[0]
    scores    = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx   = np.argsort(scores)[::-1][:top_k + 1]   # +1 vì snippet 0 tự match chính nó

    print(f"Query snippet (source_index={metadata[0]['source_index']}, "
          f"label={metadata[0]['label']}):")
    print(f"  {metadata[0].get('snippet_text','')[:120]}...\n")

    print(f"Top-{top_k} kết quả tương đồng nhất:")
    shown = 0
    for idx in top_idx:
        if idx == 0:
            continue   # bỏ chính nó
        m = metadata[idx]
        print(f"  [{shown+1}] score={scores[idx]:.4f}  "
              f"label={m['label']}  "
              f"source_idx={m['source_index']}  "
              f"snippet_idx={m['snippet_index']}")
        print(f"      {m.get('snippet_text','')[:100]}...")
        shown += 1
        if shown >= top_k:
            break


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    args   = parse_args()
    outdir = Path(args.outdir)

    print("=" * 60)
    print("  Smart Contract Corpus — TF-IDF Builder")
    print("=" * 60)

    # Step 1: Load
    corpus = load_corpus(args.input)

    # Step 2: Chunk
    print(f"\n[INFO] Bắt đầu chunking (MIN_SNIPPET_LENGTH={MIN_SNIPPET_LENGTH})...")
    snippets, metadata = build_snippets(corpus)

    # Step 3: Vectorize
    vectorizer, tfidf_matrix = vectorize(snippets, args.max_features)

    # Step 4: Save
    summary = save_artifacts(outdir, vectorizer, tfidf_matrix, snippets, metadata)

    # Step 5: Demo
    demo_retrieval(vectorizer, tfidf_matrix, metadata)

    print("\n" + "=" * 60)
    print(f"  ✓ Hoàn tất! Tất cả file đã lưu vào: {outdir.resolve()}")
    print("=" * 60)
    print()
    print("Cách tải lại để dùng trong RAG pipeline:")
    print("""
    import joblib, scipy.sparse as sp, json

    vectorizer   = joblib.load('tfidf_vectorizer.joblib')
    tfidf_matrix = sp.load_npz('tfidf_matrix.npz')
    with open('corpus_metadata.json') as f:
        metadata = json.load(f)

    # Transform một query mới:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    query = "function transfer(address to, uint256 amount) public returns (bool)"
    query_vec = vectorizer.transform([query])
    scores    = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_k_idx = np.argsort(scores)[::-1][:5]

    for idx in top_k_idx:
        print(f"score={scores[idx]:.4f}  label={metadata[idx]['label']}")
        print(metadata[idx]['snippet_text'][:200])
        print()
    """)


if __name__ == "__main__":
    main()