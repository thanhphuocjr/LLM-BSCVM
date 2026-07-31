"""
tfidf_retriever.py
==================
Bước 1.2 của RAG Pipeline: Cosine Similarity Search + Weighted Vulnerability Scoring

Dùng mô hình TF-IDF đã build từ build_tfidf_corpus.py để:
  1. Vector hóa đoạn code mới (Test Code)
  2. Tìm Top-K snippet tương đồng nhất trong corpus
  3. Tính Vulnerability Probability dựa trên weighted scoring
  4. Trả về kết quả có cấu trúc để tích hợp vào RAG pipeline

Cách chạy độc lập (demo):
    python3 tfidf_retriever.py

Cách dùng như module:
    from tfidf_retriever import SmartContractRetriever
    retriever = SmartContractRetriever()
    result    = retriever.retrieve(code_snippet, top_k=5)
"""

import json
import re
import time
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════════════════
# ★ CẤU HÌNH PATH MẶC ĐỊNH — khớp với output của build_tfidf_corpus.py
# ══════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TFIDF_DIR = str(PROJECT_ROOT / "rag" / "tfidf_store")

# Tên file (tương đối trong thư mục trên)
FILE_VECTORIZER = "tfidf_vectorizer.joblib"
FILE_MATRIX     = "tfidf_matrix.npz"
FILE_METADATA   = "corpus_metadata.json"

# ══════════════════════════════════════════════════════════════════
# Cấu hình Weighted Scoring
# ══════════════════════════════════════════════════════════════════
# Trọng số giảm dần cho Top-5, tổng = 1.0
DEFAULT_WEIGHTS = [0.50, 0.20, 0.15, 0.10, 0.05]

# Ngưỡng phân loại rủi ro cuối cùng
RISK_THRESHOLDS = {
    "Critical": 0.60, 
    "High":     0.40,  
    "Medium":   0.20,   
    "Low":      0.00,  
}


# ══════════════════════════════════════════════════════════════════
# Data classes — cấu trúc kết quả trả về
# ══════════════════════════════════════════════════════════════════
@dataclass
class RetrievedSnippet:
    """Một snippet được tìm thấy trong corpus."""
    rank:           int
    similarity:     float
    label:          str          # "Safe" | "Vulnerable"
    source_index:   int          # index trong corpus gốc
    snippet_index:  int          # thứ tự snippet trong contract đó
    snippet_length: int
    snippet_text:   str
    weight:         float = 0.0  # trọng số gán theo rank


@dataclass
class RetrievalResult:
    """Kết quả đầy đủ của một lần retrieval."""
    query_snippet:          str
    top_k:                  int
    retrieved:              list[RetrievedSnippet]
    vulnerability_score:    float          # 0.0 – 1.0
    vulnerability_prob_pct: float          # 0.0 – 100.0
    risk_level:             str            # "Low" | "Medium" | "High" | "Critical"
    vulnerable_count:       int            # số snippet Vulnerable trong Top-K
    elapsed_ms:             float


# ══════════════════════════════════════════════════════════════════
# Chunking helper (tái sử dụng logic từ build_tfidf_corpus.py)
# ══════════════════════════════════════════════════════════════════
_FUNC_HEADER = re.compile(
    r"""
    (?:^|\n)
    [ \t]*
    (?:(?:public|private|internal|external|pure|view|payable|virtual|override|
           constructor|receive|fallback|abstract)\s+)*
    function\s+\w+\s*\(
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _extract_raw_code(text: str) -> str:
    """Tách code Solidity ra khỏi markdown wrapper (```...```)."""
    pattern = r"```(?:solidiy|solidity|sol|)?\s*\n?(.*?)```"
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    return "\n\n".join(m.strip() for m in matches if m.strip()) if matches else text.strip()


def _extract_function_body(segment: str) -> str:
    """Trích function hoàn chỉnh bằng cách đếm ngoặc nhọn."""
    brace_start = segment.find("{")
    if brace_start == -1:
        return segment.strip()

    depth, in_str, str_ch, i = 0, False, None, brace_start
    while i < len(segment):
        ch = segment[i]
        if in_str:
            if ch == "\\":
                i += 2; continue
            if ch == str_ch:
                in_str = False
        else:
            if ch in ('"', "'", "`"):
                in_str, str_ch = True, ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return segment[: i + 1].strip()
        i += 1
    return segment.strip()


def chunk_code(code: str, min_length: int = 10) -> list[str]:
    """
    Tách một đoạn code Solidity thành list các function snippets.
    Nếu không tìm thấy function, trả về [code] nếu đủ dài.
    """
    raw = _extract_raw_code(code)
    headers = [m.start() for m in _FUNC_HEADER.finditer(raw)]

    if not headers:
        return [raw] if len(raw) >= min_length else []

    snippets = []
    for i, start in enumerate(headers):
        end       = headers[i + 1] if i + 1 < len(headers) else len(raw)
        candidate = raw[start:end].strip()
        body      = _extract_function_body(candidate)
        if body and len(body) >= min_length:
            snippets.append(body)

    return snippets or ([raw] if len(raw) >= min_length else [])


# ══════════════════════════════════════════════════════════════════
# CLASS CHÍNH: SmartContractRetriever
# ══════════════════════════════════════════════════════════════════
class SmartContractRetriever:
    """
    Module tìm kiếm Top-K và tính Vulnerability Score dựa trên TF-IDF.

    Tích hợp vào RAG pipeline của LLM-BSCVM:
        retriever = SmartContractRetriever()
        result    = retriever.retrieve(test_code, top_k=5)

        # Dùng result.vulnerability_prob_pct làm đầu vào cho Detector Agent
        # Dùng result.retrieved để inject context vào LLM prompt
    """

    def __init__(
        self,
        tfidf_dir: str = DEFAULT_TFIDF_DIR,
        weights:   Optional[list[float]] = None,
    ):
        """
        Args:
            tfidf_dir : Thư mục chứa 3 file output của build_tfidf_corpus.py
            weights   : Trọng số cho Top-K (mặc định [0.5, 0.2, 0.15, 0.1, 0.05])
        """
        self.tfidf_dir = Path(tfidf_dir)
        self.weights   = weights or DEFAULT_WEIGHTS
        self._validate_weights()

        print("[SmartContractRetriever] Đang tải model...")
        self._vectorizer    = self._load_vectorizer()
        self._corpus_matrix = self._load_matrix()
        self._metadata      = self._load_metadata()
        print(f"[SmartContractRetriever] Sẵn sàng — "
              f"{len(self._metadata):,} snippets trong corpus.")

    # ──────────────────────────────────────────────
    # Private: Load artifacts
    # ──────────────────────────────────────────────
    def _load_vectorizer(self):
        path = self.tfidf_dir / FILE_VECTORIZER
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy vectorizer: {path}")
        return joblib.load(path)

    def _load_matrix(self):
        path = self.tfidf_dir / FILE_MATRIX
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy matrix: {path}")
        return sp.load_npz(str(path))

    def _load_metadata(self) -> list[dict]:
        path = self.tfidf_dir / FILE_METADATA
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy metadata: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _validate_weights(self):
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError(
                f"Tổng weights phải = 1.0, hiện tại = {sum(self.weights):.4f}"
            )

    # ──────────────────────────────────────────────
    # Private: Vector hóa query
    # ──────────────────────────────────────────────
    def _vectorize_query(self, text: str):
        """Transform đoạn code mới thành TF-IDF vector."""
        return self._vectorizer.transform([text])

    # ──────────────────────────────────────────────
    # Private: Cosine Similarity + Top-K
    # ──────────────────────────────────────────────
    def _find_top_k(self, query_vec, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Tính cosine similarity và trả về top-k indices + scores.
        Dùng argpartition thay vì argsort toàn bộ → nhanh hơn ~3x với corpus lớn.
        """
        scores = cosine_similarity(query_vec, self._corpus_matrix).flatten()

        # argpartition cho top-k phần tử (không cần sort toàn bộ 101k items)
        if top_k >= len(scores):
            top_idx = np.arange(len(scores))
        else:
            top_idx = np.argpartition(scores, -top_k)[-top_k:]

        # Sort top-k theo score giảm dần
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return top_idx, scores[top_idx]

    # ──────────────────────────────────────────────
    # Private: Weighted Vulnerability Scoring
    # ──────────────────────────────────────────────
    @staticmethod
    def _classify_risk(score: float) -> str:
        for level, threshold in RISK_THRESHOLDS.items():
            if score >= threshold:
                return level
        return "Low"

    def _compute_weighted_score(
        self,
        top_idx:    np.ndarray,
        top_scores: np.ndarray,
        top_k:      int,
    ) -> tuple[float, list[RetrievedSnippet]]:
        """
        Tính Vulnerability Probability theo công thức:
            P = Σ (similarity_i × weight_i)  với i ∈ {Vulnerable snippets trong Top-K}

        Trả về (vulnerability_score, list[RetrievedSnippet])
        """
        # Pad weights nếu top_k < len(weights)
        weights = (self.weights[:top_k] +
                   [0.0] * max(0, top_k - len(self.weights)))
        # Chuẩn hóa lại để tổng = 1.0
        w_sum   = sum(weights)
        weights = [w / w_sum for w in weights] if w_sum > 0 else weights

        retrieved     : list[RetrievedSnippet] = []
        vuln_score    : float = 0.0

        for rank, (idx, sim, w) in enumerate(zip(top_idx, top_scores, weights), start=1):
            meta = self._metadata[int(idx)]
            snippet = RetrievedSnippet(
                rank           = rank,
                similarity     = float(sim),
                label          = meta["label"],
                source_index   = meta["source_index"],
                snippet_index  = meta["snippet_index"],
                snippet_length = meta["snippet_length"],
                snippet_text   = meta.get("snippet_text", ""),
                weight         = w,
            )
            retrieved.append(snippet)

            # Chỉ cộng điểm nếu snippet là Vulnerable
            if meta["label"].lower() == "vulnerable":
                vuln_score += sim * w

        return vuln_score, retrieved

    # ──────────────────────────────────────────────
    # PUBLIC: retrieve — entry point chính
    # ──────────────────────────────────────────────
    def retrieve(self, code: str, top_k: int = 5) -> RetrievalResult:
        """
        Tìm Top-K snippet tương đồng và tính Vulnerability Score.

        Args:
            code  : Đoạn code Solidity cần đánh giá (raw hoặc markdown)
            top_k : Số lượng kết quả trả về (mặc định 5, theo paper LLM-BSCVM)

        Returns:
            RetrievalResult chứa đầy đủ thông tin retrieval + scoring
        """
        t0 = time.perf_counter()

        # Bước 1: Tách function snippets từ code đầu vào
        chunks = chunk_code(code)
        query_text = " ".join(chunks) if chunks else code

        # Bước 2: Vector hóa
        query_vec = self._vectorize_query(query_text)

        # Bước 3: Top-K cosine similarity
        top_idx, top_scores = self._find_top_k(query_vec, top_k)

        # Bước 4: Weighted scoring
        vuln_score, retrieved = self._compute_weighted_score(top_idx, top_scores, top_k)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            query_snippet          = query_text[:500],
            top_k                  = top_k,
            retrieved              = retrieved,
            vulnerability_score    = vuln_score,
            vulnerability_prob_pct = min(vuln_score * 100, 100.0),
            risk_level             = self._classify_risk(vuln_score),
            vulnerable_count       = sum(1 for r in retrieved
                                        if r.label.lower() == "vulnerable"),
            elapsed_ms             = elapsed_ms,
        )

    # ──────────────────────────────────────────────
    # PUBLIC: retrieve_batch — xử lý nhiều contract
    # ──────────────────────────────────────────────
    def retrieve_batch(
        self, codes: list[str], top_k: int = 5
    ) -> list[RetrievalResult]:
        """Xử lý batch nhiều contracts liên tiếp."""
        results = []
        for i, code in enumerate(codes, 1):
            print(f"  [{i}/{len(codes)}] Processing...", end="\r")
            results.append(self.retrieve(code, top_k))
        print()
        return results

    # ──────────────────────────────────────────────
    # PUBLIC: get_context_for_llm — inject vào LLM prompt
    # ──────────────────────────────────────────────
    def get_context_for_llm(
        self,
        result: RetrievalResult,
        max_chars_per_snippet: int = 400,
    ) -> str:
        """
        Định dạng kết quả retrieval thành context block để inject vào prompt
        của Detector Agent trong LLM-BSCVM.

        Usage:
            context = retriever.get_context_for_llm(result)
            prompt  = DETECTOR_PROMPT_TEMPLATE.format(code=test_code, context=context)
        """
        lines = ["<Audited Smart Contracts>"]
        for r in result.retrieved:
            text = r.snippet_text[:max_chars_per_snippet].replace("\n", " ")
            lines.append(
                f"[Rank {r.rank} | sim={r.similarity:.4f} | "
                f"weight={r.weight:.2f} | label={r.label}]\n"
                f"{text}..."
            )
        lines.append("</Audited Smart Contracts>")
        lines.append(
            f"\n<RAG Vulnerability Signal>\n"
            f"Vulnerability Probability: {result.vulnerability_prob_pct:.1f}%  "
            f"(Risk Level: {result.risk_level})\n"
            f"Vulnerable snippets in Top-{result.top_k}: "
            f"{result.vulnerable_count}/{result.top_k}\n"
            f"</RAG Vulnerability Signal>"
        )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Pretty Print helper
# ══════════════════════════════════════════════════════════════════
def print_result(result: RetrievalResult, show_snippet: bool = True):
    """In kết quả ra terminal dạng có cấu trúc, dễ đọc."""
    SEP  = "═" * 65
    sep2 = "─" * 65

    # Header
    print(f"\n{SEP}")
    print(f"  TF-IDF RETRIEVAL RESULT")
    print(f"{SEP}")
    print(f"  Query length : {len(result.query_snippet)} chars")
    print(f"  Top-K        : {result.top_k}")
    print(f"  Elapsed      : {result.elapsed_ms:.1f} ms")
    print(sep2)

    # Top-K table
    LABEL_COLOR = {"Vulnerable": "\033[91m", "Safe": "\033[92m"}
    RESET       = "\033[0m"

    print(f"  {'Rank':<5} {'Sim':>7}  {'Weight':>7}  {'Label':<12}  {'SrcIdx':>7}  Preview")
    print(f"  {'-'*5} {'-'*7}  {'-'*7}  {'-'*12}  {'-'*7}  {'-'*20}")

    for r in result.retrieved:
        color   = LABEL_COLOR.get(r.label, "")
        preview = r.snippet_text[:50].replace("\n", " ")
        label_s = f"{color}{r.label:<12}{RESET}"
        contrib = ""
        if r.label.lower() == "vulnerable":
            contrib = f"  ← +{r.similarity * r.weight:.4f}"
        print(f"  {r.rank:<5} {r.similarity:>7.4f}  {r.weight:>7.2f}  {label_s}  "
              f"{r.source_index:>7}  {preview}...{contrib}")

    print(sep2)

    # Scoring breakdown
    print(f"\n  VULNERABILITY SCORING BREAKDOWN")
    print(f"  Formula: P = Σ(sim_i × weight_i) for Vulnerable snippets")
    print(sep2)

    for r in result.retrieved:
        if r.label.lower() == "vulnerable":
            contrib = r.similarity * r.weight
            print(f"    Rank {r.rank}: {r.similarity:.4f} × {r.weight:.2f} = {contrib:.4f}")

    print(sep2)

    # Final verdict
    RISK_COLOR = {
        "Critical": "\033[91m\033[1m",
        "High":     "\033[91m",
        "Medium":   "\033[93m",
        "Low":      "\033[92m",
    }
    risk_color = RISK_COLOR.get(result.risk_level, "")

    print(f"\n  ┌─ FINAL VERDICT {'─'*47}")
    print(f"  │  Vulnerability Score    : {result.vulnerability_score:.4f}")
    print(f"  │  Vulnerability Prob (%) : {result.vulnerability_prob_pct:.2f}%")
    print(f"  │  Risk Level             : "
          f"{risk_color}{result.risk_level}{RESET}")
    print(f"  │  Vulnerable in Top-{result.top_k}   : "
          f"{result.vulnerable_count}/{result.top_k} snippets")
    print(f"  └{'─'*63}")

    # Optional: show snippet detail
    if show_snippet and result.retrieved:
        print(f"\n  TOP-1 SNIPPET DETAIL (source_index={result.retrieved[0].source_index}):")
        print(f"  {sep2[2:]}")
        wrapped = textwrap.fill(
            result.retrieved[0].snippet_text[:600],
            width=62, initial_indent="  ", subsequent_indent="  "
        )
        print(wrapped)

    print()


# ══════════════════════════════════════════════════════════════════
# DEMO — chạy trực tiếp để kiểm tra
# ══════════════════════════════════════════════════════════════════
# Test cases demo (dict format: name + code)
TEST_CASES = [
    {
        "name": "OverflowExample (potential integer overflow/underflow)",
        "code": """
The function FuseTokenAdapterV1 from the contract wrap \n```Solidiy\nfunction wrap( uint256 amount, address recipient ) external onlyAlchemist returns (uint256) { SafeERC20.safeTransferFrom(underlyingToken, msg.sender, address(this), amount); SafeERC20.safeApprove(underlyingToken, token, amount); uint256 startingBalance = IERC20(token).balanceOf(address(this)); uint256 error; if ((error = ICERC20(token).mint(amount)) != NO_ERROR) { revert FuseError(error); } uint256 endingBalance = IERC20(token).balanceOf(address(this)); uint256 mintedAmount = endingBalance - startingBalance; SafeERC20.safeTransfer(token, recipient, mintedAmount); return mintedAmount; }\n```\n### As a Caller:\nFuseTokenAdapterV1 calls these functions:\n```\nSafeERC20.safeTransferFrom\nFuseTokenAdapterV1.address\nSafeERC20.safeApprove\nFuseTokenAdapterV1.IERC20\nFuseTokenAdapterV1.ICERC20\nFuseTokenAdapterV1.FuseError\nSafeERC20.safeTransfer\nmodifier onlyAlchemist() { if (msg.sender != alchemist) { revert Unauthorized(\"Not alchemist\"); } _; }\n
""",
    },
]


def demo(tfidf_dir: str, top_k: int = 5):
    print("\n" + "█" * 65)
    print("  SmartContractRetriever — DEMO MODE")
    print("█" * 65)

    retriever = SmartContractRetriever(tfidf_dir=tfidf_dir)

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n{'▶' * 3}  TEST CASE {i}: {tc['name']}")
        result = retriever.retrieve(tc["code"], top_k=top_k)
        print_result(result, show_snippet=(i == 1))  # chỉ show snippet ở case 1

        # Cũng in context block dành cho LLM
        if i == 1:
            print("  [LLM CONTEXT BLOCK — inject vào Detector Agent prompt]")
            print("  " + "─" * 63)
            ctx = retriever.get_context_for_llm(result)
            for line in ctx.split("\n"):
                print(f"  {line}")
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TF-IDF Retriever — Cosine Similarity + Weighted Vulnerability Scoring"
    )
    parser.add_argument(
        "--tfidf-dir",
        default=DEFAULT_TFIDF_DIR,
        help=f"Thư mục chứa output của build_tfidf_corpus.py (default: {DEFAULT_TFIDF_DIR})",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Số kết quả trả về (default: 5)")
    args = parser.parse_args()

    demo(tfidf_dir=args.tfidf_dir, top_k=args.top_k)
