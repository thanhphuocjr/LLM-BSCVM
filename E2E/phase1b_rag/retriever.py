"""
Phase 1B - local RAG retriever: TF-IDF (char 3-5 gram) similarity over the 11,869 knowledge
cases, gated by the 37 SWC signal agents (swc_agents.py). A finding is reported only when its
confidence reaches VULNERABLE_CONFIDENCE (0.60).
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot

from E2E.common.io_utils import read_json
from E2E.config import KNOWLEDGE_BASE_DIR, RISK_ORDER
from E2E.phase1b_rag.swc_agents import (
    build_agents,
    extract_signal_code,
    find_signal_hits,
    load_swc_registry,
)


# Kẹp giá trị vào đoạn [0, 1].
def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


MIN_EXACT_MATCH_CHARS = 80
MIN_SEMANTIC_ONLY_SIMILARITY = 0.60
CANDIDATE_CONFIDENCE_CAP = 0.59
MIN_EXACT_MATCH_LENGTH_RATIO = 0.8
VULNERABLE_CONFIDENCE = 0.60
MAX_FINDINGS_PER_UNIT = 10


# Kiểm tra đoạn mã cần xét gần như TRÙNG KHỚP một hàm lỗi đã biết trong KB: cả hai >= 80 ký tự, cái này chứa cái kia
# và độ dài chênh lệch không quá 20%. Mục đích: tránh hàm getter ngắn "khớp" với rất nhiều ca trong KB.
def _is_exact_code_match(known_code: str, query_code: str) -> bool:
    known = re.sub(r"\s+", " ", known_code).strip()
    query = re.sub(r"\s+", " ", query_code).strip()
    if len(known) < MIN_EXACT_MATCH_CHARS or len(query) < MIN_EXACT_MATCH_CHARS:
        return False
    if known not in query and query not in known:
        return False
    return min(len(known), len(query)) / max(len(known), len(query)) >= MIN_EXACT_MATCH_LENGTH_RATIO


# Bộ truy hồi RAG chạy local: TF-IDF (char 3-5 gram) trên 11,869 ca lỗ hổng + 37 SWC agent làm "cổng" tín hiệu.
# Kết luận lỗ hổng chỉ khi có tín hiệu mã tất định (regex của agent) hoặc trùng khớp gần như y hệt một ca đã biết.
class LocalRAGRetriever:
    """TF-IDF retrieval over SWC guidance and empirical vulnerability cases, gated by deterministic code signals."""

    # Nạp các artifact đã build sẵn trong data/knowledge_base/built/: swc_registry.json, vectorizer TF-IDF,
    # ma trận TF-IDF, knowledge_items.json; dựng 37 SWC agent; lập chỉ mục SWC -> các dòng trong ma trận.
    def __init__(self, store_dir: str | Path = KNOWLEDGE_BASE_DIR) -> None:
        self.store_dir = Path(store_dir)
        self.records = load_swc_registry(self.store_dir / "swc_registry.json")
        self.by_id = {record["id"]: record for record in self.records}
        self.agents = build_agents(self.records)
        self.vectorizer = joblib.load(self.store_dir / "swc_vectorizer.joblib")
        # Cosine similarity = tích vô hướng của các vector đã chuẩn hoá L2. Phía KB được chuẩn hoá và chuyển vị
        # MỘT lần ở đây (cùng phép tính với sklearn cosine_similarity, vốn làm lại việc này với ma trận
        # 11,869 x 60,000 ở mỗi lần gọi - khoảng 45 ms cho mỗi unit).
        self._kb_vectors_t = normalize(load_npz(self.store_dir / "swc_matrix.npz")).T.tocsr()

        # Knowledge items, one per row of the matrix
        self.knowledge_items: list[dict[str, Any]] = read_json(self.store_dir / "knowledge_items.json")

        # swc_id -> matrix row indices (ascending)
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, item in enumerate(self.knowledge_items):
            swc_id = item.get("swc") or item.get("id")
            if swc_id:
                grouped[swc_id].append(idx)
        self.swc_to_indices: dict[str, np.ndarray] = {swc: np.array(rows) for swc, rows in grouped.items()}

    # Độ giống (cosine) giữa đoạn mã và từng ca trong KB, một mảng 11,869 phần tử.
    def _similarities(self, code: str) -> np.ndarray:
        query = normalize(self.vectorizer.transform([code]))
        return safe_sparse_dot(query, self._kb_vectors_t, dense_output=True)[0]

    # Lấy top-k ca trong KB giống đoạn mã nhất (cosine similarity TF-IDF), kèm root cause, cách sửa, mã lỗi/mã đã sửa.
    # Mục đích: cung cấp ví dụ thực tế cho Fixer (không dùng để kết luận lỗi).
    def retrieve_context(self, code: str, top_k: int = 5) -> list[dict[str, Any]]:
        similarities = self._similarities(code)
        indices = similarities.argsort()[::-1][: max(1, top_k)]
        results: list[dict[str, Any]] = []
        for index in indices:
            item = self.knowledge_items[index]
            swc_id = item.get("swc") or item.get("id")
            record = self.by_id.get(swc_id, {})
            results.append(
                {
                    "swc": swc_id,
                    "title": item.get("title") or record.get("title", swc_id),
                    "root_cause": item.get("root_cause") or record.get("root_cause", ""),
                    "description": item.get("description") or record.get("description", ""),
                    "remediation": item.get("remediation") or record.get("remediation", ""),
                    "vulnerable_code": item.get("vulnerable_code", ""),
                    "fixed_code": item.get("fixed_code", ""),
                    "similarity": round(float(similarities[index]), 6),
                }
            )
        return results

    # [CỐT LÕI RAG] Với mỗi SWC agent:
    #   - tìm tín hiệu regex (find_signal_hits) + ca giống nhất cùng SWC (semantic_score) + có trùng khớp chính xác không;
    #   - không có tín hiệu, similarity < 0.28 và không trùng khớp -> bỏ qua;
    #   - confidence: trùng khớp = 0.95; có tín hiệu = 0.55*độ mạnh tín hiệu + 0.45*similarity;
    #     chỉ có similarity thì bị chặn <= 0.59 nếu similarity < 0.60 (similarity đơn thuần chỉ đúng ~51%).
    #   - chỉ báo finding khi confidence >= 0.60 (dưới ngưỡng chỉ là ứng viên, không được báo).
    # run_rag gắn thêm source="rag", unit_id, dòng. Trả về (tối đa 10) finding sắp theo mức rủi ro + confidence.
    def analyze(self, code: str) -> list[dict[str, Any]]:
        similarities = self._similarities(code)
        signal_code = extract_signal_code(code)
        findings: list[dict[str, Any]] = []

        for agent in self.agents:
            signal_hits = find_signal_hits(signal_code, agent.signal_patterns, agent.swc_id)
            rows = self.swc_to_indices.get(agent.swc_id)
            best_item: dict[str, Any] | None = None
            semantic_score = 0.0
            if rows is not None:
                best_row = int(rows[np.argmax(similarities[rows])])  # first row with the highest similarity
                semantic_score = float(similarities[best_row])
                best_item = self.knowledge_items[best_row]

            # Exact match with a known empirical vulnerable function. Both sides must be
            # non-trivial and near-identical in size: a short getter is a substring of many
            # knowledge items, and a tiny knowledge snippet is a substring of many units.
            exact_code_match = bool(best_item) and _is_exact_code_match(
                str(best_item.get("vulnerable_code") or ""), code
            )

            # Allow either deterministic regex hits, strong semantic match, or exact code match
            if not signal_hits and semantic_score < 0.28 and not exact_code_match:
                continue

            if exact_code_match:
                confidence = 0.95
            elif signal_hits:
                signal_strength = _clamp(len(signal_hits) / max(1, agent.min_signal_hits))
                confidence = _clamp(0.55 * signal_strength + 0.45 * min(1.0, semantic_score * 2.5))
            else:
                # Similarity alone (no deterministic code signal) was right only ~51% of the
                # time on the function-level test split; it may only confirm a finding when
                # the match is a near-duplicate, otherwise it stays a candidate.
                confidence = _clamp(min(1.0, semantic_score * 1.8))
                if semantic_score < MIN_SEMANTIC_ONLY_SIMILARITY:
                    confidence = min(confidence, CANDIDATE_CONFIDENCE_CAP)
            if confidence < VULNERABLE_CONFIDENCE:
                continue

            record = self.by_id.get(agent.swc_id, {})
            findings.append(
                {
                    "swc": agent.swc_id,
                    "title": agent.title,
                    "severity": agent.severity,
                    "confidence": round(confidence, 6),
                    "root_cause": (best_item.get("root_cause") if best_item else "") or record.get("root_cause", ""),
                    "description": (best_item.get("description") if best_item else "") or record.get("description", agent.description),
                    "remediation": (best_item.get("remediation") if best_item else "") or record.get("remediation", agent.remediation),
                    "vulnerable_code": best_item.get("vulnerable_code", "") if best_item else "",
                    "fixed_code": best_item.get("fixed_code", "") if best_item else "",
                }
            )

        findings.sort(
            key=lambda item: (RISK_ORDER.get(item["severity"], 0), item["confidence"]),
            reverse=True,
        )
        return findings[:MAX_FINDINGS_PER_UNIT]


# Một retriever dùng chung cho cả tiến trình: nạp KB (~0.5 s) một lần thay vì ở mỗi lần chạy RAG
# (Phase 1B, mỗi attempt của vòng sửa, kiểm chứng cuối). Các phương thức chỉ đọc nên dùng chung an toàn.
@lru_cache(maxsize=1)
def default_retriever() -> LocalRAGRetriever:
    return LocalRAGRetriever()
