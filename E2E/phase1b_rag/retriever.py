"""Phase 1B - bộ truy hồi RAG local: TF-IDF (char 3-5 gram) trên 11,869 ca lỗi, "gác cổng" bởi 37 SWC agent."""

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot

from E2E.common.io_utils import read_json
from E2E.common.regex_patterns import RE_WHITESPACE
from E2E.config import KNOWLEDGE_BASE_DIR, RISK_ORDER
from E2E.phase1b_rag.swc_agents import build_agents, extract_signal_code, find_signal_hits, load_swc_registry

MIN_EXACT_MATCH_CHARS = 80
MIN_SEMANTIC_ONLY_SIMILARITY = 0.60
CANDIDATE_CONFIDENCE_CAP = 0.59
MIN_EXACT_MATCH_LENGTH_RATIO = 0.8
VULNERABLE_CONFIDENCE = 0.60
MAX_FINDINGS_PER_UNIT = 10


# Kẹp giá trị vào đoạn [0, 1].
def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


# Mã đang xét gần như TRÙNG một hàm lỗi đã biết: cả hai >= 80 ký tự, cái này chứa cái kia, độ dài chênh <= 20%.
def _is_exact_code_match(known_code: str, query_code: str) -> bool:
    known, query = RE_WHITESPACE.sub(" ", known_code).strip(), RE_WHITESPACE.sub(" ", query_code).strip()
    if min(len(known), len(query)) < MIN_EXACT_MATCH_CHARS or (known not in query and query not in known):
        return False
    return min(len(known), len(query)) / max(len(known), len(query)) >= MIN_EXACT_MATCH_LENGTH_RATIO


# Bộ truy hồi: nạp KB đã build (registry, vectorizer, ma trận TF-IDF, knowledge_items) + 37 SWC agent.
class LocalRAGRetriever:
    # Nạp artifact trong data/knowledge_base/built/, chuẩn hoá + chuyển vị ma trận KB một lần, chỉ mục SWC -> dòng.
    def __init__(self, store_dir: str | Path = KNOWLEDGE_BASE_DIR) -> None:
        store_dir = Path(store_dir)
        self.records = load_swc_registry(store_dir / "swc_registry.json")
        self.by_id = {record["id"]: record for record in self.records}
        self.agents = build_agents(self.records)
        self.vectorizer = joblib.load(store_dir / "swc_vectorizer.joblib")
        self._kb_vectors_t = normalize(load_npz(store_dir / "swc_matrix.npz")).T.tocsr()
        self.knowledge_items: list[dict] = read_json(store_dir / "knowledge_items.json")
        grouped = defaultdict(list)
        for index, item in enumerate(self.knowledge_items):
            if swc_id := item.get("swc") or item.get("id"):
                grouped[swc_id].append(index)
        self.swc_to_indices = {swc: np.array(rows) for swc, rows in grouped.items()}

    # Cosine similarity giữa đoạn mã và từng ca trong KB (mảng 11,869 phần tử).
    def _similarities(self, code: str) -> np.ndarray:
        return safe_sparse_dot(normalize(self.vectorizer.transform([code])), self._kb_vectors_t, dense_output=True)[0]

    # Top-k ca giống đoạn mã nhất (root cause, cách sửa, mã lỗi / mã đã sửa) - ví dụ sửa lỗi cho Fixer.
    def retrieve_context(self, code: str, top_k: int = 5) -> list[dict]:
        similarities = self._similarities(code)
        results = []
        for index in similarities.argsort()[::-1][:max(1, top_k)]:
            item = self.knowledge_items[index]
            swc_id = item.get("swc") or item.get("id")
            record = self.by_id.get(swc_id, {})
            results.append({
                "swc": swc_id, "title": item.get("title") or record.get("title", swc_id),
                **{key: item.get(key) or record.get(key, "") for key in ("root_cause", "description", "remediation")},
                "vulnerable_code": item.get("vulnerable_code", ""), "fixed_code": item.get("fixed_code", ""),
                "similarity": round(float(similarities[index]), 6),
            })
        return results

    # [CỐT LÕI RAG] Mỗi SWC agent: tín hiệu regex + ca giống nhất cùng SWC + trùng khớp -> confidence; báo khi >= 0.60.
    def analyze(self, code: str) -> list[dict]:
        similarities, signal_code, findings = self._similarities(code), extract_signal_code(code), []
        for agent in self.agents:
            signal_hits = find_signal_hits(signal_code, agent.signal_patterns, agent.swc_id)
            rows = self.swc_to_indices.get(agent.swc_id)
            best_item, semantic_score = {}, 0.0
            if rows is not None:
                best_row = int(rows[np.argmax(similarities[rows])])
                semantic_score, best_item = float(similarities[best_row]), self.knowledge_items[best_row]
            exact = bool(best_item) and _is_exact_code_match(str(best_item.get("vulnerable_code") or ""), code)
            if not signal_hits and semantic_score < 0.28 and not exact:
                continue
            if exact:
                confidence = 0.95
            elif signal_hits:
                strength = _clamp(len(signal_hits) / max(1, agent.min_signal_hits))
                confidence = _clamp(0.55 * strength + 0.45 * min(1.0, semantic_score * 2.5))
            else:
                confidence = _clamp(min(1.0, semantic_score * 1.8))
                if semantic_score < MIN_SEMANTIC_ONLY_SIMILARITY:
                    confidence = min(confidence, CANDIDATE_CONFIDENCE_CAP)
            if confidence < VULNERABLE_CONFIDENCE:
                continue
            record = self.by_id.get(agent.swc_id, {})
            findings.append({
                "swc": agent.swc_id, "title": agent.title, "severity": agent.severity,
                "confidence": round(confidence, 6),
                "root_cause": best_item.get("root_cause") or record.get("root_cause", ""),
                "description": best_item.get("description") or record.get("description", agent.description),
                "remediation": best_item.get("remediation") or record.get("remediation", agent.remediation),
                "vulnerable_code": best_item.get("vulnerable_code", ""), "fixed_code": best_item.get("fixed_code", ""),
            })
        findings.sort(key=lambda item: (RISK_ORDER.get(item["severity"], 0), item["confidence"]), reverse=True)
        return findings[:MAX_FINDINGS_PER_UNIT]


# Một retriever dùng chung cho cả tiến trình (nạp KB ~0.5 s một lần thay vì mỗi lần chạy RAG).
@lru_cache(maxsize=1)
def default_retriever() -> LocalRAGRetriever:
    return LocalRAGRetriever()
