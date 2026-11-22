"""Cơ sở tri thức lỗ hổng (KB) - dịch vụ dữ liệu DÙNG CHUNG: nạp một lần, tra theo SWC hoặc theo độ giống mã (TF-IDF)."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot

from E2E.common.io_utils import read_json
from E2E.common.swc import normalize_swc_id
from E2E.config import KNOWLEDGE_BASE_DIR

# Fixer: số ca KB đưa vào prompt cho một hàm, và độ giống tối thiểu của ca "bù" (ca không cùng SWC với finding).
FIXER_CASE_LIMIT = 3
FIXER_CASE_MIN_SIMILARITY = 0.08
_CACHE_SIZE = 256


# Thêm chuỗi (đã strip) vào danh sách nếu khác rỗng và chưa có.
def _add_unique(bucket: list[str], value) -> None:
    text = str(value or "").strip()
    if text and text not in bucket:
        bucket.append(text)


# KB: 37 mục SWC (swc_registry.json) + 11,869 ca lỗi (knowledge_items.json) + vectorizer / ma trận TF-IDF đã build.
class KnowledgeBase:
    # Nạp artifact trong data/knowledge_base/built/, chuẩn hoá + chuyển vị ma trận một lần, chỉ mục SWC -> các dòng.
    def __init__(self, store_dir: str | Path = KNOWLEDGE_BASE_DIR) -> None:
        store_dir = Path(store_dir)
        self.registry = {swc: record for record in read_json(store_dir / "swc_registry.json")
                         if isinstance(record, dict) and (swc := normalize_swc_id(record.get("id")))}
        self.items: list[dict] = read_json(store_dir / "knowledge_items.json")
        self.item_swc = np.array([normalize_swc_id(item.get("swc") or item.get("id")) for item in self.items])
        self.rows_by_swc = {swc: np.flatnonzero(self.item_swc == swc) for swc in sorted(set(self.item_swc))}
        self.vectorizer = joblib.load(store_dir / "swc_vectorizer.joblib")
        self._matrix_t = normalize(load_npz(store_dir / "swc_matrix.npz")).T.tocsr()
        self._similarities, self._knowledge = {}, {}

    # Cosine similarity giữa đoạn mã và từng ca trong KB (có nhớ: RAG, Advisor và Fixer tra cùng một hàm).
    def similarities(self, code: str) -> np.ndarray:
        if code not in self._similarities:
            if len(self._similarities) >= _CACHE_SIZE:
                self._similarities.clear()
            vector = normalize(self.vectorizer.transform([code]))
            self._similarities[code] = safe_sparse_dot(vector, self._matrix_t, dense_output=True)[0]
        return self._similarities[code]

    # Tiêu đề của một SWC theo registry.
    def title(self, swc: str) -> str:
        return str(self.registry.get(swc, {}).get("title") or swc).strip()

    # Một ca KB dạng gọn (SWC, tiêu đề, root cause, mô tả, cách sửa, mã lỗi / mã đã sửa, độ giống).
    def case(self, row: int, similarity: float) -> dict:
        item, swc = self.items[row], str(self.item_swc[row])
        record = self.registry.get(swc, {})
        return {"swc": swc, "title": item.get("title") or record.get("title", swc),
                **{key: item.get(key) or record.get(key, "") for key in ("root_cause", "description", "remediation")},
                "vulnerable_code": item.get("vulnerable_code", ""), "fixed_code": item.get("fixed_code", ""),
                "similarity": round(float(similarity), 6)}

    # Với mỗi SWC có ca trong KB: (dòng của ca giống mã nhất, độ giống) - đầu vào của RAG Phase 1B.
    def nearest_by_swc(self, code: str) -> dict[str, tuple[int, float]]:
        similarities = self.similarities(code)
        return {swc: (int(rows[np.argmax(similarities[rows])]), float(similarities[rows].max()))
                for swc, rows in self.rows_by_swc.items() if len(rows)}

    # Ca giống mã nhất toàn KB (dòng, độ giống) - Phase 1 dùng SWC của ca này để phân loại hàm chỉ detector báo.
    def nearest_case(self, code: str) -> tuple[int, float]:
        similarities = self.similarities(code)
        row = int(np.argmax(similarities))
        return row, float(similarities[row])

    # [Phase 4] Ca cho Fixer khi sửa MỘT hàm: ca giống nhất của mỗi SWC của finding, rồi bù ca giống nhất toàn KB.
    def cases_for_unit(self, code: str, swcs: set[str], limit: int = FIXER_CASE_LIMIT,
                       min_similarity: float = FIXER_CASE_MIN_SIMILARITY) -> list[dict]:
        similarities = self.similarities(code)
        nearest = self.nearest_by_swc(code)
        first = sorted((nearest[swc][0] for swc in swcs if swc in nearest), key=lambda row: -similarities[row])
        rest = [int(row) for row in np.argsort(-similarities, kind="stable")[:limit + len(first)]
                if int(row) not in first and similarities[row] >= min_similarity]
        return [self.case(row, similarities[row]) for row in (first + rest)[:limit]]

    # Gom mô tả / cách sửa / checklist / mã sửa từ registry + `limit` ca của SWC (giống mã nhất nếu có code).
    def _collect(self, swc: str, code: str | None, limit: int) -> dict:
        titles, descriptions, remediations, checklist, registry_fixes, empirical_fixes = [], [], [], [], [], []
        best_pair: dict = {}
        reg = self.registry.get(swc)
        if reg:
            _add_unique(titles, reg.get("title") or swc)
            _add_unique(descriptions, reg.get("description") or reg.get("root_cause"))
            _add_unique(remediations, reg.get("remediation"))
            for text in reg.get("checklist", []) or []:
                _add_unique(checklist, text)
            for sample in reg.get("samples", []) or []:
                if isinstance(sample, dict):
                    _add_unique(registry_fixes, sample.get("fixed"))
        rows = self.rows_by_swc.get(swc, np.array([], int))
        if code and len(rows):
            rows = rows[np.argsort(-self.similarities(code)[rows], kind="stable")]
        for item in (self.items[row] for row in rows[:limit]):
            _add_unique(titles, item.get("title"))
            _add_unique(descriptions, item.get("description") or item.get("root_cause"))
            remediation = str(item.get("remediation") or "").strip()
            _add_unique(remediations, remediation)
            for text in item.get("checklist", []) or []:
                _add_unique(checklist, text)
            fixed, vulnerable = (str(item.get(key) or "").strip() for key in ("fixed_code", "vulnerable_code"))
            _add_unique(empirical_fixes, fixed)
            if (fixed or vulnerable) and not best_pair:
                best_pair = {"vulnerable_code": vulnerable, "fixed_code": fixed,
                             "root_cause": str(item.get("root_cause") or "").strip(), "remediation": remediation,
                             "severity": str(item.get("severity") or (reg.get("severity") if reg else "Medium")),
                             "cwe_ids": item.get("cwe_ids") or []}
        sample = next((item for item in (reg.get("samples", []) or []) if isinstance(item, dict)), None) \
            if reg and not best_pair else None
        if sample:
            best_pair = {"vulnerable_code": str(sample.get("vulnerable") or "").strip(),
                         "fixed_code": str(sample.get("fixed") or "").strip(),
                         "root_cause": str(sample.get("root_cause") or "").strip(),
                         "remediation": str(reg.get("remediation") or "").strip(),
                         "severity": str(reg.get("severity") or "Medium"), "cwe_ids": []}
        code_examples = empirical_fixes + [fix for fix in registry_fixes if fix not in empirical_fixes]
        return {
            "descriptions": descriptions, "remediations": remediations, "checklist": checklist,
            "code_examples": code_examples[:limit], "best_pair": best_pair,
            "risk_guidance": {
                "swc": swc, "title": titles[0] if titles else swc,
                "benchmark_severity": reg.get("severity") if reg else best_pair.get("severity", "Medium"),
                "cwe_ids": best_pair.get("cwe_ids", []),
                "remediation_summary": remediations[0] if remediations else "",
                "root_cause_summary": descriptions[0] if descriptions else "", "checklist": checklist[:6],
            },
        }

    # [Phase 2] Tri thức sửa lỗi cho một finding: theo SWC, các ca xếp theo độ giống với mã của hàm (có nhớ).
    def repair_knowledge(self, swc: str, code: str | None = None, limit: int = 3) -> dict:
        key = (normalize_swc_id(swc), code, limit)
        if key not in self._knowledge:
            if len(self._knowledge) >= _CACHE_SIZE:
                self._knowledge.clear()
            self._knowledge[key] = self._collect(*key)
        return self._knowledge[key]

    # [Phase 3] Hướng dẫn đánh giá rủi ro của một SWC (mức chuẩn, CWE, cách sửa chuẩn, checklist).
    def risk_guidance(self, swc: str) -> dict:
        return self.repair_knowledge(swc)["risk_guidance"]


# Một KB dùng chung cho cả tiến trình (nạp ~0.5 s một lần thay vì mỗi phase / mỗi contract).
@lru_cache(maxsize=1)
def default_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()
