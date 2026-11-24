"""Phase 1B - RAG phát hiện bằng truy hồi: hàm giống ca lỗi đã biết của SWC nào trong KB (TF-IDF) thì báo SWC đó."""

from E2E.common.knowledge_base import KnowledgeBase, default_knowledge_base
from E2E.common.regex_patterns import RE_WHITESPACE
from E2E.common.swc import swc_severity

MIN_EXACT_MATCH_CHARS = 80
MIN_EXACT_MATCH_LENGTH_RATIO = 0.8
EXACT_MATCH_CONFIDENCE = 0.95
# Độ giống tối thiểu để báo một SWC.
MIN_SIMILARITY = 0.40
MAX_FINDINGS_PER_UNIT = 10


# Mã đang xét gần như TRÙNG một hàm lỗi đã biết: cả hai >= 80 ký tự, cái này chứa cái kia, độ dài chênh <= 20%.
def _is_exact_code_match(known_code: str, query_code: str) -> bool:
    known, query = RE_WHITESPACE.sub(" ", known_code).strip(), RE_WHITESPACE.sub(" ", query_code).strip()
    if min(len(known), len(query)) < MIN_EXACT_MATCH_CHARS or (known not in query and query not in known):
        return False
    return min(len(known), len(query)) / max(len(known), len(query)) >= MIN_EXACT_MATCH_LENGTH_RATIO


# [CỐT LÕI RAG] Mỗi SWC: ca giống nhất; trùng khớp -> 0.95, độ giống >= 0.40 -> confidence = độ giống; <= 10 SWC.
def analyze(code: str, kb: KnowledgeBase | None = None) -> list[dict]:
    kb = kb or default_knowledge_base()
    findings = []
    for swc, (row, similarity) in kb.nearest_by_swc(code).items():
        item = kb.items[row]
        exact = _is_exact_code_match(str(item.get("vulnerable_code") or ""), code)
        if not exact and similarity < MIN_SIMILARITY:
            continue
        record = kb.registry.get(swc, {})
        findings.append({
            "swc": swc, "title": kb.title(swc), "severity": swc_severity(swc),
            "confidence": round(EXACT_MATCH_CONFIDENCE if exact else similarity, 6),
            "similarity": round(similarity, 6), "match": "exact" if exact else "similar",
            **{key: item.get(key) or record.get(key, "") for key in ("root_cause", "description", "remediation")},
            "vulnerable_code": item.get("vulnerable_code", ""), "fixed_code": item.get("fixed_code", ""),
        })
    findings.sort(key=lambda finding: finding["confidence"], reverse=True)
    return findings[:MAX_FINDINGS_PER_UNIT]


# Lớp lỗi của MỘT hàm theo ca lỗi giống nó nhất trong KB (1-NN) - Phase 1 dùng cho hàm chỉ detector báo.
def nearest_class(code: str, kb: KnowledgeBase | None = None) -> dict:
    kb = kb or default_knowledge_base()
    row, similarity = kb.nearest_case(code)
    swc = str(kb.item_swc[row])
    return {"swc": swc, "title": kb.title(swc), "severity": swc_severity(swc), "similarity": round(similarity, 6),
            "nearest_case_id": kb.items[row].get("id")}
