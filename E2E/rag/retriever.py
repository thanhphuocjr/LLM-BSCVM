from __future__ import annotations

from collections import defaultdict
import re
from pathlib import Path
from typing import Any

import joblib
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

from E2E.config import RISK_ORDER, RUNTIME_DATA_DIR
from E2E.rag.legacy.multi_agent_vuln_detector import (
    build_agents_from_registry,
    find_signal_hits,
)
from E2E.shared.io_utils import read_json


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


MIN_EXACT_MATCH_CHARS = 80
MIN_SEMANTIC_ONLY_SIMILARITY = 0.60
CANDIDATE_CONFIDENCE_CAP = 0.59
MIN_EXACT_MATCH_LENGTH_RATIO = 0.8


def _is_exact_code_match(known_code: str, query_code: str) -> bool:
    known = re.sub(r"\s+", " ", known_code).strip()
    query = re.sub(r"\s+", " ", query_code).strip()
    if len(known) < MIN_EXACT_MATCH_CHARS or len(query) < MIN_EXACT_MATCH_CHARS:
        return False
    if known not in query and query not in known:
        return False
    return min(len(known), len(query)) / max(len(known), len(query)) >= MIN_EXACT_MATCH_LENGTH_RATIO


class LocalRAGRetriever:
    """TF-IDF retrieval over SWC guidance and empirical vulnerability cases, gated by deterministic code signals."""

    def __init__(self, store_dir: str | Path = RUNTIME_DATA_DIR) -> None:
        self.store_dir = Path(store_dir)
        self.registry_path = self.store_dir / "swc_registry.json"
        self.records = read_json(self.registry_path)
        self.by_id = {record["id"]: record for record in self.records}
        self.vectorizer = joblib.load(self.store_dir / "swc_vectorizer.joblib")
        self.matrix = load_npz(self.store_dir / "swc_matrix.npz")
        self.agents = build_agents_from_registry(str(self.registry_path))

        # Load granular knowledge items (1-to-1 with rows of self.matrix)
        knowledge_items_path = self.store_dir / "knowledge_items.json"
        if knowledge_items_path.exists():
            self.knowledge_items: list[dict[str, Any]] = read_json(knowledge_items_path)
        else:
            self.knowledge_items = self.records

        # Map swc_id -> list of matrix row indices
        self.swc_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, item in enumerate(self.knowledge_items):
            swc_id = item.get("swc") or item.get("id")
            if swc_id:
                self.swc_to_indices[swc_id].append(idx)

    def retrieve_context(self, code: str, top_k: int = 5) -> list[dict[str, Any]]:
        query = self.vectorizer.transform([code])
        similarities = cosine_similarity(query, self.matrix)[0]
        indices = similarities.argsort()[::-1][: max(1, top_k)]
        results: list[dict[str, Any]] = []
        for index in indices:
            item = self.knowledge_items[index]
            swc_id = item.get("swc") or item.get("id")
            record = self.by_id.get(swc_id, {})
            results.append(
                {
                    "id": item.get("id", f"item_{index}"),
                    "swc": swc_id,
                    "title": item.get("title") or record.get("title", swc_id),
                    "severity": item.get("severity") or record.get("severity", "Medium"),
                    "root_cause": item.get("root_cause") or record.get("root_cause", ""),
                    "description": item.get("description") or record.get("description", ""),
                    "remediation": item.get("remediation") or record.get("remediation", ""),
                    "checklist": item.get("checklist") or record.get("checklist", []),
                    "vulnerable_code": item.get("vulnerable_code", ""),
                    "fixed_code": item.get("fixed_code", ""),
                    "has_code_patch": bool(item.get("has_code_patch", bool(item.get("fixed_code")))),
                    "similarity": round(float(similarities[index]), 6),
                    "source": item.get("source", "vulnerability_knowledge"),
                }
            )
        return results

    def analyze(self, code: str, max_findings: int = 10) -> dict[str, Any]:
        query = self.vectorizer.transform([code])
        similarities = cosine_similarity(query, self.matrix)[0]
        findings: list[dict[str, Any]] = []

        for agent in self.agents:
            signal_hits = find_signal_hits(code, agent.signal_patterns, agent.swc_id)
            matching_indices = self.swc_to_indices.get(agent.swc_id, [])
            if matching_indices:
                best_sub_idx = max(matching_indices, key=lambda i: similarities[i])
                semantic_score = float(similarities[best_sub_idx])
                best_item = self.knowledge_items[best_sub_idx]
            else:
                best_sub_idx = None
                semantic_score = 0.0
                best_item = None

            # Exact match with a known empirical vulnerable function. Both sides must be
            # non-trivial and near-identical in size: a short getter is a substring of many
            # knowledge items, and a tiny knowledge snippet is a substring of many units.
            exact_code_match = False
            if best_item:
                exact_code_match = _is_exact_code_match(
                    str(best_item.get("vulnerable_code") or ""), code
                )

            # Allow either deterministic regex hits, strong semantic match, or exact code match
            if not signal_hits and semantic_score < 0.28 and not exact_code_match:
                continue

            if exact_code_match:
                signal_strength = 1.0
                confidence = 0.95
            elif signal_hits:
                signal_strength = _clamp(len(signal_hits) / max(1, agent.min_signal_hits))
                confidence = _clamp(0.55 * signal_strength + 0.45 * min(1.0, semantic_score * 2.5))
            else:
                # Similarity alone (no deterministic code signal) was right only ~51% of the
                # time on the function-level test split; it may only confirm a finding when
                # the match is a near-duplicate, otherwise it stays a candidate.
                signal_strength = 0.0
                confidence = _clamp(min(1.0, semantic_score * 1.8))
                if semantic_score < MIN_SEMANTIC_ONLY_SIMILARITY:
                    confidence = min(confidence, CANDIDATE_CONFIDENCE_CAP)

            record = self.by_id.get(agent.swc_id, {})
            findings.append(
                {
                    "agent": agent.name,
                    "swc": agent.swc_id,
                    "title": agent.title,
                    "severity": agent.severity,
                    "confidence": round(confidence, 6),
                    "similarity_score": round(semantic_score, 6),
                    "signal_score": round(signal_strength, 6),
                    "signal_hits": signal_hits,
                    "vulnerable": confidence >= 0.60,
                    "candidate": confidence < 0.60,
                    "root_cause": (best_item.get("root_cause") if best_item else "") or record.get("root_cause", ""),
                    "description": (best_item.get("description") if best_item else "") or record.get("description", agent.description),
                    "remediation": (best_item.get("remediation") if best_item else "") or record.get("remediation", agent.remediation),
                    "checklist": (best_item.get("checklist") if best_item else []) or record.get("checklist", []),
                    "best_match_id": best_item.get("id") if best_item else None,
                    "vulnerable_code": best_item.get("vulnerable_code", "") if best_item else "",
                    "fixed_code": best_item.get("fixed_code", "") if best_item else "",
                    "has_code_patch": bool(best_item.get("has_code_patch", bool(best_item.get("fixed_code")))) if best_item else False,
                    "source": record.get("source", "vulnerability_knowledge"),
                }
            )

        findings.sort(
            key=lambda item: (RISK_ORDER.get(item["severity"], 0), item["confidence"]),
            reverse=True,
        )
        positives = [finding for finding in findings if finding["vulnerable"]]
        returned = (positives + [item for item in findings if not item["vulnerable"]])[:max_findings]
        risk_level = positives[0]["severity"] if positives else "None"
        score = max((item["confidence"] for item in positives), default=0.0)
        return {
            "component": "rag",
            "status": "ok",
            "engine": "local_swc_tfidf_plus_signal_gating",
            "final_verdict": "Vulnerable" if positives else "Safe",
            "risk_level": risk_level,
            "final_score": round(score, 6),
            "positive_count": len(positives),
            "candidate_count": len(findings) - len(positives),
            "top_findings": returned,
            "findings": returned,
            "retrieved_context": self.retrieve_context(code, top_k=5),
        }


def run_rag_analysis(code: str, store_dir: str | Path = RUNTIME_DATA_DIR) -> dict[str, Any]:
    return LocalRAGRetriever(store_dir).analyze(code)
