from __future__ import annotations

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


class LocalRAGRetriever:
    """TF-IDF retrieval over SWC guidance, gated by deterministic code signals."""

    def __init__(self, store_dir: str | Path = RUNTIME_DATA_DIR) -> None:
        self.store_dir = Path(store_dir)
        self.registry_path = self.store_dir / "swc_registry.json"
        self.records = read_json(self.registry_path)
        self.by_id = {record["id"]: record for record in self.records}
        self.vectorizer = joblib.load(self.store_dir / "swc_vectorizer.joblib")
        self.matrix = load_npz(self.store_dir / "swc_matrix.npz")
        self.agents = build_agents_from_registry(str(self.registry_path))

    def retrieve_context(self, code: str, top_k: int = 5) -> list[dict[str, Any]]:
        query = self.vectorizer.transform([code])
        similarities = cosine_similarity(query, self.matrix)[0]
        indices = similarities.argsort()[::-1][: max(1, top_k)]
        return [
            {
                "swc": self.records[index]["id"],
                "title": self.records[index]["title"],
                "severity": self.records[index]["severity"],
                "description": self.records[index]["description"],
                "remediation": self.records[index]["remediation"],
                "checklist": self.records[index]["checklist"],
                "similarity": round(float(similarities[index]), 6),
                "source": self.records[index]["source"],
            }
            for index in indices
        ]

    def analyze(self, code: str, max_findings: int = 10) -> dict[str, Any]:
        query = self.vectorizer.transform([code])
        similarities = cosine_similarity(query, self.matrix)[0]
        index_by_swc = {record["id"]: index for index, record in enumerate(self.records)}
        findings: list[dict[str, Any]] = []

        for agent in self.agents:
            signal_hits = find_signal_hits(code, agent.signal_patterns, agent.swc_id)
            if not signal_hits:
                continue
            index = index_by_swc.get(agent.swc_id)
            semantic_score = float(similarities[index]) if index is not None else 0.0
            signal_strength = _clamp(len(signal_hits) / max(1, agent.min_signal_hits))
            confidence = _clamp(0.72 * signal_strength + 0.28 * min(1.0, semantic_score * 4.0))
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
                    "vulnerable": confidence >= 0.62,
                    "candidate": confidence < 0.62,
                    "description": record.get("description", agent.description),
                    "remediation": record.get("remediation", agent.remediation),
                    "checklist": record.get("checklist", []),
                    "source": record.get("source"),
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
            "retrieved_context": self.retrieve_context(code, top_k=5),
        }


def run_rag_analysis(code: str, store_dir: str | Path = RUNTIME_DATA_DIR) -> dict[str, Any]:
    return LocalRAGRetriever(store_dir).analyze(code)
