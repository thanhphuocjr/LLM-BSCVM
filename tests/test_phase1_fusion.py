from types import SimpleNamespace
import unittest

from phase1_integrated_detector import build_final_result


def args(**overrides):
    defaults = {
        "llm_weight": 0.15,
        "rag_weight": 0.25,
        "static_weight": 0.60,
        "final_threshold": 0.50,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def llm(score=0.0, status="ok", verdict="Safe"):
    return {
        "status": status,
        "component": "LLM / CodeBERT model-based detection",
        "score": score,
        "verdict": verdict,
        "vulnerable_votes": 0,
        "safe_votes": 5,
    }


def rag(score=0.0, vulnerable_count=0, status="ok", verdict="Safe"):
    return {
        "status": status,
        "component": "RAG / TF-IDF retrieval-based detection",
        "backend": "tfidf",
        "score": score,
        "final_verdict": verdict,
        "risk_level": "Low",
        "top_k": 3,
        "vulnerable_count": vulnerable_count,
        "retrieved": [],
        "top_findings": [],
        "agent_results": [],
    }


def static_result(score=0.0, verdict="Safe", risk_level="Low", high_risk=False):
    high_risk_list = []
    if high_risk:
        high_risk_list.append(
            {
                "vuln_id": "SWC-107",
                "vuln_type": "reentrancy",
                "risk_level": risk_level,
                "line": 8,
                "context": "msg.sender.call{value: amount}(\"\"); balances[msg.sender] = 0;",
                "note": "External value call before state update",
            }
        )
    return {
        "status": "ok",
        "component": "Static analysis",
        "score": score,
        "static_score": score,
        "verdict": verdict,
        "risk_level": risk_level,
        "findings": [],
        "high_risk_list": high_risk_list,
        "context_issues": [],
    }


class Phase1FusionTest(unittest.TestCase):
    def test_critical_static_evidence_is_not_drowned_by_low_llm_and_rag(self):
        output = build_final_result(
            "contract VulnerableBank {}",
            args(),
            llm(score=0.000047),
            rag(score=0.0),
            static_result(score=0.4975, verdict="Vulnerable", risk_level="Critical", high_risk=True),
        )

        self.assertTrue(output["final"]["is_vulnerable"])
        self.assertGreaterEqual(output["final"]["final_score"], 0.75)
        self.assertEqual(output["final"]["fusion"]["decision_rule"], "static_critical_evidence_floor")

    def test_safe_low_signal_case_stays_safe(self):
        output = build_final_result(
            "contract SafeToken {}",
            args(),
            llm(score=0.01),
            rag(score=0.0),
            static_result(score=0.05, verdict="Safe", risk_level="Low", high_risk=False),
        )

        self.assertFalse(output["final"]["is_vulnerable"])
        self.assertLess(output["final"]["final_score"], 0.50)

    def test_codebert_error_weight_is_redistributed(self):
        output = build_final_result(
            "contract VulnerableBank {}",
            args(),
            llm(score=0.0, status="error", verdict="Error"),
            rag(score=0.0),
            static_result(score=0.4975, verdict="Vulnerable", risk_level="Critical", high_risk=True),
        )

        self.assertTrue(output["final"]["is_vulnerable"])
        self.assertEqual(output["final"]["effective_weights"]["llm"], 0.0)
        self.assertGreater(output["final"]["effective_weights"]["static_analysis"], 0.60)

    def test_tfidf_vulnerable_neighbor_is_calibrated_above_raw_score(self):
        rag_result = rag(score=0.08, vulnerable_count=1, verdict="Vulnerable")
        rag_result["retrieved"] = [
            {
                "rank": 1,
                "similarity": 0.58,
                "label": "Vulnerable",
                "source_index": 1,
                "snippet_index": 0,
                "snippet_length": 120,
                "weight": 0.58,
            }
        ]

        output = build_final_result(
            "contract Risky {}",
            args(),
            llm(score=0.0),
            rag_result,
            static_result(score=0.30, verdict="Vulnerable", risk_level="Medium", high_risk=False),
        )

        self.assertTrue(output["final"]["is_vulnerable"])
        self.assertGreater(output["final"]["score_breakdown"]["rag"]["adjusted_score"], 0.08)


if __name__ == "__main__":
    unittest.main()
