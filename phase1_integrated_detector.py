#!/usr/bin/env python3
"""
Phase 1 integrated smart-contract vulnerability detector.

This file combines the three detection components currently present in the
project:

1. Static analysis: deterministic security signals
2. RAG retrieval-based detection: similar historical/code evidence
3. CodeBERT LoRA model-based detection: probabilistic model signal

Example:
    python3 phase1_integrated_detector.py --code-file path/to/Contract.sol

Inline test:
    Edit CODE_TO_TEST below, then run:
    python3 phase1_integrated_detector.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CODEBERT_ADAPTER = PROJECT_ROOT / "Codebert" / "codebert-vuln-lora-final"
DEFAULT_CODEBERT_BASE_MODEL = "microsoft/codebert-base"
DEFAULT_RAG_KNOWLEDGE_DIR = PROJECT_ROOT / "rag" / "knowledge_store"
DEFAULT_TFIDF_DIR = PROJECT_ROOT / "rag" / "tfidf_store"
DEFAULT_RAG_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_SWC_REGISTRY = PROJECT_ROOT / "dataset" / "swc_registry.json"

LABEL2ID = {"Safe": 0, "Vulnerable": 1}
ID2LABEL = {0: "Safe", 1: "Vulnerable"}
DEFAULT_LLM_THRESHOLD = 0.40
DEFAULT_FINAL_THRESHOLD = 0.50

DEFAULT_WEIGHTS = {
    "llm": 0.15,
    "rag": 0.25,
    "static_analysis": 0.60,
}

STATIC_RISK_SCORE_FLOORS = {
    "Critical": 0.85,
    "High": 0.70,
    "Medium": 0.45,
    "Low": 0.20,
}

RAG_RISK_SCORE_FLOORS = {
    "Critical": 0.75,
    "High": 0.65,
    "Medium": 0.50,
    "Low": 0.35,
}

# Edit this block when you want to test code directly inside this file.
# CLI input still has priority:
# 1. --code-file
# 2. --code
# 3. CODE_TO_TEST
CODE_TO_TEST = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint256) public balances;

    // Nạp ETH vào contract
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    // Rút ETH - CÓ LỖ HỔNG REENTRANCY
    function withdraw() public {
        uint256 amount = balances[msg.sender];

        require(amount > 0, "No balance to withdraw");

        // Gửi ETH trước khi cập nhật số dư
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        // Cập nhật số dư sau
        balances[msg.sender] = 0;
    }

    // Xem số dư ETH trong contract
    function getContractBalance() public view returns (uint256) {
        return address(this).balance;
    }
}
""".strip()

PROMPT_VARIANTS = [
    {
        "a": "Devise a label name suitable for categorizing items as either vulnerable or safe.",
        "b": "Please review the code. Please find out if it is vulnerable.",
        "c": "The function {fn_name} from the contract {contract_name}.",
    },
    {
        "a": "Suggest a label designation that clearly identifies an item's status as either vulnerable or safe.",
        "b": "Inspect the following Solidity code. Determine if there are any vulnerabilities present.",
        "c": "Observe the method {fn_name} within the smart contract {contract_name}.",
    },
    {
        "a": "Invent a naming label that aptly segregates items into vulnerable or safe classifications.",
        "b": "Examine this Solidity script. Identify any potential security risks.",
        "c": "Review the function {fn_name} in the blockchain contract {contract_name}.",
    },
    {
        "a": "Formulate a label descriptor that bifurcates objects into categories of vulnerable and safe.",
        "b": "Please assess the provided Solidity code for any security vulnerabilities.",
        "c": "Check the procedure {fn_name} in the digital contract {contract_name}.",
    },
    {
        "a": "Propose a label nomenclature that aptly differentiates between vulnerable and safe states.",
        "b": "Evaluate the given Solidity function. Are there any security flaws?",
        "c": "Inspect the subroutine {fn_name} from the decentralized contract {contract_name}.",
    },
]

RISK_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "None": 0, "Unknown": 0}


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def extract_fn_and_contract(code: str) -> tuple[str, str]:
    fn_match = re.search(r"function\s+(\w+)\s*\(", code)
    contract_match = re.search(r"contract\s+(\w+)", code)
    fn_name = fn_match.group(1) if fn_match else "unknownFunction"
    contract_name = contract_match.group(1) if contract_match else "UnknownContract"
    return fn_name, contract_name


def build_prompt_text(code: str, variant_idx: int) -> str:
    variant = PROMPT_VARIANTS[variant_idx]
    fn_name, contract_name = extract_fn_and_contract(code)
    contract_context = variant["c"].format(fn_name=fn_name, contract_name=contract_name)
    return f"{variant['a']} {variant['b']} {contract_context}"


def resolve_torch_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_transformers_runtime(local_files_only: bool) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class CodeBertVulnerabilityDetector:
    def __init__(
        self,
        adapter_path: str | Path = DEFAULT_CODEBERT_ADAPTER,
        base_model: str = DEFAULT_CODEBERT_BASE_MODEL,
        device: str = "auto",
        max_tokens: int = 512,
        local_files_only: bool = True,
    ) -> None:
        self.adapter_path = Path(adapter_path)
        self.base_model = base_model
        self.device_name = resolve_torch_device(device)
        self.max_tokens = max_tokens
        self.local_files_only = local_files_only

        if not self.adapter_path.exists():
            raise FileNotFoundError(f"CodeBERT LoRA adapter not found: {self.adapter_path}")

        configure_transformers_runtime(local_files_only)

        import torch
        import torch.nn.functional as functional
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()

        self.torch = torch
        self.functional = functional
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.adapter_path),
            local_files_only=True,
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            use_safetensors=False,
        )
        self.model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self.model.to(self.device_name)
        self.model.eval()

    def predict_proba_single(self, code: str, variant_idx: int) -> float:
        prompt_text = build_prompt_text(code, variant_idx)
        encoding = self.tokenizer(
            prompt_text,
            code,
            max_length=self.max_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device_name)

        with self.torch.no_grad():
            outputs = self.model(**encoding)

        probabilities = self.functional.softmax(outputs.logits, dim=-1)
        return float(probabilities[0][LABEL2ID["Vulnerable"]].item())

    def analyze(self, code: str, threshold: float = DEFAULT_LLM_THRESHOLD) -> dict[str, Any]:
        prompt_probabilities = [
            self.predict_proba_single(code, index)
            for index in range(len(PROMPT_VARIANTS))
        ]
        probability = sum(prompt_probabilities) / len(prompt_probabilities)
        hard_votes = ["Vulnerable" if value >= threshold else "Safe" for value in prompt_probabilities]

        return {
            "status": "ok",
            "component": "LLM / CodeBERT model-based detection",
            "score": round(probability, 6),
            "verdict": "Vulnerable" if probability >= threshold else "Safe",
            "threshold": threshold,
            "vulnerable_probability": round(probability, 6),
            "safe_probability": round(1.0 - probability, 6),
            "prompt_probabilities": [round(value, 6) for value in prompt_probabilities],
            "prompt_votes": hard_votes,
            "vulnerable_votes": hard_votes.count("Vulnerable"),
            "safe_votes": hard_votes.count("Safe"),
            "adapter_path": str(self.adapter_path),
            "base_model": self.base_model,
            "device": self.device_name,
            "local_files_only": self.local_files_only,
        }


def run_llm_component(args: argparse.Namespace, code: str) -> dict[str, Any]:
    if args.skip_llm:
        return skipped_component("LLM / CodeBERT model-based detection")
    detector = CodeBertVulnerabilityDetector(
        adapter_path=args.codebert_adapter,
        base_model=args.codebert_base_model,
        device=args.device,
        max_tokens=args.max_tokens,
        local_files_only=not args.allow_model_download,
    )
    return detector.analyze(code, threshold=args.llm_threshold)


def map_tfidf_rag_result(result: Any) -> dict[str, Any]:
    retrieved = [
        {
            "rank": item.rank,
            "similarity": round(float(item.similarity), 6),
            "label": item.label,
            "source_index": item.source_index,
            "snippet_index": item.snippet_index,
            "snippet_length": item.snippet_length,
            "weight": round(float(item.weight), 6),
        }
        for item in result.retrieved
    ]
    score = float(result.vulnerability_score or 0.0)
    return {
        "status": "ok",
        "component": "RAG / TF-IDF retrieval-based detection",
        "backend": "tfidf",
        "score": round(score, 6),
        "final_score": round(score, 6),
        "final_verdict": "Vulnerable" if result.vulnerable_count > 0 else "Safe",
        "risk_level": result.risk_level,
        "top_k": result.top_k,
        "vulnerable_count": result.vulnerable_count,
        "vulnerability_prob_pct": round(float(result.vulnerability_prob_pct), 6),
        "elapsed_ms": round(float(result.elapsed_ms), 3),
        "retrieved": retrieved,
        "top_findings": [],
        "agent_results": [],
    }


def run_tfidf_rag_component(args: argparse.Namespace, code: str) -> dict[str, Any]:
    from rag.tfidf_retriever import SmartContractRetriever

    retriever = SmartContractRetriever(tfidf_dir=str(args.tfidf_dir))
    return map_tfidf_rag_result(retriever.retrieve(code, top_k=args.rag_top_k))


def run_semantic_rag_component(args: argparse.Namespace, code: str) -> dict[str, Any]:
    if args.skip_rag:
        return skipped_component("RAG / Retrieval-based detection")

    from rag.multi_agent_vuln_detector import run_multi_agent_analysis

    result = run_multi_agent_analysis(
        code=code,
        knowledge_dir=str(args.rag_knowledge_dir),
        model_name=args.rag_model_name,
        device=args.device,
        max_seq_length=args.rag_max_seq_length,
        max_input_chars=args.rag_max_input_chars,
        top_k=args.rag_top_k,
        swc_registry_path=str(args.swc_registry),
        max_findings=args.rag_max_findings,
        include_all_agents=args.include_all_agents,
    )
    result = to_jsonable(result)
    result.update(
        {
            "status": "ok",
            "component": "RAG / Semantic multi-agent retrieval-based detection",
            "backend": "semantic",
            "score": round(float(result.get("final_score", 0.0) or 0.0), 6),
        }
    )
    return result


def run_rag_component(args: argparse.Namespace, code: str) -> dict[str, Any]:
    if args.skip_rag:
        return skipped_component("RAG / Retrieval-based detection")
    if args.rag_backend == "tfidf":
        return run_tfidf_rag_component(args, code)
    return run_semantic_rag_component(args, code)


def cleanup_torch_memory() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass
    except Exception:
        return


def serialize_static_result(result: dict[str, Any]) -> dict[str, Any]:
    serialized = to_jsonable(result)
    findings = []
    for vuln_type, finding in serialized.get("findings", {}).items():
        item = dict(finding)
        item["vuln_type"] = item.get("vuln_type") or vuln_type
        findings.append(item)
    serialized["findings"] = findings
    serialized["score"] = round(float(serialized.get("static_score", 0.0) or 0.0), 6)
    serialized["component"] = "Static analysis"
    serialized["status"] = "ok"
    return serialized


def run_static_component(args: argparse.Namespace, code: str) -> dict[str, Any]:
    if args.skip_static:
        return skipped_component("Static analysis")

    from static_analysis.static_analyzer import run_static_analysis

    return serialize_static_result(run_static_analysis(code))


def skipped_component(name: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "component": name,
        "score": 0.0,
        "verdict": "Skipped",
    }


def errored_component(name: str, error: Exception) -> dict[str, Any]:
    return {
        "status": "error",
        "component": name,
        "score": 0.0,
        "verdict": "Error",
        "error": f"{type(error).__name__}: {error}",
    }


def run_component(
    name: str,
    runner: Any,
    args: argparse.Namespace,
    code: str,
) -> dict[str, Any]:
    try:
        return runner(args, code)
    except Exception as error:
        if args.fail_on_component_error:
            raise
        return errored_component(name, error)
    finally:
        cleanup_torch_memory()


def code_window(code: str, line: int, radius: int = 2) -> dict[str, Any]:
    lines = code.splitlines()
    if line < 1 or line > len(lines):
        return {"start_line": None, "end_line": None, "code": ""}

    start = max(1, line - radius)
    end = min(len(lines), line + radius)
    numbered = [
        f"{line_no}: {lines[line_no - 1]}"
        for line_no in range(start, end + 1)
    ]
    return {
        "start_line": start,
        "end_line": end,
        "code": "\n".join(numbered),
    }


def highest_risk(*levels: str | None) -> str:
    clean_levels = [level or "None" for level in levels]
    return max(clean_levels, key=lambda level: RISK_ORDER.get(level, 0), default="None")


def matching_agent_result(rag_result: dict[str, Any], swc: str | None, agent: str | None) -> dict[str, Any]:
    for result in rag_result.get("agent_results", []) or []:
        if swc and result.get("swc") == swc:
            return result
        if agent and result.get("agent") == agent:
            return result
    return {}


def best_evidence(agent_result: dict[str, Any], swc: str | None) -> dict[str, Any]:
    evidence = agent_result.get("evidence", []) or []
    for item in evidence:
        if swc and item.get("Swc") == swc:
            return item
    return evidence[0] if evidence else {}


def build_vulnerability_report(
    code: str,
    rag_result: dict[str, Any],
    static_result: dict[str, Any],
) -> list[dict[str, Any]]:
    vulnerabilities: dict[str, dict[str, Any]] = {}

    for finding in rag_result.get("top_findings", []) or []:
        swc = finding.get("swc") or "Unknown"
        agent_result = matching_agent_result(rag_result, swc, finding.get("agent"))
        evidence = best_evidence(agent_result, swc)
        key = swc if swc != "Unknown" else finding.get("title", "Unknown")
        vulnerabilities[key] = {
            "swc": swc,
            "title": finding.get("title") or evidence.get("Title"),
            "severity": finding.get("severity") or agent_result.get("severity") or "Unknown",
            "description": evidence.get("Vulnerable"),
            "best_practice": evidence.get("BestPractice") or finding.get("best_practice"),
            "best_practice_checklist": evidence.get("BestPracticeChecklist") or finding.get("checklist", []),
            "rag_confidence": finding.get("confidence"),
            "rag_similarity_score": finding.get("similarity_score"),
            "rag_signal_hits": finding.get("signal_hits", []),
            "code_patterns_from_rag": evidence.get("CodePatterns", []),
            "related_code": [],
            "components": ["RAG"],
        }

    for item in static_result.get("high_risk_list", []) or []:
        swc = item.get("vuln_id") or "Unknown"
        key = swc if swc != "Unknown" else item.get("vuln_type", "Unknown")
        if key not in vulnerabilities:
            vulnerabilities[key] = {
                "swc": swc,
                "title": item.get("vuln_type"),
                "severity": item.get("risk_level") or "Unknown",
                "description": None,
                "best_practice": None,
                "best_practice_checklist": [],
                "rag_confidence": None,
                "rag_similarity_score": None,
                "rag_signal_hits": [],
                "code_patterns_from_rag": [],
                "related_code": [],
                "components": [],
            }

        report_item = vulnerabilities[key]
        report_item["severity"] = highest_risk(report_item.get("severity"), item.get("risk_level"))
        if "Static analysis" not in report_item["components"]:
            report_item["components"].append("Static analysis")
        report_item["related_code"].append(
            {
                "source": "Static analysis",
                "line": item.get("line"),
                "note": item.get("note"),
                "context": item.get("context"),
                "window": code_window(code, int(item.get("line") or 0)),
            }
        )

    return sorted(
        vulnerabilities.values(),
        key=lambda item: (
            RISK_ORDER.get(item.get("severity", "Unknown"), 0),
            float(item.get("rag_confidence") or 0.0),
        ),
        reverse=True,
    )


def validate_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Component weights must sum to 1.0, got {total:.6f}")
    return weights


def component_is_active(result: dict[str, Any]) -> bool:
    return result.get("status") == "ok"


def calibrated_llm_score(llm_result: dict[str, Any]) -> float:
    if not component_is_active(llm_result):
        return 0.0

    score = clamp(float(llm_result.get("score", 0.0) or 0.0))
    vulnerable_votes = int(llm_result.get("vulnerable_votes", 0) or 0)
    total_votes = vulnerable_votes + int(llm_result.get("safe_votes", 0) or 0)
    if total_votes > 0 and vulnerable_votes > 0:
        vote_ratio = vulnerable_votes / total_votes
        score = max(score, 0.25 + 0.65 * vote_ratio)
    return clamp(score)


def calibrated_tfidf_rag_score(rag_result: dict[str, Any]) -> float:
    score = clamp(float(rag_result.get("score", 0.0) or 0.0))
    vulnerable_count = int(rag_result.get("vulnerable_count", 0) or 0)
    top_k = max(1, int(rag_result.get("top_k", 1) or 1))
    if vulnerable_count <= 0:
        return score

    vulnerable_similarities = [
        float(item.get("similarity", 0.0) or 0.0)
        for item in rag_result.get("retrieved", []) or []
        if str(item.get("label", "")).lower() == "vulnerable"
    ]
    best_vulnerable_similarity = max(vulnerable_similarities, default=0.0)
    vulnerable_ratio = min(1.0, vulnerable_count / top_k)
    evidence_score = 0.25 + 0.25 * vulnerable_ratio + 0.35 * best_vulnerable_similarity
    return clamp(max(score, evidence_score, RAG_RISK_SCORE_FLOORS.get(rag_result.get("risk_level"), 0.0)))


def calibrated_semantic_rag_score(rag_result: dict[str, Any]) -> float:
    score = clamp(float(rag_result.get("score", 0.0) or 0.0))
    if rag_result.get("final_verdict") == "Vulnerable":
        score = max(score, RAG_RISK_SCORE_FLOORS.get(rag_result.get("risk_level"), 0.65))
    elif int(rag_result.get("all_candidate_count", 0) or 0) > 0:
        score = max(score, 0.45)
    return clamp(score)


def calibrated_rag_score(rag_result: dict[str, Any]) -> float:
    if not component_is_active(rag_result):
        return 0.0
    if rag_result.get("backend") == "tfidf":
        return calibrated_tfidf_rag_score(rag_result)
    return calibrated_semantic_rag_score(rag_result)


def calibrated_static_score(static_result: dict[str, Any]) -> float:
    if not component_is_active(static_result):
        return 0.0

    score = clamp(float(static_result.get("score", static_result.get("static_score", 0.0)) or 0.0))
    high_risk_count = len(static_result.get("high_risk_list", []) or [])
    context_issue_count = len(static_result.get("context_issues", []) or [])
    verdict_is_vulnerable = static_result.get("verdict") == "Vulnerable"

    if verdict_is_vulnerable:
        risk_floor = STATIC_RISK_SCORE_FLOORS.get(static_result.get("risk_level"), 0.30)
        score = max(score, risk_floor)

    if high_risk_count > 1:
        score += min(0.10, 0.02 * (high_risk_count - 1))
    if context_issue_count > 0:
        score = max(score, min(1.0, 0.50 + 0.05 * context_issue_count))

    return clamp(score)


def calibrate_component_scores(
    llm_result: dict[str, Any],
    rag_result: dict[str, Any],
    static_result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "llm": {
            "raw_score": clamp(float(llm_result.get("score", 0.0) or 0.0)),
            "adjusted_score": calibrated_llm_score(llm_result),
            "active": component_is_active(llm_result),
            "verdict": llm_result.get("verdict"),
            "status": llm_result.get("status"),
        },
        "rag": {
            "raw_score": clamp(float(rag_result.get("score", 0.0) or 0.0)),
            "adjusted_score": calibrated_rag_score(rag_result),
            "active": component_is_active(rag_result),
            "verdict": rag_result.get("final_verdict") or rag_result.get("verdict"),
            "status": rag_result.get("status"),
        },
        "static_analysis": {
            "raw_score": clamp(float(static_result.get("score", static_result.get("static_score", 0.0)) or 0.0)),
            "adjusted_score": calibrated_static_score(static_result),
            "active": component_is_active(static_result),
            "verdict": static_result.get("verdict"),
            "status": static_result.get("status"),
        },
    }


def resolve_effective_weights(
    configured_weights: dict[str, float],
    calibrated_scores: dict[str, dict[str, Any]],
) -> dict[str, float]:
    active_total = sum(
        configured_weights[name]
        for name, item in calibrated_scores.items()
        if item["active"] and configured_weights[name] > 0
    )
    if active_total <= 0:
        return {name: 0.0 for name in configured_weights}
    return {
        name: (configured_weights[name] / active_total if calibrated_scores[name]["active"] else 0.0)
        for name in configured_weights
    }


def apply_evidence_guardrails(
    fusion_score: float,
    rag_result: dict[str, Any],
    static_result: dict[str, Any],
    calibrated_scores: dict[str, dict[str, Any]],
) -> tuple[float, str]:
    score = fusion_score
    reason = "weighted_fusion"
    static_risk = static_result.get("risk_level")
    high_risk_count = len(static_result.get("high_risk_list", []) or [])
    static_score = calibrated_scores["static_analysis"]["adjusted_score"]
    rag_score = calibrated_scores["rag"]["adjusted_score"]

    if static_result.get("verdict") == "Vulnerable" and high_risk_count > 0:
        if static_risk == "Critical" and static_score >= 0.80:
            score = max(score, 0.75)
            reason = "static_critical_evidence_floor"
        elif static_risk == "High" and static_score >= 0.65:
            score = max(score, 0.62)
            reason = "static_high_evidence_floor"

    if (
        static_result.get("verdict") == "Vulnerable"
        and (rag_result.get("final_verdict") == "Vulnerable" or rag_score >= 0.45)
    ):
        score = max(score, 0.60)
        if reason == "weighted_fusion":
            reason = "static_rag_corroboration_floor"

    return clamp(score), reason


def build_final_result(
    code: str,
    args: argparse.Namespace,
    llm_result: dict[str, Any],
    rag_result: dict[str, Any],
    static_result: dict[str, Any],
) -> dict[str, Any]:
    weights = validate_weights(
        {
            "llm": args.llm_weight,
            "rag": args.rag_weight,
            "static_analysis": args.static_weight,
        }
    )

    calibrated_scores = calibrate_component_scores(llm_result, rag_result, static_result)
    effective_weights = resolve_effective_weights(weights, calibrated_scores)
    fusion_score = sum(
        effective_weights[name] * calibrated_scores[name]["adjusted_score"]
        for name in effective_weights
    )
    final_score, decision_rule = apply_evidence_guardrails(
        fusion_score,
        rag_result,
        static_result,
        calibrated_scores,
    )
    verdict = "Vulnerable" if final_score >= args.final_threshold else "Safe"
    vulnerabilities = build_vulnerability_report(code, rag_result, static_result)
    detected_risk_level = highest_risk(rag_result.get("risk_level"), static_result.get("risk_level"))
    final_risk_level = "None"
    if verdict == "Vulnerable":
        final_risk_level = detected_risk_level if vulnerabilities else "Unknown"

    return {
        "final": {
            "is_vulnerable": verdict == "Vulnerable",
            "verdict": verdict,
            "final_score": round(final_score, 6),
            "decision_threshold": args.final_threshold,
            "risk_level": final_risk_level,
            "weights": weights,
            "effective_weights": {name: round(value, 6) for name, value in effective_weights.items()},
            "fusion": {
                "strategy": "calibrated_weighted_fusion_with_evidence_floors",
                "weighted_score_before_guardrails": round(fusion_score, 6),
                "decision_rule": decision_rule,
            },
            "score_breakdown": {
                "llm": {
                    "raw_score": calibrated_scores["llm"]["raw_score"],
                    "adjusted_score": round(calibrated_scores["llm"]["adjusted_score"], 6),
                    "configured_weight": weights["llm"],
                    "weight": round(effective_weights["llm"], 6),
                    "weighted_score": round(effective_weights["llm"] * calibrated_scores["llm"]["adjusted_score"], 6),
                    "verdict": calibrated_scores["llm"]["verdict"],
                    "status": calibrated_scores["llm"]["status"],
                },
                "rag": {
                    "raw_score": calibrated_scores["rag"]["raw_score"],
                    "adjusted_score": round(calibrated_scores["rag"]["adjusted_score"], 6),
                    "configured_weight": weights["rag"],
                    "weight": round(effective_weights["rag"], 6),
                    "weighted_score": round(effective_weights["rag"] * calibrated_scores["rag"]["adjusted_score"], 6),
                    "verdict": calibrated_scores["rag"]["verdict"],
                    "status": calibrated_scores["rag"]["status"],
                },
                "static_analysis": {
                    "raw_score": calibrated_scores["static_analysis"]["raw_score"],
                    "adjusted_score": round(calibrated_scores["static_analysis"]["adjusted_score"], 6),
                    "configured_weight": weights["static_analysis"],
                    "weight": round(effective_weights["static_analysis"], 6),
                    "weighted_score": round(
                        effective_weights["static_analysis"] * calibrated_scores["static_analysis"]["adjusted_score"],
                        6,
                    ),
                    "verdict": calibrated_scores["static_analysis"]["verdict"],
                    "status": calibrated_scores["static_analysis"]["status"],
                },
            },
        },
        "vulnerabilities": vulnerabilities if verdict == "Vulnerable" else [],
        "components": {
            "llm": llm_result,
            "rag": rag_result,
            "static_analysis": static_result,
        },
    }


def read_code(args: argparse.Namespace) -> str:
    if args.code_file:
        return Path(args.code_file).read_text(encoding="utf-8")
    if args.code:
        return args.code
    if CODE_TO_TEST:
        return CODE_TO_TEST
    raise ValueError("Provide Solidity code with --code-file, --code, or CODE_TO_TEST.")


def write_or_print_output(output: dict[str, Any], output_path: str | None) -> None:
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if output_path:
        Path(output_path).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


def run_component_only(args: argparse.Namespace, code: str) -> dict[str, Any]:
    if args.component_only == "llm":
        return run_component(
            "LLM / CodeBERT model-based detection",
            run_llm_component,
            args,
            code,
        )
    if args.component_only == "rag":
        return run_component(
            "RAG / Retrieval-based detection",
            run_rag_component,
            args,
            code,
        )
    if args.component_only == "static":
        return run_component(
            "Static analysis",
            run_static_component,
            args,
            code,
        )
    raise ValueError(f"Unknown component: {args.component_only}")


def component_command(component: str, args: argparse.Namespace, code_file: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--component-only",
        component,
        "--code-file",
        str(code_file),
        "--device",
        args.device,
    ]

    if component == "llm":
        command.extend(
            [
                "--llm-threshold",
                str(args.llm_threshold),
                "--codebert-adapter",
                str(args.codebert_adapter),
                "--codebert-base-model",
                str(args.codebert_base_model),
                "--max-tokens",
                str(args.max_tokens),
            ]
        )
        if args.allow_model_download:
            command.append("--allow-model-download")
    elif component == "rag":
        command.extend(
            [
                "--rag-backend",
                args.rag_backend,
                "--rag-knowledge-dir",
                str(args.rag_knowledge_dir),
                "--tfidf-dir",
                str(args.tfidf_dir),
                "--rag-model-name",
                args.rag_model_name,
                "--swc-registry",
                str(args.swc_registry),
                "--rag-top-k",
                str(args.rag_top_k),
                "--rag-max-findings",
                str(args.rag_max_findings),
                "--rag-max-seq-length",
                str(args.rag_max_seq_length),
                "--rag-max-input-chars",
                str(args.rag_max_input_chars),
            ]
        )
        if args.include_all_agents:
            command.append("--include-all-agents")
    return command


def parse_component_stdout(stdout: str) -> dict[str, Any]:
    rendered = stdout.strip()
    if not rendered:
        raise ValueError("Component process returned empty stdout.")
    try:
        return json.loads(rendered)
    except json.JSONDecodeError:
        start = rendered.find("{")
        if start < 0:
            raise
        return json.loads(rendered[start:])


def isolated_component_result(
    component: str,
    component_name: str,
    args: argparse.Namespace,
    code_file: Path,
) -> dict[str, Any]:
    command = component_command(component, args, code_file)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        error = RuntimeError((completed.stderr or completed.stdout or "").strip())
        if args.fail_on_component_error:
            raise error
        return errored_component(component_name, error)
    try:
        return parse_component_stdout(completed.stdout)
    except Exception as error:
        if args.fail_on_component_error:
            raise
        return errored_component(component_name, error)


def temporary_code_file(code: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".sol",
        prefix="phase1_input_",
        encoding="utf-8",
        delete=False,
    )
    with handle:
        handle.write(code)
    return Path(handle.name)


def should_isolate(args: argparse.Namespace, component: str) -> bool:
    if args.no_isolate_heavy_components or args.component_only:
        return False
    if component == "llm" and args.skip_llm:
        return False
    if component == "rag" and args.skip_rag:
        return False
    return component in {"llm", "rag"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine CodeBERT, RAG, and static analysis into one phase-1 verdict."
    )
    parser.add_argument("--code-file", default=None, help="Path to a Solidity source file. Highest priority.")
    parser.add_argument("--code", default=None, help="Raw Solidity code string. Overrides CODE_TO_TEST.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--component-only", choices=["llm", "rag", "static"], default=None, help=argparse.SUPPRESS)

    parser.add_argument("--llm-weight", type=float, default=DEFAULT_WEIGHTS["llm"])
    parser.add_argument("--rag-weight", type=float, default=DEFAULT_WEIGHTS["rag"])
    parser.add_argument("--static-weight", type=float, default=DEFAULT_WEIGHTS["static_analysis"])
    parser.add_argument("--final-threshold", type=float, default=DEFAULT_FINAL_THRESHOLD)
    parser.add_argument("--llm-threshold", type=float, default=DEFAULT_LLM_THRESHOLD)

    parser.add_argument("--codebert-adapter", default=str(DEFAULT_CODEBERT_ADAPTER))
    parser.add_argument("--codebert-base-model", default=DEFAULT_CODEBERT_BASE_MODEL)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--device", default="cpu", help="auto, cpu, cuda, or mps.")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face downloads for the CodeBERT base model.",
    )

    parser.add_argument("--rag-knowledge-dir", default=str(DEFAULT_RAG_KNOWLEDGE_DIR))
    parser.add_argument("--tfidf-dir", default=str(DEFAULT_TFIDF_DIR))
    parser.add_argument(
        "--rag-backend",
        choices=["tfidf", "semantic"],
        default="semantic",
        help="Use semantic embedding RAG by default. tfidf remains available only for fallback/debug.",
    )
    parser.add_argument("--rag-model-name", default=DEFAULT_RAG_MODEL_NAME)
    parser.add_argument("--swc-registry", default=str(DEFAULT_SWC_REGISTRY))
    parser.add_argument("--rag-top-k", type=int, default=1)
    parser.add_argument("--rag-max-findings", type=int, default=3)
    parser.add_argument("--rag-max-seq-length", type=int, default=512)
    parser.add_argument("--rag-max-input-chars", type=int, default=8000)
    parser.add_argument(
        "--include-all-agents",
        action="store_true",
        help="Semantic RAG only: include raw results for all SWC agents.",
    )
    parser.add_argument(
        "--no-isolate-heavy-components",
        action="store_true",
        help="Run CodeBERT/RAG inside the main Python process instead of separate subprocesses.",
    )

    parser.add_argument("--skip-llm", action="store_true", help="Debug only: skip CodeBERT.")
    parser.add_argument("--skip-rag", action="store_true", help="Debug only: skip RAG.")
    parser.add_argument("--skip-static", action="store_true", help="Debug only: skip static analysis.")
    parser.add_argument(
        "--memory-safe",
        action="store_true",
        help="Run embedding RAG on CPU with serialized execution and conservative input limits.",
    )
    parser.add_argument(
        "--fail-on-component-error",
        action="store_true",
        help="Raise immediately if any component fails instead of returning an error block.",
    )
    return parser.parse_args()


def apply_memory_safe_mode(args: argparse.Namespace) -> argparse.Namespace:
    if not args.memory_safe:
        return args
    args.rag_backend = "semantic"
    args.device = "cpu"
    args.rag_top_k = min(args.rag_top_k, 1)
    args.rag_max_input_chars = min(args.rag_max_input_chars, 6000)
    args.include_all_agents = False
    args.no_isolate_heavy_components = False
    return args


def run_pipeline_component(
    component: str,
    component_name: str,
    runner: Any,
    args: argparse.Namespace,
    code: str,
    code_file: Path,
) -> dict[str, Any]:
    if should_isolate(args, component):
        return isolated_component_result(component, component_name, args, code_file)
    return run_component(component_name, runner, args, code)


def main() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    args = apply_memory_safe_mode(parse_args())
    code = read_code(args)
    if args.component_only:
        write_or_print_output(run_component_only(args, code), args.output)
        return

    created_temp_code_file = args.code_file is None
    code_file = Path(args.code_file) if args.code_file else temporary_code_file(code)
    try:
        llm_result = run_pipeline_component(
            "llm",
            "LLM / CodeBERT model-based detection",
            run_llm_component,
            args,
            code,
            code_file,
        )
        rag_result = run_pipeline_component(
            "rag",
            "RAG / Retrieval-based detection",
            run_rag_component,
            args,
            code,
            code_file,
        )
        static_result = run_pipeline_component(
            "static",
            "Static analysis",
            run_static_component,
            args,
            code,
            code_file,
        )
        output = build_final_result(code, args, llm_result, rag_result, static_result)
        write_or_print_output(output, args.output)
    finally:
        if created_temp_code_file:
            code_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
