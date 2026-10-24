"""
E2E/shared/phases.py
──────────────────────────────────────────────────────────────────────────────
Định nghĩa nguồn chân lý duy nhất (Single Source of Truth) cho các Phase
trong toàn bộ hệ thống LLM-BSCVM (UI, logs, orchestrator, reporter).
Giải quyết triệt để lỗi U-01, U-02 (tên phase phân tán, nhãn không trung thực).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any, Dict, List


PHASES: List[Dict[str, Any]] = [
    {
        "id": "phase0",
        "key": "ast",
        "step": 0,
        "tag": "PHASE 0",
        "label": "Phase 0: AST Parsing",
        "title": "Solidity AST & Code Units",
        "engine": "Function AST Parser",
        "short_name": "AST Parser",
        "description": "Deconstructs contract into isolated functions, interfaces, and modifier units.",
    },
    {
        "id": "phase1a",
        "key": "static",
        "step": "1A",
        "tag": "PHASE 1A",
        "label": "Phase 1A: Static Analysis",
        "title": "Regex Rules Static Analyzer",
        "engine": "Regex rules",
        "short_name": "Static Rules",
        "description": "Scans for known vulnerable Solidity syntax patterns and anti-patterns.",
    },
    {
        "id": "phase1b",
        "key": "rag",
        "step": "1B",
        "tag": "PHASE 1B",
        "label": "Phase 1B: RAG Knowledge",
        "title": "Vulnerability RAG Retrieval",
        "engine": "TF-IDF Knowledge Store",
        "short_name": "RAG Knowledge",
        "description": "Retrieves closest vulnerability cases and remediation guidance from curated KB.",
    },
    {
        "id": "phase1c",
        "key": "detector",
        "step": "1C",
        "tag": "PHASE 1C",
        "label": "Phase 1C: Neural Detector",
        "title": "Neural Classifier",
        "engine": "Qwen2.5-Coder-14B LoRA",
        "short_name": "Neural Detector",
        "description": "Evaluates functions for vulnerability likelihood using fine-tuned model.",
    },
    {
        "id": "fusion",
        "key": "fusion",
        "step": "1 FUSED",
        "tag": "PHASE 1 FUSED",
        "label": "Phase 1 Fused: Multi-Modal Fusion",
        "title": "Multi-Modal Decision Fusion",
        "engine": "Calibrated Decision Fusion",
        "short_name": "Signal Fusion",
        "description": "Fuses static, RAG, and neural signals to produce finding-level verdicts.",
    },
    {
        "id": "phase2",
        "key": "advisor",
        "step": 2,
        "tag": "PHASE 2",
        "label": "Phase 2: Repair Advisor",
        "title": "Root-Cause & Triage Advisor",
        "engine": "LLM Advisor Agent",
        "short_name": "Repair Advisor",
        "description": "Performs triage, filters false positives, and devises repair strategies.",
    },
    {
        "id": "phase3",
        "key": "assessor",
        "step": 3,
        "tag": "PHASE 3",
        "label": "Phase 3: Risk Assessor",
        "title": "CVSS 3.1 & Impact Assessment",
        "engine": "CVSS 3.1 Standard Assessor",
        "short_name": "Risk Assessor",
        "description": "Evaluates exploitability metrics and calculates CVSS 3.1 base scores.",
    },
    {
        "id": "phase4",
        "key": "fixer",
        "step": 4,
        "tag": "PHASE 4",
        "label": "Phase 4: Patch Fixer",
        "title": "Automated Remediation",
        "engine": "Generative Fixer Agent",
        "short_name": "Patch Fixer",
        "description": "Synthesizes minimal semantic patches adhering to contract invariants.",
    },
    {
        "id": "phase5",
        "key": "verifier",
        "step": 5,
        "tag": "PHASE 5",
        "label": "Phase 5: Patch Verifier",
        "title": "AST Behavioral & Multi-Tier Verification",
        "engine": "solc + AST Guard & Slither",
        "short_name": "Patch Verifier",
        "description": "Verifies patch compilation, observable state invariants, and security tools.",
    },
    {
        "id": "phase6",
        "key": "reporter",
        "step": 6,
        "tag": "PHASE 6",
        "label": "Phase 6: Executive Reporter",
        "title": "Audit Report Compilation",
        "engine": "Deterministic Reporter",
        "short_name": "Executive Reporter",
        "description": "Compiles comprehensive multi-format audit reports (HTML, PDF, Markdown).",
    },
]

PHASE_MAP: Dict[str, Dict[str, Any]] = {p["id"]: p for p in PHASES}
PHASE_KEY_MAP: Dict[str, Dict[str, Any]] = {p["key"]: p for p in PHASES}
PHASE_TAG_MAP: Dict[str, Dict[str, Any]] = {p["tag"]: p for p in PHASES}


def get_phase_meta(phase_id_or_key: str) -> Dict[str, Any]:
    return (
        PHASE_MAP.get(phase_id_or_key)
        or PHASE_KEY_MAP.get(phase_id_or_key)
        or {"id": phase_id_or_key, "label": phase_id_or_key, "engine": "Unknown"}
    )
