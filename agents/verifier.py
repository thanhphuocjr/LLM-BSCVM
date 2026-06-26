"""
verifier.py
──────────────────────────────────────────────────────────────────────────────
Patch Verification Agent (Verifier) — agent #5 of the LLM-BSCVM framework.

Consumes the Fixer's repaired contract (phase 4) and confirms the patch is
actually correct before it is trusted, combining two complementary signals:

    1. Static re-detection  -> re-run the phase-1 detector on the fixed contract
       and check that every TARGETED vulnerability (by SWC id) is gone and that
       no NEW vulnerability was introduced. Deterministic, no LLM. (Injected by
       the phase-5 driver so the agent stays free of subprocess concerns.)

    2. Adversarial LLM review -> for each targeted vulnerability, ask a security
       reviewer whether the fix truly removes the root cause in the repaired
       code, actively trying to construct a bypass. Plus one functionality /
       public-interface preservation check (the patch must not break behaviour).

The two signals are fused into a per-vulnerability status (fixed /
partially_fixed / not_fixed / regressed) and an overall verdict
(PASS / NEEDS_REVIEW / FAIL), mirroring the paper's closed-loop "detect ->
repair -> verify" idea: a fix is only accepted once verification confirms it.

Decompose -> one verification per targeted vulnerability
Retrieve  -> reuses the Fixer's change summaries + the static re-detection result
Generate  -> adversarial fix-validity judgement via a generative LLM (Gemini)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

try:
    from .llm_client import build_llm_client
except ImportError:  # allow running as a script
    from llm_client import build_llm_client

RISK_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "None": 0, "Unknown": 0}

# Combined per-vulnerability outcomes.
STATUS_FIXED = "fixed"
STATUS_PARTIAL = "partially_fixed"
STATUS_NOT_FIXED = "not_fixed"
STATUS_REGRESSED = "regressed"  # fix introduced a new problem for this finding

VERIFIER_SYSTEM_INSTRUCTION = (
    "You are an adversarial smart-contract security reviewer validating a patch. "
    "Given an original vulnerability and the COMPLETE repaired contract, decide "
    "whether the fix genuinely removes the root cause. Be skeptical: actively try "
    "to construct an exploit or bypass that still works against the repaired code. "
    "A fix is only 'fixed' if you cannot find any residual exploitation path; if a "
    "partial mitigation remains exploitable under some condition it is "
    "'partially_fixed'; if the vulnerability is essentially untouched it is "
    "'not_fixed'. Judge ONLY the repaired code you are given, not what the patch "
    "notes claim. Return ONLY valid JSON matching the requested schema."
)

VERIFY_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": [STATUS_FIXED, STATUS_PARTIAL, STATUS_NOT_FIXED]},
        "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
        "bypass_scenario": {"type": "string"},
        "residual_risk": {"type": "string"},
        "justification": {"type": "string"},
    },
    "required": ["status", "confidence", "bypass_scenario", "residual_risk", "justification"],
    "propertyOrdering": ["status", "confidence", "bypass_scenario", "residual_risk", "justification"],
}

FUNCTIONALITY_SYSTEM_INSTRUCTION = (
    "You are a Solidity code reviewer checking that a security patch preserved the "
    "contract's original behaviour. Compare the original and repaired contracts and "
    "decide whether the public interface (function signatures, visibility, events) "
    "and the intended functionality are preserved, and whether the patch introduced "
    "any NEW bug or vulnerability. Return ONLY valid JSON matching the schema."
)

FUNCTIONALITY_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "interface_preserved": {"type": "boolean"},
        "functionality_preserved": {"type": "boolean"},
        "introduced_issues": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["interface_preserved", "functionality_preserved", "introduced_issues", "notes"],
    "propertyOrdering": ["interface_preserved", "functionality_preserved", "introduced_issues", "notes"],
}

# Matches public/external function and event signatures for a cheap interface diff.
_SIGNATURE_RE = re.compile(
    r"\b(function|event|constructor)\b[^\{;]*", re.IGNORECASE
)


@dataclass
class VulnVerification:
    swc: str
    vulnerability_name: str
    risk_level: str
    repair_priority: int
    reported_addressed: bool          # what the Fixer claimed
    still_detected: bool | None       # static re-detection (None = not run)
    llm_status: str                   # adversarial LLM verdict
    status: str                       # fused final status
    confidence: str
    bypass_scenario: str
    residual_risk: str
    justification: str
    error: str | None = None


@dataclass
class VerificationResult:
    agent: str = "Verifier (Patch Verification)"
    llm_model: str = ""
    overall_verdict: str = "NEEDS_REVIEW"   # PASS | NEEDS_REVIEW | FAIL
    target_count: int = 0
    fixed_count: int = 0
    not_fixed_count: int = 0
    resolution_rate: float = 0.0
    redetection_ran: bool = False
    introduced_vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    removed_signatures: List[str] = field(default_factory=list)
    interface_preserved: bool | None = None
    functionality_preserved: bool | None = None
    introduced_issues: List[str] = field(default_factory=list)
    verifications: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    error: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Input adaptation
# ──────────────────────────────────────────────────────────────────────────────
def _targets_of(fix_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The vulnerabilities the Fixer set out to repair (its `changes` list)."""
    changes = fix_result.get("changes")
    if isinstance(changes, list):
        return [c for c in changes if isinstance(c, dict)]
    return []


def _vulnerabilities_of(detection_result: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not detection_result:
        return []
    vulns = detection_result.get("vulnerabilities")
    return [v for v in vulns if isinstance(v, dict)] if isinstance(vulns, list) else []


def _norm_swc(value: object) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def extract_signatures(code: str) -> List[str]:
    """Normalised public function / event / constructor signatures for diffing."""
    signatures: set[str] = set()
    for match in _SIGNATURE_RE.finditer(code or ""):
        signature = " ".join(match.group(0).split())
        # Drop private/internal helpers: interface preservation only cares about
        # the externally observable surface.
        if re.search(r"\b(private|internal)\b", signature, re.IGNORECASE):
            continue
        signatures.add(signature)
    return sorted(signatures)


# ──────────────────────────────────────────────────────────────────────────────
# Verifier agent
# ──────────────────────────────────────────────────────────────────────────────
class VerifierAgent:
    def __init__(self, llm: Any | None = None, backend: str = "auto") -> None:
        self.llm = llm or build_llm_client(backend)

    # ── per-vulnerability adversarial review ─────────────────────────────────
    @staticmethod
    def _build_vuln_prompt(fixed_code: str, target: Dict[str, Any], still_detected: bool | None) -> str:
        schema = (
            "{\n"
            '  "status": "fixed|partially_fixed|not_fixed",\n'
            '  "confidence": "High|Medium|Low",\n'
            '  "bypass_scenario": string,   // a still-working exploit, or "" if none found\n'
            '  "residual_risk": string,     // what (if anything) remains exploitable\n'
            '  "justification": string      // grounded in the repaired code\n'
            "}"
        )
        detector_note = ""
        if still_detected is True:
            detector_note = (
                "NOTE: the static detector STILL flags this vulnerability class in the "
                "repaired code — scrutinise whether the fix is real.\n"
            )
        elif still_detected is False:
            detector_note = (
                "NOTE: the static detector no longer flags this vulnerability class, but "
                "confirm independently that the root cause is gone.\n"
            )
        return "\n".join(
            [
                "Verify whether the following vulnerability is genuinely fixed in the repaired contract.",
                "",
                "<Vulnerability>",
                f"SWC: {target.get('swc') or 'Unknown'}",
                f"Name: {target.get('vulnerability_name') or 'Unknown vulnerability'}",
                f"Risk level: {target.get('risk_level') or 'Unknown'}",
                f"Patch note from the fixer: {str(target.get('change_summary') or '(none)')[:600]}",
                "</Vulnerability>",
                "",
                detector_note.rstrip(),
                "<RepairedContract>",
                "```solidity",
                (fixed_code or "").strip()[:12000],
                "```",
                "</RepairedContract>",
                "",
                "Return ONLY a JSON object with exactly this schema:",
                schema,
                "Do not wrap the JSON in markdown fences.",
            ]
        )

    def _verify_one(
        self, fixed_code: str, target: Dict[str, Any], still_detected: bool | None
    ) -> VulnVerification:
        swc = target.get("swc") or "Unknown"
        name = target.get("vulnerability_name") or "Unknown vulnerability"
        base = dict(
            swc=swc,
            vulnerability_name=name,
            risk_level=target.get("risk_level") or "Unknown",
            repair_priority=int(target.get("repair_priority") or 0),
            reported_addressed=bool(target.get("addressed", False)),
            still_detected=still_detected,
        )

        try:
            data = self.llm.generate_json(
                self._build_vuln_prompt(fixed_code, target, still_detected),
                VERIFIER_SYSTEM_INSTRUCTION,
                response_schema=VERIFY_RESPONSE_SCHEMA,
            )
        except Exception as error:  # noqa: BLE001 - reported per finding
            return VulnVerification(
                **base,
                llm_status="",
                status=STATUS_NOT_FIXED if still_detected else STATUS_PARTIAL,
                confidence="Low",
                bypass_scenario="",
                residual_risk="",
                justification="",
                error=f"{type(error).__name__}: {error}",
            )

        llm_status = str(data.get("status") or "").strip().lower()
        if llm_status not in {STATUS_FIXED, STATUS_PARTIAL, STATUS_NOT_FIXED}:
            llm_status = STATUS_PARTIAL

        return VulnVerification(
            **base,
            llm_status=llm_status,
            status=self._fuse_status(llm_status, still_detected),
            confidence=str(data.get("confidence") or "").strip() or "Low",
            bypass_scenario=str(data.get("bypass_scenario") or "").strip(),
            residual_risk=str(data.get("residual_risk") or "").strip(),
            justification=str(data.get("justification") or "").strip(),
        )

    @staticmethod
    def _fuse_status(llm_status: str, still_detected: bool | None) -> str:
        """Combine the deterministic detector signal with the LLM judgement.

        The detector is conservative on recall; the LLM is the semantic check.
        We only declare a finding 'fixed' when neither signal objects.
        """
        if still_detected is True:
            # Detector still sees it: at best a partial fix, never 'fixed'.
            return STATUS_NOT_FIXED if llm_status == STATUS_NOT_FIXED else STATUS_PARTIAL
        # Detector clear (or not run) -> trust the adversarial LLM verdict.
        return llm_status

    # ── functionality / interface preservation ──────────────────────────────
    def _check_functionality(self, original_code: str, fixed_code: str) -> Dict[str, Any]:
        prompt = "\n".join(
            [
                "Compare the two contracts and judge whether the patch preserved behaviour.",
                "",
                "<OriginalContract>",
                "```solidity",
                (original_code or "").strip()[:10000],
                "```",
                "</OriginalContract>",
                "",
                "<RepairedContract>",
                "```solidity",
                (fixed_code or "").strip()[:10000],
                "```",
                "</RepairedContract>",
                "",
                "Return ONLY a JSON object with exactly this schema:",
                "{\n"
                '  "interface_preserved": boolean,\n'
                '  "functionality_preserved": boolean,\n'
                '  "introduced_issues": [string],   // new bugs/vulns the patch added, [] if none\n'
                '  "notes": string\n'
                "}",
                "Do not wrap the JSON in markdown fences.",
            ]
        )
        return self.llm.generate_json(
            prompt, FUNCTIONALITY_SYSTEM_INSTRUCTION, response_schema=FUNCTIONALITY_RESPONSE_SCHEMA
        )

    # ── orchestration ────────────────────────────────────────────────────────
    def verify(
        self,
        original_code: str,
        fix_result: Dict[str, Any],
        redetection_result: Dict[str, Any] | None = None,
        original_detection: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Verify the Fixer's patch.

        original_code       — the pre-repair contract.
        fix_result          — the Fixer (phase-4) output (its `fixed_code` + `changes`).
        redetection_result  — phase-1 detection re-run on the FIXED code (optional but
                              strongly recommended; supplies the deterministic signal).
        original_detection  — phase-1 detection on the ORIGINAL code, used to attribute
                              newly-introduced vulnerabilities.
        """
        fixed_code = str(fix_result.get("fixed_code") or "").strip()
        targets = _targets_of(fix_result)

        if not fixed_code:
            return asdict(
                VerificationResult(
                    llm_model=getattr(self.llm, "model_name", ""),
                    target_count=len(targets),
                    overall_verdict="FAIL",
                    error="Fixer produced no fixed_code to verify.",
                )
            )

        # Deterministic re-detection signal: which SWCs are still flagged after the fix?
        redetection_ran = redetection_result is not None
        still_detected_swcs = {
            _norm_swc(v.get("swc")) for v in _vulnerabilities_of(redetection_result)
        }
        original_swcs = {_norm_swc(v.get("swc")) for v in _vulnerabilities_of(original_detection)}
        target_swcs = {_norm_swc(t.get("swc")) for t in targets}

        # Vulnerabilities present after the fix that were NOT targets and not in the
        # original detection → newly introduced by the patch (a regression).
        introduced = [
            v
            for v in _vulnerabilities_of(redetection_result)
            if _norm_swc(v.get("swc")) not in original_swcs
            and _norm_swc(v.get("swc")) not in target_swcs
        ]

        # Per-vulnerability adversarial verification.
        verifications: List[VulnVerification] = []
        for target in targets:
            still = (
                _norm_swc(target.get("swc")) in still_detected_swcs if redetection_ran else None
            )
            verifications.append(self._verify_one(fixed_code, target, still))

        # Cheap deterministic interface diff (dropped public surface).
        original_sigs = set(extract_signatures(original_code))
        fixed_sigs = set(extract_signatures(fixed_code))
        removed_signatures = sorted(original_sigs - fixed_sigs)

        # LLM functionality / interface preservation check.
        interface_preserved: bool | None = None
        functionality_preserved: bool | None = None
        introduced_issues: List[str] = []
        notes = ""
        func_error: str | None = None
        try:
            func = self._check_functionality(original_code, fixed_code)
            interface_preserved = bool(func.get("interface_preserved"))
            functionality_preserved = bool(func.get("functionality_preserved"))
            introduced_issues = [str(x) for x in (func.get("introduced_issues") or []) if str(x).strip()]
            notes = str(func.get("notes") or "").strip()
        except Exception as error:  # noqa: BLE001
            func_error = f"{type(error).__name__}: {error}"

        # A dropped public signature is hard evidence the interface changed,
        # regardless of what the LLM concluded.
        if removed_signatures and interface_preserved is not False:
            interface_preserved = False

        fixed_count = sum(1 for v in verifications if v.status == STATUS_FIXED)
        not_fixed_count = sum(1 for v in verifications if v.status in (STATUS_NOT_FIXED, STATUS_REGRESSED))
        target_count = len(verifications)
        resolution_rate = round(fixed_count / target_count, 4) if target_count else 0.0

        overall = self._overall_verdict(
            target_count=target_count,
            fixed_count=fixed_count,
            not_fixed_count=not_fixed_count,
            introduced=introduced,
            introduced_issues=introduced_issues,
            interface_preserved=interface_preserved,
            functionality_preserved=functionality_preserved,
        )

        return asdict(
            VerificationResult(
                llm_model=getattr(self.llm, "model_name", ""),
                overall_verdict=overall,
                target_count=target_count,
                fixed_count=fixed_count,
                not_fixed_count=not_fixed_count,
                resolution_rate=resolution_rate,
                redetection_ran=redetection_ran,
                introduced_vulnerabilities=introduced,
                removed_signatures=removed_signatures,
                interface_preserved=interface_preserved,
                functionality_preserved=functionality_preserved,
                introduced_issues=introduced_issues,
                verifications=[asdict(v) for v in verifications],
                notes=notes,
                error=func_error,
            )
        )

    @staticmethod
    def _overall_verdict(
        *,
        target_count: int,
        fixed_count: int,
        not_fixed_count: int,
        introduced: List[Dict[str, Any]],
        introduced_issues: List[str],
        interface_preserved: bool | None,
        functionality_preserved: bool | None,
    ) -> str:
        """PASS only when every target is fixed and nothing regressed."""
        broke_contract = (
            bool(introduced)
            or bool(introduced_issues)
            or interface_preserved is False
            or functionality_preserved is False
        )
        if not_fixed_count > 0 or broke_contract:
            return "FAIL"
        if target_count and fixed_count == target_count:
            return "PASS"
        return "NEEDS_REVIEW"


def run_verifier(
    original_code: str,
    fix_result: Dict[str, Any],
    redetection_result: Dict[str, Any] | None = None,
    original_detection: Dict[str, Any] | None = None,
    backend: str = "auto",
) -> Dict[str, Any]:
    return VerifierAgent(backend=backend).verify(
        original_code, fix_result, redetection_result, original_detection
    )
