"""Identifier of one finding, shared by fusion, Advisor, Assessor, Fixer and Verifier outputs."""

from __future__ import annotations

from typing import Any


def finding_key(finding: dict[str, Any]) -> str:
    """Stable identifier shared by fusion, Advisor, Assessor and Verifier outputs."""
    return str(
        finding.get("finding_id")
        or f"{finding.get('swc') or 'SWC-100'}:{finding.get('unit_id') or 'contract'}"
    )
