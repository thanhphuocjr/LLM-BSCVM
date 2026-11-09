"""Identifier of one finding, shared by fusion, Advisor, Assessor, Fixer and Verifier outputs."""

from __future__ import annotations

from typing import Any


# Nhiệm vụ: trả về khoá định danh của một finding: dùng "finding_id" nếu có, không thì "<SWC>:<unit_id>"
# (vd "SWC-107:Bank.withdraw").
# Mục đích: cùng một lỗ hổng được nối xuyên suốt fusion -> Advisor -> Assessor -> Fixer -> Verifier
# bằng đúng một khoá, để tra kết quả của phase trước khi xử lý phase sau.
def finding_key(finding: dict[str, Any]) -> str:
    """Stable identifier shared by fusion, Advisor, Assessor and Verifier outputs."""
    return str(
        finding.get("finding_id")
        or f"{finding.get('swc') or 'SWC-100'}:{finding.get('unit_id') or 'contract'}"
    )
