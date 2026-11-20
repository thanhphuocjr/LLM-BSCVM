"""Khoá định danh một finding, dùng chung cho fusion, Advisor, Assessor, Fixer, Verifier."""


# Khoá của finding: finding_id nếu có, không thì "<SWC>:<unit_id>" (vd "SWC-107:Bank.withdraw").
def finding_key(finding: dict) -> str:
    return str(finding.get("finding_id") or f"{finding.get('swc') or 'SWC-100'}:{finding.get('unit_id') or 'contract'}")
