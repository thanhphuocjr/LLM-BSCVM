"""Danh mục SWC dùng chung mọi phase: chuẩn hoá mã SWC và mức nghiêm trọng mặc định theo SWC."""

from E2E.common.regex_patterns import RE_SWC_ID

# Mức nghiêm trọng theo SWC (SWC không có trong bảng = Medium).
SEVERITY_BY_SWC = {
    "SWC-104": "Critical", "SWC-105": "Critical", "SWC-106": "Critical", "SWC-107": "Critical", "SWC-112": "Critical",
    "SWC-115": "Critical", "SWC-101": "High", "SWC-113": "High", "SWC-114": "High", "SWC-121": "High",
    "SWC-122": "High", "SWC-124": "High", "SWC-126": "High", "SWC-128": "High",
}


# Chuẩn hoá chuỗi bất kỳ về "SWC-xxx" ("swc-107: Reentrancy" -> "SWC-107").
def normalize_swc_id(value: object) -> str:
    match = RE_SWC_ID.search(str(value or "").upper())
    return match.group(0) if match else str(value or "").strip().upper()


# Mức nghiêm trọng mặc định của một SWC.
def swc_severity(swc: str) -> str:
    return SEVERITY_BY_SWC.get(swc, "Medium")
