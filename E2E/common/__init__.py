"""
Phần dùng chung cho nhiều phase.

gpu_client.py      HTTP tới GPU server: health, detect (Phase 1C/5), generate (Phase 2/3/4).
security_tools.py  chạy Slither native (Phase 1A, Phase 5), chuẩn hoá finding.
solc.py            chọn solc theo pragma, biên dịch, so với bản gốc (baseline).
findings.py        finding_key: định danh một finding giữa các phase.
safe_run.py        chạy một thành phần, lỗi thành status=error thay vì dừng cả pipeline.
io_utils.py        đọc/ghi JSON, SHA-256, thời gian UTC.
regex_patterns.py  MỌI regex mà các hàm Phase 0 -> 6 dùng, đặt tên theo nội dung (RE_XXX, re_xxx(name)).
"""
