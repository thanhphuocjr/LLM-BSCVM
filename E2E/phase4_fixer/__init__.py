"""
Phase 4 - Fixer.

fixer.py               prompt + gọi LLM + vá từng hàm (patch_unit).
patching.py            tách hàm thay thế từ câu trả lời của model, ghép vào contract theo unit id.
behavior_guard.py      bản vá phải giữ chuyển ETH, ghi trạng thái, phát event của hàm gốc.
"""
