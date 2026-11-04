"""
Phase 1C - detector neural Qwen2.5-Coder-14B + LoRA v13 (phân loại nhị phân từng hàm).

detector.py  gửi từng unit lên GPU server (/api/v1/detect), tóm tắt fail-closed.
Code huấn luyện / phục vụ model nằm ở gpu_server/ (train_qwen25_v13.py, serve_v13_api.py).
"""
