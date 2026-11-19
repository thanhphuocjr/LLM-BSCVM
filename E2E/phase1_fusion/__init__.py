"""
Phase 1 (hợp nhất) - kết hợp static + RAG + detector.

fusion.py                 fuse_findings (theo từng finding, cho các agent) và fuse_detection (kết luận contract).
fusion_calibration.json   trọng số logistic (fit trên v12 val+test thật) + ngưỡng tau_agent (chọn theo F2).
"""
