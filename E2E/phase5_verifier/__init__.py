"""
Phase 5 - kiểm chứng bản vá.

candidate_check.py     trong vòng sửa (gpu_pipeline): compile so với bản gốc, detector chấm lại, static/RAG chạy lại.
final_verification.py  kiểm chứng cuối (orchestrator): ma trận tool + hành vi + static/RAG, ra overall_verdict.
tool_matrix.py         Slither (+ Mythril, SmartCheck qua SmartBugs) trên bản gốc và bản vá.
behavior_verifier.py   so sánh AST trước/sau để phát hiện hồi quy hành vi.
"""
