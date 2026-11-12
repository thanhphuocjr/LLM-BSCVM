"""
Luồng chạy chính.

orchestrator.py   Phase 0 -> 6 cho một contract (CLI: python -m E2E.run <file.sol>, web app gọi lại CLI).
gpu_pipeline.py   Phase 1C -> 5 (chạy local, gọi model trên GPU server qua HTTP): detector, fusion, Advisor, Assessor, Fixer + vòng sửa <= 3 lần.
"""
