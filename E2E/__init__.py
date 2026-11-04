"""
LLM-BSCVM - hệ thống kiểm toán lỗ hổng smart contract Solidity (phần chạy trên máy local).

Thư mục theo đúng thứ tự pipeline:
  pipeline/          luồng chạy: orchestrator (Phase 0 -> 6) + gpu_pipeline (Phase 1C -> 5)
  phase0_parsing/    tách contract thành code unit (hàm, constructor, fallback/receive)
  phase1a_static/    phân tích tĩnh: luật regex/AST + Slither
  phase1b_rag/       RAG: xây cơ sở tri thức + truy hồi TF-IDF + 37 SWC agent
  phase1c_detector/  detector Qwen2.5-Coder-14B + LoRA v13 (gọi GPU server)
  phase1_fusion/     hợp nhất static + RAG + detector theo từng finding
  phase2_advisor/    Advisor: lọc false positive, nguyên nhân, hướng sửa
  phase3_assessor/   Assessor: CVSS v3.1, mức rủi ro, thứ tự ưu tiên
  phase4_fixer/      Fixer: vá từng hàm, guard hành vi, vá dự phòng
  phase5_verifier/   kiểm chứng bản vá: compile, detector, static/RAG, Slither, AST
  phase6_reporter/   báo cáo Markdown / HTML / PDF
  common/            dùng chung: client GPU server, solc, chạy Slither/SmartBugs, IO
  experiments/       chạy thực nghiệm: vòng 1 (batch_audit) và vòng 2 (repair_round2)
  evaluation/        đánh giá: benchmark hồi quy, static/RAG trên test split, 9 tool
  data_tools/        công cụ dữ liệu: bỏ comment SmartBugs-Wild, manifest toàn vẹn
"""

from .config import E2E_ROOT

__all__ = ["E2E_ROOT"]
