"""
Phase 1B - RAG trên cơ sở tri thức lỗ hổng.

build_knowledge_base.py  (bước offline) xây cơ sở tri thức: data/knowledge_base/source -> built.
swc_agents.py            37 SWC agent: mẫu tín hiệu + điều kiện kích hoạt cho từng SWC.
retriever.py             truy hồi TF-IDF (char 3-5 gram) các ca lỗi tương tự + khớp chính xác đoạn mã.
rag_runner.py            run_rag: chạy retriever cho từng unit, trả finding + ngữ cảnh cho Fixer.
"""
