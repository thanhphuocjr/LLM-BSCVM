# Smart Contract Vulnerability RAG

Project được chia thành 4 khu vực chính:

- `static_analysis/`: phân tích tĩnh bằng rule/regex cho Solidity.
- `rag/`: pipeline RAG, TF-IDF retriever, knowledge store và multi-agent SWC retriever.
- `codelama_results/`: script và output phân loại SWC bằng LLM/CodeLlama-style workflow.
- `dataset/`: toàn bộ dữ liệu cần thiết cho project.

## Dataset

- `dataset/raw/solodit/solodit.json`: Solodit nguyên bản, được giữ lại.
- `dataset/raw/dappscan/dappscan.json`: Dappscan nguyên bản, được giữ lại.
- `dataset/swc_registry.json`: registry SWC dùng cho agent/query.
- `dataset/processed/`: dữ liệu đã xử lý hoặc làm giàu.
- `dataset/auxiliary/`: dữ liệu phụ trợ khi merge/build dataset.
- `dataset/tools/`: script xử lý dataset.

## Lệnh chính

Chạy multi-agent RAG retriever:

```bash
python3 rag/retrieve_best_vuln.py --code-file path/to/Contract.sol
```

Mặc định lệnh này chạy đủ 37 SWC agents nhưng chỉ trả về top 3 finding liên quan nhất. Nếu cần đổi số lượng finding:

```bash
python3 rag/retrieve_best_vuln.py --code-file path/to/Contract.sol --max-findings 5
```

Nếu cần debug toàn bộ 37 agent:

```bash
python3 rag/retrieve_best_vuln.py --code-file path/to/Contract.sol --include-all-agents
```

Chạy retriever một query kiểu cũ:

```bash
python3 rag/retrieve_best_vuln.py --single-query --query "Check reentrancy" --code-file path/to/Contract.sol
```

Rebuild knowledge store:

```bash
python3 rag/build_knowledge_store.py
```

Rebuild TF-IDF store:

```bash
python3 rag/build_tfidf_corpus.py
```
