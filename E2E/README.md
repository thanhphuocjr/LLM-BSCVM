# CodeLlama Smart Contract Audit E2E

Kho này tự chứa toàn bộ runtime cần cho khóa luận: static analysis đã khôi
phục, RAG local, adapter CodeLlama detector, dữ liệu nguồn, notebook Kaggle,
orchestrator sáu phase và bộ sinh báo cáo audit.

## Quick Start Kaggle

Chạy từ thư mục gốc `/Users/phuocthanh/Documents/RAG`:

```bash
.venv/bin/python -m pip install -r E2E/requirements-local.txt
.venv/bin/python -m E2E.rag.build_store
.venv/bin/python -m E2E.run E2E/examples/VulnerableBank.sol
```

Lệnh trên tạo request local, submit notebook private lên Kaggle, tải
`result.json`, rồi hoàn tất verification/report ở local. Nếu chỉ muốn debug
offline không submit Kaggle, chạy thêm `--backend local_fast` hoặc
`--backend local`.

## Full E2E Through Kaggle

Notebook:

```text
E2E/kaggle/codellama_e2e.ipynb
```

Metadata kernel đã đặt:

- private kernel: `thanhphuocjr/codellama-e2e-smart-contract-audit`
- adapter source: output của `ntpuet/codellama`
- detector base: `metaresearch/codellama/PyTorch/7b-hf/1`
- agent model: `metaresearch/codellama/PyTorch/7b-instruct-hf/1`
- GPU cố định `NvidiaTeslaT4`; Internet chỉ bật để nâng `bitsandbytes>=0.46.1`
  khi Kaggle image cài bản cũ

Sau khi chấp nhận license Code Llama trên Kaggle, chạy một lệnh:

```bash
.venv/bin/python -m E2E.run path/to/Contract.sol
```

Orchestrator sẽ tạo request, nhúng request vào notebook private, push kernel,
poll trạng thái và tải `result.json`. Sau mỗi patch, local chạy lại Static/RAG
và kết hợp với CodeLlama detector trên fixed code. Nếu một trong ba kênh còn
`Vulnerable`, bằng chứng Static/RAG cùng lý do từ LLM verifier được gửi lại cho
Fixer ở attempt kế tiếp. Vòng lặp dừng ngay khi cả ba kênh `Safe` và không bao
giờ vượt quá 3 attempts; kết quả attempt cuối vẫn được compile, chạy SmartBugs
9-tool matrix và đưa vào báo cáo dù patch pass hay fail.

Artifact từng attempt nằm trong
`E2E/runs/<run_id>/repair_attempts/attempt-<n>/`. Chế độ `--remote-result` thủ
công xử lý đúng một kết quả; vòng lặp tự động ba lượt được điều phối bởi
backend mặc định `kaggle`. Cờ `--submit-kaggle` vẫn còn để tương thích.

Mặc định Kaggle run đã bật các tối ưu tốc độ không bỏ detector/fixer chính:
skip generator khi cả local evidence và CodeLlama detector đều `Safe`, bỏ lượt
LLM reporter, bỏ lượt LLM explanation phụ, giảm generation budget còn 900 token
và poll Kaggle mỗi 8 giây. Khi cần bản narrative đầy đủ hơn, dùng thêm
`--llm-reporter` hoặc `--llm-detector-explanation`. Khi muốn nhanh hơn nữa cho
debug patch, dùng `--fast-agents` để advisor/assessor lấy fallback tất định và
chỉ giữ CodeLlama generation cho fixer/verifier.

Phase 5 mặc định chạy đủ SmartBugs 9-tool matrix cho Kaggle/full runs. Nếu cần
chạy nhanh kiểu Slither-only để debug, truyền rõ `--tools slither`.

Chế độ tách rời khi muốn thao tác Kaggle thủ công:

```bash
# 1. Tạo request local
.venv/bin/python -m E2E.run path/to/Contract.sol --run-id thesis-demo --backend local_fast

# 2. Submit request đã tạo
.venv/bin/python -m E2E.kaggle_job E2E/runs/thesis-demo/request.json

# 3. Ghép output và hoàn tất report
.venv/bin/python -m E2E.run path/to/Contract.sol \
  --run-id thesis-demo \
  --remote-result E2E/runs/thesis-demo/kaggle_result.json
```

## Phase Visibility

Mỗi phase có JSON riêng trong `E2E/runs/<run_id>/`. Báo cáo cuối có cùng khung
7 phần với mẫu đã cung cấp, đồng thời thêm bảng trạng thái phase, source hash,
compiler/security-tool evidence và limitations.

1. Static detector: pattern + context checks, chạy local.
2. RAG detector: TF-IDF trên 37 tài liệu SWC + signal gating, chạy local.
3. CodeLlama detector: adapter sequence classification theo function, chạy Kaggle.
4. Advisor/Assessor/Fixer: CodeLlama Instruct sinh JSON có kiểm tra hình dạng.
5. Verifier: model review có lý do khi `Vulnerable` + detector lại fixed code +
   static/RAG + feedback loop tối đa 3 attempts + compiler + SmartBugs 9-tool
   matrix + kiểm tra bảo toàn hành vi quan sát được.
6. Reporter: CodeLlama tạo narrative; local renderer quyết định cấu trúc và số liệu.

Kết quả chạy thật, số đo và negative control v5/v6 được ghi tại
[`VALIDATION.md`](VALIDATION.md).

## Data and Models

Tài nguyên runtime đều nằm trong `E2E/`:

- `models/codellama-vuln-detector/`: adapter cuối và threshold `0.25`.
- `data/training/detect_v4_functionlevel.jsonl`: 3,558 function, cân bằng nhãn.
- `data/source/vulnerability-knowledge/`: 13,628 mẫu tri thức lỗ hổng thực tế có mã nguồn và bản vá (nguồn tri thức duy nhất).
- `data/source/raw/`: DAppSCAN và iAudit/TrustLLM nguyên liệu, kèm provenance.
- `data/runtime/`: registry chuẩn hóa và TF-IDF artifacts.
- `data_pipeline/dataset_build/`: pipeline tái tạo dataset function-level.
- `training/codellama/`: notebook/script huấn luyện CodeLlama v2/v4 đã dùng để đối chiếu.
- `rag/legacy/`: mã RAG cũ được phục hồi để đối chiếu.

Adapter binary và corpus lớn được giữ vật lý trong E2E nhưng bị loại khỏi Git
thông thường. `data_manifest.json` lưu kích thước và SHA-256 để kiểm tra toàn vẹn.

Đánh giá lại hai detector local trên split test:

```bash
.venv/bin/python -m E2E.tools.evaluate_local_detectors
```

Kết quả hiện được lưu ở
`data/training/reports/local_detector_eval.json`: static F1 khoảng `0.493`,
RAG F1 khoảng `0.259` với precision khoảng `0.615`. Vì vậy fusion hoàn chỉnh
đặt trọng số CodeLlama `0.65`, static `0.20`, RAG `0.15`; local detectors là
kênh bằng chứng và gán SWC, không thay thế classifier.

## Known Limits

- Adapter là binary classifier, không tự gán SWC; attribution dùng static/RAG.
- Run v6 cho thấy adapter chưa hiệu chuẩn tốt trên contract mẫu: function
  `deposit` an toàn vẫn có xác suất vulnerable gần `0.99`. Khi chỉ classifier
  còn dương tính nhưng các kiểm tra tất định đều pass, patch được ghi
  `Inconclusive`, không tự động chấp nhận hay phủ nhận bằng chứng.
- Validation macro F1 của cấu hình adapter khoảng `0.777`; dữ liệu DAppSCAN khó
  hơn Solodit, nên cần báo domain gap trong luận văn.
- SWC Registry không còn được duy trì tích cực và không bao phủ toàn bộ DeFi,
  oracle, cross-chain hoặc business-logic attack.
- Patch do model sinh luôn phải qua manual review và test suite của dự án.
- Contract có import ngoài hoặc pragma không có compiler tương ứng có thể làm
  compile/Slither ở trạng thái `partial`; lỗi này được ghi rõ trong report.
- `manifest.status=complete` chỉ có nghĩa mọi phase đã chạy xong; xem
  `verification.overall_verdict` để biết patch được chấp nhận, từ chối hay còn
  bất đồng.
