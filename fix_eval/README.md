# fix_eval — đánh giá bản vá trên tập test (KB không chứa test, tắt kiểm chứng)

Thư mục này nằm **ngoài pipeline chính** (`E2E/`). Mục đích: lấy tập test của cơ sở tri thức (KB), chạy từng mẫu
qua pipeline Phase 0 → 6 sao cho bản vá không thể "chép" lời giải có sẵn trong KB, rồi so sánh bản vá với
ground truth (phần so sánh do người đánh giá tự làm).

## Các bước

```bash
# Bước 1: tách tập test + build KB không chứa test (tất định, chạy lại ra cùng kết quả)
.venv/bin/python -m fix_eval.split_test_set

# Bước 2: chạy pipeline cho từng mẫu (resume được; dừng êm bằng: touch data/results/fix_eval/STOP)
.venv/bin/python -m fix_eval.run_fix_eval
.venv/bin/python -m fix_eval.run_fix_eval --limit 3          # chạy thử
.venv/bin/python -m fix_eval.run_fix_eval --ids know_01334   # chạy lại đúng mẫu này
.venv/bin/python -m fix_eval.run_fix_eval --export-only      # chỉ xuất lại CSV
```

Chạy nền qua đêm (không cho máy ngủ, giữ chạy khi đóng terminal):

```bash
nohup caffeinate -is arch -arm64 .venv/bin/python -m fix_eval.run_fix_eval > data/results/fix_eval/runner.out 2>&1 &
tail -f data/results/fix_eval/progress.log
```

## Dữ liệu

| File | Nội dung |
| --- | --- |
| `data/fix_eval/test_set_ground_truth.json` | 2,276 mẫu `split = "test"` của `final-vulnerability-knowledge.json`, đủ mọi trường (`is_vulnerable`, `swc_id`, `root_cause`, `mitigation_guidance`, `fixed_code`, ...) + `detector_v12_split` |
| `data/fix_eval/test_input.json` | đầu vào pipeline: **chỉ** `id`, `source_code` (iAudit chạy trước, kaggle_ds2 sau) |
| `data/knowledge_base/source/final-vulnerability-knowledge.no_test.json` | 11,352 mẫu train + val |
| `data/knowledge_base/built_no_test/` | KB build từ file trên (9,933 ca; KB chính `built/` có 11,869) |
| `data/fix_eval/split_manifest.json` | số lượng, SHA-256, kiểm tra rò rỉ (id / mã lỗi / mã đã sửa của test trong KB mới = 0) |
| `data/results/fix_eval/fix_eval_results.csv` | **kết quả**: một dòng / mẫu (cột mô tả ở `CSV_COLUMNS` trong `run_fix_eval.py`) |
| `data/results/fix_eval/runs/<id>/` | artifact đầy đủ của từng mẫu (phase0 … phase6, `pipeline.log`) |
| `data/results/fix_eval/results.jsonl`, `summary.json`, `progress.log` | kết quả thô, thống kê, tiến độ |

## Cách chạy mỗi mẫu

- Giống `python -m E2E.run <id>.sol --no-verify --max-fix-attempts 1`, trong cùng một tiến trình.
- `RAG_KNOWLEDGE_BASE_DIR=data/knowledge_base/built_no_test` trỏ RAG (Phase 1B), Advisor/Assessor (Phase 2/3) và
  ví dụ sửa lỗi của Fixer (Phase 4) sang KB không chứa test. Script dừng nếu bất kỳ nơi nào còn đọc KB khác.
- `--no-verify` tắt Phase 5: Fixer vá đúng một lần, không compile / Slither / chấm lại detector, verdict
  `Not verified: ...`. Mặc định của pipeline (web app, CLI) vẫn bật kiểm chứng.
- Lời gọi GPU server bị lỗi (Advisor/Assessor vốn lặng lẽ dùng tri thức KB thay thế) được đếm; mẫu đó được chạy lại.

## Lưu ý khi đọc kết quả

- `vulnerability_detected` = còn ít nhất một finding sau fusion Phase 1 **và** sau khi Advisor loại báo nhầm;
  `fixed_code` chỉ có khi Fixer vá thành công (`patch_status = patched / partially_patched`).
- Mẫu là một hàm (hoặc vài hàm) đứng riêng, không phải contract hoàn chỉnh: Slither không compile được nên không
  chạy; phát hiện dựa vào luật static, RAG, detector và Advisor.
- Detector Qwen v13 đã học trên split train của dataset v12: 550/692 mẫu iAudit và 284/1,584 mẫu kaggle_ds2 của
  tập test này nằm trong train của detector (cột `detector_v12_split` trong file ground truth).
- kaggle_ds2 là dữ liệu tổng hợp từ 5 template (tên biến có hậu tố ngẫu nhiên): KB không chứa test vẫn còn các ca
  cùng template (khác tên) cho mọi mẫu kaggle. Mẫu iAudit (thực tế) không có ca cùng template trong KB.
