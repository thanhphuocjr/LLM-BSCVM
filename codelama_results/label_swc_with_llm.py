import os, json, time, re
from pathlib import Path
from groq import Groq

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Đổi biến môi trường sang GROQ_API_KEY
API_KEY       = os.getenv("GROQ_API_KEY")
# Sử dụng Llama 3.3 70B của Groq - cực kỳ thông minh và nhanh
MODEL_NAME = "llama-3.1-8b-instant"
DATASET_PATH  = PROJECT_ROOT / "dataset" / "raw" / "solodit" / "solodit.json"
SWC_PATH      = PROJECT_ROOT / "dataset" / "swc_registry.json"
OUTPUT_PATH   = PROJECT_ROOT / "codelama_results" / "dataset_swc_labeled.json"
PROGRESS_PATH = PROJECT_ROOT / "codelama_results" / "swc_progress.json"

BATCH_SIZE          = 20
INTER_BATCH_DELAY   = 5    # Groq rất nhanh, 5 giây nghỉ giữa các batch là đủ an toàn
RETRY_LIMIT         = 3
# ──────────────────────────────────────────────────────────────────────────────

system_instruction = (
    "You are an academic smart contract security researcher. "
    "Classify audit findings into SWC weakness categories. "
    "This is purely academic classification work."
)

client: Groq | None = None
swc_id_set: set[str] = set()
swc_reference = ""


def load_inputs() -> list[dict]:
    global client, swc_id_set, swc_reference

    if not API_KEY:
        raise ValueError("Chưa tìm thấy GROQ_API_KEY. Vui lòng chạy lệnh 'export GROQ_API_KEY=...'")

    client = Groq(api_key=API_KEY)
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    with open(SWC_PATH, encoding="utf-8") as f:
        swc_registry = json.load(f)

    swc_ids = [e["id"] for e in swc_registry]
    swc_id_set = set(swc_ids)
    swc_reference = "\n".join(f"{e['id']}: {e['title']}" for e in swc_registry)
    return dataset

# ── Checkpoint — only cache confirmed SWC IDs ─────────────────────────────────
def load_progress() -> dict:
    if Path(PROGRESS_PATH).exists():
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        cleaned = {k: v for k, v in data.items() if v in swc_id_set or v == "UNKNOWN"}
        if len(cleaned) < len(data):
            print(f"  Cleaned {len(data)-len(cleaned)} invalid cached entries")
        return cleaned
    return {}

def save_progress(progress: dict):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)

# ── Batch classify ────────────────────────────────────────────────────────────
def classify_batch(batch: list[dict]) -> dict[str, str]:
    """
    Send up to BATCH_SIZE samples in a single API call using Groq.
    """
    if client is None:
        raise RuntimeError("Groq client is not initialized. Call load_inputs() from main first.")

    items = "\n\n".join(
        f"[{i+1}] ID={s['id']}\nAUDIT FINDING: {s['completion'][:600]}"
        for i, s in enumerate(batch)
    )

    prompt = f"""Classify each audit finding below into one SWC category.

SWC CATEGORIES:
{swc_reference}

AUDIT FINDINGS TO CLASSIFY:
{items}

INSTRUCTIONS:
- For each item, reply on a new line in this exact format:  [N] SWC-XXX
- Use UNKNOWN if no category matches.
- Output ONLY the numbered results, nothing else.

Example output:
[1] SWC-107
[2] SWC-101
[3] UNKNOWN
"""

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            # Gọi API của Groq
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                model=MODEL_NAME,
                temperature=0.0,
                max_tokens=20 * BATCH_SIZE, # Groq dùng 'max_tokens' thay vì 'max_output_tokens'
            )

            text = chat_completion.choices[0].message.content

            if not text:
                print(f"  Empty response for batch, attempt {attempt}")
                if attempt == RETRY_LIMIT:
                    return {}
                time.sleep(5)
                continue

            results = {}
            for line in text.strip().splitlines():
                m = re.search(r'\[(\d+)\]\s*(SWC-\d+|UNKNOWN)', line.upper())
                if m:
                    idx  = int(m.group(1)) - 1
                    swc  = m.group(2)
                    if 0 <= idx < len(batch):
                        sid = batch[idx]["id"]
                        results[sid] = swc

            return results

        except Exception as e:
            err = str(e)
            if "429" in err:
                print(f"  Rate limited by Groq (attempt {attempt}/{RETRY_LIMIT}) — waiting 10s ...")
                time.sleep(10)
            else:
                print(f"  Error attempt {attempt}: {e}")
                if attempt < RETRY_LIMIT:
                    time.sleep(10)

    return {}

# ── Output ────────────────────────────────────────────────────────────────────
def write_output(dataset, progress):
    output = []
    for s in dataset:
        ns = dict(s)
        if s.get("label") == "vulnerable":
            ns["swc"] = progress.get(s["id"])
        else:
            ns["swc"] = None
        output.append(ns)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    labeled = sum(1 for s in output if s["swc"] and s["swc"] != "UNKNOWN")
    unknown = sum(1 for s in output if s["swc"] == "UNKNOWN")
    pending = sum(1 for s in output if s.get("label") == "vulnerable" and s["swc"] is None)
    safe    = sum(1 for s in output if s.get("label") == "safe")
    print(f"\n✅ Saved → {OUTPUT_PATH}")
    print(f"   Labeled  (SWC)  : {labeled}")
    print(f"   UNKNOWN         : {unknown}")
    print(f"   Pending (retry) : {pending}")
    print(f"   Safe (null)     : {safe}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    dataset = load_inputs()
    progress   = load_progress()
    vulnerable = [s for s in dataset if s.get("label") == "vulnerable"]
    remaining  = [s for s in vulnerable if s["id"] not in progress]

    batches = [remaining[i:i+BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]

    print(f"Model       : {MODEL_NAME}")
    print(f"Dataset     : {len(dataset)} total  |  Vulnerable: {len(vulnerable)}")
    print(f"Cached      : {len(progress)}  |  To process: {len(remaining)}")
    print(f"Batch size  : {BATCH_SIZE}  |  Total batches: {len(batches)}")
    print(f"Delay/batch : {INTER_BATCH_DELAY}s")
    print(f"Est. time   : ~{len(batches)*INTER_BATCH_DELAY/60:.0f} min")
    print("=" * 60)

    done = len(progress)
    try:
        for b_idx, batch in enumerate(batches):
            print(f"\nBatch {b_idx+1}/{len(batches)} ({len(batch)} samples) ...", end=" ", flush=True)

            results = classify_batch(batch)

            # Save results
            progress.update(results)
            save_progress(progress)

            done += len(results)
            classified = [f"{sid}→{swc}" for sid, swc in results.items()]
            print(f"got {len(results)}/{len(batch)} results")
            for r in classified:
                print(f"  {r}")

            labeled_so_far = sum(1 for v in progress.values() if v != "UNKNOWN")
            print(f"  Progress: {done}/{len(vulnerable)} | Labeled: {labeled_so_far}")

            if b_idx < len(batches) - 1:
                time.sleep(INTER_BATCH_DELAY)

    except KeyboardInterrupt:
        print("\n[!] Dừng khẩn cấp bởi người dùng.")
    except Exception as e:
        print(f"\n[!] Lỗi không mong muốn: {e}")

    write_output(dataset, progress)

if __name__ == "__main__":
    main()
