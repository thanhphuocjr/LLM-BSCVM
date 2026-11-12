#!/usr/bin/env python3
"""
serve_v13_api.py
══════════════════════════════════════════════════════════════════════════════
Unified AI Inference Server (FastAPI) for LLM-BSCVM (Version 13).
Deploy chuyên dụng trên GPU 0 (NVIDIA RTX 4070 Ti SUPER 16GB).

Kiến trúc "All-in-One":
  - Dùng chung 1 instance mô hình Qwen2.5-Coder-14B-Instruct trong VRAM (~8.5GB).
  - POST /api/v1/detect   (Phase 1C, chấm lại bản vá ở Phase 5): LoRA v13 + score head + line_head,
    phân loại nhị phân, xác suất theo ngưỡng tau = 0.700.
  - POST /api/v1/generate (Phase 2 Advisor, 3 Assessor, 4 Fixer): tắt adapter, dùng model Instruct gốc.
  - GET  /api/v1/health   trạng thái server, GPU, VRAM.
Client phía máy local: E2E/common/gpu_client.py.

Cổng mặc định: 6868
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from safetensors import safe_open

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("v13_api_server")

# ─── Cấu hình mặc định ───
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPTER_DIR = SCRIPT_DIR / "qwen25-v13-14b-detector-final"
if not DEFAULT_ADAPTER_DIR.exists():
    # Thử tìm ở thư mục models của project
    alt = SCRIPT_DIR.parent / "models" / "qwen25-v13-14b-detector-final"
    if alt.exists():
        DEFAULT_ADAPTER_DIR = alt

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
DEFAULT_THRESHOLD = 0.700
MAX_LEN = 1024

INSTRUCTION_PROMPT = (
    "Analyze the following Solidity smart contract code for security vulnerabilities. "
    "Determine whether this function is Safe or Vulnerable:\n\n"
)

SERVER_START_TIME = time.time()

# ─── Pydantic Models ───
# Body của POST /api/v1/detect: mã một hàm + ngưỡng (tuỳ chọn).
class DetectRequest(BaseModel):
    code: str = Field(..., description="Solidity code or function snippet to analyze")
    threshold: Optional[float] = Field(default=DEFAULT_THRESHOLD, description="Vulnerability classification threshold")


# Kết quả detect: verdict, P(vulnerable), P(safe), ngưỡng, điểm từng dòng (line_scores), thời gian.
class DetectResponse(BaseModel):
    status: str = "ok"
    verdict: str
    score: float
    vulnerable_probability: float
    safe_probability: float
    threshold: float
    line_scores: Optional[List[float]] = None
    elapsed_ms: float


# Body của POST /api/v1/generate: prompt, system prompt, temperature, top_p, max_tokens.
class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt or instructions")
    system_prompt: Optional[str] = Field(default="", description="Optional system prompt")
    max_tokens: Optional[int] = Field(default=1500, description="Max generated tokens")
    temperature: Optional[float] = Field(default=0.2, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, description="Nucleus sampling top_p")


# Kết quả sinh văn bản: response + thời gian.
class ChatResponse(BaseModel):
    status: str = "ok"
    response: str
    elapsed_ms: float


# ─── FastAPI App ───
app = FastAPI(
    title="LLM-BSCVM v13 Unified AI Inference Server",
    description="Dedicated GPU API for Smart Contract Vulnerability Detection & Multi-Agent Repair",
    version="13.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in os.environ.get("SERVER_GPU_CORS_ORIGINS", "").split(",") if origin],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Optional shared secret: when SERVER_GPU_API_KEY is set, every endpoint except the
# health check requires the header `X-API-Key`.
API_KEY = os.environ.get("SERVER_GPU_API_KEY", "").strip()
PUBLIC_PATHS = {"/", "/health", "/api/v1/health"}


# Middleware: nếu đặt SERVER_GPU_API_KEY thì mọi endpoint (trừ "/" và health) bắt buộc header X-API-Key đúng -> sai thì 401.
@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if API_KEY and request.url.path not in PUBLIC_PATHS and request.headers.get("X-API-Key") != API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Missing or invalid X-API-Key"})
    return await call_next(request)


# FastAPI runs sync endpoints in a thread pool; the single GPU model must not be used
# by two requests at once (VRAM spikes, interleaved adapter enable/disable).
GPU_LOCK = threading.Lock()

# ─── Global Model State ───
# Giữ MỘT bản model duy nhất trong VRAM: Qwen2.5-Coder-14B-Instruct 4-bit + LoRA v13 + score_head (phân loại)
# + line_head (điểm từng dòng). Dùng chung cho cả detect (bật LoRA) và generate (tắt LoRA) -> chỉ tốn ~8.5 GB VRAM.
class ModelContainer:
    # Khởi tạo trạng thái rỗng (thiết bị, tokenizer, model, các head, ngưỡng), is_ready=False.
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.causal_model = None
        self.score_head = None
        self.line_head = None
        self.base_model_name = DEFAULT_BASE_MODEL
        self.adapter_dir = DEFAULT_ADAPTER_DIR
        self.best_threshold = DEFAULT_THRESHOLD
        self.is_ready = False

    # Nạp model khi server khởi động: đọc ngưỡng từ threshold_config.json -> tokenizer -> backbone 4-bit NF4 -> gắn LoRA
    # -> đọc score_head từ adapter_model.safetensors (KHÔNG có thì từ chối khởi động, tránh head ngẫu nhiên)
    # -> line_head.pt nếu có -> is_ready=True.
    def load(self, base_model_name: str = DEFAULT_BASE_MODEL, adapter_dir: Path = DEFAULT_ADAPTER_DIR):
        start_t = time.time()
        self.base_model_name = base_model_name
        self.adapter_dir = Path(adapter_dir)

        logger.info("════════════════════════════════════════════════════════════════════")
        logger.info("  KHỞI TẠO MÔ HÌNH V13 TRÊN GPU 0")
        logger.info(f"  Target Device : {self.device}")
        logger.info(f"  Base Model    : {self.base_model_name}")
        logger.info(f"  Adapter Dir   : {self.adapter_dir}")
        logger.info("════════════════════════════════════════════════════════════════════")

        # 1. Đọc threshold config nếu có
        cfg_file = self.adapter_dir / "threshold_config.json"
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
                self.best_threshold = float(cfg.get("best_threshold", DEFAULT_THRESHOLD))
                logger.info(f"Loaded threshold config: best_threshold = {self.best_threshold}")
            except Exception as e:
                logger.warning(f"Could not read threshold_config: {e}")

        # 2. Tokenizer
        from transformers import AutoTokenizer
        logger.info("==> [1/3] Đang nạp tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_dir), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # 3. Model CausalLM với Quantization 4-bit (chỉ tốn ~8.5GB VRAM)
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        quant_config = None
        if "cuda" in self.device:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        logger.info(f"==> [2/3] Đang nạp backbone từ {self.base_model_name}...")
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
            device_map={"": 0} if "cuda" in self.device else None,
            trust_remote_code=True,
        )

        logger.info(f"==> [3/3] Đang gắn LoRA adapter từ {self.adapter_dir}...")
        self.causal_model = PeftModel.from_pretrained(base, str(self.adapter_dir))
        if "cuda" not in self.device:
            self.causal_model.to(self.device)
        self.causal_model.eval()

        # 4. Trích xuất Score Head từ adapter_model.safetensors để làm Detector
        safetensor_file = self.adapter_dir / "adapter_model.safetensors"
        hidden_size = base.config.hidden_size  # 5120 đối với Qwen 14B
        self.score_head = nn.Linear(hidden_size, 2, bias=False).to(self.device)

        score_loaded = False
        if safetensor_file.exists():
            try:
                with safe_open(str(safetensor_file), framework="pt") as f:
                    for k in f.keys():
                        if "score.weight" in k or "classifier.weight" in k:
                            w = f.get_tensor(k)
                            self.score_head.weight = nn.Parameter(w.to(dtype=torch.float32, device=self.device))
                            score_loaded = True
                            logger.info(f"  • Đã nạp thành công score_head tensor: {k} {w.shape}")
                            break
            except Exception as e:
                logger.warning(f"Không thể đọc score_head từ safetensors: {e}")

        if not score_loaded:
            # A randomly initialised head would silently return meaningless probabilities.
            raise RuntimeError(
                f"Không tìm thấy score_head (score.weight/classifier.weight) trong {safetensor_file}; "
                "từ chối khởi động detector với head ngẫu nhiên."
            )
        self.score_head.eval()

        # 5. Line head nếu có
        line_head_file = self.adapter_dir / "line_head.pt"
        if line_head_file.exists():
            try:
                self.line_head = nn.Linear(hidden_size, 1).to(self.device)
                self.line_head.load_state_dict(torch.load(str(line_head_file), map_location=self.device))
                self.line_head.eval()
                logger.info(f"  • Đã nạp thành công line_head từ {line_head_file}")
            except Exception as e:
                logger.warning(f"Could not load line_head.pt: {e}")

        self.is_ready = True
        elapsed = time.time() - start_t
        logger.info(f"✅ KHỞI ĐỘNG XONG MÔ HÌNH V13 TRONG {elapsed:.1f}s! ĐÃ SẴN SÀNG PHỤC VỤ API.")


CONTAINER = ModelContainer()


# ─── API Endpoints ───
# GET /: thông tin chung của server.
@app.get("/")
def root():
    return {
        "service": "LLM-BSCVM v13 Unified AI Inference Server",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
        "endpoints": [
            "POST /api/v1/detect (Phase 1C classification)",
            "POST /api/v1/generate (Phase 2-4 generative agents)",
        ],
    }


# GET /api/v1/health: trạng thái (ready/loading), GPU, VRAM, tên model. Client local dùng để biết server sẵn sàng chưa.
@app.get("/api/v1/health")
@app.get("/health")
def health_check():
    vram_used = 0.0
    vram_total = 0.0
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / (1024 * 1024)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    return {
        "status": "ready" if CONTAINER.is_ready else "initializing",
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_used_mb": round(vram_used, 1),
        "vram_total_mb": round(vram_total, 1),
        "model_name": "qwen25-v13-14b-detector-final",
        "base_model": CONTAINER.base_model_name,
        "best_threshold": CONTAINER.best_threshold,
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
    }


# [PHASE 1C trên server] POST /api/v1/detect: ghép INSTRUCTION_PROMPT + mã (giống hệt lúc train), cắt MAX_LEN token,
# forward lấy hidden state của token CUỐI -> score_head -> softmax -> P(vulnerable); line_head -> xác suất từng token
# -> gộp về từng dòng mã (qua offset của tokenizer). verdict = Vulnerable nếu P >= ngưỡng. Dùng GPU_LOCK (1 request/lần).
@app.post("/api/v1/detect", response_model=DetectResponse)
def detect_vulnerability(req: DetectRequest):
    if not CONTAINER.is_ready:
        raise HTTPException(status_code=503, detail="Mô hình đang khởi tạo, vui lòng thử lại sau vài giây.")

    start_t = time.time()
    code = req.code.strip()
    input_text = f"{INSTRUCTION_PROMPT}{code}"

    with GPU_LOCK, torch.no_grad():
        encoded = CONTAINER.tokenizer(
            input_text,
            truncation=True,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        inputs = encoded.to(CONTAINER.device)

        # Forward pass qua CausalLM với output_hidden_states=True
        outputs = CONTAINER.causal_model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[-1]  # [1, seq_len, 5120]

        # Single unpadded sequence: the pooled token is the last one, as in training.
        last_token_idx = inputs["input_ids"].shape[-1] - 1
        last_hidden = hidden_states[0, last_token_idx].unsqueeze(0)
        logits = CONTAINER.score_head(last_hidden.float())
        probs = torch.softmax(logits, dim=-1)[0]
        prob_vuln = float(probs[1].item())
        prob_safe = float(probs[0].item())

        # Line scores: map each token back to its source line through the tokenizer
        # offsets (same alignment as the line labels used in training).
        line_scores = None
        if CONTAINER.line_head is not None:
            try:
                token_probs = torch.sigmoid(CONTAINER.line_head(hidden_states.float()).squeeze(-1))[0].tolist()
                prefix_len = len(INSTRUCTION_PROMPT)
                line_starts = [0]
                for index, char in enumerate(code):
                    if char == "\n":
                        line_starts.append(index + 1)
                line_scores = [0.0] * len(line_starts)
                for probability, (start, end) in zip(token_probs, offsets):
                    if end <= start or start < prefix_len:
                        continue
                    position = start - prefix_len
                    line = max(0, bisect.bisect_right(line_starts, position) - 1)
                    line_scores[line] = max(line_scores[line], round(float(probability), 4))
            except Exception as e:
                logger.warning(f"Line head error: {e}")

    threshold = req.threshold if req.threshold is not None else CONTAINER.best_threshold
    verdict = "Vulnerable" if prob_vuln >= threshold else "Safe"
    elapsed = (time.time() - start_t) * 1000

    return DetectResponse(
        status="ok",
        verdict=verdict,
        score=round(prob_vuln, 6),
        vulnerable_probability=round(prob_vuln, 6),
        safe_probability=round(prob_safe, 6),
        threshold=threshold,
        line_scores=line_scores,
        elapsed_ms=round(elapsed, 2),
    )


# Tham số sinh: temperature <= 0 -> greedy (do_sample=False); ngược lại sample với temperature + top_p (mặc định 0.9).
def _sampling_kwargs(temperature: Optional[float], top_p: Optional[float]) -> Dict[str, Any]:
    """temperature=0 means greedy decoding (0.0 must not fall back to the 0.2 default)."""
    temperature = 0.2 if temperature is None else float(temperature)
    if temperature <= 0.0:
        return {"do_sample": False}
    return {"do_sample": True, "temperature": temperature, "top_p": 0.9 if top_p is None else top_p}


# [PHASE 2/3/4 trên server] POST /api/v1/generate: áp chat template, từ chối prompt > 4096 token
# (HTTP 413), sinh văn bản với LoRA TẮT (disable_adapter -> model Instruct gốc), trả câu trả lời.
@app.post("/api/v1/generate", response_model=ChatResponse)
def generate_text(req: ChatRequest):
    if not CONTAINER.is_ready:
        raise HTTPException(status_code=503, detail="Mô hình đang khởi tạo, vui lòng thử lại sau.")

    start_t = time.time()
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})

    prompt_text = CONTAINER.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_tokens = CONTAINER.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(prompt_tokens) > 4096:
        raise HTTPException(
            status_code=413,
            detail=f"Prompt context length exceeded: {len(prompt_tokens)} tokens > 4096. Client must budget contract context.",
        )

    inputs = CONTAINER.tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=False,
    ).to(CONTAINER.device)

    input_len = inputs["input_ids"].shape[1]
    disable_adapter_ctx = (
        CONTAINER.causal_model.disable_adapter()
        if hasattr(CONTAINER.causal_model, "disable_adapter")
        else contextlib.nullcontext()
    )
    with GPU_LOCK, torch.no_grad(), disable_adapter_ctx:
        output_ids = CONTAINER.causal_model.generate(
            **inputs,
            **_sampling_kwargs(req.temperature, req.top_p),
            max_new_tokens=req.max_tokens or 1500,
            pad_token_id=CONTAINER.tokenizer.pad_token_id or CONTAINER.tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_len:]
    response_text = CONTAINER.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    elapsed = (time.time() - start_t) * 1000

    return ChatResponse(
        status="ok",
        response=response_text,
        elapsed_ms=round(elapsed, 2),
    )


# Điểm vào: đọc tham số (host, port 6868, base model, adapter), nạp model rồi chạy uvicorn.
def main():
    parser = argparse.ArgumentParser(description="Start v13 Unified AI Inference Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface (0.0.0.0 for LAN/Tailscale access; set SERVER_GPU_API_KEY when exposed)",
    )
    parser.add_argument("--port", type=int, default=6868, help="Port (mặc định 6868)")
    parser.add_argument("--adapter-dir", default=str(DEFAULT_ADAPTER_DIR), help="Đường dẫn adapter")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Tên base model HF")
    args = parser.parse_args()

    # Nạp model
    CONTAINER.load(base_model_name=args.base_model, adapter_dir=Path(args.adapter_dir))

    # Chạy uvicorn
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
