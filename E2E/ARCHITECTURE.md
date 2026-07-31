# E2E Architecture

## Execution Boundary

```text
Local machine
  Solidity parser
      |
      +-- Phase 1A: restored static analyzer
      +-- Phase 1B: local SWC TF-IDF RAG
      |
      +-- request.json (source hash + units + local evidence)
                         |
                         v
Private Kaggle GPU kernel
  Phase 1C: CodeLlama-7b-hf + QLoRA sequence-classification adapter
  unload detector
  CodeLlama-7b-Instruct-hf, loaded once:
      Phase 2 Advisor
      Phase 3 Assessor
      Phase 4 Fixer
      Phase 5 adversarial Verifier
      Phase 6 narrative draft
  unload generator
  reload detector and classify fixed functions
                         |
                         v
Local machine
  fuse original evidence
  static/RAG redetection of fixed code
  compile + Slither
  observable-behavior preservation guard
  deterministic Markdown/HTML/PDF report
```

## Why Two CodeLlama Models

The trained adapter has `task_type=SEQ_CLS`. It is suitable for binary
Safe/Vulnerable classification and is not a text-generation model. Advisor,
Assessor, Fixer, Verifier, and Reporter therefore use the instruction-tuned
CodeLlama variation. The base `CodeLlama-7b-hf` is kept for adapter compatibility.

## Failure Semantics

- `Vulnerable`: positive evidence crosses the fusion threshold.
- `Safe`: all required detectors completed and the score remains below threshold.
- `Inconclusive`: evidence is incomplete, including a missing/failed Kaggle detector.

No combination of skipped or failed components is allowed to produce `Safe`.

Patch verification has a separate outcome from the original audit verdict:

- rejected when compilation fails, local findings remain, or behavior regresses;
- inconclusive when deterministic checks pass but the binary detector remains positive;
- accepted only when redetection, compiler, Slither, and behavior checks agree.

The deterministic report layer overrides model prose that conflicts with these
outcomes. A completed run is not deployment approval.

## Artifact Contract

Every run is isolated under `runs/<run_id>/` and records:

| Artifact | Purpose |
| --- | --- |
| `phase0_input.json` | immutable source, SHA-256, parsed units |
| `phase1a_static.json` | deterministic local static evidence |
| `phase1b_rag.json` | local retrieval and SWC signal evidence |
| `request.json` | Kaggle request contract |
| `kaggle_result.json` | detector and agent result contract |
| `phase1_fused.json` | calibrated detector fusion |
| `phase2_advisor.json` | root cause and remediation |
| `phase3_assessor.json` | severity, likelihood, impact |
| `phase4_fixer.json` | generated function patches |
| `phase5_verification.json` | redetection, compiler, Slither, behavior guard, model review |
| `phase6_audit_report.*` | final Markdown, HTML, and PDF |
| `manifest.json` | run status and entry points |

The request and result are versioned by `schema_version=1.0.0` and bound to the
source through SHA-256.

## Reproducibility Boundary

`data/source/raw/` contains the DAppSCAN and iAudit/TrustLLM inputs used by the
function-level dataset builder. `data_pipeline/dataset_build/` contains the
restored extraction, negative sampling, parsing, balancing, and leakage-control
scripts with paths redirected into E2E. The final training notebooks and scripts
are retained under `training/codellama/`.
