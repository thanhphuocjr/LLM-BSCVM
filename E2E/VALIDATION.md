# E2E Validation Record

## Latest Full Run

The reference full-system run is `runs/kaggle-smoke-v6/`.

| Item | Result |
| --- | --- |
| Pipeline status | `complete` |
| Remote CodeLlama runtime | 215,624 ms |
| Original fused verdict | Vulnerable, score 0.930016 |
| Confirmed local finding | SWC-107 reentrancy in `VulnerableBank.withdraw` |
| Generated repair | Checks-effects-interactions; state update moved before value call |
| Compilation | Passed with Solidity 0.8.34 |
| Fixed-code static/RAG | No confirmed finding |
| Slither | No reentrancy or locked-Ether finding; one informational low-level-call finding |
| Behavior preservation | Passed |
| Final patch verdict | Inconclusive because the binary CodeLlama detector remained positive |

The generated v6 patch preserved the ETH transfer and its success check. The
local verifier did not accept deployment because the fine-tuned classifier
continued to score the safe `deposit` function at 0.989986 and the repaired
`withdraw` function at 0.937612.

`manifest.status=complete` means every configured phase completed. It does not
mean that a generated patch was accepted.

## Regression Evidence

Run `kaggle-smoke-v5` is retained as a negative control. CodeLlama removed the
ETH transfer while trying to eliminate reentrancy. The strengthened verifier
rejected that patch for two independent reasons:

- the patched declaration removed an observable ETH value call;
- Slither reported `locked-ether`.

This run demonstrates why compilation and a generative verifier alone are not
sufficient patch-acceptance criteria.

## Detector Evaluation

The local test-split evaluation is stored in
`data/training/reports/local_detector_eval.json`.

| Detector | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Restored static detector | 0.5361 | 0.4564 | 0.4931 |
| Local RAG detector | 0.6154 | 0.1641 | 0.2591 |

The adapter training configuration records validation F1 near 0.777. The v6
function scores show that aggregate validation performance is not sufficient
evidence of calibration on unseen contract styles. The adapter should be
recalibrated on a held-out, source-disjoint set containing simple safe functions
before its output is used as an automatic patch-acceptance gate.

## Automated Checks

Run the local regression suite from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s E2E/tests -v
```

The current suite contains 10 tests covering parsing, patch placement,
static/RAG controls, fail-closed fusion, Kaggle request injection, malformed
agent JSON, no-patch verification, behavioral regression, detector
disagreement, report normalization, and PDF-producing integration flow.

## Primary Artifacts

- `runs/kaggle-smoke-v6/manifest.json`
- `runs/kaggle-smoke-v6/kaggle_result.json`
- `runs/kaggle-smoke-v6/phase4_fixer.json`
- `runs/kaggle-smoke-v6/phase5_verification.json`
- `runs/kaggle-smoke-v6/phase6_audit_report.pdf`
