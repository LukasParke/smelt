# ExLlamaV2 / EXL2 format

| | |
|---|---|
| Source | turboderp — 2024 |
| Link | https://github.com/turboderp/exllamav2 |
| Added | 2026-08-23 |
| Tags | #quant #format #local |

## Summary
- Consumer-GPU B=1 decode specialist; hand-written fused GEMV path, minimal batching sophistication.
- EXL2: measurement-driven mixed-bpw encoding stored INSIDE safetensors: packed q_weight (int32 nibbles), q_scale (fp16 group maxima), q_invmeta metadata tensors; average bpw selectable per model with per-tensor variation.

## Relevance to SMELT
- EXL2 proves quant-convention-inside-standard-container pattern (PLAN §7 loader matrix rows).
