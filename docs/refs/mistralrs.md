# mistral.rs (+ candle)

| | |
|---|---|
| Source | LBuehler / Hugging Face — ongoing |
| Link | https://github.com/EricLBuehler/mistral.rs (candle: https://github.com/huggingface/candle) |
| Added | 2026-08-23 |
| Tags | #rust #engine |

## Summary
- Rust serving stack on candle tensor lib: GGUF+safetensors, ISQ quantize-at-load, speech/multimodal extras.
- Perf ceiling set by substrate defaults: cuBLASLt default-algo gemm_strided_batched_ex lands ~10x under vLLM on BF16 MoE prefill; no graph replay.

## Relevance to SMELT
- Primary evidence for D1: own GemmPlan algo-cache seam instead of inheriting library defaults (PLAN §0/§12).
