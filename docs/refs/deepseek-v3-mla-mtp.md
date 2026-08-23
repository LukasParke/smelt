# DeepSeek-V3: MLA + aux-loss-free MoE + MTP

| | |
|---|---|
| Source | DeepSeek-AI — Dec 2024 |
| Link | https://arxiv.org/abs/2412.19437 |
| Added | 2026-08-23 |
| Tags | #architecture #mla #moe |

## Summary
- MLA: compress KV to latent c_kv=512 + decoupled RoPE key 64 (576/row); absorbed decode kernels skip decompression; prefill uses MHA tiles transiently.
- MoE: 256 routed top-8 + 1 shared, sigmoid+bias routing, aux-loss-free load balancing via per-expert bias updates.
- MTP: extra multi-token-prediction module -> auto-enable speculative drafting ~1.5-1.8x.
- Trained in FP8 at scale; GLM/Qwen follow same family patterns.

## Relevance to SMELT
- PLAN §6 row M6; MLA pages hold latents directly, rel err <1e-2 gate vs decompressed reference (PLAN §8/§12).
