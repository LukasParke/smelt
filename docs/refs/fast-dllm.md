# Fast-dLLM

| | |
|---|---|
| Source | NVIDIA — 2025 |
| Link | https://github.com/NVlabs/Fast-dLLM |
| Added | 2026-08-23 |
| Tags | #diffusion #kv |

## Summary
- Approximate block-wise KV caching for diffusion LMs + confidence-aware parallel decoding; up to 27x speedup claims on dLLM workloads.

## Relevance to SMELT
- Buffer-KV approximate cache w/ divergence-adaptive refresh adopted in PLAN §10.3; honest >=5x wall-clock gate (PLAN §17).
