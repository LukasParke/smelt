# KTransformers

| | |
|---|---|
| Source | KVCache-AI — 2025 |
| Link | https://github.com/kvcache-ai/ktransformers |
| Added | 2026-08-23 |
| Tags | #hybrid #moe #local |

## Summary
- Desktop CPU-GPU hybrid for DeepSeek-class MoE: GPU runs attention/KV/shared experts; routed experts execute on CPU with AMX/AVX-512 blocked GEMV.
- Template-based injection into HF transformers; per-layer placement by cost model, overridable.
- Key limitation (FreeToken analysis): expert placement frozen at load/prefill time misses shifted routing.

## Relevance to SMELT
- Direct ancestor of PLAN §13 --cpu-moe mode; Fiddler-style cost model adopted (PLAN §11); static-placement lesson motivates LRU hot cache.
