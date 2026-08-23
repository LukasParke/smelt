# llama.cpp + ggml

| | |
|---|---|
| Source | Gerganov et al., community — 2023.. |
| Link | https://github.com/ggml-org/llama.cpp (ggml: https://github.com/ggml-org/ggml) |
| Added | 2026-08-23 |
| Tags | #engine #local #quant |

## Summary
- Graph interpreter, not server-first design: model forward = ggml_cgraph built CPU-side per step, dispatched to backends (CPU AVX-512/NEON, CUDA, Metal, Vulkan, ROCm, SYCL).
- Center of gravity is quantization formats: k-quants/i-quants with repacked CPU kernels, fused-dequant MMQ/MMVQ CUDA GEMV, quantized KV (q8_0/q4_0), IQ dequant-on-load fallback.
- KV cache: per-layer contiguous buffers, unified cache w/ SWA handling, prompt cache for prefix reuse; no paged/refcounted allocation.
- Server: parallel slots + continuous batching + GBNF grammars + draft/self-spec/lookahead speculation.
- Sweet spot B=1..4 latency and quality-per-bit; high-concurrency throughput collapses vs paged-cache engines.
- Shells: Ollama (Go daemon+registry) and LM Studio wrap it; not engines themselves.

## Relevance to SMELT
- --n-cpu-moe semantics + block_q4_Kx8 repack lineage reused in PLAN §11/§13 hybrid path (license-clean).
- GGUF subset reader in PLAN §7; k-quant MMVQ kernels in PLAN §12 catalog.
