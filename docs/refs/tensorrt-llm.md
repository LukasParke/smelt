# TensorRT-LLM

| | |
|---|---|
| Source | NVIDIA — ongoing |
| Link | https://github.com/NVIDIA/TensorRT-LLM |
| Added | 2026-08-23 |
| Tags | #engine #aot #closed-stack |

## Summary
- Ahead-of-time compiled 'engines': offline graph capture + kernel fusion via plugin system; quantization baked in (FP8/NVFP4).
- C++ executor: in-flight batching, paged KV, spec decode (Medusa/ReDrafter/EAGLE/MTP), MPI TP/PP.
- Lowest overhead when supported; costs: enormous build surface, slow model coverage, no embeddability.
- Perf ceiling reference for what AOT buys vs runtime JIT (Jan.ai Jul-2026 benchmark vs llama.cpp: https://jan.ai/post/benchmarking-nvidia-tensorrt-llm).

## Relevance to SMELT
- Validates D2 seam: AOT fatbin release path compiles the SAME .cu sources; NVRTC keeps single-binary UX (PLAN §12).
