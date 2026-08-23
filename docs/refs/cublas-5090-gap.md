# Beating cuBLAS on RTX 5090

| | |
|---|---|
| Source | cloudrift — 2025/2026 |
| Link | https://www.cloudrift.ai/blog/beating-cublas-on-rtx-5090 |
| Added | 2026-08-23 |
| Tags | #kernels #gemm #blackwell |

## Summary
- Measured cuBLAS FMA-pipe utilization 33-42% pinned to sizes 256-8192 on GB202; custom kernel reaches ~68%.
- Consumer Blackwell lacks cuBLAS tuning attention datacenter parts get.

## Relevance to SMELT
- Justifies profiler-gated custom GEMM escalation post-M3 through GemmPlan registration (PLAN §12).
