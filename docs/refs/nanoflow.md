# NanoFlow: Intra-Device Parallelism

| | |
|---|---|
| Source | Zhu et al., Duke/Stanford — OSDI 2024 |
| Link | https://arxiv.org/abs/2408.12757 |
| Added | 2026-08-23 |
| Tags | #serving #kernels |

## Summary
- Splits a single GPU into pipeline-parallel nanobatches overlapping compute/attention/host ops.
- Closest academic attack on the same SM-idle gaps custom kernels + streams address.

## Relevance to SMELT
- Copy-stream double-buffering in PLAN §5/§11 pursues identical overlap goals without pipeline complexity.
