# FlashAttention 1/2/3

| | |
|---|---|
| Source | Dao et al. — 2022/2023/2024 |
| Link | https://arxiv.org/abs/2205.14135 / https://arxiv.org/abs/2307.08691 / https://arxiv.org/abs/2407.08608 |
| Added | 2026-08-23 |
| Tags | #kernels #attention |

## Summary
- Exact attention via tiling + online softmax: never materialize NxN, IO-aware passes over HBM.
- FA2: reorder loop, better parallelism; FA3: Hopper wgmma/TMA async pipeline (~1.5-2x FA2 on H100).
- sm_120 caveat: no tcgen05/wgmma — FA3-style async designs don't port; synchronous mma.sync + TMA double-buffering instead.

## Relevance to SMELT
- PLAN §12 kernel catalog: FA2-style varlen prefill gate >=170 TFLOP/s BF16 on GB202.
