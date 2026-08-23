# NVFP4 / MXFP4 on Blackwell

| | |
|---|---|
| Source | NVIDIA — 2025.. |
| Link | https://developer.nvidia.com/blog/tag/nvfp4/ |
| Added | 2026-08-23 |
| Tags | #quant #blackwell #flagship |

## Summary
- NVFP4: E2M1 values, two-level scaling — 16-elem microblocks scaled by FP8 E4M3 + tensor-level FP32 scale; native tensor-core MMA.
- MXFP4: 32-elem microblocks, E8M0 shared exponents; gpt-oss ships native MXFP4 weights.
- Jul-2026 reality check: NVFP4 not yet faster than FP8 end-to-end; GeForce PTX exposure of block-scaled mma.sync needs verification.

## Relevance to SMELT
- D3 sequencing consequence: NVFP4 isolated behind sealed quant seam at M8 (PLAN §0, §12 gates >=630 TFLOP/s).
