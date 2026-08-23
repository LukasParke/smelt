# Marlin W4A16 kernels

| | |
|---|---|
| Source | Frantar/Aminabadi et al. — 2024 |
| Link | https://github.com/IST-DASLab/marlin |
| Added | 2026-08-23 |
| Tags | #quant #kernels |

## Summary
- Tensor-core-efficient W4A16 GEMM: repacked nibble+scales layout friendly to ldmatrix; near-lossless fp16 speedup at batch<=16ish.
- Key lesson: permute AT CONVERSION TIME so runtime load is pure DMA.

## Relevance to SMELT
- D9 + smelt-layout Tensor<T,L>: MarlinRepacked layout typed at converter boundary (PLAN §3.5/§7).
