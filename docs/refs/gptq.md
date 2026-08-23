# GPTQ

| | |
|---|---|
| Source | Frantar et al. — ICLR 2023 |
| Link | https://arxiv.org/abs/2210.17323 |
| Added | 2026-08-23 |
| Tags | #quant #payload |

## Summary
- Post-training weight quant via approximate second-order (Hessian) error compensation, group size 128 symmetric int4 typical.
- Distribution convention inside safetensors/bin: qweight (uint32-packed nibbles), qzeros (packed zero points), scales (fp16), g_idx (group permutation map).

## Relevance to SMELT
- W4A16 g128 tier (PLAN §7): Marlin-style repack at convert time; ppl gate ±0.15 Wiki2.
