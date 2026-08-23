# GGML k-quant / i-quant block anatomy

| | |
|---|---|
| Source | ggml-org — 2023.. |
| Link | https://github.com/ggml-org/ggml/blob/master/docs/gguf.md |
| Added | 2026-08-23 |
| Tags | #quant #payload |

## Summary
- Q4_0: 32-elem blocks, fp16 delta + 16B nibbles (4.5bpw). Q8_0: fp16 delta + 32 int8 (8.5bpw).
- K-quants: 256-elem superblocks with per-subblock (typically 16) secondary scales — Q4_K ~4.5bpw, Q6_K 210B/256elem (6.5625bpw); designed so different tensor roles get different types automatically (mixed bpw predates EXL2).
- i-quants: codebook/vector schemes (IQ1_S ~1.5-1.75bpw, IQ2/IQ3, IQ4_XS 4.25bpw) from importance-matrix training; best quality-per-bit, slowest to dequant.
- MXFP4: 32-elem microblocks, shared E8M0 power-of-2 exponent + E2M1 values ~=4.25bpw; maps to Blackwell block-scaled MMA natively.

## Relevance to SMELT
- Numeric truth for these lives in smelt-dtype only, property-tested (PLAN §3 rule 2); MMVQ fused-dequant kernels serve Q4_K/Q6_K/IQ4_XS tiers (PLAN §7/§12).
