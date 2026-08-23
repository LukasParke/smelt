# AWQ

| | |
|---|---|
| Source | Lin et al. — MLSys 2024 |
| Link | https://arxiv.org/abs/2306.00945 |
| Added | 2026-08-23 |
| Tags | #quant #payload |

## Summary
- Activation-aware channel scaling protects salient channels; 4-bit weights, fp16 activations.
- Same packed-tensor convention family as GPTQ (qweight/qzeros/scales); zero-point semantics differ.

## Relevance to SMELT
- Shares one Marlin-family kernel path with GPTQ/HQQ in PLAN §7 — transport formats != execution formats.
