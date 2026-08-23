# BD3-LM (block diffusion)

| | |
|---|---|
| Source | Arriola et al., Cornell — 2025 |
| Link | https://arxiv.org/abs/2502.06768 |
| Added | 2026-08-23 |
| Tags | #diffusion |

## Summary
- Blockwise discrete diffusion enabling KV-cache reuse and arbitrary-length generation like AR models.
- Theory link between block diffusion and autoregression.

## Relevance to SMELT
- Background for DiffusionStep packing decisions; supports block_length request param semantics (PLAN §10.3).
