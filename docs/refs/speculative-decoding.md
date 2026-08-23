# Speculative Decoding (draft & verify)

| | |
|---|---|
| Source | Leviathan et al. (Google) / Chen et al. (DeepMind) — ICML 2023 |
| Link | https://arxiv.org/abs/2211.17192 |
| Added | 2026-08-23 |
| Tags | #spec #decoding |

## Summary
- Cheap draft model proposes k tokens; target verifies in one forward; rejection sampling guarantees EXACT target distribution.
- Value collapses with batch size: EAGLE-3 6.5x @ B=1 -> 1.38x @ B=64.

## Relevance to SMELT
- D6 gating: per-request EWMA acceptance + batch cutoff + auto-disable near B~32 with <2% regression at disable (PLAN §10.2).
