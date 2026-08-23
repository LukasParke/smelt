# PowerInfer

| | |
|---|---|
| Source | Song et al., SJTU IPADS — SOSP 2023 |
| Link | https://github.com/SJTU-IPADS/PowerInfer |
| Added | 2026-08-23 |
| Tags | #hybrid #local |

## Summary
- Exploits neuron activation skew (hot/cold) in dense models: hot neurons on GPU, cold on CPU with locality-aware scheduling.

## Relevance to SMELT
- Dense-model counterpart of PLAN §13 hybrid doctrine (bandwidth-first).
