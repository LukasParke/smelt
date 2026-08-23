# s1: Simple test-time scaling (budget forcing)

| | |
|---|---|
| Source | Muennighoff et al., Stanford — 2025 |
| Link | https://arxiv.org/abs/2501.19393 |
| Added | 2026-08-23 |
| Tags | #reasoning #control |

## Summary
- Force end-thinking at token budget by injecting end-think transition; or extend thinking by appending 'Wait'.
- Terminates think blocks exactly at limit with natural transitions.

## Relevance to SMELT
- PLAN §14 budget forcing consumes M2 pause/inject primitives; exit criterion mirrors s1 semantics per-model.
