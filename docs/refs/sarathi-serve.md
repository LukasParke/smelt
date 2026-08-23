# Sarathi-Serve: Chunked Prefill

| | |
|---|---|
| Source | Agrawal et al., MSR/Georgia Tech — OSDI 2024 (Sarathi: 2302.01318) |
| Link | https://arxiv.org/abs/2403.02310 |
| Added | 2026-08-23 |
| Tags | #serving #scheduling #prefill |

## Summary
- Split long prefills into chunks and piggyback them with decodes in one step: prefill never stalls decode.
- Token-budget scheduling protects TPOT tail; basis of vLLM/SGLang default chunked prefill.
- Original Sarathi (2302.01318) introduced chunked-prefill co-scheduling; Sarathi-Serve made it a serving system.

## Relevance to SMELT
- PLAN §9 token-budget loop + chunk boundaries in BatchPlan; invariant: chunks never split in-flight verify rounds (PLAN §9.3).
