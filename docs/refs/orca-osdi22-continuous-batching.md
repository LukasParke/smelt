# Orca: Iteration-Level Scheduling

| | |
|---|---|
| Source | Yu et al., UW/MSR — OSDI 2022 |
| Link | https://www.usenix.org/conference/osdi22/presentation/yu |
| Added | 2026-08-23 |
| Tags | #serving #scheduling |

## Summary
- Continuous batching: admission/eviction decisions per decode iteration, not per request lifetime.
- Selective batching: batch attention across sequences, run memory-bound GEMMs per-sequence when shapes diverge.
- Every modern server inherits iteration-level scheduling.

## Relevance to SMELT
- Scheduler tick loop (PLAN §9) is Orca lineage; StepPolicy executes exactly one iteration.
