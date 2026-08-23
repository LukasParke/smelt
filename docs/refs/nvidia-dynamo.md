# NVIDIA Dynamo

| | |
|---|---|
| Source | NVIDIA — 2025 |
| Link | https://github.com/ai-dynamo/dynamo |
| Added | 2026-08-23 |
| Tags | #orchestration #disaggregation |

## Summary
- Serving orchestrator: routes across TRT-LLM/vLLM workers, manages prefill/decode disaggregation, KV-aware routing, NIXL transfer.

## Relevance to SMELT
- Confirms orchestration is separable from engine — SMELT stays embeddable library-first (PLAN §1).
