# DistServe: Prefill/Decode Disaggregation

| | |
|---|---|
| Source | Zhong et al., PKU/UCSD/SJTU — OSDI 2024 |
| Link | https://arxiv.org/abs/2401.09670 |
| Added | 2026-08-23 |
| Tags | #serving #disaggregation |

## Summary
- Prefill and decode have opposite resource profiles (compute-bound vs bandwidth-bound) and interfere when collocated.
- Separate pools with KV transfer between them; optimizes joint 'goodput' subject to SLOs.

## Relevance to SMELT
- Out of scope for v0.x (single-node), but AttnBackend/TopologyMap seams anticipate it (PLAN §1 non-goals).
