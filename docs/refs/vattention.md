# vAttention: Dynamic KV Memory via CUDA VMM

| | |
|---|---|
| Source | Prabhu et al., UIUC/AMD — 2025 |
| Link | https://arxiv.org/abs/2505.00289 |
| Added | 2026-08-23 |
| Tags | #kv #memory |

## Summary
- Replaces fixed physical pages with CUDA virtual-memory reservations (cuMemMap): contiguous virtual KV, demand-backed physical backing.
- Removes fragmentation AND removes block-table indirection cost; kernels see dense tensors.
- Compatible layer over paged attention kernels.

## Relevance to SMELT
- Alternative to PLAN §8 page tables; Stable-VA reservation (PLAN §5) could adopt VMM backing later without API churn.
