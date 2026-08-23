# Mooncake: KV-Cache-Centric Disaggregated Cluster

| | |
|---|---|
| Source | Qin et al., Moonshot AI — FAST 2025 |
| Link | https://arxiv.org/abs/2407.00079 |
| Added | 2026-08-23 |
| Tags | #serving #disaggregation #kv |

## Summary
- Production cluster for Kimi: KV cache treated as a first-class distributed resource pooled over RDMA.
- Global scheduler matches requests to nodes holding reusable prefixes; underload/overload games.

## Relevance to SMELT
- Long-term direction evidence for prefix-cache economics; validates session soft-pin idea (PLAN §8).
