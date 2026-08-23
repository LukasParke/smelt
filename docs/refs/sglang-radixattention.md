# SGLang / RadixAttention

| | |
|---|---|
| Source | Zheng et al., LMSYS/UC Berkeley — 2024; SGLang project |
| Link | https://arxiv.org/abs/2312.07104 (repo https://github.com/sgl-project/sglang, docs https://docs.sglang.ai) |
| Added | 2026-08-23 |
| Tags | #serving #kv #prefix |

## Summary
- Radix tree over token prefixes maps to KV page chains: automatic cross-request prefix reuse with LRU eviction and fork support.
- Zero-overhead scheduler: CPU planning of batch N+1 overlaps GPU execution of batch N (vLLM V0 serialized these).
- DeepSeek-class strengths: absorbed-latent MLA decode kernels, DP attention, expert parallelism with EPLB rebalancing, MTP speculation.
- Hierarchical cache (HiCache) spills cold radix subtrees to host/disk with coherence counters; PD disaggregation across nodes.
- Started as frontend DSL for structured programming of LLM calls; grew into full engine.

## Relevance to SMELT
- PLAN §8 adopts the pool/policy split SGLang validated: radix PrefixTree above request-agnostic KvPool.
- Leaf-LRU eviction + host tier + generation counters mirror HiCache semantics (PLAN §8, D8).
