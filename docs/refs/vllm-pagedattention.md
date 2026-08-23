# vLLM / PagedAttention

| | |
|---|---|
| Source | Kwon et al., UC Berkeley — SOSP 2023; vLLM project |
| Link | https://arxiv.org/abs/2309.06180 (repo https://github.com/vllm-project/vllm, docs https://docs.vllm.ai) |
| Added | 2026-08-23 |
| Tags | #serving #kv #engine |

## Summary
- KV cache managed like OS virtual memory: fixed-size blocks (16 tok), per-sequence block tables, refcounts, copy-on-write.
- Near-zero fragmentation, cheap prefix sharing, preemption via swap-or-recompute.
- V1 rewrite (2025): 1D scheduler with persistent per-request state, piecewise CUDA graphs (captured around attention only), async output processing, overlap scheduling.
- Ecosystem breadth: widest model coverage, attention backend abstraction (FA/FlashInfer/Triton), TP/PP/EP multi-node via Ray+NCCL, KV connectors (LMCache/NIXL), spec decode (ngram/EAGLE/Medusa), xgrammar/outlines structured output.

## Key mechanisms
- Paged blocks decouple logical token positions from physical GPU memory — beams/branches fork via COW page clones.
- Prefix caching hashes block chains; automatic reuse across requests sharing prefixes.

## Relevance to SMELT
- PLAN §8 KvPool/PrefixTree splits vLLM's monolith into pool + policy layer (proven SGLang split).
- Graph-memory budget calibration (≤2.5 GB) taken from vLLM production numbers (PLAN §12).
