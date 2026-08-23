# FreeToken: Edge-Native MoE Serving

| | |
|---|---|
| Source | Yang et al., UC Berkeley Sky Computing Lab — Aug 2026 |
| Link | https://arxiv.org/html/2608.16157v1 (repo https://github.com/FlashML-org/FreeToken, https://flashml.ai) |
| Added | 2026-08-23 |
| Tags | #moe #edge #offload #agent |

## Summary
- Treats a consumer machine as a unified elastic platform: full routed-expert pool in pinned host RAM (source of truth); all spare VRAM = one elastic LRU cache of whole (layer,expert) slots tracking the router's working set.
- Prefill activates nearly every expert -> full-layer double buffering: stream layer l+1 over PCIe while computing layer l; falls back to on-demand when slots can't spare two layers.
- Decode misses split between PCIe cache-fill and in-place CPU execution by measured bandwidths: q* ~= m * B_P/B_H (residual-bandwidth argument, exact merge of partial sums).
- Semantic-aware state checkpoints: recurrent/hybrid-layer states anchored at special-token boundaries (think blocks, tool calls, turns) attached to radix prefix tree nodes -> agent context edits re-prefill only the suffix.
- Elastic VRAM: expert-cache rebuilt under revised budget at scheduler safe points without restart; load-into-final-layout-then-pin fast bootstrap; cold first request by construction.
- Results: 284B DeepSeek-V4-Flash 22-25 tok/s on RTX 5090; GLM-5.2 753B on one workstation GPU at 2x llama.cpp throughput; worst-case TTFT <44s vs >150s baselines; 1.3-2.3x decode across 6 machines vs llama.cpp/Ollama/KTransformers/MoE-Infinity.

## Key mechanisms
- CUDA-graph-compatible LRU cache machinery keeps miss-detection/victim-selection/CPU branch inside captured graphs (paper §4.1).

## Relevance to SMELT
- Closest system cousin to PLAN §11: validates TopologyMap-as-policy, generation counters, KV-headroom arbitration (kv_min_free_pages analog).
- q* bandwidth-split policy worth grafting into PLAN §13 miss-path policy; semantic anchors complement PLAN §14 pause/retract primitives.
