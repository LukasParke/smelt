# Language Models are Few-Shot Learners (GPT-3)

| | |
|---|---|
| Source | Brown et al., OpenAI — NeurIPS 2020 |
| Link | https://arxiv.org/abs/2005.14165 |
| Added | 2026-08-23 |
| Tags | #fundamentals #scaling |

## Summary
- 175B decoder-only causal LM; demonstrated in-context learning: task specification via prompt examples, no gradient updates.
- Established decoder-only + scale as the generative lineage over encoder-decoder.

## Relevance to SMELT
- Prompt/prefix reuse economics (PLAN §8 PrefixTree) exist because few-shot prompts dominate real traffic.
