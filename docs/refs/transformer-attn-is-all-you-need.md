# Attention Is All You Need

| | |
|---|---|
| Source | Vaswani et al., Google Brain/Research — NeurIPS 2017 |
| Link | https://arxiv.org/abs/1706.03762 |
| Added | 2026-08-23 |
| Tags | #fundamentals #architecture |

## Summary
- Introduces the Transformer: recurrence deleted, self-attention carries all sequence modeling.
- Scaled dot-product attention softmax(QK^T/sqrt(d_k))V connects any two positions in one step (O(1) path length vs RNN O(n)).
- Multi-head attention, sinusoidal position encodings, residual+LayerNorm, FFN blocks, encoder-decoder.
- Training parallelizes fully across the time axis — the property that made scaling exponential.

## Relevance to SMELT
- Substrate of every architecture in PLAN §6; MLA/GQA/SWA/MoE are all deltas on this block.
