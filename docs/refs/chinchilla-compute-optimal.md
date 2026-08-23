# Training Compute-Optimal LLMs (Chinchilla)

| | |
|---|---|
| Source | Hoffmann et al., Google DeepMind — NeurIPS 2022 |
| Link | https://arxiv.org/abs/2203.15556 |
| Added | 2026-08-23 |
| Tags | #fundamentals #scaling |

## Summary
- Revised scaling law: ~20 tokens per parameter at compute-optimal; prior models were badly undertrained.
- Data quality/curation became as decisive as parameter count.

## Relevance to SMELT
- Informs expected active-parameter footprints of modern MoE targets (Qwen3-A3B class) in PLAN §17 roofline math.
