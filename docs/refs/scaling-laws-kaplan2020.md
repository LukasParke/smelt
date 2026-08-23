# Scaling Laws for Neural Language Models

| | |
|---|---|
| Source | Kaplan et al., OpenAI — 2020 |
| Link | https://arxiv.org/abs/2001.08361 |
| Added | 2026-08-23 |
| Tags | #fundamentals #scaling |

## Summary
- Test loss is a power law in compute, dataset size, parameters — predictable across orders of magnitude.
- Large models are more sample-efficient; compute-optimal allocation favors bigger models over more data (later corrected by Chinchilla).

## Relevance to SMELT
- Justifies treating model size as a continuous engineering dial; informs target-model matrix in PLAN §6.
