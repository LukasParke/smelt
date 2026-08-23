# Dream 7B

| | |
|---|---|
| Source | HKU/Huawei — 2025 |
| Link | https://github.com/DreamLM/Dream |
| Added | 2026-08-23 |
| Tags | #diffusion |

## Summary
- 7B diffusion LM matching same-size AR quality; defines remask taxonomy (Random/Origin/MaskGitPlus/TopkMargin/Entropy/ConfidenceThreshold/BlockFixed).
- Request params: gen_length, steps, block_length, confidence threshold, early EOS.

## Relevance to SMELT
- Remask strategy enum copied verbatim into PLAN §10.3.
