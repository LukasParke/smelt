# GGUF

| | |
|---|---|
| Source | ggml-org — 2023.. |
| Link | https://github.com/ggml-org/ggml/blob/master/docs/gguf.md (deep dive: https://deepwiki.com/ggml-org/ggml/2.6-gguf-file-format) |
| Added | 2026-08-23 |
| Tags | #format #container |

## Summary
- Single-file portable container: magic 'GGUF' + version + typed KV metadata (architecture hyperparams, FULL tokenizer vocab/scores/types) + tensor info table (name/dims/dtype enum/aligned offset) + raw data section.
- Alignment boundary (general.alignment, default 32B) between header and data.
- Container-native quant enums: legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1; K-quants Q2_K..Q8_K; i-quants IQ1_S..IQ4_XS; MXFP4_MOE (gpt-oss); TQ ternary; BF16.
- Embedding tokenizer makes one-file distribution possible; tradeoff: rigid enum taxonomy owned by ggml.

## Relevance to SMELT
- PLAN §7: native subset reader (F32/BF16/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS/MXFP4); unknown type -> hard error naming nearest supported.
