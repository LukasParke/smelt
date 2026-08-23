# safetensors

| | |
|---|---|
| Source | Hugging Face — 2022.. |
| Link | https://github.com/huggingface/safetensors |
| Added | 2026-08-23 |
| Tags | #format #container |

## Summary
- No-pickle container: 8-byte LE u64 header length + JSON header (tensor name -> dtype/shape/data_offsets) + contiguous raw bytes.
- Zero-copy mmap loading; dtypes incl BF16/F16/F32/I8/U8/F8_E5M2/F8_E4M3; __metadata__ free-form dict.
- HF sharding convention: model-0000X-of-0000N.safetensors + model.safetensors.index.json {metadata.total_size, weight_map}.
- Quant payloads ride as extra tensors (scales/zeros/qweights) + quantization_config in config.json — format stays dumb.

## Relevance to SMELT
- PLAN §7 primary interchange: native mmap zero-copy loader incl. FP8 scaled checkpoints.
