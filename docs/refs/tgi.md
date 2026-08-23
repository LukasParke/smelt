# Text Generation Inference (TGI)

| | |
|---|---|
| Source | Hugging Face — 2022..2026 |
| Link | https://github.com/huggingface/text-generation-inference |
| Added | 2026-08-23 |
| Tags | #serving #history |

## Summary
- Rust router + gRPC Python workers; flash-attn, continuous batching, Medusa/lookup spec, bnb/AWQ/Marlin quants.
- Archived to maintenance mode March 2026; HF recommends vLLM/SGLang/llama.cpp/MLX (https://www.tekblueprint.org/blog/ai/llm-inference-frameworks-operations/).

## Relevance to SMELT
- Cautionary tale for server-first architectures; SMELT's library-core/thin-shells rule (PLAN §3.4) hedges this.
