# Reference Library (running list)

Maintenance convention: append-only. New entries go under the matching topic heading,
one line each: `[Name](url) — what it is / why it matters`. Mark additions made after
initialization with `(added YYYY-MM-DD)`.

Initialized: 2026-08-23.

## LLM fundamentals

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the Transformer paper (Google, NeurIPS 2017); self-attention replaces recurrence; substrate of every current LLM.
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — scale → in-context learning; decoder-only lineage wins for generation.
- [Scaling Laws for Neural LMs (Kaplan et al.)](https://arxiv.org/abs/2001.08361) — loss predicts from compute/params/tokens; made scaling an engineering discipline.
- [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) — token:param ratio guidance (~20:1); Google DeepMind.
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) — intermediate steps unlock reasoning; seed of the reasoning-model era.
- [InstructGPT / RLHF](https://arxiv.org/abs/2203.02155) — instruction tuning + preference optimization turned completers into assistants.
- [DPO](https://arxiv.org/abs/2305.18290) — direct preference optimization; RLHF-free post-training staple.

## Serving systems — datacenter

- [Orca (OSDI'22)](https://www.usenix.org/conference/osdi22/presentation/yu) — iteration-level (continuous) batching; every modern server inherits this.
- [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) — OS-paged KV cache: block tables, COW, near-zero fragmentation. [repo](https://github.com/vllm-project/vllm) · [docs](https://docs.vllm.ai)
- [SGLang](https://arxiv.org/abs/2312.07104) — RadixAttention prefix tree + frontend DSL; grew into full engine. [repo](https://github.com/sgl-project/sglang) · [docs](https://docs.sglang.io)
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310) — chunked prefill piggybacking decodes; token-budget scheduling origin.
- [DistServe](https://arxiv.org/abs/2401.09670) — prefill/decode disaggregation; goodput-optimized.
- [Mooncake](https://arxiv.org/abs/2407.00079) — KV-cache-centric disaggregated cluster (Moonshot/Kimi production).
- [NanoFlow](https://arxiv.org/abs/2408.12757) — intra-device pipeline parallelism (compute/attention/host ops overlap).
- [vAttention](https://arxiv.org/abs/2505.00289) — KV memory via CUDA virtual-memory APIs instead of fixed blocks.
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — AOT-compiled engines, plugin kernel system, in-flight batching; closed-stack perf ceiling.
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) — serving orchestrator: disagg routing across TRT-LLM/vLLM workers.
- [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) — HF's Rust-router + Python-worker server; archived to maintenance mode Mar 2026 ([context](https://www.tekblueprint.org/blog/ai/llm-inference-frameworks-operations/)).
- [LMDeploy / TurboMind](https://github.com/InternLM/lmdeploy) — FasterTransformer-lineage C++ engine, persistent-thread scheduling; AWQ-first.
- [DeepSpeed-FastGen](https://github.com/microsoft/DeepSpeed) — Dynamic SplitFuse (chunked prefill precursor); largely superseded.

## Serving systems — local / edge

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — ggml graph runtime, quant-format-first kernels, ubiquitous backends; desktop-AI default. ([ggml](https://github.com/ggml-org/ggml))
- [Ollama](https://github.com/ollama/ollama) — distribution/UX layer over llama.cpp (Go daemon + model registry).
- [LM Studio](https://lmstudio.ai) — GUI/local-server shell over llama.cpp runtimes.
- [KTransformers](https://github.com/kvcache-ai/ktransformers) — desktop CPU-GPU hybrid for DeepSeek-class MoE; AMX/AVX-512 CPU experts, template injection.
- [FreeToken](https://arxiv.org/html/2608.16157v1) — edge-native MoE serving (Berkeley Sky, Aug 2026): CPU-resident expert pool + elastic GPU expert cache; full-layer double-buffered prefill; q\*=m·B_P/B_H miss-split between PCIe fill and in-place CPU exec; semantic recurrent-state checkpoints surviving agent context edits; runtime VRAM re-budgeting. [repo](https://github.com/FlashML-org/FreeToken) · [site](https://flashml.ai) · [coverage](https://www.marktechpost.com/2026/08/23/meet-freetoken-an-edge-native-moe-serving-engine-that-runs-753b-glm-5-2-on-a-single-workstation-gpu/)
- [MoE-Infinity](https://github.com/TerryEcho/MoE-Infinity) — activation-aware expert offloading predecessor.
- [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) — hot/cold neuron locality for consumer-GPU dense inference.
- [ExLlamaV2](https://github.com/turboderp/exllamav2) · [ExLlamaV3](https://github.com/turboderp/exllamav3) — consumer-GPU B=1 decode specialists; EXL2/EXL3 mixed-bpw formats.
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) — Rust serving stack on candle; ISQ quantize-at-load. ([candle](https://github.com/huggingface/candle))
- [MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) — Apple unified-memory arrays framework; Mac counterpart of llama.cpp.
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) — TVM machine-learning compilation; compile-model-to-portable-code philosophy (WebGPU/mobile).

## Kernels & attention

- [FlashAttention](https://arxiv.org/abs/2205.14135) · [FA2](https://arxiv.org/abs/2307.08691) · [FA3](https://arxiv.org/abs/2407.08608) — IO-aware exact attention; tiling + online softmax.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — kernel library powering vLLM/SGLang attention paths.
- [CUTLASS](https://github.com/NVIDIA/cutlass/blob/main/README.md) — NVIDIA template kernel library; bring-up comparison baseline.
- [Beating cuBLAS on RTX 5090](https://www.cloudrift.ai/blog/beating-cublas-on-rtx-5090) — measured cuBLAS FMA-pipe gaps on GB202; motivation for own GEMM plans.

## Quantization & weight formats

- [GPTQ](https://arxiv.org/abs/2210.17323) · [AWQ](https://arxiv.org/abs/2306.00945) · [HQQ](https://github.com/dropbox/hqq) — W4-family weight-quant methods.
- [Marlin](https://github.com/IST-DASLab/marlin) — W4A16 repacked-layout GEMM kernels; permute-at-conversion lesson.
- [SmoothQuant](https://arxiv.org/abs/2211.10438) · [QuaRot](https://arxiv.org/abs/2404.00456) — activation-outlier handling for W8A8/W4A8.
- [NVFP4 @ NVIDIA](https://developer.nvidia.com/blog/tag/nvfp4/) — 4-bit two-level block scaling for Blackwell tensor cores.
- [GGUF spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) — single-file portable weight container + metadata KV.

## Speculative decoding & reasoning serving

- [Fast Inference via Speculative Decoding](https://arxiv.org/abs/2211.17192) — draft-verify foundation.
- [Medusa](https://arxiv.org/abs/2401.10774) · [EAGLE](https://github.com/SafeAILab/EAGLE) — multi-head / feature-aware drafting trees.
- [DeepSeek-V3 tech report](https://arxiv.org/abs/2412.19437) — MLA, MoE aux-loss-free routing, Multi-Token Prediction heads.
- [s1: budget forcing](https://arxiv.org/abs/2501.19393) — test-time thinking control via forced end-think injection.
- [llguidance](https://raw.githubusercontent.com/guidance-ai/llguidance/main/README.md) · [XGrammar-2](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation) — constrained decoding backends.
- [vLLM reasoning outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html) · [SGLang separate reasoning](https://docs.sglang.io/docs/advanced_features/separate_reasoning.md) — reasoning/content stream separation conventions.

## Diffusion LMs

- [LLaDA](https://github.com/ml-gsai/LLaDA) · [Dream](https://github.com/DreamLM/Dream) · [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) · [BD3-LM](https://arxiv.org/abs/2502.06768) — masked diffusion text generation + approximate KV caching for dLLMs.

## Related project docs

- [PLAN.md](../PLAN.md) — SMELT engineering plan; §21 has the primary research corpus mapped to decisions.
