# Reference Index

Atomic notes live in `docs/refs/` — one note per paper/system/format.
This file is the index; keep it sorted by topic, newest additions at section bottoms.

**Maintenance rules (while planning):**
1. New reference -> new note file in `docs/refs/<kebab-name>.md` using the standard template
   (metadata table: Source / Link / Added / Tags, then Summary, Key mechanisms, Relevance to SMELT).
2. Add exactly one index line here: `- [Title](refs/<file>.md) — one-liner · [source](url)`.
3. Never grow this file into content; it is a table of contents only.
4. Cross-link related notes inside notes (`See also:` line), not here.

## Fundamentals

- [Attention Is All You Need](refs/transformer-attn-is-all-you-need.md) — the Transformer; self-attention replaces recurrence · [arXiv 1706.03762](https://arxiv.org/abs/1706.03762)
- [GPT-3 few-shot](refs/gpt3-few-shot.md) — scale → in-context learning · [arXiv 2005.14165](https://arxiv.org/abs/2005.14165)
- [Scaling Laws (Kaplan)](refs/scaling-laws-kaplan2020.md) — loss as power law in compute/params/data · [arXiv 2001.08361](https://arxiv.org/abs/2001.08361)
- [Chinchilla compute-optimal](refs/chinchilla-compute-optimal.md) — ~20 tok:param at optimum · [arXiv 2203.15556](https://arxiv.org/abs/2203.15556)
- [Chain-of-Thought](refs/chain-of-thought-wei2022.md) — intermediate steps unlock reasoning · [arXiv 2201.11903](https://arxiv.org/abs/2201.11903)
- [InstructGPT / RLHF](refs/instructgpt-rlhf.md) — instruction tuning + preference RL · [arXiv 2203.02155](https://arxiv.org/abs/2203.02155)
- [DPO](refs/dpo.md) — closed-form preference optimization · [arXiv 2305.18290](https://arxiv.org/abs/2305.18290)

## Serving engines — datacenter

- [Orca](refs/orca-osdi22-continuous-batching.md) — iteration-level continuous batching origin · [OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu)
- [vLLM / PagedAttention](refs/vllm-pagedattention.md) — OS-paged KV, V1 rewrite, ecosystem breadth · [arXiv 2309.06180](https://arxiv.org/abs/2309.06180)
- [SGLang / RadixAttention](refs/sglang-radixattention.md) — radix prefix sharing, overlap scheduler, MLA lead · [arXiv 2312.07104](https://arxiv.org/abs/2312.07104)
- [Sarathi-Serve](refs/sarathi-serve.md) — chunked prefill + token-budget scheduling · [arXiv 2403.02310](https://arxiv.org/abs/2403.02310)
- [DistServe](refs/distserve-pd-disagg.md) — prefill/decode disaggregation, goodput · [arXiv 2401.09670](https://arxiv.org/abs/2401.09670)
- [Mooncake](refs/mooncake-kv-centric.md) — KV-centric disaggregated cluster (Kimi prod) · [arXiv 2407.00079](https://arxiv.org/abs/2407.00079)
- [NanoFlow](refs/nanoflow.md) — intra-device pipeline overlap · [arXiv 2408.12757](https://arxiv.org/abs/2408.12757)
- [vAttention](refs/vattention.md) — CUDA VMM-backed KV without fixed pages · [arXiv 2505.00289](https://arxiv.org/abs/2505.00289)
- [TensorRT-LLM](refs/tensorrt-llm.md) — AOT compiled engines, plugin kernels · [repo](https://github.com/NVIDIA/TensorRT-LLM)
- [NVIDIA Dynamo](refs/nvidia-dynamo.md) — multi-engine serving orchestrator · [repo](https://github.com/ai-dynamo/dynamo)
- [TGI](refs/tgi.md) — HF server; archived Mar 2026 · [repo](https://github.com/huggingface/text-generation-inference)
- [LMDeploy / TurboMind](refs/lmdeploy-turbomind.md) — FasterTransformer-lineage C++ engine · [repo](https://github.com/InternLM/lmdeploy)
- [DeepSpeed-FastGen](refs/deepspeed-fastgen.md) — Dynamic SplitFuse precursor · [repo](https://github.com/microsoft/DeepSpeed)

## Serving engines — local / edge

- [llama.cpp + ggml](refs/llamacpp-ggml.md) — graph interpreter, quant-first kernels, ubiquitous · [repo](https://github.com/ggml-org/llama.cpp)
- [KTransformers](refs/ktransformers.md) — desktop CPU-GPU MoE hybrid, static placement · [repo](https://github.com/kvcache-ai/ktransformers)
- [FreeToken](refs/freetoken.md) — elastic expert cache + q\* bandwidth-split miss policy + semantic state anchors · [arXiv 2608.16157](https://arxiv.org/html/2608.16157v1)
- [MoE-Infinity](refs/moe-infinity.md) — activation-aware expert offloading · [repo](https://github.com/TerryEcho/MoE-Infinity)
- [PowerInfer](refs/powerinfer.md) — hot/cold neuron CPU-GPU dense hybrid · [repo](https://github.com/SJTU-IPADS/PowerInfer)
- [ExLlamaV2 / EXL2](refs/exllamav2-exl2.md) — mixed-bpw consumer decode specialist · [repo](https://github.com/turboderp/exllamav2)
- [ExLlamaV3 / EXL3](refs/exllamav3-exl3.md) — QTIP trellis quant, Rust rewrite · [repo](https://github.com/turboderp/exllamav3)
- [mistral.rs + candle](refs/mistralrs.md) — Rust stack; default-algo GEMM ceiling evidence · [repo](https://github.com/EricLBuehler/mistral.rs)
- [MLX](refs/mlx.md) — Apple unified-memory framework · [repo](https://github.com/ml-explore/mlx)
- [MLC-LLM](refs/mlc-llm.md) — TVM compile-to-portable-code · [repo](https://github.com/mlc-ai/mlc-llm)

## Kernels

- [FlashAttention 1/2/3](refs/flashattention-1-2-3.md) — IO-aware exact attention lineage · [FA](https://arxiv.org/abs/2205.14135) / [FA2](https://arxiv.org/abs/2307.08691) / [FA3](https://arxiv.org/abs/2407.08608)
- [FlashInfer](refs/flashinfer.md) — plan/run kernel library behind vLLM/SGLang · [repo](https://github.com/flashinfer-ai/flashinfer)
- [CUTLASS](refs/cutlass.md) — NVIDIA template GEMM library · [repo](https://github.com/NVIDIA/cutlass)
- [Beating cuBLAS on RTX 5090](refs/cublas-5090-gap.md) — cuBLAS FMA-pipe gap on GB202 · [blog](https://www.cloudrift.ai/blog/beating-cublas-on-rtx-5090)

## Weight formats & quantization

- [safetensors](refs/safetensors.md) — no-pickle container; JSON header + raw blob; sharding index · [repo](https://github.com/huggingface/safetensors)
- [GGUF](refs/gguf.md) — single-file container w/ typed KV metadata + native tokenizer · [spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGML k-quant/i-quant anatomy](refs/ggml-kquant-formats.md) — block layouts from Q4_0 to IQ4_XS/MXFP4 · [spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GPTQ](refs/gptq.md) — Hessian-compensated int4; packed-tensor convention · [arXiv 2210.17323](https://arxiv.org/abs/2210.17323)
- [AWQ](refs/awq.md) — activation-aware channel scaling · [arXiv 2306.00945](https://arxiv.org/abs/2306.00945)
- [HQQ](refs/hqq.md) — calibration-free fast quant · [repo](https://github.com/dropbox/hqq)
- [Marlin](refs/marlin.md) — W4A16 repacked-layout tensor-core GEMM · [repo](https://github.com/IST-DASLab/marlin)
- [SmoothQuant](refs/smoothquant.md) — outlier migration for W8A8 · [arXiv 2211.10438](https://arxiv.org/abs/2211.10438)
- [QuaRot / SpinQuant](refs/quarot-spinquant.md) — rotation-based outlier suppression · [arXiv 2404.00456](https://arxiv.org/abs/2404.00456)
- [NVFP4 / MXFP4](refs/nvfp4-mxfp4.md) — Blackwell block-scaled 4-bit tiers · [NVIDIA](https://developer.nvidia.com/blog/tag/nvfp4/)

## Speculation & reasoning serving

- [Speculative decoding](refs/speculative-decoding.md) — draft-verify, exact distribution · [arXiv 2211.17192](https://arxiv.org/abs/2211.17192)
- [Medusa](refs/medusa.md) — multi-head tree drafting · [arXiv 2401.10774](https://arxiv.org/abs/2401.10774)
- [EAGLE 1/2/3](refs/eagle.md) — feature-space drafting trees · [repo](https://github.com/SafeAILab/EAGLE)
- [DeepSeek-V3 report](refs/deepseek-v3-mla-mtp.md) — MLA latent KV + MTP heads + routing · [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- [s1 budget forcing](refs/s1-budget-forcing.md) — forced end-think injection · [arXiv 2501.19393](https://arxiv.org/abs/2501.19393)
- [llguidance](refs/llguidance.md) — Rust grammar engine, default backend · [repo](https://github.com/guidance-ai/llguidance)
- [XGrammar-2](refs/xgrammar2.md) — cached cross-request grammars for agentic loads · [blog](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation)

## Diffusion LMs

- [LLaDA](refs/llada.md) — masked diffusion LM family · [repo](https://github.com/ml-gsai/LLaDA)
- [Dream 7B](refs/dream.md) — remask taxonomy reference · [repo](https://github.com/DreamLM/Dream)
- [Fast-dLLM](refs/fast-dllm.md) — approximate buffer-KV + parallel decoding for dLLMs · [repo](https://github.com/NVlabs/Fast-dLLM)
- [BD3-LM](refs/bd3-lm.md) — block diffusion with AR-like KV reuse · [arXiv 2502.06768](https://arxiv.org/abs/2502.06768)

## Project research & experiments (in-house, 2026-08-23)

- [Format-evolution patterns](refs/format-evolution-patterns.md) — the five mechanisms surviving systems share · survey doc: [extensibility-mechanisms](research/extensibility-mechanisms.md)
- [Living models: plasticity + live mutation + hardware frontier](refs/ttt-titans-fast-weights.md) · [live-adapters](refs/live-adapters-serving.md) · [pim-frontier](refs/pim-hardware-frontier.md) — survey doc: [living-models-plasticity](research/living-models-plasticity.md)
- [Perf lab experiments](../experiments/perf-lab/) — bandwidth/roofline/atoms/living-engine benches + adversarially verified claims · results + methods: [experiments/README](../docs/experiments/README.md)

## Project specs

- [SMT v1 format spec](format/smt-v1-spec.md) — content-addressed, execution-layout-aware container; numeric-atom space; portable graph IR; delta composition (in design)
- [SMT v2 open-core & living-weights spec](format/smt-v2-living-spec.md) — tagged refs, in-file ExtDefs w/ verified expression DSL, tiered resolution, dialect packs, living-weights substrate; supersedes v1 extensibility model
