# SMELT — Engineering Plan

**A multipurpose, highly performant LLM/GLM/RLM inference engine in pure Rust.**
Status: approved plan v1.1 (2026-08-23). Derived from an 11-area research sweep, two independent architecture proposals ("Forge" = max perf ceiling, "Kindling" = fastest end-to-end), a scored judge panel (winner: **Kindling sequencing with Forge grafts**, 39 vs 32), and an adversarial completeness-critic round whose findings are folded into this revision.

---

## 0. Decision Record

| # | Decision | Choice | Why |
|---|----------|--------|-----|
| D1 | Tensor/GPU foundation | Own tensor type + cuBLASLt **GemmPlan algo-cache**; hand-written kernels register into the same plan table | mistral.rs's candle-based stack lands ~10x under vLLM on BF16 MoE prefill because of default-algo `gemm_strided_batched_ex`; owning the plan layer is cheap and is the perf floor. Full cuBLASLt replacement (Forge's bet) stays profiler-gated post-M3 |
| D2 | Kernel compilation | **NVRTC-primary**, disk-cached PTX keyed by source hash; build.rs/nvcc AOT fatbins opt-in release feature; same embedded `.cu` sources feed both paths | Single-binary distribution without toolkit install; cold-start JIT killed by cached PTX; CI audits SASS (`ptxas -v`, spill assertions) on either path |
| D3 | Sequencing | BF16-eager first; FP8/W4A16 by M3; NVFP4 flagship at **M8** | Jul-2026 reports: NVFP4 not yet faster than FP8 end-to-end; GeForce PTX exposure of block-scaled MMA needs verification. Quant tier is a sealed seam — deferring costs nothing architecturally |
| D4 | Generation unification | One scheduler-visible `StepPolicy` step; AR / speculative-tree / masked-diffusion are interchangeable policies | dLLMs become first-class iterations without forking batching machinery. Known leakiest abstraction: `DiffusionStep` may reject batches it cannot pack |
| D5 | Forward contract | Every forward returns `(logits, final_hidden_features)` from day one | EAGLE-family drafters need target features mid-forward; retrofitting is invasive. Cost: one retained buffer |
| D6 | Speculation gating | Per-request EWMA acceptance gate + global batch-size cutoff | Speculative value collapses with batch (EAGLE-3: 6.5x @ B=1 → 1.38x @ B=64); ungated drafts hurt throughput |
| D7 | KV quantization | FP8-E4M3 KV with per-head scales at M2, before exotic weight quants | Cheapest large memory lever: ~2x capacity at negligible quality cost |
| D8 | Expert placement | Abstract `TopologyMap` (device → resident expert ids); `--cpu-moe`, HBM hot-expert LRU cache, future multi-GPU EP are policies over one module | KTransformers/Fiddler lesson: data-movement direction is a per-layer cost-model decision, not a hardcoded path |
| D9 | Formats | safetensors-first mmap loader + GGUF subset reader; offline `smelt convert` repack to kernel-tile `.smt` packs; transport formats ≠ execution formats | Marlin lesson: permuted nibbles+scales on disk make load pure DMA; one kernel family serves GPTQ/AWQ/HQQ/GGUF alike |
| D10 | Constrained decoding | llguidance default, XGrammar optional backend behind one trait; process-wide compiled-grammar cache; masks applied as fused GPU-side bitmask, never float-logit patching | llguidance is Rust-native (~50 µs/token @ 128k vocab); XGrammar-2 wins JSON-heavy agentic loads (~50% cross-request reuse at 50 tools) |

### Graft Traceability

Each judge-mandated graft from the losing variant maps to a named home here:

| Forge graft | Where it lives in this plan |
|---|---|
| Standalone pure `smelt-dtype` crate | §3 crate rules, M0 |
| Layout-typed tensors at converter boundary (`Tensor<T,L>`) | §7 formats/repack, §4 |
| AOT fatbin release path + `ptxas -v` SASS/spill CI audits | §12 toolchain, M1 (CI job), M9 (release) |
| pause/retract/continue + thinking-budget injection hook | §9 primitives (built M2), consumed by §14 budget forcing (M6) |
| Measured swap-vs-recompute crossover heuristic | §8 preemption policy |
| XGrammar backend + cross-request grammar cache | §14, M6 |
| Stable-VA determinism + sleep/wake reservation | §5 arena design, reserved M0 |
| Roofline-fraction (% of computed ceiling) exit criteria | §17 method; every performance-tier gate (kernel/memory tiers in M1/M3/M4/M8) carries %-of-ceiling form; behavioral gates carry absolute/suite measures |

---

## 1. Mission & Positioning

SMELT is simultaneously:

- a **library** (`smelt-engine` + subsystem crates, no HTTP required),
- a **server** (OpenAI-compatible REST+SSE single binary),
- a **CLI** (interactive generation, conversion, benchmarking).

Serving targets: autoregressive dense models, DeepSeek-class MLA+MoE models, GLM-4.5-family hybrid-thinking MoE ("GLM"), gpt-oss MXFP4-native MoE, masked-diffusion LMs (LLaDA/Dream), and reasoning-LM workloads ("RLM": o1/R1/GLM-thinking/Qwen3-thinking class — long CoT, thinking budgets, partial rollouts; recurrent-memory and retrieval-LM lines explicitly out of scope).

Positioning vs incumbents:

| Engine | Gap SMELT exploits |
|---|---|
| llama.cpp | CPU-first graph runtime; CUDA decode leaves perf on floor; limited continuous-batching/prefix-cache sophistication |
| vLLM / SGLang | Python process model, heavy deploy; not embeddable as a Rust library |
| TensorRT-LLM | Closed, NVIDIA-only, enormous build surface |
| mistral.rs / candle | Proof Rust serving works; default-algo GEMM and missing paged-KV/graph-replay cap throughput |
| ExLlama / KTransformers | Single-purpose; no unified scheduler across AR/spec/diffusion |

Non-goals (v0.x): training/fine-tuning, multimodal encoders beyond text-projector hooks, multi-node TP/EP (seams reserved, not built), Windows-first support, non-NVIDIA backends before CUDA certifies (Metal/ROCm/Vulkan follow).

## 2. Ground Truth

Dev/target machine (measured 2026-08-23):

| Component | Fact | Consequence |
|---|---|---|
| GPU | RTX 5090, 32 GB, compute cap 12.0 (sm_120), driver 610.43 | Consumer Blackwell: **no tcgen05/TMEM/wgmma** — synchronous `mma.sync` blocks issuing warp. TMA (`cp.async.bulk.tensor`) IS present. Requires CUDA ≥ 12.8; target 13.x. >1 PFLOP/s NVFP4 MMA verified on this chip class (sparse/MAC-convention marketing figure; see §12 for standard-counting gates) |
| HBM | GDDR7 ≈ 1792 GB/s peak | Decode roofline: `tok/s = active_weight_bytes / 1523 GB/s` (85%-efficiency planning figure) |
| CPU | Ryzen 9 7950X3D: V-cache CCD = cores {0-7,16-23} (96 MB L3), second CCD plain 32 MB; DDR5-4800 dual-channel measured **~50.3 GB/s NT-store triad** (65% eff), ~33 GB/s single-thread triad; AVX-512 incl. `avx512_vnni`+`avx512_bf16`, ~150 GFLOP/s/core FP32, ~2.3 TFLOP/s all-core; no AMX | CPU decode purely bandwidth-bound: ≤5 GB DRAM/token for 10+ tok/s comfort; pin expert-GEMM threads to V-cache CCD; hugepages available; AVX2 fallback compiled for non-AVX512 hosts |
| PCIe | Gen5 x16, literature ~55-60 GB/s effective pinned H2D (on-box microbench mandatory M0) | 15 MB expert slab ≈ 0.25-0.35 ms transfer — hideable by double-buffering against grouped-GEMM compute |
| Toolchain | rustc/cargo 1.98; **no nvcc installed yet** | Prerequisite: install CUDA toolkit 13.x before M0 kernel work; cudarc dynamic-loads libcuda so users need only driver ≥ 570 |

Environment facts shaping the plan: 64 GB host RAM (pinned-host backing for experts/KV tiers); heavy current swap usage — close workloads before benchmarking.

## 3. System Architecture

```mermaid
flowchart TB
    subgraph binaries
        SRV[smelt-server<br/>axum REST+SSE]
        CLI[smelt-cli]
    end
    subgraph engine["smelt-engine"]
        EH[EngineHandle facade]
        SCH[smelt-schedule<br/>token-budget loop · chunked prefill<br/>priorities · preemption · pause/retract]
        GEN[smelt-generate<br/>StepPolicy: AR | SpecTree | Diffusion<br/>samplers · Drafter trait · GrammarConstraint]
        MOE[smelt-moe<br/>MoEConfig · TopologyMap<br/>expert LRU cache · prefetch]
        KV[smelt-kv<br/>paged refcounted pool · COW<br/>PrefixTree radix cache]
    end
    subgraph models["model layer"]
        MOD[smelt-models<br/>Llama · Qwen · Mistral · Gemma · Phi<br/>DeepSeek-V3 MLA+MTP · GLM-4.5 · gpt-oss<br/>Mixtral · Qwen3-MoE · LLaDA/Dream]
        FMT[smelt-formats<br/>safetensors mmap · HF index · GGUF subset<br/>fp8 checkpoints · tokenizer glue]
        LAY[smelt-layout<br/>repack → .smt kernel-tile packs]
    end
    subgraph runtime["runtime layer"]
        CORE[smelt-core<br/>Tensor&lt;T&gt; · arenas · GemmPlan algo-cache]
        DT[smelt-dtype<br/>fp8 · E8M0 · MX/NVFP4 blocks · k-quants<br/>pure + property-tested]
        KER[smelt-kernels<br/>.cu sources · NVRTC registry<br/>autotune records]
        CUDA[smelt-cuda<br/>sole unsafe boundary over cudarc<br/>graphs · streams · pinned pools]
    end
    SRV & CLI --> EH --> SCH --> GEN
    GEN --> MOD
    MOD --> KV & MOE
    SCH --> KV
    FMT --> LAY --> MOD
    MOD & KV & MOE & GEN --> CORE
    CORE --> DT
    CORE --> KER --> CUDA
```

Crate rules:

1. `smelt-cuda` is the **only** crate containing `unsafe` FFI (cudarc 0.19.x substrate: driver/NVRTC/cuBLASLt/CUPTI/NCCL; raw FFI reserved for `cuGraphExecKernelNodeSetParams`, mbarrier/TMA intrinsics).
2. `smelt-dtype` is pure, `no_std`-capable, property-tested (rounding rules, microblock packing, scale math). Numeric truth lives nowhere else.
3. Downstream crates hold only handles from `smelt-cuda`; no implicit allocation in hot paths.
4. Server and CLI are thin shells over `EngineHandle`; tests use the handle directly.
5. `smelt-layout` types tensors by layout at the converter boundary (`Tensor<T, L>` with L ∈ {RowMajor, MarlinRepacked, MxBlocked, CpuInterleaved}); a wrongly-laid-out tensor cannot reach a kernel. Runtime hot path uses erased descriptors resolved before dispatch (zero per-token cost).

## 4. Core Abstractions

| Abstraction | Signature / shape | Notes |
|---|---|---|
| `Tensor<T>` | arena-backed device view: shape, strides, dtype phantom (+ layout phantom at conversion boundaries) | No implicit allocs; planned handles only |
| `GemmPlan` | cuBLASLt heuristic query + algo cache keyed `(M,N,K,dtype)`; custom kernels register into the same table | Callers never know who executed a matmul; seam where profiler-gated custom kernels replace cuBLASLt call sites without API churn |
| `KernelRegistry` | source-hash → compiled CUfunction + tuned config; NVRTC primary, PTX disk cache, AOT fatbin opt-in | Autotune records persisted to config table |
| `WeightSource` | `fn tensors() -> Iterator<NamedView>` | safetensors-sharded and GGUF readers produce zero-copy mmapped views; fp8-scaled checkpoint views carry scale sidecars |
| `ModelArch` | `fn forward(ctx, &BatchPlan) -> ForwardOut`; `ForwardOut { logits, final_hidden }` | Feature plumbing for EAGLE baked in day one (D5) |
| `AttnBackend` | FlashInfer-shaped two-phase: `plan(&BatchPlan) -> PlanMeta`, then `run(q, &KvPool, meta)` | Impls: Fa2Varlen prefill, SplitKFlashDecode, MlaAbsorbedDecode, TreeVerify, BidirectionalFull (diffusion) |
| `KvPool` | typed slabs (bf16 / fp8-e4m3+per-head scales / fp4-reserved), 16-token configurable pages, u32 block tables, atomic refcounts, COW; exposes `alloc_pages / clone_seq / commit_scratch / free` | Request-agnostic; knows byte/token constants only |
| `PrefixTree` | radix tree over token ids → Arc'd page chains; leaf-LRU eviction, lock-on-match, session soft-pin; host tier with generation-counter coherence | Sharing/refcounting policy layer above KvPool (proven vLLM/SGLang split) |
| `BatchPlan` | per-step schedule output: sequence set, page tables, chunk boundaries, draft trees, phase mix | Single input consumed by attention backend AND captured graphs |
| `StepPolicy` | `fn step(&mut self, batch, engine) -> StepOutput { committed_tokens, kv_writes, telemetry }` | The ONLY thing the scheduler executes (D4) |
| `Drafter` | `propose(ctx) -> CandidateTree`, `update(accepted)` | Ngram/PLD (zero-config default), DraftModel, Medusa, EAGLE-3, MTP auto-enable |
| `MoEConfig` | `{n_routed, n_shared, top_k, expert_hidden, scoring: softmax \| sigmoid+bias, norm_topk_prob}` | Verified configs differ sharply across DeepSeek-V3/Qwen3/GLM-4.5/gpt-oss/Mixtral/Llama-4 |
| `TopologyMap` | device → resident expert ids (HBM-cache / pinned-host / staging); placement-policy resolver | `--cpu-moe` is a policy; future EP reuses it unchanged (D8) |
| `GrammarConstraint` | `compile(schema) -> Grammar; mask(state,pos) -> Bitmask; advance(tok); fork()/rewind()` | llguidance default; XGrammar binding for agentic loads (M6); cross-request compile cache keyed hash(schema+tokenizer+vocab) |
| `ReasoningParser` | streaming state machine over **token IDs**; per-model tag tables (deepseek_r1, qwen3, glm45, gpt-oss channels) | Emits typed Reasoning\|Content deltas; gates grammar constraints until `</think>`; feeds budget forcing |

## 5. Execution Model

Two phases per iteration:

- **PLAN (CPU)**: scheduler forms a `BatchPlan` mixing decode rows + prefill chunks under a token budget (default 8192 tokens/step); builds attention metadata; computes grammar masks concurrently with prior GPU work; resolves expert topology pointers.
- **RUN (GPU)**: decode batches replay from CUDA graphs captured per bucket (batch size ∈ {1,2,4,8,…,64}; spec adds tree node-count buckets {8,16,32,64}); all tensors live at fixed addresses in a graph-owned arena — block tables/routing indices/logits mutated **in place**, so replay never needs parameter updates. Prefill runs eager varlen (captured chunk graphs above 4k tokens). A dedicated copy stream serves MoE prefetch and KV host-tier migration, event-fenced against compute.

Streams/allocator: stream-ordered caching allocator (`cudaMallocAsync` semantics) + pinned-host pool inside `smelt-cuda`. At load, HBM partitions into weights / KV / workspace / expert-cache budgets, runtime-adjustable within safety margins.

**Stable-VA determinism reservation (from M0)**: virtual-address regions for graph arenas and weight slabs are reserved up front, enabling deterministic-replay mode and sleep/wake of RL rollouts without allocator relayout later.

## 6. Model Coverage Matrix

| Family | Attention | KV form | Extras | Milestone |
|---|---|---|---|---|
| Llama 2/3.x, Qwen2/3 dense, Mistral, Phi | GQA | paged bf16→fp8 | SWA variants (windowed second pool) | M1-M2 |
| Gemma 2/3 | GQA + interleaved SWA | dual pools, eager beyond-window free | logit soft-capping (gemma2) | M3 |
| Mixtral 8x7B/8x22B | GQA | standard | softmax routing, norm_topk_prob | M4 |
| Qwen3-MoE 30B-A3B / 235B-A22B | GQA | standard | sigmoid+bias routing, 128 experts top-8 | M4 |
| gpt-oss-20b/120b | GQA | standard | **MXFP4-native weights**, 128 experts top-4 | M4 |
| DeepSeek-V3/R1 class | **MLA** (c_kv 512 + rope 64 latent) | latent pages (576 elem/row) | 256 routed + 1 shared, sigmoid+bias; **MTP heads auto-enable** | M6 |
| GLM-4.5 / Air | GQA | standard | hybrid thinking blocks, MoE | M6 |
| LLaDA-8B / 1.5, Dream-7B, LLaDA2.0-mini, LLaDA-MoE-7B-A1B | **bidirectional full-seq** | whole-sequence page sets | remask-policy enum; Fast-dLLM approximate buffer-KV | M7 (LLaDA2.0-flash 100B = stretch when weights land) |
| Embedding/reranker pooling | any above | — | mean/last-token pooling heads | M3 (cheap add) |

New-arch onboarding contract: config struct + layer builder + weight-name mapping + golden-logit fixture; nothing else.

## 7. Weight Formats & Quantization Stack

Loader matrix (support level):

| Format | Level | Notes |
|---|---|---|
| Safetensors (+ HF sharded index.json) | Native, mmap zero-copy | Primary interchange; includes **FP8-E4M3 scaled checkpoints** (weights + scale sidecars) |
| GGUF | Native reader: F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS/MXFP4 | Metadata KV parsing. IQ1-IQ3: CPU dequant-to-BF16 fallback at load (explicitly not first-class GPU kernels) |
| PyTorch .bin / EXL2/EXL3 | Via `smelt convert` offline only | Never runtime pickle parsing |
| `.smt` pack (own) | Native | Kernel-tile-aligned repack output; single-file manifest embeds tokenizer, chat template, model config, quant manifest, provenance hash — GGUF-style one-file portability |

Unsupported-format behavior is explicit and tabulated: unknown GGUF type → hard error naming nearest supported type; safetensors dtype outside matrix → error unless `--dequant-to-bf16` fallback requested.

Execution tiers (per-layer dispatchable, mixed within one model):

| Tier | Weights×Acts | Path | Milestone |
|---|---|---|---|
| BF16/FP16 | baseline | cuBLASLt via GemmPlan | M1 bring-up + reference |
| FP8-E4M3 W8A8 | scaled, activation scales per-tensor default, per-channel option, SmoothQuant-style static calibration optional | cuBLASLt → custom | M3 |
| W4A16 g128 | GPTQ/AWQ/HQQ checkpoints | Marlin-style repacked nibbles+scales, own GEMV/MMVQ kernels | M3 |
| GGUF k-quants | Q8_0/Q4_K/Q6_K/IQ4_XS | MMVQ fused-dequant | M3-M4 |
| MXFP4 | native Blackwell tensor cores | block-scale-32 + E8M0/E4M3 scales | M4 |
| NVFP4 | tensor cores, two-level scaling (16-elem blocks + FP8 scales) | flagship | M8 |

Optional accuracy upgrades (post-M3, opt-in flags): QuaRot/SpinQuant-style Hadamard rotation preprocessing in `smelt convert` for outlier-heavy checkpoints; per-tensor mixed-bpw defaults à la Unsloth dynamic quants (converter suggests per-layer bpw from sensitivity pass).

Numeric truth (packing, scales, rounding) lives in `smelt-dtype` only. Perplexity harness gates every tier: WikiText-2 delta within ±0.15 of published GPTQ references; NVFP4 ≤2% vs BF16 on WikiText-2 + one coding eval.

KV quantization: FP8-E4M3 with per-head scales from M2 (per-tensor fallback). Capacity math (concrete, Llama-3-8B: 32 layers, 8 KV heads × 128 head_dim, K+V): 2048 elems × 2 B = **4 KB/layer/tok → 128 KB/tok BF16 → 64 KB/tok FP8**. Budget: 32 GB − 16 GB weights − ~3 GB workspace/graphs ≈ 13 GB KV ⇒ ~200k tokens FP8 (2x the ~100k-token BF16 envelope) at identical quality budget.

## 8. KV Cache Subsystem

- One `KvPool` per model: typed slabs cut into pages (16 tok default, 16-256 configurable), atomic refcounts, **copy-on-write** on radix-hit partial pages and beam forks.
- Block tables live in pinned staging mirrored to a fixed device address → captured graphs read them without recapture.
- MLA models store latents directly in pages; decode runs absorbed-latent kernels, prefill decompresses transiently to MHA tiles (validated vs decompressed-reference math, rel err < 1e-2).
- Eviction: leaf-LRU on radix tree; optional host tier migrates cold subtrees asynchronously with generation counters invalidating stale copies.
- SWA hybrids: second windowed pool frees beyond-window pages eagerly.
- Preemption: recompute-by-default; **swap-vs-recompute chosen per request by measured crossover** — `remaining_context × recompute_ms` vs pinned-copy PCIe time (microbenched at M0/M2, refreshed per driver release).

## 9. Scheduler & Serving Core

- Sarathi-style **token-budget loop**: each tick admits requests up to budget, mixing chunked-prefill slices with decodes → protects TPOT tail.
- Priority queues order admission; prefix-aware batching groups queued requests sharing prefixes to maximize radix hits.
- Primitives (built M2, consumed everywhere later): `pause / continue / retract` per request — one mechanism serving RL partial rollout, long-tail mitigation, and thinking-budget forcing.
- Targets: tick planning < 100 µs at 64 running requests; TTFT for 2k-token request queued behind decode ≤ 400 ms.

### Scheduler Invariants Under Speculation (hard rules, fuzz-tested)

1. **Draft KV writes are scratch**: tree-verify forwards write drafted-token pages tagged scratch/COW; rejection sampling commits the accepted suffix atomically (`commit_scratch`) — rejected branches free instantly, never leak into the prefix tree.
2. **Preemption mid-spec-round**: retract drops the whole candidate tree (scratch pages freed), keeps committed prefix; the request resumes AR and may re-enter speculation next round. No partially-committed trees, ever.
3. **Chunked prefill collision**: a new prefill chunk never splits an in-flight verify round; the scheduler defers admission to the tick boundary. Drafter state pauses (no proposals) while its request is mid-prefill; resumes after.
4. **Retract semantics**: retract = drop scratch + optionally evict committed pages (recompute later); invariant asserted: token accounting balances exactly across admit/commit/reject/retract events (scheduler fuzzer property).

## 10. Generation Policies

One scheduler-visible step; three families:

1. **ArStep** — incremental decode, graph-replayed per bucket; fused sampler (temperature/top-k/top-p/min-p/penalties/grammar-mask in one pass).
2. **SpecTreeStep** — `Drafter::propose` → padded candidate tree (node buckets, captured graph each) → ONE target forward verifies all branches via TreeVerify attention → lossless rejection sampling commits accepted suffix → `Drafter::update`. Depth/tree budget modulated online by EWMA acceptance with hysteresis across pre-captured tiers; auto-disable near B≈32 (net-positive gate: <2% regression at disable point). Grammar masks apply **per verify step**: the constraint state forks across the candidate tree, advances along accepted suffixes, rewinds to the last common accepted node on rejection (`fork()/rewind()` in `GrammarConstraint`).
   - Drafter roster: NgramDrafter/PromptLookup (CPU hash-table, zero-config default; big wins on RAG/code/agentic copy loads) → MtpDrafter (checkpoint MTP auto-enable; typically 1.5-1.8x, model-dependent) → DraftModelDrafter (aux stream) → Eagle3Drafter (consumes `ForwardOut.final_hidden`) → MedusaDrafter (trait-compatible; implemented if community demand, else documented non-goal).
   - Correctness invariant: acceptance divergence vs greedy baseline exactly zero on fixed seeds regardless of drafter.
3. **DiffusionStep** (LLaDA/Dream family) — each denoise pass is a step; full bidirectional forwards over preallocated `[prompt | buffer]`; prompt-side K/V reused exactly; buffer-side approximately cached Fast-dLLM-style with periodic refresh (refresh cadence adaptive to divergence metric). May reject batches it cannot pack (D4 escape hatch).
   - Remask strategy enum matching Dream's taxonomy: `Random`, `Origin`, `MaskGitPlus`, `TopkMargin`, `Entropy`, `ConfidenceThreshold(tau)`, `BlockFixed`.
   - Request parameters surfaced: `gen_length`, `steps`, `block_length`, `confidence_threshold`, `early_stop_eos`.
   - Token-budget accounting: one denoise round books `batch_tokens = seq_len` against the scheduler budget (full-sequence forwards); documented as the reason dLLM requests co-schedule poorly with dense AR batches at high occupancy.
   - `get_log_likelihood` endpoint (ELBO estimator) ships with M7.

Sampling extras: structured-output bitmask upload on the copy stream overlapping the forward; spec-decode draft-tree traversal through the XGrammar backend.

## 11. MoE Subsystem & Expert Cache

- Router logits → top-k on GPU → sort/align tokens per expert (vLLM moe_align pattern) → **grouped GEMM** once per layer over sorted ragged tiles. No-drop routing default; capacity-factor capping opt-in prefill-only.
- Shared experts always resident, computed early, overlapped with routed dispatch.
- **Expert residency**:
  - Experts stored kernel-optimal-packed in **pinned host RAM**, each slab carrying a generation counter (same coherence scheme as the KV host tier — stale copies invalidate on refetch);
  - bounded HBM hot-expert cache holds working-set slabs, evicted **LRU baseline** with frequency/recency-hybrid upgrade path;
  - residency arbitration: the load-time partitioner assigns HBM byte budgets [weights | KV | workspace | expert-cache]; at runtime the **expert cache may borrow free KV headroom but never exceed `kv_min_free_pages`**, so KV growth pressure evicts experts first, requests never OOM from cache greed;
  - router precedes FFN ⇒ free within-layer prefetch window: next-needed experts DMA into alternate staging on the copy stream while grouped GEMM consumes the current buffer (double-buffer, event-fenced);
  - Pre-gated-MoE router-driven prediction = upgrade path (needs retrained gates; optional per model).
- Hybrid CPU mode (`--cpu-moe`): GPU keeps attention + KV + shared experts + first-N MoE layers; remaining routed experts execute in-place on the V-cache CCD with AVX-512-VNNI blocked GEMV (llama.cpp `--cpu-moe` semantics, license-clean repack lineage). Per-layer placement decided by a Fiddler-style cost model, logged, overridable. Per-tensor expert bpw selection first-class: cold experts q3/q4, hot q6/q8.
- Telemetry (metric sources defined, not vibes): `expert_cache_hit_rate` = grouped-GEMM launches served fully-resident / total, per layer; `expert_fence_wait_ms` histogram = copy-stream event wait per layer; `dram_bytes_per_token` estimate from slab fetch counts. Miss-path bound: at 55% hit rate worst-case modeled layer stall = (misses × 0.35 ms) − hidden transfer overlap; fence-wait target < 0.03 ms/layer at ≥90% prefetch coverage, with the 55%-hit-rate case explicitly budgeted (≥25 tok/s gpt-oss target below assumes it).

## 12. Kernel Program (sm_120)

Chip truths honored everywhere: synchronous `mma.sync` m16n8k* (no wgmma/tcgen05 producer-consumer machinery — ported async designs measure slower on GB202), TMA + cp.async double-buffering, ~99 KB smem/SM, 255-reg limit, CUDA 13.x toolchain, `sm_120a` arch code, driver ≥ 570 gate.

Hand-written catalog (each gated on roofline-fraction exit criteria):

| Kernel | Purpose | Gate |
|---|---|---|
| FA2-style varlen paged prefill | causal + GQA + chunked prefill | ≥170 TFLOP/s BF16 (≈80% of the ~210 TFLOP/s dense peak; standard FLOP counting throughout this plan) |
| Split-K flash-decode | GQA paged KV, L2-aware CTA swizzle, sized for 170-SM occupancy | ≥85% of BW roofline at B=1-4 |
| GEMV/MMVQ fused-dequant | W4A16 + GGUF k-quant tiers | ≥340 tok/s 8B W4A16 |
| FP8 W8A8 scaled GEMM | workhorse tier | ≥330 TFLOP/s (≈79% of ~419 peak); 180 tok/s 8B |
| MXFP4 / NVFP4 block-scaled MMA | flagship tiers (M4/M8) | NVFP4 ≥630 TFLOP/s (≈75% of ~840 dense peak); ≥280-330 tok/s 8B |
| MLA absorbed-latent decode + MHA-mode prefill | DeepSeek-class | rel err < 1e-2 vs reference |
| Grouped GEMM + sort/align | MoE dispatch | see §17 targets |
| Tree-verify attention | speculative batches, block-sparse causal over tree topology | MTP ≥1.5x / EAGLE-3 ≥1.8x at B=1 |
| Bidirectional full-seq attention | diffusion LMs | parity vs reference outputs |
| Fused sampler | temp/top-k/top-p/min-p/penalties/bitmask | ≤50 µs total at 128k vocab |
| AVX-512 CPU GEMM/GEMV | Zen4 hybrid path | ≥40% DDR5 peak on expert microbench |

Toolchain: embedded `.cu` sources → NVRTC at startup (disk-cached PTX keyed by source hash); `build.rs` nvcc fatbins behind opt-in release feature compiling THE SAME sources; vendored CUTLASS headers for bring-up comparison only. CI runs `ptxas -v` SASS audits asserting expected MMA variants and zero register spills on the canonical shape set. Custom kernels enter through the `GemmPlan` registration seam — profiler-gated replacement of cuBLASLt call sites (reference point: beating cuBLAS on 5090 reaches ~68% FMA-pipe utilization where cuBLAS pins sizes 256-8192 at 33-42%).

CUDA-graph discipline: buckets padded, never recaptured; graph memory table ≤ 2.5 GB at max bucket set (vLLM calibration: 42 shapes ≈ 2.4 GB); replay overhead < 20 µs/step.

## 13. CPU & Hybrid Path

- Bandwidth-first doctrine (measured on-box): DDR5-4800 ≈ 50 GB/s effective ⇒ offloaded-MoE budgets: ≤5 GB DRAM/token for 10+ tok/s comfort, ≤25 GB for the ≥2 tok/s floor.
- Repack-once blocked layouts (8-row-interleaved, `block_q4_Kx8`-style) so runtime GEMV needs no transpose.
- Thread affinity: expert GEMM workers pinned to V-cache CCD {0-7,16-23}; hugepages for weight arenas; NT stores for streaming writes; AVX2 fallback build for non-AVX512 hosts.
- Streaming-batch trigger mode (switch hybrid↔offload at measured concurrency ~8-16) deferred until M6 telemetry exists.

## 14. Constrained Decoding & Reasoning Serving

- Grammar backends behind `GrammarConstraint`: **llguidance default** (Rust-native, ~50 µs/token, no startup cost) from M3; **XGrammar binding** at M6 for JSON-schema-heavy agentic loads (cross-request ~50% substructure reuse at 50 tools, 80x compile speedups); process-wide cache keyed hash(schema+tokenizer+vocab); automatic fallback llguidance→XGrammar or reverse when compile blowup detected.
- Masks are `u64` bitmasks fused into the GPU sampler, uploaded pinned-async during forward — never serial float-logit patching. Under speculation: fork/advance/rewind semantics per §10.2.
- Reasoning serving ("RLM"):
  - Token-ID streaming parser with per-model tag tables; `reasoning_content` deltas on chat-completions; Responses-API reasoning items (M6).
  - Thinking toggles (`enable_thinking`, GLM hybrid thinking) plumbed through chat templates.
  - **Budget forcing**: count think-block tokens per request; at limit, force-inject end-transition token sequence via scheduler injection hook (pause/inject primitives from M2). Exit criterion: terminates think blocks exactly at limit with natural-transition injection, verified per-model.
  - Radix KV makes multi-turn reasoning + n-sample branching nearly free after turn one — hit-rate metrics exposed.
  - Partial rollout/resume: pause/retract primitives + stable-VA sleep/wake.

## 15. Server API Surface

- `/v1/chat/completions`, `/v1/completions`: SSE chunks byte-level compatible with OpenAI Python SDK suite (first chunk role delta; subsequent chunks single content/reasoning deltas; terminal `finish_reason`; usage on final chunk); tool-call parsing conventions; stop-sequence/token-healing handled pre-detokenizer.
- `/v1/responses` (M6): `previous_response_id` chaining, reasoning items, background mode mapped onto pause/continue primitives.
- `/health`, `/v1/models`, `/metrics` — Prometheus instrument names aligned with vLLM conventions: `smelt_ttft_seconds` (request-admit → first token), `smelt_tpot_seconds` (inter-token, per stream), `smelt_queue_depth`, `smelt_prefix_hit_rate`, `smelt_kv_usage_ratio`, `smelt_expert_cache_hit_rate`, `smelt_expert_fence_wait_ms`. OTel spans with `gen_ai.*` attributes.
- Auth: bearer-token minimal multi-tenant (single-key v0.x).
- Deployment: single static binary; dynamic-loads libcuda; ships cached PTX (opt-in fatbins); fresh-machine install-to-serving < 5 min.
- License gate: `docs/MODELS.md` matrix maintained in-repo; CI fails on new model config lacking license entry (non-commercial weights excluded from shipped presets, loader remains neutral).

## 16. Testing, Observability, CI

- **Golden-logits harness** from M1: dumped PyTorch reference activations per arch; max-abs-diff + argmax-agreement thresholds per milestone.
- Kernel unit tests vs naive Rust reference on randomized tensors, per shape bucket.
- Paged-vs-naive-KV bit-exactness on fixed seeds (whole-memory-system canary).
- Perplexity harness (WikiText-2) gating every quant tier.
- Speculation: zero-divergence invariant + scheduler-invariant fuzz properties (§9).
- Soak: 24-48 h mixed-workload crash-free + flat KV bytes before tag.
- Benchmarks locked-clock (`nvidia-smi --lock-gpu-clock`); matrix = model × quant × concurrency × io-len; persisted baselines vs llama.cpp/vLLM/KTransformers on identical hardware; CI regression gates compare against the persisted autotune/bench table.
- Tracing: OTel spans; CUPTI wired into `smelt bench` for kernel-level attribution.

## 17. Performance Method & Targets

Roofline doctrine: decode B=1 is pure weight-streaming — `tok/s = active_weight_bytes / 1523 GB/s` at the 85%-efficiency planning figure. FLOP math uses standard counting (FMA = 2 ops; device peaks: BF16 ≈210, FP8 ≈419, NVFP4 ≈840 TFLOP/s dense). Every **performance-tier** exit (kernel/memory tiers, M1/M3/M4/M8) states a measured fraction of a computed ceiling alongside the absolute number; behavioral exits (scheduling, correctness, policy isolation) use absolute or suite-defined measures.

| Workload | Ceiling math | Commit |
|---|---|---|
| Llama-3-8B BF16 B=1 | 16 GB / 1.79 TB/s → 95 tok/s | ≥85 (≥89% ceiling) |
| Llama-3-8B FP8 | 8.5 GB → 210 tok/s | ≥180 (86%) |
| Llama-3-8B NVFP4 (M8) | 4.6 GB → ~330-389 tok/s | ≥280 (stretch 330) |
| Qwen3-30B-A3B Q4_K_M resident | active ≈3.3B params ≈ 1.9-2.2 GB/tok at ~4.7bpw avg → ~700-800 tok/s theoretical; llama.cpp measured 366 (~45% eff), sparkinfer 485 (~60%) | M4 ≥300 conservative (resident, incl. routing overhead); stretch ≥485 requires beating sparkinfer efficiency via fused MMVQ + resident experts — tracked explicitly, not silently |
| 8B FP8 prefill 8k tok | 2NP = 131 TFLOP ÷ ~330 TFLOP/s eff ≈ 400 ms | TTFT ≤500 ms idle; ≥20k tok/s |
| TPOT tail | chunked prefill + token budget | p99 @ B=32 ≤ 1.5x B=1 |
| Speculation | EAGLE-3/MTP B=1 | EAGLE-3 ≥1.8x, checkpoint MTP ≥1.5x wall-clock (numeric); net-positive verified through B=8 in exit tests; measured crossover expected near B≈32 where auto-disable triggers (<2% regression at disable) |
| gpt-oss-120b partial residency | 63 GB MXFP4 experts, top-4/128; ≥55% HBM hit rate with miss-path budget per §11 | ≥25 tok/s B=1 |
| DeepSeek-class hybrid CPU-expert | ≤20 GB DRAM/tok ÷ 50 GB/s | ≥2.5 tok/s floor; GLM-4.5-Air hybrid ≥15 tok/s B=1 |
| Dream-7B + Fast-dLLM-style caching | paper claims up to 27.6x | ≥5x honest wall-clock vs naive denoise on fixed prompt suite (stretch 8x) |
| Graph replay | vs ~100 eager launches | <20 µs/step overhead |

## 18. Roadmap

| Phase | Scope | Exit criteria (hard, measurable) |
|---|---|---|
| **M0 Ignition** | Install CUDA 13.x toolkit; workspace scaffolding; `smelt-cuda` boundary (context/stream/arena/pinned/NVRTC+cache, stable-VA reservation); `smelt-dtype`; smoke kernel; PCIe microbench; CI device inventory | `cargo run -p smelt-cli -- selftest` compiles a kernel via NVRTC, launches, validates on the 5090, green in CI; pinned H2D GB/s measured and recorded |
| **M1 First light** | `smelt-formats` (safetensors+HF index+GGUF subset+fp8 checkpoints), tokenizer, `smelt-core` Tensor+GemmPlan, eager BF16 Llama forward, naive KV, greedy+fused sampler, golden-logits harness, SASS-audit CI job (ptxas -v, spill assertions) | Llama-3-8B BF16 coherent CLI generation; max abs logit diff <0.05 vs reference over 100 tokens; 16 GB mmap load <30 s; ≥65% FMA-pipe utilization on bring-up GEMM vs cloudrift 68% reference |
| **M2 Engine heart** | KvPool+COW+PrefixTree LRU, FP8 per-head-scale KV, AttnBackend trait + FA2 varlen + split-K flash-decode, token-budget scheduler + chunked prefill + continuous batching + preemption (swap-vs-recompute heuristic), pause/retract/continue primitives, axum server REST+SSE, Prometheus basics, scheduler-invariant fuzzer | 32 concurrent chat streams co-batched; paged==naive logits bit-exact; 8k TTFT <1 s; 1 h soak flat KV bytes; fuzz properties green 10⁶ events |
| **M3 Speed of light** | CUDA-graph buckets + fixed-address arenas; FP8 W8A8 tier (+activation-scale options); `smelt convert` W4A16 repack + GEMV/MMVQ; mixed-format per-layer dispatch; Gemma 2/3 interleaved-SWA dual pools; llguidance constrained decoding with think-phase gating; reasoning-parser tag tables; embedding pooling | 8B BF16 B=1 ≥85 tok/s locked-clock (≥89% ceiling); W4A16 ≥250; replay overhead <20 µs; constrained JSON parses 100% over 10k generations; ppl delta ≤±0.15 Wiki2 on GPTQ refs |
| **M4 Mixture ignition** | `smelt-moe`: generic MoEConfig, router+sort/align, grouped GEMM, shared experts, expert cache (LRU, pinned backing, generation counters, HBM budget w/ `kv_min_free_pages` arbitration), double-buffer prefetch; Mixtral + Qwen3-MoE + gpt-oss-20b | Qwen3-30B-A3B resident ≥300 tok/s (stretch 485 per §17); prefetch hit-rate >95% AND `expert_fence_wait_ms` <0.03/layer at B≤8 (distinct metrics, §11); gpt-oss-20b MXFP4 serves |
| **M5 Guess well** | StepPolicy refactor; Drafter trait + Ngram default; MtpDrafter auto-enable; Eagle3 consuming hidden features; tree-verify kernel + node buckets; acceptance EWMA gating + telemetry; scratch-page commit protocol (§9) | ≥1.5x B=1 wall-clock with checkpoint MTP, ≥1.8x with EAGLE-3 on supported models; ≥1.3x code workload ngram-only; zero accepted-token divergence on fixed seeds; auto-disable crossover measured + documented (<2% regression at disable) |
| **M6 Big game** | MLA latent KV + absorbed decode + MHA-mode prefill; DeepSeek-V3-class + GLM-4.5/Air + large Qwen3-MoE; hybrid `--cpu-moe` + cost model; thinking-budget forcing via injection; Responses API; XGrammar backend + grammar cache | GLM-4.5-Air ≥15 tok/s B=1 hybrid on 32 GB; MLA rel err <1e-2; CPU expert microbench ≥40% DDR5 peak; thinking models stream separated reasoning deltas at 30k CoT with parser <2% step time; budget-forcing terminates think blocks exactly at limit; BFCL-style strict-schema tool loop passes |
| **M7 Non-causal** | DiffusionStep + bidirectional backend; LLaDA/Dream/LLaDA-MoE configs; remask enum (§10.3); Fast-dLLM approximate buffer-KV; dLLM request params + `get_log_likelihood`; constrained decoding verified under diffusion; diffusion-aware SSE | LLaDA-8B-Instruct serves via same endpoint with gen_length/steps/block_length knobs honored; ≥5x wall-clock vs naive denoise on fixed suite; AR regression suite untouched (policy isolation proof) |
| **M8 Flagship numerics** | NVFP4 tensor-core path end-to-end; accuracy campaign; autotuner persistence; host-DRAM KV tier generalization; full comparison campaign vs llama.cpp/vLLM | NVFP4 8B ≥280 tok/s (stretch 330); ppl delta ≤2% vs BF16; published locked-clock scorecard |
| **M9 Ship it** | Hardening: fuzzed loaders, adversarial mixes, graceful OOM; docs/book; packaging (single binary + cached PTX, opt-in fatbins); 48 h soak; changelog; license matrix lint | All M1-M8 gates re-verified in CI; fresh-machine install-to-serving <5 min; v0.1 tagged |

Sequencing logic: user-visible artifact (streaming server) exists by M2 (~week 10); every later phase lands on a working engine. Custom-GEMM-beats-cuBLASLt escalation remains profiler-gated post-M3. Metal/ROCm/Vulkan and distributed TP/EP deliberately post-v0.1 — seams (`TopologyMap`, `AttnBackend`, dtype/layout split) anticipate them.

## 19. Risks & Mitigations

| Risk | Sev | Mitigation |
|---|---|---|
| sm_120 PTX/toolchain churn (GeForce block-scaled-MMA exposure settling) | High | FP8-first; NVFP4 isolated in M8 behind sealed quant seam; verify PTX ISA docs before kernel commitment |
| Solo-dev scope collapse | High | Hard milestone exits; every phase ships something runnable |
| Diffusion-as-step-policy strains paged-causal assumptions | Med | Whole-sequence page ownership, unpackable-batch rejection; dedicated seam test from M7 start |
| Speculative scratch-page leaks corrupt prefix cache | Med | §9 invariants + scheduler fuzzer property (token accounting balances) |
| NVRTC lacks nvcc link-time optimizations | Med | Single-TU kernel design; AOT shares sources; delta benched each release |
| cudarc safe-graph API gaps (exec-node updates) | Med | Raw FFI contained in unsafe boundary; track upstream |
| FP8-KV quality drift on some models | Med | Per-tensor scale fallback; ppl harness gates |
| Host RAM pressure (pinned pools on 64 GB) | Med | Configurable pinned budgets; host-tier telemetry |
| Weight licensing contamination | Low | Clean-room repack lineage (llama.cpp-proven MIT); MODELS.md lint gate |
| Upstream dependency rot (cudarc/tokenizers majors) | Low | Pinned versions; unsafe-boundary isolation makes swaps mechanical |

## 20. Open Questions

1. Exact sm_120a PTX surface for NVFP4/MXFP4 block-scaled `mma.sync` — verify against CUDA 13.x PTX ISA before M8 commitment.
2. Pinned-vs-pageable optimum for host-tier KV migration on this chipset (M0/M2 microbench).
3. EAGLE-3 draft heads: community checkpoints sufficient at launch, or must we distribute per-model trained heads?
4. LLaDA2.0-flash (100B MoE diffusion) as M7 stretch once weights land.
5. Optimal page size per model class (16 vs 32) — autotune sweep decides the default.

## 21. Research Appendix (primary sources)

**Formats**: [GGUF spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) · [GGUF deep dive](https://deepwiki.com/ggml-org/ggml/2.6-gguf-file-format) · [gpt-oss announcement](https://huggingface.co/blog/welcome-openai-gpt-oss)
**Quantization**: [Marlin](https://github.com/IST-DASLab/marlin) · [GPTQ](https://arxiv.org/abs/2306.00978) · [HQQ](https://github.com/dropbox/hqq) · [NVFP4 @ NVIDIA](https://developer.nvidia.com/blog/tag/nvfp4/) · [vLLM quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html)
**Attention/KV**: [PagedAttention](https://arxiv.org/abs/2309.06180) · [SGLang radix cache](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) · [FlashAttention-3](https://arxiv.org/abs/2407.08608)
**Decoding acceleration**: [Speculative decoding](https://arxiv.org/abs/2211.17192) · [Medusa](https://arxiv.org/abs/2401.10774) · [EAGLE](https://github.com/SafeAILab/EAGLE) · [Sarathi-Serve chunked prefill](https://arxiv.org/abs/2302.01318)
**Diffusion LMs**: [LLaDA](https://github.com/ml-gsai/LLaDA) · [Dream](https://github.com/DreamLM/Dream) · [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) · [BD3-LM block diffusion](https://arxiv.org/abs/2502.06768)
**MoE serving**: [DeepSeek-V3 tech report](https://arxiv.org/abs/2412.19437) · [Pre-gated MoE](https://arxiv.org/abs/2508.10925) · [mixtral-offloading](https://github.com/dvmazur/mixtral-offloading) · [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
**Rust stack**: [cudarc](https://github.com/coreylowman/cudarc) · [mistral.rs architecture](https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/docs/src/content/docs/developer/architecture.md) · [candle](https://github.com/huggingface/candle)
**Blackwell kernels**: [Blackwell + CUDA 12.9 family features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/) · [Beating cuBLAS on RTX 5090](https://www.cloudrift.ai/blog/beating-cublas-on-rtx-5090) · [CUTLASS](https://github.com/NVIDIA/cutlass/blob/main/README.md) · [GeForce vs datacenter Blackwell](https://rfriedmann.de/blog/what-the-5090-lacks-vs-datacenter/)
**CPU/hybrid**: [llama.cpp repack sources](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cpu/arch/x86/repack.cpp) · [KTransformers](https://github.com/kvcache-ai/ktransformers)
**Serving API**: [llguidance](https://raw.githubusercontent.com/guidance-ai/llguidance/main/README.md) · [XGrammar-2](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation) · [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
**Reasoning serving**: [vLLM reasoning outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html) · [SGLang separate reasoning](https://docs.sglang.io/docs/advanced_features/separate_reasoning.md) · [s1 budget forcing](https://arxiv.org/abs/2501.19393) · [GLM-4.5](https://huggingface.co/zai-org/GLM-4.5)

Full corpus with per-technique detail: `local://smelt-research.json`. Architecture variants + judgement: `local://smelt-arch-{a,b}.json`, `local://smelt-arch-judgement.json`. Critic round: `local://smelt-critic.json`.
