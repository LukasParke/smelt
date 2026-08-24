# Model Lab — SMT v2-lite end-to-end proof on real GPT-2 weights

Code: `experiments/model-lab/` · Evidence log: `results/model-lab.log` (exit 0)
Model: `openai-community/gpt2` (124M params) · Host: Ryzen 9 7950X3D / DDR5-4800 / NVMe

## What was built (the pipeline)

```
HF safetensors (548 MB fp32)
   │  pack bin: atom encode (core.i8.b32.f16scale for big matrices,
   │            core.f16 for small/1D), Conv1D→[out,in] transpose AT CONVERT TIME,
   │            per-tensor BLAKE3-128 digests, graph-as-data op list,
   │            embedded tokenizer.json, merkle content_id
   ▼
gpt2-q8.smt (151.5 MB)      gpt2-f16.smt (262.7 MB, canonical comparison pack)
   │  run bin: mmap load → merkle verify → all-tensor digest verify →
   │           BPE encode/decode (hand-rolled, validated exact vs reference ids) →
   │           fused-dequant GEMV executor over KV cache → greedy generation
   ▼
correctness probes + benchmarks + live engine swap
```

## Proof points (all in `results/model-lab.log`)

1. **Integrity**: `tensor_digest_mismatches=0`; merkle `content_id` verifies; load+mmap+verify = 302 ms for 151 MB.
2. **Tokenizer exactness**: `"The capital of France is"` → `[464, 3139, 286, 4881, 318]` — identical to reference GPT-2 tokenization; `"Once"` → `[7454]`.
3. **Model correctness (objective)**: teacher-forced NLL/token = **2.40** (in-domain prose), 2.52 (scientific prose) — healthy GPT-2-small numbers from OUR executor reading OUR container.
4. **Model correctness (behavioral)**: greedy outputs are canonical GPT-2 — e.g. *"The capital of France is → the capital of France, and it is the capital of France…"* (the well-known greedy repetition attractor).
5. **Atoms preserve the model**: f16-pack vs q8-pack forced-continuation divergence: top-1 agreement **91.3 %** across 23 positions × 50 257-way softmax, mean |Δlogit| 0.238, NLL Δ = −0.047 (quantization noise ≈ sampling noise).
6. **Living engine on real weights**: two engines (different atoms) behind one RwLock slot; serving path swapped q8→f16 at step 12 of an ongoing generation with critical section **0.35 µs** — text continues coherently across the switch.

## Benchmarks

| Metric | Value | Notes |
|---|---|---|
| Pack conversion | 340 ms (q8) / 511 ms (f16) | 548 MB source incl. transpose+quantize+hash |
| Load + integrity verify | 302 ms | mmap; full per-tensor digest pass |
| Prefill (sequential steps) | 21.0 tok/s (47.6 ms/tok) | one token per step, no batching — engine-level optimization deliberately out of scope |
| Decode B=1 | **20.9 tok/s** (47.8 ms/tok) | 12-layer naive kernels, 8-thread matmuls |
| Roofline floor | 2.6 ms/tok (126 MB active @ 50.7 GB/s measured DRAM BW) | achieved = **5.5 % of bandwidth ceiling** |

The 18× gap to the roofline is honest and expected: attention loops over positions serially per head,
no flash-tiling, no vectorized i8 paths beyond `mul_add`, thread spawn per matmul call. It bounds what
the *format* costs (≈0: payloads are DMA-ready blocks) versus what *kernel engineering* buys — which is
SMELT's M1–M4 job, not the container's.

## Scope notes / honest limits

- GRAPH section carries the op list as data and the loader resolves it, but ops dispatch to hand-written
  Rust kernels (native fast path); the v2 expression-JIT/interpreter tiers are spec-stage (spec §8).
- GELU uses the tanh approximation (matches most inference stacks; ~1e-3 rel vs erf-exact).
- Prefill runs sequentially through the same `step()` as decode; chunked/batched prefill is future work.
- Weights and packs are gitignored (>100 MB); reproduce via the download + commands below.

## Reproduce

```bash
cd experiments/model-lab/assets
curl -sL -O https://huggingface.co/openai-community/gpt2/resolve/main/{config.json,tokenizer.json,model.safetensors}
cd .. && cargo build --release
./target/release/pack            # builds gpt2-q8.smt + gpt2-f16.smt
./target/release/run             # correctness + benchmarks + swap demo > results log
```
