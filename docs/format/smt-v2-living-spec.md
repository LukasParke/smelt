# SMT v2 — Open Core & Living Weights

**Status**: design spec, supersedes SMT v1's extensibility model (`smt-v1-spec.md`). Binary section
architecture from v1 is preserved; everything about *how unknown things load* changes.
**Evidence chain**: research sweep `docs/research/extensibility-mechanisms.md` · judge panel scores in
session record · on-machine experiments `docs/experiments/` (verified claims marked ✅ below).

---

## 0. Why v2 exists

V1's closed registries (fixed atom axes, fixed opcodes, fixed hw-profile ids) share a property with
every system that died of rigidity: **unknown = failure**. The cross-system survey found no surviving
extensible system that relies on a closed registry alone; all converge on five mechanisms
(unknown-region preservation, self-describing payloads, capability negotiation, tiered fallback,
versioned extension packs). V2 adopts all five and adds one thing none of them had: **the container is
aware the engine is mutable while running**, so extension packs and weight deltas are runtime events,
not install-time decisions.

Design panel outcome (2 competing proposals, 3 independent judges): Openness-design won
future-proofness (8.7/10) but lost safety (7.7 vs 8.7); Conservatism-design won safety/perf/complexity.
V2 is the judged hybrid.

## 1. Mechanism: tagged references (replaces bare enum ids)

Every reference to an atom, opcode, transform, or hardware profile becomes:

```
Ref { namespace: u16, name: StrRef|Inline, version: u32 }
```

- namespace 0 = core registry (v1 atoms/ops, frozen, append-only).
- Other namespaces resolve through EXTDEFS (in-file) or dialect packs (installed).
- Unknown tag ⇒ never a guessable-enum misread; it names itself. (ONNX lesson: immutable
  `(domain, op_type, since_version)` triples survive forever.)

## 2. Mechanism: EXTDEFS — self-describing definitions in-file

New section carrying records:

```
ExtDef {
  ref: Ref, kind: ATOM_DECODE | OP_COMPOSE | TRANSFORM | SCALEFN,
  flags: PORTABLE | DETERMINISTIC | OPTIONAL_SUBTREE,
  body: ExprTree | SubgraphRef | Recipe,     // see below
  digest: BLAKE3-256,                         // what signatures cover
}
```

`ExprTree` (for ATOM_DECODE / SCALEFN) is a typed expression over a **closed primitive set**:
`load_idx, load_scale, lookup_codebook, mul, add, fma, exp2, cast, rne_round, clamp`. No loops, no
arbitrary memory access, bounded tensor extents ⇒ statically verifiable in linear time (eBPF-verifier
lesson), compilable by NVRTC to PTX (disk-cached, same pipeline as core kernels), or executable by the
CPU reference interpreter. This is deliberately NOT a language: no control flow, no pointers, no I/O.
Anything needing more than expressions must ship as engine code — the format says so loudly instead of
pretending.

`OP_COMPOSE` defines new layer types as subgraphs over already-known ops (macro expansion at graph
build). A 2027 "MXFP6 three-level-scale" atom is ~15 expression nodes; a "liquid continuous-time cell"
with exotic state is an OP_COMPOSE over primitives plus declared state shapes — both load without an
engine release.

## 3. Mechanism: tiered resolution (the load-time decision tree)

For each referenced capability:

```
1. core/native registered            -> fastest known kernel
2. signed dialect pack installed     -> native kernel from pack
3. in-file ExtDef + valid signature  -> JIT compile (NVRTC), cache keyed by digest
4. in-file ExtDef, unsigned          -> CPU interpreter only (deterministic, safe)
5. declared OPTIONAL_SUBTREE         -> skip subtree, log resolution map
6. otherwise                         -> hard error naming name@version + digest
```

Steps 3–4 mean an untrusted file can never execute anything beyond verified pure expressions on the CPU
path. META carries a capability string; the loader emits a per-file resolution map
(`native: 214, pack: 12, jit: 3, interp: 0, skipped: 1`) — observability instead of silent drift.

## 4. Mechanism: dialect packs

Signed bundles (ed25519 over pack root hash; engine config lists trusted keys) distributing shared
ExtDefs independently of engine releases — the MLIR-dialect distribution model. Packs hot-install **at
runtime**: new kernels enter the GemmPlan registry via the same pointer-swap path proven in the
living-engine experiment ✅ (critical section 0.02–231 µs class, zero restarts). In-flight requests
finish on old code; new steps pick up new code.

## 5. Mechanism: preservation & evolution rules

- All sections TLV; unknown types are **skipped AND retained** on rewrite (protobuf rule) — a v3 field
  survives a v2 round-trip.
- Codified compatibility whitelist (Cap'n Proto lesson): patch = bugfix w/o layout change; minor =
  append sections/refs; major = reinterpretation allowed. LLVM bitcode is the named counter-example:
  forward-compat promises are refused because they freeze internal evolution — SMT instead freezes only
  the *reference grammar*, not implementations.
- EVAL cards remain mandatory for quantized packs (converter-measured ppl Δ, sensitivity ranking).

## 6. Living weights — the architecture

Five substrate properties, each mapped to an implemented-or-precedented mechanism:

| Brain property | SMT/engine mechanism | Precedent / evidence |
|---|---|---|
| Long-term memory (stable synapses) | Immutable CAS tensors, content-addressed, signed | HF Xet dedup default since 2025; OCI artifact registries |
| Plastic synaptic state | **State layers**: `state_read/state_write/delta_update` ops with declared shapes (state_schema); TTT/Titans-lineage hidden states as first-class tensors | TTT (2407.04620), Titans (2501.00663); production liquid models |
| Neurogenesis (growth) | Expert slot pools with preallocated headroom; runtime resize | FreeToken runtime re-budgeting; measured swap cost ✅ 231 µs |
| Ongoing learning | Delta streams applied online (LoRA-class overlays composed against base digests) | S-LoRA/Punica/dLoRA shipping today; measured loss-bend ✅ mid-serving |
| Sleep (consolidation) | Background compaction: deltas → new canonical generation, re-materialize hot layouts, atomic generation switch | CAS makes generations cheap; ServerlessLLM locality lessons |

Neuromodulation maps to global scalar knobs (learning-rate-like gains on state updates) exposed through
the scheduler, not baked into weights. Catastrophic-forgetting mitigations (orthogonal-gradient /
replay families) are *policy* above this substrate, out of format scope.

**What this is not**: not a training format, not autograd, not continual-learning algorithmry. It is
the storage/execution contract that makes those systems expressible without format migrations.

## 7. Performance doctrine (measured anchors)

- Decode stays bandwidth-bound: DRAM stream 50.7 GB/s ✅, naive GEMV B=1 52.6 GB/s ✅ — every format
  decision optimizes bytes-moved-per-token first (expert slots, materializations, KV tiers).
- Compute-bound regimes reward kernel quality, not format: naive GEMM B=64 reached only 67 GFLOP/s due
  to W re-streaming — loop order and blocking are engine concerns the format must not constrain away
  (materializations carry layouts, graphs stay free).
- Atom-space numeric results ✅ inform defaults: prefer sub-16 scale trees for outlier-heavy checkpoints
  (two-level 0.104 vs flat-E8M0 0.167 rms-rel on outlier mixtures); E8M0-flat (MXFP4-class) remains
  correct for natively-blockscaled hardware paths.
- Hardware frontier (survey): FLOPs grow ~3×/2yr vs bandwidth ~1.4×/2yr; HBM4 ramping 2026; JEDEC HBF
  (high-bandwidth flash) standard published Aug 2026 → expert pools will tier across VRAM/HBM/DRAM/HBF;
  TopologyMap + slot tables are the ready seam. PIM/CXL/analog remain niche or edge-scale; software
  attacks (caching, placement, quantization, scheduling) own the 2026-27 window.

## 8. Honest limits

1. Single-skeleton atom decoder is design-stage; experiments validate the coordinate space numerically
   (three hand-written decoders), not the unified implementation. M1 gate includes building it.
2. Kernel hot-swap proved the mutation mechanism (zero pause) but showed no throughput delta in the
   harness (FLOPs not gated through swappable path) — the claim is scoped to mechanism, not speedup.
3. Step-latency tail (35 ms single samples from OS/allocator jitter) means mutation-cost metrics must
   use critical-section time, not step time.
4. ExprTree verification is specified, not implemented; until then unsigned extdefs never leave the
   interpreter (fail-safe default).

## 9. Migration

v1 packs: registry ids → tagged refs mechanically; EXTDEFS absent = fully native path. safetensors/GGUF
→ existing converter emits canonical atoms + optional materializations; GGUF quant enums map to atoms
(v1 §3 table unchanged). No breaking change to v1 files: they are valid v2 files with an empty
extension surface.
