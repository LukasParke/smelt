# SMT v1 — Model Container Specification

**Status**: draft v1 (2026-08-23). Extends PLAN.md §7 (`.smt` pack) into a full format design.
**One-liner**: a content-addressed, execution-layout-aware, self-describing model container whose
quantization is a *space* instead of an enum list, and whose architecture is *data* instead of Python code.

---

## 0. Why a new format (gap analysis)

| Existing | Fatal gap |
|---|---|
| safetensors | Dumb blob map: no quant awareness, no architecture, no dedup, no alignment contract |
| PyTorch `.bin` | Pickle (RCE), torch-coupled, no random access |
| GGUF | Right instincts (one file, native quant, tokenizer inside) but closed enum taxonomy, ad-hoc metadata, no execution layouts, no content addressing, C-owned |
| ONNX | Graph good, weight story terrible, expressiveness graveyard, nobody ships frontier models in it |
| EXL2/EXL3 | Great packing, locked to one consumer, still plain safetensors underneath |

SMT keeps safetensors' simplicity and GGUF's portability, then adds the three things nobody ships:
**(1) execution-layout materializations**, **(2) a parameterized numeric-atom space**, **(3) content-addressed composition (deltas)** — plus a portable architecture IR so adding a model doesn't require an engine release.

## 1. Design principles

1. **Canonical + materialized.** Every tensor has exactly one canonical encoding (source truth) and zero or more
   *materializations*: byte-exact kernel-consumable layouts per hardware profile. Load = pick materialization, else
   mmap canonical and repack once.
2. **Quantization is a coordinate space, not a list.** A `NumericAtom` is a point on six orthogonal axes.
   Every legacy format (Q4_K, IQ4_XS, MXFP4, NVFP4, GPTQ, EXL2) maps to an atom. One templated decoder serves all;
   NVRTC instantiates per-atom kernels from the same `.cu` sources (PLAN D2).
3. **Architecture is data.** A small typed graph IR describes forward computation from registered atoms.
   Unknown op → hard error naming nearest supported op (GGUF lesson). No Turing-completeness, no escapes except
   explicit `custom_op` marked non-portable.
4. **Content-address everything.** Merkle root (`content_id`) + per-tensor BLAKE3-128. Dedup, deltas, partial
   download verification, signing — all fall out of hashing, cost ~nothing at write time.
5. **DMA-honest alignment.** Expert slots padded to 2 MiB, tensors ≥256 B aligned, hugepage hints — because
   PCIe-transfer efficiency is the decode bottleneck for >VRAM MoE (FreeToken lesson).
6. **Skip what you don't know.** Sections are length-prefixed TLV; readers MUST ignore unknown section types.
   Feature evolution never breaks old loaders.
7. **Measure at convert time.** The converter emits its own eval card (ppl delta, sensitivity ranking,
   recommended per-layer atoms). Roofline-fraction culture (PLAN §17) baked into the artifact.

Non-goals v1: training/optimizer state, encryption/DRM, autograd IR, ONNX interop, executable bytecode in-file.

## 2. Binary layout

All integers little-endian. CRC32C per section payload.

```
┌──────────────────────────────────────────────────────────┐
│ SUPERHEADER — fixed 128 B                                │
│   magic        u32 = "SMT\x01"                           │
│   version      u16                                       │
│   header_len   u16                                       │
│   feature_bits u64        (HAS_GRAPH|HAS_DELTAS|SIG|...) │
│   index_off    u64       ── section directory            │
│   index_len    u64                                       │
│   content_id   BLAKE3-256  (Merkle over section digests) │
│   hw_hint      u32         (profile the pack targets)    │
│   reserved     ...                                       │
├──────────────────────────────────────────────────────────┤
│ SECTIONS — TLV: {type u32, flags u32, len u64, crc u32}  │
│   INDEX · META · GRAPH · TOKENIZER · TENSORS             │
│   CODEBOOKS · DELTAS · TRANSFORMS · EVAL · SIG           │
└──────────────────────────────────────────────────────────┘
```

### Section catalog

| Type | Contents | Required |
|---|---|---|
| INDEX | directory: {section_type → off,len,digest}; the ONLY section a router reads first | yes |
| META | config: architecture params, generation defaults, chat template, license manifest, source provenance, `state_schema` (recurrent/hybrid state shapes for checkpointing) | yes |
| GRAPH | architecture IR (§5) | yes unless `arch: external` |
| TOKENIZER | verbatim `tokenizer.json` (+ optional compiled trie blob) | yes |
| TENSORS | tensor table + payloads (§4) | yes |
| CODEBOOKS | shared codebooks referenced by atoms (i-quant style), globally deduplicated | opt |
| DELTAS | adapter/fine-tune overlays (§6) | opt |
| TRANSFORMS | materialization recipes (§4.3) | opt |
| EVAL | converter measurements: ppl Δ, per-layer sensitivity, recommended atom profile, golden-logit checksum | SHOULD |
| SIG | ed25519 signature over `content_id` + key id | opt |

## 3. NumericAtom — the scale lattice

Six axes define every quantization scheme in existence today:

```rust
struct NumericAtom {
    val:      ValDtype,   // I2 I3 I4 I6 I8 | F4_E2M1 | F8_E4M3 | F8_E5M2 | BF16 | F16 | F32
    block:    u16,        // values per microblock: 0(per-elem) 16 32 64 128 256
    scale:    ScaleTree,  // none | flat(F16|E8M0|E4M3) | two_level{super, sub_dtype: SCALE6|F16}
    zero:     ZeroMode,   // symmetric | sub_min_f16 | packed_nibbles(GPTQ-style) | codebook
    codebook: Option<Ref>,// shared CODEBOOKS entry (i-quant vector schemes)
    order:    LayoutTag,  // canonical row-major nibble order (materializations override)
}
```

Legacy schemes as atoms — proof the space suffices:

| Legacy | Atom coordinates |
|---|---|
| Q4_0 | `{I4, 32, flat F16, symmetric}` |
| Q8_0 | `{I8, 32, flat F16, symmetric}` |
| Q4_K | `{I4, 256, two_level(sub 16, SCALE6, d+dmin), sub_min}` |
| Q6_K | `{I6, 256, two_level(sub 16, F16), symmetric}` |
| IQ4_XS | `{I4, 256, two_level(SCALE6), codebook IQ4_NL}` |
| MXFP4 | `{F4_E2M1, 32, flat E8M0, symmetric}` — Blackwell MMA native |
| NVFP4 | `{F4_E2M1, 16, flat E4M3 + F32 tensor scale}` |
| GPTQ g128 | `{I4, 128, flat F16, packed_nibbles + g_idx}` |
| BF16 baseline | `{BF16, 0, none, symmetric}` |

Consequences:
- **One decoder skeleton**: `dequant(idx) = f(val, codebook) ⊗ Πscales`; NVRTC generates the per-atom kernel at
  startup, disk-cached PTX keyed by atom hash (PLAN D2). New scheme = new coordinates, not new engine release.
- **Mixed granularity free**: per-tensor atoms + EVAL's recommended profile = dynamic-quants à la carte.
- Round-trip rule: converter MUST publish the atom coordinates of every tensor in META's quant manifest.

## 4. Tensor table

### 4.1 Records

```
TensorRecord {
  name          String        // path-like: "l12.moe.e047.ffn.w13"
  shape         [u32]         // logical, canonical order
  atom          NumericAtom
  payload       {off, len}    // within TENSORS, 256B-aligned
  digest        BLAKE3-128
  expert_slot   Option<u32>   // membership in a slot group
}
```

### 4.2 Expert slots (first-class MoE)

Experts are entities, not loose tensors: all members of `(layer, expert)` are contiguous and the slot is padded to
a **2 MiB multiple** → one pinned-host slab, one `cp.async.bulk` per fetch, LRU-cacheable as a unit.
Slot table lives in META (`moe_topology`: n_layers, n_routed, slot_bytes, suggested HBM budget, `kv_min_free`
arbitration hint). Directly serves FreeToken/KTransformers-style hierarchies and PLAN §11.

### 4.3 Materializations

```
Materialization {
  hw_profile u32        // registry: CPU_AVX512_ZEN4, CUDA_SM120_MARLIN, CUDA_SM90, ...
  recipe     RecipeRef  // TRANSFORMS entry
  payload    {off, len} // byte-exact kernel-ready layout
}
Recipe = [ (Permute[p]) | InterleaveRows(g,n) | PadTo(dim,mult) | TransposeTiles(r,c)
         | PackNibbles(w,ord) | FuseScales(mode) | CastRound(atom') ]*
```

Determinism contract: same canonical bytes + recipe version ⇒ identical output bytes (property-tested).
Marlin lesson institutionalized: the permuted-nibble+scales layout ships **in the file**; runtime load is pure DMA
(PLAN D9, `Tensor<T,L>` at the boundary). Typical pack ships canonical + 1–2 hot profiles; others regenerated
on demand via recipe.

## 5. Graph IR — the "neural network syntax"

Typed dataflow DAG over registered atoms. Textual surface syntax (canonical form in-file is compact TLV):

```text
graph glm45_air (tok: i32[B,S]) -> logits: f32[B,S,V] {
  h   = embed(tok, t.tok_emb @bf16)
  h   = rmsnorm(h, t.l0.norm_in, eps=1e-5)
  // attention atom: GQA causal, optional window
  a   = attn.causal_gqa(h, t.l0.wq @q6k, t.l0.wk @q6k, t.l0.wv @q6k,
                        t.l0.wo @bf16, n_kv=8, head_dim=128, rope{theta=1e6})
  h   = h + a
  // MoE atom: router policy + slot table + shared expert
  moe = moe(h, gate=t.l12.router @bf16,
            mode=sigmoid_bias, top_k=8, n_routed=160, norm_topk=true,
            slots=t.l12.slots @slot(q4k), shared=(t.l12.sh_up, t.l12.sh_down))
  ...
  logits = linear(rmsnorm(h, t.norm_out), t.head @bf16)
}
```

Op registry (v1, closed): `embed · linear · rmsnorm · softmax · silu · gelu · geglu · rope ·
attn.causal_gqa{window?} · attn.mla{ckv,k_rope} · attn.bidir_full · moe · add · mul · concat · split ·
gather · cast · logits · custom_op(!nonportable)`.

Rules:
- Shapes symbolic: `B, S` resolved at load; static per-request otherwise.
- Weight references bind to tensor-table names + atom; the IR never contains weights.
- `attn.*` atoms map 1:1 onto AttnBackend impls (PLAN §4) — the IR is scheduler-facing, not kernel-facing.
- Unknown opcode or arity → loader error listing supported set. `custom_op` runs only if the consuming engine
  was built with that op registered (META declares it; portability flag false).
- New model families = new GRAPH + TENSORS with zero engine code, **if** composed of known atoms.
  Honest boundary: genuinely novel layers need an engine release; the IR shrinks that surface from "model support"
  to "one op addition".

## 6. Deltas — adapters and fine-tune composition

```
DeltaRecord {
  target_digest BLAKE3-128     // base tensor this applies to
  kind: LowRank {A, B, alpha}  // LoRA: rank-r factors as ordinary tensors
      | SparseCsr {values, indices, indptr}
      | Replace {off, len}     // retuned tensor segment
}
```

Load-time composition: resolve target by digest (verify!), apply, produce composite tensor.
A 100 MB LoRA becomes a standalone `.smt` containing only DELTAS+META, valid iff base `content_id` matches —
the fine-tune ecosystem stops shipping 16 GB forks. Stacked deltas compose left-to-right; cycles impossible
(DAG by construction: each delta names its base digest).

## 7. Integrity, provenance, distribution

- `content_id` = BLAKE3 over sorted section digests → the file *is* its own checksum chain.
- Per-tensor digests enable: partial-download verification, cross-checkpoint dedup (same base model,
  different finetunes share unchanged tensors), delta targeting.
- SIG section signs `content_id` (ed25519). License manifest in META is machine-checked
  (feeds PLAN §15 MODELS.md lint).
- Distribution convention: CAS mirrors serve `<blake3hex>.smt`; HTTP range requests fetch INDEX → needed
  sections/components only (text-only inference skips vision tensors; server prefetches expert slots lazily).
- EVAL card travels with weights: `ppl_delta`, `kl_golden`, per-layer sensitivity ranking,
  `recommended_atoms`, converter version — consumers can reject sloppy quantizations mechanically.

## 8. Loader algorithm (reference)

```
open → read superheader (verify magic/version/content_id if SIG)
read INDEX → decide relevance: GRAPH atoms ⊆ supported? hw_profile match?
mmap TENSORS
for each needed tensor:
    pick materialization(hw_profile) else schedule canonical→recipe repack (background threads)
    view = mmap slice (zero-copy)
build expert slot table → hand TopologyMap its resident sets (PLAN §8/§11)
compile missing atom kernels via NVRTC (PTX cache hit expected)
instantiate graph → EngineHandle ready
```

Cold-start target: config+graph parse <50 ms; first token gated only on weight-touch latency, never on
full-file scan (sections are independent; nothing forces whole-file reads).

## 9. Versioning & evolution

- `version` bumps only for superheader/INDEX semantics changes; everything else evolves via new section types,
  new atom coordinates, new opcodes — all ignorable-or-errorable by old readers per rules above.
- Registries (hw_profiles, opcodes, ValDtypes) are append-only, spec-controlled, mirrored into
  `smelt-dtype` constants — one truth, property-tested (PLAN §3 rule 2).

## 10. Rejected ideas (and why)

| Idea | Why rejected |
|---|---|
| Embed compiled cubins/PTX | Engine concern, not model concern; PTX cache rides with the engine (D2) |
| Store KV snapshots / conversation state | Runtime state ≠ weights; `state_schema` in META covers shape contracts only |
| Full ONNX-compatible IR | Expressiveness trap; closed-atom registry stays auditable and schedulable |
| Encrypted sections | DRM is hostile to the openness that makes weight distribution work; licensing = manifest+signature |
| General autodiff/training graph | Non-goal forever (PLAN §1) |

## 11. Open questions

1. META encoding: JSON (debuggable, big) vs CBOR (compact) — lean JSON until size hurts.
2. String interning for tensor names (10⁵–10⁶ names on MoE monsters) — defer, measure first.
3. Should GRAPH carry sampler/chat defaults or stay compute-pure? Lean compute-pure.
4. Compressed sections (zstd) for network transport vs leaving it to CAS mirrors — defer decision to M1 data.

## 12. Relationship to PLAN.md

Supersedes the §7 `.smt` bullet's sketch with a concrete design; preserves its stated properties
(one file, embedded tokenizer/config/quant-manifest/provenance) and adds: numeric-atom space (strengthens D9's
"transport ≠ execution" into a mechanism), execution materializations (strengthens `smelt-layout`'s `Tensor<T,L>`
boundary), expert slots (serves §11 directly), graph IR (new capability, feeds §6 onboarding contract:
config+graph+mapping+fixture becomes config-in-graph+mapping+fixture), deltas (new ecosystem capability).
