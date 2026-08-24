# Perf Lab — experiments, methodology, verified claims

Code: `experiments/perf-lab/` (std-only Rust, no deps beyond `parking_lot`). Reproduce:

```bash
cd experiments/perf-lab
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/bw       > ../../docs/experiments/results/bw.log
./target/release/roofline > ../../docs/experiments/results/roofline.log
./target/release/atoms    > ../../docs/experiments/results/atoms.log
./target/release/living   > ../../docs/experiments/results/living.log
```

Machine: Ryzen 9 7950X3D, DDR5-4800 dual-channel, 64 GB, CachyOS 7.1.5. Logs are committed evidence;
each carries a UTC run stamp.

## E1 — Memory hierarchy (bw.log)

| Level | Result |
|---|---|
| L1 32 KB 1t | 97.8 GB/s |
| L2 1 MB 1t | 93.2 GB/s |
| L3 96 MB MT | 643.3 GB/s (cached regime) |
| DRAM 4 GiB MT read | **50.7 GB/s** |
| Stream kernel R+W | 30.3 GB/s |

Methodology note: all passes timed inside one thread-scope so spawn cost is amortized; an earlier
per-pass-timed variant produced garbage for small buffers (fixed in commit history).

## E2 — Arithmetic-intensity sweep (roofline.log)

W=4096² fp32 (64 MB), naive-but-unrolled kernels, 32 threads:

| B | GB/s | GFLOP/s |
|---|---|---|
| 1 | **52.6** | 26.3 |
| 8 | 8.5 | 33.7 |
| 64 | 2.2 | 67.2 |

Findings: (a) B=1 saturates DRAM bandwidth with plain code — decode bandwidth-boundness is structural,
not a kernel-quality artifact; (b) absolute GFLOP/s at B≥8 is *understated by loop order*: this naive
kernel re-streams W once per batch row (traffic ∝ B·M·K), which is precisely why real engines block and
cache — the format layer must not constrain layout freedom (SMT v2 §7).

## E3 — NumericAtom space validation (atoms.log)

N=2²⁰ per distribution, 5 distributions × 4 encodings, rms-relative-error + cosine:

| Distribution | bf16 ref | q4_sym b32 | mxfp4 e8m0 | two-level sub16 |
|---|---|---|---|---|
| gauss01 | .00166 | .0971 | .1150 | **.0853** |
| laplace | .00166 | .1207 | .1268 | **.1010** |
| outlier mix | .00165 | .1436 | .1672 | **.1039** |
| uniform ±3 | .00170 | .0682 | .1102 | **.0652** |
| gauss×10⁻³ | .00166 | .0970 | .1144 | **.0852** |

Findings: (a) strict ordering two-level < q4_sym < mxfp4 on gaussian data; (b) **scale-invariance**
confirmed — errors identical at 10⁻³ amplitude (E8M0/F16 scale trees adapt); (c) pre-registered
expectation FALSIFIED on outliers: flat per-32 E8M0 scales handle rare huge elements *worst* (.167)
because one outlier destroys resolution for its 31 block-mates; per-16 sub-scales isolate them (.104).
Design consequence: default atom recommendation for outlier-heavy checkpoints = sub-scale trees; matches
SmoothQuant/QuaRot literature direction. (d) internal consistency: cos ≈ 1/√(1+rel²) holds to ~5 decimals
across all rows (adversarial check).

Scope limit: three hand-written decoders were measured; the unified single-skeleton decoder is design-stage (v2 §8).

## E4 — Living-engine demo (living.log)

12 s serving run, D=2048 batch=128 rank-k predictor fitting a rank-4 target; mutations injected live:

| t | Event | Critical section | Effect |
|---|---|---|---|
| 3s | grow expert pool +3 | **231 µs** | capacity added, steps continue (rate dips as work grows 1→4 experts: 936→~270 steps/s) |
| 6s | hot kernel swap scalar→blocked | **0.02 µs** | mechanism proven; throughput delta not measurable (most FLOPs ungated) — scoped honestly |
| 9s | weight delta (exact components) | **21 µs** | loss curve bends mid-serving: rel-MSE 0.975 → 0.567 by t=12s |

5383 steps, zero restarts, zero dropped steps. Step-latency tail spikes to 35 ms observed (single
samples, OS/allocator jitter) ⇒ mutation cost is measured as critical-section time, never step time.

## Adversarial verification

5 published claims × 3 independent skeptic agents (numerical-checker / scope-hunter / methods-reproducer),
default-refute stance, against committed logs:

| Claim | Verdict |
|---|---|
| bw-doctrine (50.7 GB/s confirms §13 figure) | ✅ 3/3 sustained |
| gemv-bound (B=1 bandwidth-bound structurally) | ✅ 3/3 sustained |
| atom-space (ordering + falsified-outlier-expectation) | ⚠️ 1/3 — numbers confirmed unanimously; refutations target phrasing "one decoder skeleton" (design-stage). Corrected wording adopted in v2 spec §8. |
| living-mutation (bounded pauses + delta bend + scoped kernel claim) | ✅ 2/3 sustained |
| research-consensus (five mechanisms) | ✅ 2/3 sustained with correction: collective taxonomy across systems (subsets each), reworded in research doc |
