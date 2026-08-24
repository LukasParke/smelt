# Research: Living models — plasticity, live mutation, hardware frontier

Source: blind scout sweep (2026-08-23); citations inline. Feeds: SMT v2 spec §6–7.

## A. Architectures that keep learning after deployment

Adoption tiers for an inference engine on commodity GPUs:

**Adopt now (<12 mo, no new hardware)**
- **Live adapter serving**: S-LoRA ([2311.03285](https://arxiv.org/abs/2311.03285)) unified batching over
  thousands of adapters; Punica SGMV/BGMV kernels ([2310.18547](https://arxiv.org/pdf/2310.18547));
  dLoRA (OSDI'24) dynamic peer/fusion selection with migration costs
  ([usenix.org/osdi24/wu-bingyang](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)).
  vLLM/SGLang ship dynamic LoRA load/unload today.
- **Runtime expert-pool resizing**: FreeToken re-budgets VRAM between expert cache and KV at scheduler
  safe points without restart — the growth primitive exists in production-grade code.
- **Test-time compute as growth**: o-series/R1-style thinking scales capability without touching weights
  ([2501.12948](https://arxiv.org/abs/2501.12948)); already the default serving mode for reasoning models.

**Adopt next (12–24 mo, engineering-heavy)**
- **State-carrying layers as learned memory**: TTT-Linear/TTT-MLP ([2407.04620](https://arxiv.org/abs/2407.04620)),
  Titans ([2501.00663](https://arxiv.org/abs/2501.00663)) — hidden state is a small neural network updated by a
  learned online rule. Format implication: state shapes and update-rule references belong in the model
  contract (SMT `state_schema` + `state_*` ops), not in engine code.
- **Fast-weight lineage**: Schmidhuber 1991 → Ba fast weights ([1610.06258](https://arxiv.org/abs/1610.06258))
  → HyperNetworks; theory mature, kernels emerging.
- **True test-time weight updates**: ARC-TTT style per-task gradient bursts
  ([2411.07279](https://arxiv.org/abs/2411.07279)) — needs pause/retract primitives SMELT already plans (M2).
- **Continual-learning practice**: orthogonal gradients ([1910.07104](https://arxiv.org/abs/1910.07104)),
  A-GEM, replay — policy layer above the substrate; prevents forgetting when deltas stream continuously.

**Needs new hardware**
- On-chip plasticity: Loihi 2 programmable learning rules, Hala Point scale prototypes
  ([2511.01553](https://arxiv.org/abs/2511.01553)) — co-located memory+compute buys orders-of-magnitude
  efficiency software cannot reach; irrelevant to GPU engines this decade.
- Liquid/CFC commercial proof-point: LFM2 ships on phones ([liquid.ai](https://www.liquid.ai/blog/lfm2-24b-a2b)) —
  architecture-level plasticity is commercially viable, i.e., the format must not assume static graphs forever.

## B. Systems mutating served models live (mechanisms to copy first)

Priority order from the survey:
1. **Unified paged pool spanning KV + adapter/delta segments** (S-LoRA's punica-style paging).
2. **Kernel-level multi-adapter GEMV** (Punica) so delta composition isn't a separate pass.
3. **Dynamic fusion policy** (dLoRA): fuse hot adapters into base weights, keep peers cold, migrate under load — measured on vLLM-class stack.
4. **Warm-start locality** (ServerlessLLM, OSDI'24): checkpoint placement for seconds→sub-second boot.
5. **RL rollout resync loops** (OpenRLHF×vLLM, veRL/HybridFlow [2409.19256](https://arxiv.org/abs/2409.19256)): production systems push new policy weights into running inference workers every training step — the existence proof that "weights change while serving" is routine infrastructure.
6. **Content-addressed stores**: HF Xet chunk-dedup default since 2025; OCI artifacts for models — Git-like weight versioning is deployed reality, validating SMT's CAS core.

## C. Hardware frontier 2026–2030 (where bottlenecks move)

- **The wall widens on schedule**: compute ~3×/2yr vs bandwidth ~1.4×/2yr; GB200-class 10 PFLOPS NVFP4
  against 8 TB/s HBM3e ([nvidia.com/gb200-nvl72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/));
  HBM4 volume ramp 2026 ([SK hynix](https://news.skhynix.com/en/sk-hynix-completes-worlds-first-hbm4-development-and-shipping-of-sample/)).
  Decode bandwidth-bound doctrine strengthens, not weakens.
- **New capacity tier**: JEDEC HBF (high-bandwidth flash, stacked NAND) standard published Aug 2026,
  products late-decade ([SK hynix FMS 2026](https://news.skhynix.com/en/hbf-at-fms-2026/)) — expert pools
  will tier VRAM/HBM/DRAM/HBF; TopologyMap seam ready.
- **PIM measured, still niche**: real speedups exist but small and host-bound
  ([2604.27808](https://arxiv.org/abs/2604.27808); NeuPIMs [2403.00579](https://arxiv.org/abs/2403.00579)).
- **CXL tiering**: bounded latency tax buys capacity elasticity ([2303.15375](https://arxiv.org/abs/2303.15375)) — KV/expert host-tier design precedent.
- **Deterministic-SRAM bets consolidated**: Groq absorbed by NVIDIA (Dec 2025); analog chips ship at edge
  (NorthPole research class), not frontier scale — flexibility-preserving software stacks win the window.
- **NVFP4 became a training format**; Rubin doubles down on low-precision — quant-tier seams (SMT atoms)
  must expect new numeric formats every cycle, reinforcing open-core extensibility.

**Net**: through 2027 the attackable bottlenecks are all software — placement, caching, quantization,
scheduling, delta logistics. That is exactly SMT's scope; hardware co-design remains out of scope until
HBF/PIM commodity tiers materialize.
