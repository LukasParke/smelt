# Live adapter serving: S-LoRA / Punica / dLoRA / ServerlessLLM

| | |
|---|---|
| Source | UC Berkeley; MSR etc. — 2023..2024 |
| Link | https://arxiv.org/abs/2311.03285 ; https://arxiv.org/pdf/2310.18547 ; https://www.usenix.org/conference/osdi24/presentation/wu-bingyang ; https://www.usenix.org/conference/osdi24/presentation/fu |
| Added | 2026-08-23 |
| Tags | #serving #delta #lora #production |

## Summary
- S-LoRA: unified paging/batching over thousands of adapters.
- Punica: kernel-level multi-tenant LoRA GEMV (SGMV/BGMV) - delta composition inside the kernel, not a separate pass.
- dLoRA (OSDI'24): dynamic peer/fusion selection + request migration during serving.
- ServerlessLLM (OSDI'24): checkpoint-placement locality cuts warm-start to sub-second class.
- RL-rollout infra (OpenRLHF x vLLM, veRL/HybridFlow 2409.19256) pushes new policy weights into running inference workers every training step - weights-changing-while-serving is routine production infrastructure.

## Relevance to SMELT
- Validates SMT v2 living-weights substrate: CAS base + delta streams + hot-swap is deployed reality, not speculation.
