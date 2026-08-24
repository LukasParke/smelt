# PIM / CXL / HBF / deterministic-SRAM hardware frontier

| | |
|---|---|
| Source | SK hynix, Samsung, UPMEM, JEDEC; Groq/NVIDIA consolidation — 2024..2026 |
| Link | https://news.skhynix.com/en/hbf-at-fms-2026/ ; https://arxiv.org/abs/2604.27808 ; https://arxiv.org/abs/2303.15375 ; https://www.nvidia.com/en-us/data-center/gb200-nvl72/ |
| Added | 2026-08-23 |
| Tags | #hardware #memory-wall |

## Summary
- Memory wall widens on schedule: compute ~3x/2yr vs bandwidth ~1.4x/2yr.
- JEDEC HBF (high-bandwidth flash) standard published Aug 2026 -> expert pools will tier VRAM/HBM/DRAM/HBF late-decade.
- PIM real but small/host-bound; CXL tiering has bounded latency tax; Groq absorbed by NVIDIA Dec 2025 (flexible stacks won).

## Relevance to SMELT
- Confirms software-side attacks (placement/caching/quantization/scheduling) own the 2026-27 window; TopologyMap + slot tables are the ready seam for HBF tiers.
