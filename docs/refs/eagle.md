# EAGLE / EAGLE-2 / EAGLE-3

| | |
|---|---|
| Source | Li et al. — 2024..2025 |
| Link | https://github.com/SafeAILab/EAGLE |
| Added | 2026-08-23 |
| Tags | #spec |

## Summary
- Drafts in feature space one step ahead of target's top layer; EAGLE-2 dynamic trees, EAGLE-3 multi-layer feature fusion + training-inference mismatch reduction.
- Needs TARGET hidden features mid-forward — the reason ForwardOut returns (logits, final_hidden).

## Relevance to SMELT
- D5 forward contract exists for Eagle3Drafter (PLAN §0/§4); tree-verify kernel gate >=1.8x @ B=1 (PLAN §12).
