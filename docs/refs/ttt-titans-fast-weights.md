# Test-time training layers & learned memory (TTT / Titans / fast weights)

| | |
|---|---|
| Source | Sun et al. 2024; Behrouz et al. 2024; Schmidhuber 1991 -> Ba 2016 |
| Link | https://arxiv.org/abs/2407.04620 ; https://arxiv.org/abs/2501.00663 ; https://arxiv.org/abs/1610.06258 |
| Added | 2026-08-23 |
| Tags | #plasticity #state #architecture |

## Summary
- Hidden state as a small neural network updated by a LEARNED online rule (TTT-Linear/MLP); Titans adds surprise-based memory + forgetting gate.
- Fast-weight lineage: weights-as-state updated at inference time since 1991.
- Engineering implication: state shape + update-rule reference belong in the model contract.

## Relevance to SMELT
- Motivates SMT v2 state_schema + state_read/state_write/delta_update ops; plasticity becomes a storage contract, not engine code.
