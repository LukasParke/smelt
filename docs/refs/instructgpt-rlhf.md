# Training LMs to follow instructions with human feedback (InstructGPT)

| | |
|---|---|
| Source | Ouyang et al., OpenAI — 2022 |
| Link | https://arxiv.org/abs/2203.02155 |
| Added | 2026-08-23 |
| Tags | #post-training #alignment |

## Summary
- SFT on demonstrations -> RLHF (reward model + PPO) turns text completers into instruction-following assistants.
- Alignment recipe behind ChatGPT-class products.

## Relevance to SMELT
- Chat-completions surface semantics (PLAN §15) assume post-trained behavior: reasoning_content separation, tool-call parsing.
