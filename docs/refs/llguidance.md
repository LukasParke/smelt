# llguidance

| | |
|---|---|
| Source | Microsoft guidance-ai — ongoing |
| Link | https://github.com/guidance-ai/llguidance |
| Added | 2026-08-23 |
| Tags | #constrained #grammar |

## Summary
- Rust-native constrained decoding engine: JSON-schema/regex/context-free grammars compiled to token masks; ~50us/token @ 128k vocab, no startup cost.

## Relevance to SMELT
- Default GrammarConstraint backend from M3 (D10); think-phase gating until </think> (PLAN §14).
