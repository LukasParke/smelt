# XGrammar-2

| | |
|---|---|
| Source | MLC team — May 2026 |
| Link | https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation |
| Added | 2026-08-23 |
| Tags | #constrained #grammar |

## Summary
- Fast grammar compilation with cross-request substructure reuse (~50% reuse at 50 tools), large compile speedups; wins JSON-heavy agentic loads.
- Process-wide compile cache keyed hash(schema+tokenizer+vocab).

## Relevance to SMELT
- Optional backend behind GrammarConstraint trait at M6; automatic fallback either direction on compile blowup (PLAN §14).
