# Format-evolution patterns (MLIR/protobuf/ONNX/eBPF/Cap'nProto/WASM)

| | |
|---|---|
| Source | Cross-system survey — see docs/research/extensibility-mechanisms.md for synthesis |
| Link | https://mlir.llvm.org/docs/BytecodeFormat/ ; https://protobuf.dev/programming-guides/encoding/ ; https://capnproto.org/language.html ; https://github.com/onnx/onnx/blob/main/docs/Versioning.md |
| Added | 2026-08-23 |
| Tags | #extensibility #format-design |

## Summary
- Five mechanisms recur across every surviving extensible system: unknown-region preservation; self-describing payloads; versioned registries+adapters; capability negotiation with tiered fallback; independently distributed extension packs.
- LLVM bitcode is the counter-example: forward-compat refused because frozen formats freeze evolution.
- Cap'n Proto codifies safe-change whitelist -> mechanically lintable evolution rules.

## Relevance to SMELT
- SMT v2 adopts all five mechanisms; v1 closed registries demoted to fast-path cache.
