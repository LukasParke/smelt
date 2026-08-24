# Research: How extensible systems absorb unknown futures

Source: blind scout sweep (2026-08-23), full JSON in session artifacts; citations inline.
Feeds: SMT v2 spec §1–5.

## The taxonomy (what survives)

Five mechanisms recur across every production system investigated. No system relies on a closed
registry alone.

| Mechanism | Canonical instance | Citation |
|---|---|---|
| Unknown-region preservation | protobuf skips-and-retains unknown fields; 25 yrs at Google scale | [protobuf.dev encoding](https://protobuf.dev/programming-guides/encoding/) |
| Self-describing payloads | `google.protobuf.Any {type_url, value}`; Cap'n Proto AnyPointer | [protobuf Any](https://protobuf.dev/reference/protobuf/google.protobuf/#any); [capnproto.org/encoding](https://capnproto.org/encoding.html) |
| Versioned registries + adapters | ONNX immutable `(domain, op_type, since_version)` + mandatory upgrade adapters; vendor domains as sanctioned extension route; opset 27 / IR v13 current | [ONNX Versioning.md](https://github.com/onnx/onnx/blob/main/docs/Versioning.md) |
| Per-dialect versioning & upconversion | MLIR bytecode: per-dialect versioning, cross-dialect upconversion, third-party dialects first-class | [MLIR BytecodeFormat](https://mlir.llvm.org/docs/BytecodeFormat/); [DefiningDialects](https://mlir.llvm.org/docs/DefiningDialects/) |
| Safe portable program carriers | eBPF CO-RE relocation against self-describing kernel types; verifier-gated | [Meta BPF blog](https://facebookmicrosites.github.io/bpf/blog/2020/02/19/bpf-portability-and-co-re.html) |
| Semver'd interface packages | WASM Component Model / WIT, WASI 0.2 shipped Feb 2024 | [WIT design](https://raw.githubusercontent.com/WebAssembly/component-model/main/design/mvp/WIT.md) |
| Symbolic shape/dtype IR | TVM Relax (Unity mainline): symbolic shapes, partial lowering | [Relax abstraction](https://tvm.apache.org/docs/deep_dive/relax/abstraction.html) |

## The cautionary tale

LLVM bitcode explicitly refuses IR backward-compatibility ("we don't promise bitcode compatibility")
because frozen formats freeze internal evolution ([DeveloperPolicy](https://llvm.org/docs/DeveloperPolicy.html#ir-backwards-compatibility)).
Lesson for SMT: freeze only the *reference grammar* (TLV framing, Ref structure, ExprTree primitives),
never implementations — and keep everything else skippable.

## Codified evolution rules worth copying

Cap'n Proto publishes an explicit safe-change whitelist (fields numbered consecutively may change
type in bounded ways; never renumber; etc.) — [language.html#evolution](https://capnproto.org/language.html).
SMT v2 §5 adopts the same discipline: patch/minor/major each get a mechanical definition so a CI lint
can enforce them.

## Capability negotiation + tiered fallback, observed

ONNX Runtime's execution-provider model is the cleanest serving-side instance:
`GetCapability()` assigns subgraphs to providers by priority list ending at CPUExecutionProvider —
exactly tiered resolution with a universal floor ([docs](https://onnxruntime.ai/docs/execution-providers/)).
SMT v2 §3 copies the shape: native → pack → JIT → interpreter → declared-skip → named error.
