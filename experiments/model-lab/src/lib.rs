//! model-lab: SMT v2-lite end-to-end proof on real GPT-2 weights.
pub mod adapter;
pub mod atoms;
pub mod bpe;
pub mod format;
pub mod gpu;
pub mod adapter_v2;
pub mod delta_v2;
pub mod gpu_backward;
pub mod tape;
pub mod consolidate;
pub mod gpt2;
