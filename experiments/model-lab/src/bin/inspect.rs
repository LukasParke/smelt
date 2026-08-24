use model_lab::gpt2::Engine;
fn main() {
    for p in ["assets/gpt2-q8.smt", "assets/gpt2-f16.smt"] {
        let e = Engine::load(p);
        println!("{p}: n_embd={} n_layer={} vocab={}", e.meta.n_embd, e.meta.n_layer, e.meta.vocab);
        for n in ["wte.weight", "wpe.weight", "h.0.attn.c_attn.weight", "h.0.ln_1.weight"] {
            let r = &e.t[n];
            println!("  {n}: shape={:?} atom={} len={}", r.shape, r.atom, r.len);
        }
    }
}
