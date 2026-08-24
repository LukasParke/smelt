//! bwparity — Agent B validation gate for the GPU backward kernels (v2 contract,
//! "GPU BACKWARD"). T=4 real-model micro-sequence; computes the trunk input
//! gradient two ways:
//!   (1) GPU chain via the "bwd" kernels: dlogits -> head (f32-uploaded wte) ->
//!       ln_f -> all decoder layers -> embedding output;
//!   (2) CPU central finite differences of the CPU Engine forward wrt the SAME
//!       embedding-row entries (32 sampled coordinates, h=1e-2 relative).
//! Reports max rel diff (<5e-2 target) and PASS/FAIL.
//! If the GPU is unavailable (exclusive mode), falls back to the always-compiled
//! Rust mirror backend, runs every kernel FD self-check through it, and marks the
//! GPU run as skipped. Deterministic seeds throughout.
const T: usize = 4;

fn tlen() -> usize {
    std::env::var("SMT_T").ok().and_then(|v| v.parse().ok()).unwrap_or(T)
}
use model_lab::gpt2::{argmax, empty_kv, Engine};
use model_lab::gpu_backward::{
    fwd_gelu_scalar, trunk_input_grad, Backend, BwdOps, KernelCheck, MirrorOps, Rng, TrunkActs,
    TrunkWeights, WhichKernel,
};


const N_COORDS: usize = 32;
const REL_DELTA: f32 = 1e-2;

// ---------------------------------------------------------------------------
// Local f32 weight copies + threaded local gemv (needed so finite differences
// can mutate a wte coordinate without touching the pack-backed Engine).
// ---------------------------------------------------------------------------

struct LayerW {
    g1: Vec<f32>,
    b1: Vec<f32>,
    w_cattn: Vec<f32>, // [3E, E]
    b_cattn: Vec<f32>,
    w_aproj: Vec<f32>, // [E, E]
    b_aproj: Vec<f32>,
    g2: Vec<f32>,
    b2: Vec<f32>,
    w_fc: Vec<f32>, // [F, E]
    b_fc: Vec<f32>,
    w_mproj: Vec<f32>, // [E, F]
    b_mproj: Vec<f32>,
    fc_rows: usize,
}

struct LocalWeights {
    e: usize,
    wte: Vec<f32>, // [V, E]; FD perturbs single coordinates in place (with restore)
    wpe_rows: Vec<Vec<f32>>,
    layers: Vec<LayerW>,
    wf: Vec<f32>,
    bf: Vec<f32>,
}

impl LocalWeights {
    fn from_engine(eng: &Engine) -> Self {
        let m = &eng.meta;
        let mut layers = Vec::with_capacity(m.n_layer);
        for l in 0..m.n_layer {
            let fc_name = format!("h.{l}.mlp.c_fc.weight");
            let fc_rows = eng.t[&fc_name].shape[0];
            layers.push(LayerW {
                g1: eng.vec_f32(&format!("h.{l}.ln_1.weight")),
                b1: eng.vec_f32(&format!("h.{l}.ln_1.bias")),
                w_cattn: eng.vec_f32(&format!("h.{l}.attn.c_attn.weight")),
                b_cattn: eng.vec_f32(&format!("h.{l}.attn.c_attn.bias")),
                w_aproj: eng.vec_f32(&format!("h.{l}.attn.c_proj.weight")),
                b_aproj: eng.vec_f32(&format!("h.{l}.attn.c_proj.bias")),
                g2: eng.vec_f32(&format!("h.{l}.ln_2.weight")),
                b2: eng.vec_f32(&format!("h.{l}.ln_2.bias")),
                w_fc: eng.vec_f32(&fc_name),
                b_fc: eng.vec_f32(&format!("h.{l}.mlp.c_fc.bias")),
                w_mproj: eng.vec_f32(&format!("h.{l}.mlp.c_proj.weight")),
                b_mproj: eng.vec_f32(&format!("h.{l}.mlp.c_proj.bias")),
                fc_rows,
            });
        }
        let wpe_rows = (0..tlen()).map(|p| eng.vec_row("wpe.weight", p as u32)).collect();
        LocalWeights {
            e: m.n_embd,
            wte: eng.vec_f32("wte.weight"),
            wpe_rows,
            layers,
            wf: eng.vec_f32("ln_f.weight"),
            bf: eng.vec_f32("ln_f.bias"),
        }
    }
}

/// Threaded row-major gemv on local slices (same decomposition pattern as Engine::matmul).
fn gemv_local(w: &[f32], x: &[f32], out: &mut [f32]) {
    let cols = x.len();
    let rows = out.len();
    debug_assert_eq!(w.len(), rows * cols);
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let threads = 8.min(rows);
    if threads <= 1 {
        for r in 0..rows {
            let mut acc = 0f32;
            for j in 0..cols {
                acc += w[r * cols + j] * x[j];
            }
            out[r] = acc;
        }
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        let mut oi = out.chunks_mut(per);
        let mut th = 0usize;
        while let Some(oc) = oi.next() {
            let lo = th * per;
            th += 1;
            handles.push(s.spawn(move || {
                for (ri, r) in (lo..lo + oc.len()).enumerate() {
                    let row = &w[r * cols..r * cols + cols];
                    let mut acc = 0f32;
                    for j in 0..cols {
                        acc += row[j] * x[j];
                    }
                    oc[ri] = acc;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
}

/// f64 end-to-end forward (same math as forward_local) so central differences
/// measure the true derivative instead of f32 rounding noise.
struct LocalWeightsF64 {
    e: usize,
    wte: Vec<f64>,
    wpe_rows: Vec<Vec<f64>>,
    g1: Vec<Vec<f64>>, b1: Vec<Vec<f64>>,
    w_cattn: Vec<Vec<f64>>, b_cattn: Vec<Vec<f64>>,
    w_aproj: Vec<Vec<f64>>, b_aproj: Vec<Vec<f64>>,
    g2: Vec<Vec<f64>>, b2: Vec<Vec<f64>>,
    w_fc: Vec<Vec<f64>>, b_fc: Vec<Vec<f64>>,
    w_mproj: Vec<Vec<f64>>, b_mproj: Vec<Vec<f64>>,
    fc_rows: usize,
    wf: Vec<f64>, bf: Vec<f64>,
}

impl LocalWeightsF64 {
    fn from_engine(eng: &Engine) -> Self {
        let m = &eng.meta;
        let mut g1 = Vec::new(); let mut b1 = Vec::new();
        let mut w_cattn = Vec::new(); let mut b_cattn = Vec::new();
        let mut w_aproj = Vec::new(); let mut b_aproj = Vec::new();
        let mut g2 = Vec::new(); let mut b2 = Vec::new();
        let mut w_fc = Vec::new(); let mut b_fc = Vec::new();
        let mut w_mproj = Vec::new(); let mut b_mproj = Vec::new();
        let fc_rows = eng.t["h.0.mlp.c_fc.weight"].shape[0];
        for l in 0..m.n_layer {
            g1.push(eng.vec_f32(&format!("h.{l}.ln_1.weight")).iter().map(|&v| v as f64).collect());
            b1.push(eng.vec_f32(&format!("h.{l}.ln_1.bias")).iter().map(|&v| v as f64).collect());
            w_cattn.push(eng.vec_f32(&format!("h.{l}.attn.c_attn.weight")).iter().map(|&v| v as f64).collect());
            b_cattn.push(eng.vec_f32(&format!("h.{l}.attn.c_attn.bias")).iter().map(|&v| v as f64).collect());
            w_aproj.push(eng.vec_f32(&format!("h.{l}.attn.c_proj.weight")).iter().map(|&v| v as f64).collect());
            b_aproj.push(eng.vec_f32(&format!("h.{l}.attn.c_proj.bias")).iter().map(|&v| v as f64).collect());
            g2.push(eng.vec_f32(&format!("h.{l}.ln_2.weight")).iter().map(|&v| v as f64).collect());
            b2.push(eng.vec_f32(&format!("h.{l}.ln_2.bias")).iter().map(|&v| v as f64).collect());
            w_fc.push(eng.vec_f32(&format!("h.{l}.mlp.c_fc.weight")).iter().map(|&v| v as f64).collect());
            b_fc.push(eng.vec_f32(&format!("h.{l}.mlp.c_fc.bias")).iter().map(|&v| v as f64).collect());
            w_mproj.push(eng.vec_f32(&format!("h.{l}.mlp.c_proj.weight")).iter().map(|&v| v as f64).collect());
            b_mproj.push(eng.vec_f32(&format!("h.{l}.mlp.c_proj.bias")).iter().map(|&v| v as f64).collect());
        }
        LocalWeightsF64 {
            e: m.n_embd,
            wte: eng.vec_f32("wte.weight").iter().map(|&v| v as f64).collect(),
            wpe_rows: (0..tlen()).map(|p| eng.vec_row("wpe.weight", p as u32).iter().map(|&v| v as f64).collect()).collect(),
            g1, b1, w_cattn, b_cattn, w_aproj, b_aproj, g2, b2, w_fc, b_fc, w_mproj, b_mproj,
            fc_rows,
            wf: eng.vec_f32("ln_f.weight").iter().map(|&v| v as f64).collect(),
            bf: eng.vec_f32("ln_f.bias").iter().map(|&v| v as f64).collect(),
        }
    }
}

fn gemv_f64(w: &[f64], x: &[f64], out: &mut [f64]) {
    let cols = x.len();
    let rows = out.len();
    for v in out.iter_mut() { *v = 0.0; }
    let threads = 8.min(rows);
    if threads <= 1 {
        for r in 0..rows {
            let mut acc = 0f64;
            for j in 0..cols { acc += w[r * cols + j] * x[j]; }
            out[r] = acc;
        }
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        let mut oi = out.chunks_mut(per);
        let mut th = 0usize;
        while let Some(oc) = oi.next() {
            let lo = th * per;
            th += 1;
            handles.push(s.spawn(move || {
                for (ri, r) in (lo..lo + oc.len()).enumerate() {
                    let row = &w[r * cols..r * cols + cols];
                    let mut acc = 0f64;
                    for j in 0..cols { acc += row[j] * x[j]; }
                    oc[ri] = acc;
                }
            }));
        }
        for hh in handles { hh.join().unwrap(); }
    });
}

fn layernorm_f64(x: &[f64], w: &[f64], b: &[f64], eps: f64) -> Vec<f64> {
    let d = x.len();
    let mean = x.iter().sum::<f64>() / d as f64;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / d as f64;
    (0..d).map(|i| (x[i] - mean) / (var + eps).sqrt() * w[i] + b[i]).collect()
}

fn gelu_f64(v: f64) -> f64 {
    0.5 * v * (1.0 + (0.7978845608028654 * (v + 0.044715 * v * v * v)).tanh())
}

/// Full-sequence causal forward in f64; returns per-position logits.
fn forward_local_f64(lw: &LocalWeightsF64, eps: f64, n_head: usize, ids: &[u32]) -> Vec<Vec<f64>> {
    let e = lw.e;
    let hd = e / n_head;
    let t_len = ids.len();
    let mut x: Vec<Vec<f64>> = (0..t_len)
        .map(|p| {
            let base = ids[p] as usize * e;
            let mut v = lw.wte[base..base + e].to_vec();
            for i in 0..e { v[i] += lw.wpe_rows[p][i]; }
            v
        })
        .collect();
    let scale = 1.0 / (hd as f64).sqrt();
    for l in 0..lw.g1.len() {
        let mut qkv_all: Vec<Vec<f64>> = Vec::with_capacity(t_len);
        for p in 0..t_len {
            let h1 = layernorm_f64(&x[p], &lw.g1[l], &lw.b1[l], eps);
            let inter3 = lw.b_cattn[l].len();
            let mut qkv = vec![0f64; inter3];
            gemv_f64(&lw.w_cattn[l], &h1, &mut qkv);
            for i in 0..inter3 { qkv[i] += lw.b_cattn[l][i]; }
            qkv_all.push(qkv);
        }
        for p in 0..t_len {
            let mut ao = vec![0f64; e];
            for head in 0..n_head {
                let o = head * hd;
                let q = &qkv_all[p][o..o + hd];
                let mut scores = vec![0f64; p + 1];
                let mut mx = f64::MIN;
                for j in 0..=p {
                    let kj = &qkv_all[j][e + o..e + o + hd];
                    let mut d = 0f64;
                    for tt in 0..hd { d += q[tt] * kj[tt]; }
                    scores[j] = d * scale;
                    mx = mx.max(scores[j]);
                }
                let mut sum = 0f64;
                for s in scores.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
                for s in scores.iter_mut() { *s /= sum; }
                for j in 0..=p {
                    let vj = &qkv_all[j][2 * e + o..2 * e + o + hd];
                    for tt in 0..hd { ao[o + tt] += scores[j] * vj[tt]; }
                }
            }
            let mut proj = vec![0f64; e];
            gemv_f64(&lw.w_aproj[l], &ao, &mut proj);
            for i in 0..e { x[p][i] += proj[i] + lw.b_aproj[l][i]; }
        }
        for p in 0..t_len {
            let h2 = layernorm_f64(&x[p], &lw.g2[l], &lw.b2[l], eps);
            let f_rows = lw.fc_rows;
            let mut fc = vec![0f64; f_rows];
            gemv_f64(&lw.w_fc[l], &h2, &mut fc);
            for i in 0..f_rows { fc[i] += lw.b_fc[l][i]; }
            for v in fc.iter_mut() { *v = gelu_f64(*v); }
            let mut mo = vec![0f64; e];
            gemv_f64(&lw.w_mproj[l], &fc, &mut mo);
            for i in 0..e { x[p][i] += mo[i] + lw.b_mproj[l][i]; }
        }
    }
    (0..t_len)
        .map(|p| {
            let xf = layernorm_f64(&x[p], &lw.wf, &lw.bf, eps);
            let v = lw.wte.len() / e;
            let mut logits = vec![0f64; v];
            gemv_f64(&lw.wte, &xf, &mut logits);
            logits
        })
        .collect()
}

/// Identical formula to Engine::layernorm.
fn layernorm_local(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let d = x.len();
    let mean = x.iter().sum::<f32>() / d as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
    for i in 0..d {
        x[i] = (x[i] - mean) / (var + eps).sqrt() * w[i] + b[i];
    }
}

/// Full-sequence causal forward over fully-local weights; returns per-position
/// logits. Same op order/formulas as Engine::step.
fn forward_local(lw: &LocalWeights, eps: f32, n_head: usize, ids: &[u32]) -> Vec<Vec<f32>> {
    let e = lw.e;
    let hd = e / n_head;
    let t_len = ids.len();

    let mut x: Vec<Vec<f32>> = (0..t_len)
        .map(|p| {
            let base = ids[p] as usize * e;
            let mut v = lw.wte[base..base + e].to_vec();
            for i in 0..e {
                v[i] += lw.wpe_rows[p][i];
            }
            v
        })
        .collect();

    let mut qkv_all: Vec<Vec<f32>> = Vec::with_capacity(t_len);
    let scale = 1.0 / (hd as f32).sqrt();

    for layer in &lw.layers {
        // ---- attention ----
        qkv_all.clear();
        for p in 0..t_len {
            let mut h1 = x[p].clone();
            layernorm_local(&mut h1, &layer.g1, &layer.b1, eps);
            let inter3 = layer.b_cattn.len();
            let mut qkv = vec![0f32; inter3];
            gemv_local(&layer.w_cattn, &h1, &mut qkv);
            for i in 0..inter3 {
                qkv[i] += layer.b_cattn[i];
            }
            qkv_all.push(qkv);
        }
        for p in 0..t_len {
            let mut ao = vec![0f32; e];
            for head in 0..n_head {
                let o = head * hd;
                let q = &qkv_all[p][o..o + hd];
                let mut scores = vec![0f32; p + 1];
                let mut mx = f32::MIN;
                for j in 0..=p {
                    let kj = &qkv_all[j][e + o..e + o + hd];
                    let mut d = 0f32;
                    for tt in 0..hd {
                        d += q[tt] * kj[tt];
                    }
                    scores[j] = d * scale;
                    mx = mx.max(scores[j]);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }
                for j in 0..=p {
                    let vj = &qkv_all[j][2 * e + o..2 * e + o + hd];
                    for tt in 0..hd {
                        ao[o + tt] += scores[j] * vj[tt];
                    }
                }
            }
            let mut proj = vec![0f32; e];
            gemv_local(&layer.w_aproj, &ao, &mut proj);
            for i in 0..e {
                x[p][i] += proj[i] + layer.b_aproj[i];
            }
        }

        // ---- mlp ----
        for p in 0..t_len {
            let mut h2 = x[p].clone();
            layernorm_local(&mut h2, &layer.g2, &layer.b2, eps);
            let f_rows = layer.fc_rows;
            let mut fc = vec![0f32; f_rows];
            gemv_local(&layer.w_fc, &h2, &mut fc);
            for i in 0..f_rows {
                fc[i] += layer.b_fc[i];
            }
            for v in fc.iter_mut() {
                *v = fwd_gelu_scalar(*v);
            }
            let mut mo = vec![0f32; e];
            gemv_local(&layer.w_mproj, &fc, &mut mo);
            for i in 0..e {
                x[p][i] += mo[i] + layer.b_mproj[i];
            }
        }
    }

    (0..t_len)
        .map(|p| {
            let mut xf = x[p].clone();
            layernorm_local(&mut xf, &lw.wf, &lw.bf, eps);
            let v = lw.wte.len() / e;
            let mut logits = vec![0f32; v];
            gemv_local(&lw.wte, &xf, &mut logits);
            logits
        })
        .collect()
}

/// Linear readout loss L = Σ_p Σ_v dlogits[p][v] * logits[p][v].
fn loss_of(dlogits: &[Vec<f32>], logits: &[Vec<f32>]) -> f64 {
    let mut l = 0f64;
    for p in 0..logits.len() {
        for v in 0..logits[p].len() {
            l += (dlogits[p][v] * logits[p][v]) as f64;
        }
    }
    l
}

fn loss_of_f64(dlogits: &[Vec<f32>], logits: &[Vec<f64>]) -> f64 {
    let mut l = 0f64;
    for p in 0..logits.len() {
        for v in 0..logits[p].len() {
            l += dlogits[p][v] as f64 * logits[p][v];
        }
    }
    l
}

// ---------------------------------------------------------------------------
// CPU forward WITH stash — uses Engine primitives (matmul/layernorm/vec_*) so the
// stashed activations are bit-identical to Engine::step numerics.
// ---------------------------------------------------------------------------

fn forward_stash(eng: &Engine, ids: &[u32]) -> (Vec<Vec<f32>>, TrunkActs) {
    let m = &eng.meta;
    let t_len = ids.len();
    let e = m.n_embd;
    let hd = e / m.n_head;
    let mut acts = TrunkActs::default();

    let mut x: Vec<Vec<f32>> = (0..t_len)
        .map(|p| {
            let mut v = eng.vec_row("wte.weight", ids[p]);
            let wpe = eng.vec_row("wpe.weight", p as u32);
            for i in 0..e {
                v[i] += wpe[i];
            }
            v
        })
        .collect();

    for l in 0..m.n_layer {
        let g1 = eng.vec_f32(&format!("h.{l}.ln_1.weight"));
        let b1 = eng.vec_f32(&format!("h.{l}.ln_1.bias"));
        let cattn = format!("h.{l}.attn.c_attn.weight");
        let aproj = format!("h.{l}.attn.c_proj.weight");
        let g2 = eng.vec_f32(&format!("h.{l}.ln_2.weight"));
        let b2 = eng.vec_f32(&format!("h.{l}.ln_2.bias"));
        let fc_name = format!("h.{l}.mlp.c_fc.weight");
        let f_rows = eng.t[&fc_name].shape[0];

        let mut x_in_l = vec![0f32; t_len * e];
        let mut x_mid_l = vec![0f32; t_len * e];
        let mut qkv_l = vec![0f32; t_len * 3 * e];
        let mut attn_out_l = vec![0f32; t_len * e];

        let scale = 1.0 / (hd as f32).sqrt();
        for p in 0..t_len {
            x_in_l[p * e..(p + 1) * e].copy_from_slice(&x[p]);
            let mut h1 = x[p].clone();
            eng.layernorm(&mut h1, &g1, &b1, m.ln_eps);
            let inter3 = eng.t[&cattn].shape[0];
            let mut qkv = vec![0f32; inter3];
            eng.matmul(&cattn, &h1, &mut qkv);
            let qb = eng.vec_f32(&format!("h.{l}.attn.c_attn.bias"));
            for i in 0..inter3 {
                qkv[i] += qb[i];
            }
            qkv_l[p * 3 * e..(p + 1) * 3 * e].copy_from_slice(&qkv);

            // causal attention over positions 0..=p (identical math to Engine::step)
            for head in 0..m.n_head {
                let o = head * hd;
                let q = &qkv[o..o + hd];
                let mut scores = vec![0f32; p + 1];
                let mut mx = f32::MIN;
                for j in 0..=p {
                    let kj = &qkv_l[j * 3 * e + e + o..j * 3 * e + e + o + hd];
                    let mut d = 0f32;
                    for tt in 0..hd {
                        d += q[tt] * kj[tt];
                    }
                    scores[j] = d * scale;
                    mx = mx.max(scores[j]);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }
                for j in 0..=p {
                    let vj = &qkv_l[j * 3 * e + 2 * e + o..j * 3 * e + 2 * e + o + hd];
                    for tt in 0..hd {
                        attn_out_l[p * e + o + tt] += scores[j] * vj[tt];
                    }
                }
            }

            let mut proj = vec![0f32; e];
            eng.matmul(&aproj, &attn_out_l[p * e..(p + 1) * e], &mut proj);
            let pb = eng.vec_f32(&format!("h.{l}.attn.c_proj.bias"));
            for i in 0..e {
                x[p][i] += proj[i] + pb[i];
            }
            // post-attention residual == pre-LN_2 input
            x_mid_l[p * e..(p + 1) * e].copy_from_slice(&x[p]);
        }

        let mut fc_pre_l = vec![0f32; t_len * f_rows];
        for p in 0..t_len {
            let mut h2 = x[p].clone();
            eng.layernorm(&mut h2, &g2, &b2, m.ln_eps);
            let mut fc = vec![0f32; f_rows];
            eng.matmul(&fc_name, &h2, &mut fc);
            let fb = eng.vec_f32(&format!("h.{l}.mlp.c_fc.bias"));
            for i in 0..f_rows {
                let v = fc[i] + fb[i];
                fc_pre_l[p * f_rows + i] = v; // pre-GELU stash
                fc[i] = fwd_gelu_scalar(v);
            }
            let mo_name = format!("h.{l}.mlp.c_proj.weight");
            let mut mo = vec![0f32; e];
            eng.matmul(&mo_name, &fc, &mut mo);
            let mb = eng.vec_f32(&format!("h.{l}.mlp.c_proj.bias"));
            for i in 0..e {
                x[p][i] += mo[i] + mb[i];
            }
        }

        acts.x_in.push(x_in_l);
        acts.x_mid.push(x_mid_l);
        acts.qkv.push(qkv_l);
        acts.fc_pre.push(fc_pre_l);
        // Post-LN stashes are part of the frozen activation layout (site taps need
        // them later); the pure input-gradient chain consumes pre-LN inputs only.
        acts.h1.push(vec![0f32; t_len * e]);
        acts.h2.push(vec![0f32; t_len * e]);
    }

    let lnf_x: Vec<f32> = x.iter().flat_map(|v| v.iter().copied()).collect();
    let wf = eng.vec_f32("ln_f.weight");
    let bf = eng.vec_f32("ln_f.bias");
    let mut lnf_out_l = vec![0f32; t_len * e];
    let logits: Vec<Vec<f32>> = (0..t_len)
        .map(|p| {
            let mut xf = x[p].clone();
            eng.layernorm(&mut xf, &wf, &bf, m.ln_eps);
            lnf_out_l[p * e..(p + 1) * e].copy_from_slice(&xf);
            let mut logits = vec![0f32; m.vocab];
            eng.matmul("wte.weight", &xf, &mut logits);
            logits
        })
        .collect();
    acts.lnf_x = lnf_x;
    acts.lnf_out = lnf_out_l;
    (logits, acts)
}

// ---------------------------------------------------------------------------

fn main() {
    let eng = Engine::load("assets/gpt2-q8.smt");
    let m = eng.meta.clone();

    // Deterministic real-model micro-sequence.
    let text = "The capital of France is";
    let mut ids = eng.bpe.encode(text);
    assert!(!ids.is_empty(), "empty tokenization");
    let t_len = T();
    while ids.len() < t_len {
        ids.push(ids[0]);
    }
    ids.truncate(t_len);
    println!("BWP ids={ids:?} text={text:?}");

    // ---- CPU forward with stash, validated against Engine::step ----
    let (stash_logits, acts) = forward_stash(&eng, &ids);
    let mut kv = empty_kv(&eng);
    let mut mad = 0f64;
    let mut agree = 0usize;
    for p in 0..tlen() {
        let step_logits = eng.step(ids[p], p, &mut kv);
        for (a, b) in stash_logits[p].iter().zip(step_logits.iter()) {
            mad = mad.max((a - b).abs() as f64);
        }
        agree += (argmax(&stash_logits[p]) == argmax(&step_logits)) as usize;
    }
    let stash_ok = mad < 2e-2 && agree == T;
    println!(
        "STASH_VS_STEP max_abs_diff={mad:.3e} argmax_agree={agree}/{T} {}",
        if stash_ok { "PASS" } else { "FAIL" }
    );

    // ---- deterministic sparse random dlogits ("onehot-ish"), linear readout loss ----
    let mut rng = Rng::new(20260824);
    let mut dlogits: Vec<Vec<f32>> = vec![vec![0f32; m.vocab]; T];
    for p in 0..tlen() {
        for _ in 0..8 {
            let idx = rng.below(m.vocab);
            dlogits[p][idx] = rng.f32();
        }
    }

    {
            // stash logits for position 0..T were produced by Engine-consistent forward; rebuild
    let stash_ref: Vec<Vec<f32>> = {
        let (lg, _) = forward_stash(&eng, &ids);
        lg
    };
    let lg_local = forward_local_f64(&LocalWeightsF64::from_engine(&eng), m.ln_eps as f64, m.n_head, &ids);
        let mut md = 0f64;
        for p in 0..ids.len() {
            for i in 0..m.vocab {
                md = md.max((lg_local[p][i] - stash_ref[p][i] as f64)).abs();
            }
        }
        println!("LOCALF64_VS_STASH_LOGITS max_abs={md:.6}");
    }
    let tw = TrunkWeights::from_engine(&eng);

    // ---- backend detect + kernel FD self-checks through EVERY available backend ----
    let mut gpu_skipped_reason: Option<String> = None;
    let mut backend: Option<Backend> = match Backend::detect(&eng) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            gpu_skipped_reason = Some(e);
            None
        }
    };

    let mut checks: Vec<KernelCheck> = Vec::new();
    for k in WhichKernel::ALL {
        checks.push(model_lab::gpu_backward::selfcheck_kernel(&mut MirrorOps, k));
    }
    if let Some(b) = backend.as_mut() {
        let ops: &mut dyn BwdOps = b;
        for k in WhichKernel::ALL {
            checks.push(model_lab::gpu_backward::selfcheck_kernel(ops, k));
        }
    }
    for c in &checks {
        println!("{}", c.line());
    }
    let checks_ok = checks.iter().all(|c| c.pass);

    // ---- trunk input gradient through whichever backend we have ----
    let dx = match backend.as_mut() {
        Some(b) => {
            let ops: &mut dyn BwdOps = b;
            println!("BWPARITY_BACKEND={}", ops.name());
            trunk_input_grad(ops, &tw, &ids, &acts, &dlogits)
        }
        None => {
            println!("BWPARITY_BACKEND=cpu-mirror");
            println!(
                "GPU_RUN=SKIPPED reason={}",
                gpu_skipped_reason.as_deref().unwrap_or("unknown")
            );
            let mut mirror = MirrorOps;
            trunk_input_grad(&mut mirror, &tw, &ids, &acts, &dlogits)
        }
    };

    // ---- central finite differences wrt sampled embedding-row entries ----
    let mut lw64 = LocalWeightsF64::from_engine(&eng);
    struct Row {
        tok: u32,
        idx: usize,
        analytic: f64,
        fd: f64,
        rel: f64,
    }
    let mut rows: Vec<Row> = Vec::new();
    for c in 0..N_COORDS {
        let ti = c % T;
        let tok = ids[ti];
        let idx = (c * 97 + 13) % m.n_embd;
        let off = tok as usize * m.n_embd + idx;
        let v0 = lw64.wte[off];
        // relative step with an absolute floor
        let h = REL_DELTA as f64 * v0.abs().max(0.05);

        // perturb in place (single coordinate), evaluate loss, restore
        lw64.wte[off] = v0 + h;
        let lp = loss_of_f64(&dlogits, &forward_local_f64(&lw64, m.ln_eps as f64, m.n_head, &ids));
        lw64.wte[off] = v0 - h;
        let lm = loss_of_f64(&dlogits, &forward_local_f64(&lw64, m.ln_eps as f64, m.n_head, &ids));
        lw64.wte[off] = v0;
        let fd = (lp - lm) / (2.0 * h);

        // analytic: input path through this token's embedding row PLUS the tied
        // head path (dlogits[p][tok] * lnf_out[p][idx], all positions).
        let input_path: f64 = (0..tlen())
            .filter(|&p| ids[p] == tok)
            .map(|p| dx[p][idx] as f64)
            .sum();
        let head_path: f64 = (0..tlen())
            .map(|p| (dlogits[p][tok as usize] * acts.lnf_out[p * m.n_embd + idx]) as f64)
            .sum();
        let analytic = input_path + head_path;
        rows.push(Row { tok, idx, analytic, fd, rel: 0.0 });
    }

    // global-scale denominator across all FD values (mirrors library rel_max policy)
    let fscale = rows.iter().map(|r| r.fd.abs()).fold(0f64, f64::max).max(1e-30) * 1e-3;
    let mut max_rel = 0f64;
    for r in rows.iter_mut() {
        r.rel =
            (r.analytic - r.fd).abs() / r.analytic.abs().max(r.fd.abs()).max(fscale);
        max_rel = max_rel.max(r.rel);
    }
    println!("--- BWPARITY table (analytic vs central-FD, h={REL_DELTA} relative) ---");
    for (c, r) in rows.iter().enumerate() {
        println!(
            "BWP {:02} tok={:<6} idx={:<4} analytic={:+.6e} fd={:+.6e} rel={:.2e}",
            c, r.tok, r.idx, r.analytic, r.fd, r.rel
        );
    }
    let parity_pass = max_rel < 5e-2;
    println!(
        "BWPARITY max_rel={max_rel:.3e} target=<5e-2 {}",
        if parity_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "ARGMAX_AGREE final-logits stash-vs-step {agree}/{T} {}",
        if agree == T { "PASS" } else { "FAIL" }
    );

    let overall = parity_pass && checks_ok && stash_ok && agree == T;
    println!(
        "BWPARITY_OVERALL={} (parity={parity_pass} kernel_checks={checks_ok} stash={stash_ok})",
        if overall { "PASS" } else { "FAIL" }
    );
    if !overall {
        std::process::exit(1);
    }
}
