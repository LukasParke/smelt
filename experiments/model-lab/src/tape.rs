//! CPU exact reverse-mode tape over the full GPT-2 block stack (SMT v2 plasticity, Agent A).
//!
//! Forward mirrors `gpt2::Engine::step` semantics for the base trunk (same kernels:
//! `Engine::layernorm`, `Engine::matmul`, identical attention op order) so logits match the
//! decode path closely. Adapters enter at the two PRE-LINEAR taps per layer per the SITE MODEL:
//!   y = W(x + BAx)      (merged form (W + BA)x — exact)
//! with ROUTED block-diagonal subspaces per `adapter_v2::RouteSpec` (inactive fact slices
//! contribute exactly zero, and their gradients stay exactly zero).
//!
//! Backward is hand-written exact reverse-mode differentiation in f32:
//!   dlogits <- softmax/CE, head input-grad (transposed dequantized wte gemv),
//!   ln_f backward, then per layer (reverse): mlp branch (c_proj^T, gelu-tanh',
//!   c_fc^T, adapter, ln_2 backward), attention branch (c_proj^T, causal-prefix
//!   softmax backward with cross-position dk/dv accumulation, c_attn^T, adapter, ln_1
//!   backward), residuals summed into the incoming residual-stream grad.
//!
//! Weight gradients are NOT produced (frozen base): only per-site (dA, dB) and dlogits.
//! Embedding-row grads are available via [`TapeModel::wte_row_grads`] (used by tapecheck's
//! wte finite-difference gate); the gather path skips them during normal backward.
#![allow(dead_code)]

use crate::adapter_v2::{fact_active, AdapterV2, SiteKind};
use crate::atoms::{f16_to_f32, ATOM_F16, ATOM_Q8};
use crate::gpt2::Engine;
use std::collections::HashMap;

pub const GELU_TANH_C: f32 = 0.7978845608028654;
pub const GELU_TANH_A: f32 = 0.044715;

/// Which tap a site hangs off. Reuses `adapter_v2::SiteKind` so site plans and adapters
/// address each other unambiguously (`AdapterV2::site_index(kind, layer)`).
pub type SiteTap = SiteKind;

/// One adapter attachment point: a tap (kind) on a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SiteSpec {
    pub kind: SiteTap,
    pub layer: usize,
}

impl SiteSpec {
    pub fn attn(layer: usize) -> Self {
        Self { kind: SiteKind::AttnIn, layer }
    }
    pub fn mlp(layer: usize) -> Self {
        Self { kind: SiteKind::MlpIn, layer }
    }
    pub fn name(&self) -> String {
        match self.kind {
            SiteKind::AttnIn => format!("SiteAttn{{{}}}", self.layer),
            SiteKind::MlpIn => format!("SiteMlp{{{}}}", self.layer),
        }
    }
}

/// Default site plan: BOTH kinds on every layer, attn tap before mlp tap, layers ascending.
pub fn default_sites(n_layer: usize) -> Vec<SiteSpec> {
    let mut v = Vec::with_capacity(n_layer * 2);
    for l in 0..n_layer {
        v.push(SiteSpec::attn(l));
        v.push(SiteSpec::mlp(l));
    }
    v
}

/// Gradients for one site, already summed over the sequence (pre-lr).
/// Layouts match `adapter_v2::SiteAdapter` exactly:
///   ga: [r*d] slice-major in route order (each slice row-major [s_i, d])
///   gb: [d*r] column-major (b[col*d + row]); routed col ranges are contiguous blocks.
/// Entries outside the active fact slices are exactly zero.
#[derive(Clone, Debug)]
pub struct SiteGrad {
    pub site_idx: usize,
    pub ga: Vec<f32>,
    pub gb: Vec<f32>,
}

pub struct BackwardOut {
    pub per_site: Vec<SiteGrad>,
    pub dlogits: Vec<Vec<f32>>,
}

/// Dequantized f32 copies of the trunk matrices needed by backward (input-grads only).
/// Forward deliberately does NOT use these — it goes through `Engine::matmul` on the packed
/// payloads so base-trunk logits track `Engine::step` bit-for-bit. The q8/f16 gemv is linear
/// in its input, so dequantized `W^T` is the exact adjoint of the forward kernel.
pub struct LayerWeights {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub cattn_w: Vec<f32>, // [2304, 768]
    pub cattn_b: Vec<f32>,
    pub aproj_w: Vec<f32>, // attn.c_proj [768, 768]
    pub aproj_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub cfc_w: Vec<f32>, // mlp.c_fc [3072, 768]
    pub cfc_b: Vec<f32>,
    pub mproj_w: Vec<f32>, // mlp.c_proj [768, 3072]
    pub mproj_b: Vec<f32>,
    pub cattn_rows: usize,
    pub cfc_rows: usize,
}

pub struct WeightCache {
    pub layers: Vec<LayerWeights>,
    pub ln_f_w: Vec<f32>,
    pub ln_f_b: Vec<f32>,
}

/// Activations saved by forward, consumed by backward. All per-layer buffers are flat
/// `[T * row_len]`, position-major.
pub struct TapeCache {
    pub t: usize,
    pub ids: Vec<u32>,
    pub d: usize,
    pub inter3: usize,
    pub inter: usize,
    /// residual stream entering each layer (post previous residual add)
    pub xin: Vec<Vec<f32>>,
    /// ln_1 outputs (attn tap input)
    pub h1: Vec<Vec<f32>>,
    /// fused qkv (post-bias), [T * 3d]; q at [0..d), k at [d..2d), v at [2d..3d)
    pub qkv: Vec<Vec<f32>>,
    /// residual stream after the attention branch
    pub xmid: Vec<Vec<f32>>,
    /// ln_2 outputs (mlp tap input)
    pub h2: Vec<Vec<f32>>,
    /// mlp.c_fc pre-gelu (bias included), [T * inter]
    pub fc_pre: Vec<Vec<f32>>,
    /// final hidden, pre-ln_f / post-ln_f
    pub xfin: Vec<f32>,
    pub xf: Vec<f32>,
    pub logits: Vec<Vec<f32>>,
}

pub struct TapeModel<'a> {
    pub eng: &'a Engine,
    pub sites: &'a [SiteSpec],
    pub r: usize,
    pub wc: WeightCache,
    /// Lazily built f64 widening of the trunk (see `ce_loss_f64`); materialized only when an
    /// FD gate runs. ~1 GB for gpt2-small.
    pub f64c: std::sync::OnceLock<F64Cache>,
}

impl<'a> TapeModel<'a> {
    /// Builds the model and dequantizes every trunk matrix backward needs (~330 MB f32 for
    /// gpt2-small). Deterministic; no RNG.
    pub fn new(eng: &'a Engine, sites: &'a [SiteSpec], r: usize) -> Self {
        let n = eng.meta.n_layer;
        let mut layers = Vec::with_capacity(n);
        for l in 0..n {
            layers.push(LayerWeights {
                ln1_w: eng.vec_f32(&format!("h.{l}.ln_1.weight")),
                ln1_b: eng.vec_f32(&format!("h.{l}.ln_1.bias")),
                cattn_w: eng.vec_f32(&format!("h.{l}.attn.c_attn.weight")),
                cattn_b: eng.vec_f32(&format!("h.{l}.attn.c_attn.bias")),
                aproj_w: eng.vec_f32(&format!("h.{l}.attn.c_proj.weight")),
                aproj_b: eng.vec_f32(&format!("h.{l}.attn.c_proj.bias")),
                ln2_w: eng.vec_f32(&format!("h.{l}.ln_2.weight")),
                ln2_b: eng.vec_f32(&format!("h.{l}.ln_2.bias")),
                cfc_w: eng.vec_f32(&format!("h.{l}.mlp.c_fc.weight")),
                cfc_b: eng.vec_f32(&format!("h.{l}.mlp.c_fc.bias")),
                mproj_w: eng.vec_f32(&format!("h.{l}.mlp.c_proj.weight")),
                mproj_b: eng.vec_f32(&format!("h.{l}.mlp.c_proj.bias")),
                cattn_rows: eng.t[&format!("h.{l}.attn.c_attn.weight")].shape[0],
                cfc_rows: eng.t[&format!("h.{l}.mlp.c_fc.weight")].shape[0],
            });
        }
        let wc = WeightCache {
            layers,
            ln_f_w: eng.vec_f32("ln_f.weight"),
            ln_f_b: eng.vec_f32("ln_f.bias"),
        };
        Self { eng, sites, r, wc, f64c: std::sync::OnceLock::new() }
    }

    fn adapter_idx(&self, ad: &AdapterV2, s: &SiteSpec) -> Option<usize> {
        ad.site_index(s.kind, s.layer)
    }

    /// Forward over the whole sequence (causal-prefix attention, position p attends 0..=p).
    /// `active` selects which fact slices of each adapter participate.
    pub fn forward(
        &self,
        ids: &[u32],
        ad: &AdapterV2,
        active: &[u64],
    ) -> (Vec<Vec<f32>>, TapeCache) {
        self.forward_with_overrides(ids, ad, active, &HashMap::new())
    }

    /// Forward with optional overrides of dequantized wte rows (token id -> row). Used by
    /// tapecheck to finite-difference embedding rows without mutating the mmapped pack.
    /// Empty map => identical to `forward`.
    pub fn forward_with_overrides(
        &self,
        ids: &[u32],
        ad: &AdapterV2,
        active: &[u64],
        wte_rows: &HashMap<u32, Vec<f32>>,
    ) -> (Vec<Vec<f32>>, TapeCache) {
        self.forward_overrides_both(ids, ad, active, wte_rows, &HashMap::new())
    }

    /// Like `forward_with_overrides` but also allows overriding wpe rows
    /// (position -> row). Debug/FD hook: wpe perturbations isolate the embedding-path
    /// gradient (no tied-head aliasing).
    pub fn forward_overrides_both(
        &self,
        ids: &[u32],
        ad: &AdapterV2,
        active: &[u64],
        wte_rows: &HashMap<u32, Vec<f32>>,
        wpe_rows: &HashMap<u32, Vec<f32>>,
    ) -> (Vec<Vec<f32>>, TapeCache) {
        let m = &self.eng.meta;
        let d = m.n_embd;
        let hd = d / m.n_head;
        let t_len = ids.len();
        assert!(t_len <= m.n_ctx, "sequence longer than n_ctx");
        let scale = 1.0 / (hd as f32).sqrt();

        // Resolve per-layer tap -> adapter index (site plan may be any order/subset).
        let attn_site: Vec<Option<usize>> = (0..m.n_layer)
            .map(|l| {
                self.sites
                    .iter()
                    .position(|s| s.kind == SiteKind::AttnIn && s.layer == l)
                    .and_then(|si| self.adapter_idx(ad, &self.sites[si]))
            })
            .collect();
        let mlp_site: Vec<Option<usize>> = (0..m.n_layer)
            .map(|l| {
                self.sites
                    .iter()
                    .position(|s| s.kind == SiteKind::MlpIn && s.layer == l)
                    .and_then(|si| self.adapter_idx(ad, &self.sites[si]))
            })
            .collect();

        let mut xin = vec![vec![0f32; t_len * d]; m.n_layer];
        let mut h1 = vec![vec![0f32; t_len * d]; m.n_layer];
        let mut qkv = vec![vec![0f32; t_len * self.wc.layers[0].cattn_rows]; m.n_layer];
        let mut xmid = vec![vec![0f32; t_len * d]; m.n_layer];
        let mut h2 = vec![vec![0f32; t_len * d]; m.n_layer];
        let mut fc_pre = vec![vec![0f32; t_len * self.wc.layers[0].cfc_rows]; m.n_layer];
        let mut xfin = vec![0f32; t_len * d];
        let mut xf = vec![0f32; t_len * d];
        let mut logits_all: Vec<Vec<f32>> = Vec::with_capacity(t_len);

        for p in 0..t_len {
            let tok = ids[p];
            // ---- embeddings (override-aware) ----
            let mut x = match wte_rows.get(&tok) {
                Some(row) => row.clone(),
                None => self.eng.vec_row("wte.weight", tok),
            };
            let wpe = match wpe_rows.get(&(p as u32)) {
                Some(row) => row.clone(),
                None => self.eng.vec_row("wpe.weight", p as u32),
            };
            for i in 0..d {
                x[i] += wpe[i];
            }

            for layer in 0..m.n_layer {
                let lw = &self.wc.layers[layer];
                xin[layer][p * d..(p + 1) * d].copy_from_slice(&x);

                // ---- attention branch ----
                let mut h = x.clone();
                self.eng.layernorm(&mut h, &lw.ln1_w, &lw.ln1_b, m.ln_eps);
                h1[layer][p * d..(p + 1) * d].copy_from_slice(&h);
                if let Some(&Some(ai)) = attn_site.get(layer) {
                    // tap is PRE-linear: qkv = W(h + BA_h)
                    ad.apply_at(ai, &mut h, active);
                }

                let inter3 = lw.cattn_rows;
                let mut q = vec![0f32; inter3];
                let cattn_name = format!("h.{layer}.attn.c_attn.weight");
                self.eng.matmul(&cattn_name, &h, &mut q);
                for i in 0..inter3 {
                    q[i] += lw.cattn_b[i];
                }
                qkv[layer][p * inter3..(p + 1) * inter3].copy_from_slice(&q);

                // causal-prefix multi-head attention over positions 0..=p
                // (op order mirrors Engine::step exactly)
                let ql = &qkv[layer][p * inter3..(p + 1) * inter3];
                let mut attn_out = vec![0f32; d];
                for head in 0..m.n_head {
                    let o = head * hd;
                    let qv = &ql[o..o + hd];
                    let mut scores = vec![0f32; p + 1];
                    let mut maxs = f32::MIN;
                    for pi in 0..=p {
                        let kp = &qkv[layer][pi * inter3 + d + o..pi * inter3 + d + o + hd];
                        let mut dot = 0f32;
                        for tt in 0..hd {
                            dot += qv[tt] * kp[tt];
                        }
                        scores[pi] = dot * scale;
                        maxs = maxs.max(scores[pi]);
                    }
                    let mut sum = 0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - maxs).exp();
                        sum += *s;
                    }
                    for s in scores.iter_mut() {
                        *s /= sum;
                    }
                    for pi in 0..=p {
                        let vp = &qkv[layer][pi * inter3 + 2 * d + o..pi * inter3 + 2 * d + o + hd];
                        for tt in 0..hd {
                            attn_out[o + tt] += scores[pi] * vp[tt];
                        }
                    }
                }

                let mut proj = vec![0f32; d];
                let proj_name = format!("h.{layer}.attn.c_proj.weight");
                self.eng.matmul(&proj_name, &attn_out, &mut proj);
                for i in 0..d {
                    x[i] += proj[i] + lw.aproj_b[i];
                }
                xmid[layer][p * d..(p + 1) * d].copy_from_slice(&x);

                // ---- mlp branch ----
                let mut h2v = x.clone();
                self.eng.layernorm(&mut h2v, &lw.ln2_w, &lw.ln2_b, m.ln_eps);
                h2[layer][p * d..(p + 1) * d].copy_from_slice(&h2v);
                if let Some(&Some(mi)) = mlp_site.get(layer) {
                    ad.apply_at(mi, &mut h2v, active);
                }

                let inter = lw.cfc_rows;
                let mut fc = vec![0f32; inter];
                let fc_name = format!("h.{layer}.mlp.c_fc.weight");
                self.eng.matmul(&fc_name, &h2v, &mut fc);
                for i in 0..inter {
                    let v = fc[i] + lw.cfc_b[i];
                    fc_pre[layer][p * inter + i] = v; // PRE-gelu, bias included (backward needs this)
                    fc[i] = 0.5 * v * (1.0 + (GELU_TANH_C * (v + GELU_TANH_A * v * v * v)).tanh());
                }

                let mut mo = vec![0f32; d];
                let mo_name = format!("h.{layer}.mlp.c_proj.weight");
                self.eng.matmul(&mo_name, &fc, &mut mo);
                for i in 0..d {
                    x[i] += mo[i] + lw.mproj_b[i];
                }
            }

            // final layernorm + tied head (cache keeps pre-ln_f and post-ln_f hiddens)
            xfin[p * d..(p + 1) * d].copy_from_slice(&x);
            let mut xfv = x;
            self.eng.layernorm(&mut xfv, &self.wc.ln_f_w, &self.wc.ln_f_b, m.ln_eps);
            xf[p * d..(p + 1) * d].copy_from_slice(&xfv);
            let logits = head_forward(self.eng, &xfv, wte_rows);
            logits_all.push(logits);
        }

        let cache = TapeCache {
            t: t_len,
            ids: ids.to_vec(),
            d,
            inter3: self.wc.layers[0].cattn_rows,
            inter: self.wc.layers[0].cfc_rows,
            xin,
            h1,
            qkv,
            xmid,
            h2,
            fc_pre,
            xfin,
            xf,
            logits: logits_all.clone(),
        };
        (logits_all, cache)
    }

    /// Exact reverse-mode backward. `targets[i] == usize::MAX` marks unsupervised positions
    /// (weight 0 in the mean-CE loss). Returns per-site grads (summed over positions,
    /// masked to `active`) and dlogits; the residual-stream/embedding-side input grad is
    /// available from [`Self::backward_full`].
    pub fn backward(&self, cache: &TapeCache, targets: &[usize], ad: &AdapterV2, active: &[u64]) -> BackwardOut {
        self.backward_full(cache, targets, ad, active).0
    }

    /// Like [`Self::backward`] but also returns `dx_embed[pos]` — the gradient wrt the
    /// position-p embedding input `wte[ids[p]] + wpe[p]` (needed for wte-row FD gates).
    pub fn backward_full(
        &self,
        cache: &TapeCache,
        targets: &[usize],
        ad: &AdapterV2,
        active: &[u64],
    ) -> (BackwardOut, Vec<Vec<f32>>) {
        let m = &self.eng.meta;
        let d = m.n_embd;
        let hd = d / m.n_head;
        let nl = m.n_layer;
        let t_len = cache.t;
        let inter3 = cache.inter3;
        let inter = cache.inter;
        let scale = 1.0 / (hd as f32).sqrt();
        assert_eq!(targets.len(), t_len, "targets must cover the sequence");

        // ---- dlogits: softmax - onehot, scaled by 1/N_sup ----
        let nsup = targets.iter().filter(|&&x| x != usize::MAX).count();
        assert!(nsup > 0, "no supervised positions");
        let inv_n = 1.0 / nsup as f32;
        let mut dlogits: Vec<Vec<f32>> = Vec::with_capacity(t_len);
        for p in 0..t_len {
            let mut dz = softmax_f32(&cache.logits[p]);
            if targets[p] != usize::MAX {
                for v in dz.iter_mut() {
                    *v *= inv_n;
                }
                dz[targets[p]] -= inv_n;
            } else {
                dz.iter_mut().for_each(|x| *x = 0.0);
            }
            dlogits.push(dz);
        }

        // ---- head: dxh[p] = wte^T dz[p] (transposed dequantized gemv over the payload) ----
        let mut dx = vec![0f32; t_len * d]; // running grad wrt residual stream, per position
        for p in 0..t_len {
            let mut dxh = vec![0f32; d];
            head_input_grad(self.eng, &dlogits[p], &mut dxh);
            // ln_f backward
            let mut dx_fin = vec![0f32; d];
            ln_backward(
                &cache.xfin[p * d..(p + 1) * d],
                &self.wc.ln_f_w,
                &dxh,
                m.ln_eps,
                &mut dx_fin,
            );
            dx[p * d..(p + 1) * d].copy_from_slice(&dx_fin);
        }

        // per-site accumulators
        let mut acc: HashMap<usize, (Vec<f32>, Vec<f32>)> = HashMap::new();

        for layer in (0..nl).rev() {
            let lw = &self.wc.layers[layer];
            let ql = &cache.qkv[layer];
            let attn_ai = self
                .sites
                .iter()
                .position(|s| s.kind == SiteKind::AttnIn && s.layer == layer)
                .and_then(|si| self.adapter_idx(ad, &self.sites[si]).map(|ai| (si, ai)));
            let mlp_mi = self
                .sites
                .iter()
                .position(|s| s.kind == SiteKind::MlpIn && s.layer == layer)
                .and_then(|si| self.adapter_idx(ad, &self.sites[si]).map(|mi| (si, mi)));

            let mut dkacc = vec![0f32; t_len * d];
            let mut dvacc = vec![0f32; t_len * d];

            // Descending query sweep: when position j is processed, dkacc[j]/dvacc[j] already
            // contain every contribution from queries p > j.
            for p in (0..t_len).rev() {
                let dp = &dx[p * d..(p + 1) * d];

                // ===== mlp branch =====
                let mut dfc = vec![0f32; inter];
                lin_input_grad(&lw.mproj_w, d, inter, dp, &mut dfc); // mlp.c_proj^T
                for i in 0..inter {
                    dfc[i] *= gelu_tanh_bwd(cache.fc_pre[layer][p * inter + i]);
                }
                let mut du_fc = vec![0f32; d];
                lin_input_grad(&lw.cfc_w, inter, d, &dfc, &mut du_fc); // mlp.c_fc^T
                let dh2 = match mlp_mi {
                    Some((si, mi)) => {
                        let sa = &ad.sites[mi];
                        let e = acc.entry(si).or_insert_with(|| {
                            (vec![0f32; sa.r * sa.d], vec![0f32; sa.d * sa.r])
                        });
                        let mut dh2 = du_fc.clone();
                        adapter_backward(sa, &cache.h2[layer][p * d..(p + 1) * d], &du_fc, active, &mut e.0, &mut e.1, &mut dh2);
                        dh2
                    }
                    _ => du_fc,
                };
                let mut dx_mlp_in = vec![0f32; d];
                ln_backward(
                    &cache.xmid[layer][p * d..(p + 1) * d],
                    &lw.ln2_w,
                    &dh2,
                    m.ln_eps,
                    &mut dx_mlp_in,
                );
                // The mlp branch consumes the attention branch's OUTPUT (xmid), so the
                // mlp feedback must be folded into the residual-stream grad BEFORE the
                // attention branch reads it: dao = c_proj^T (dp + dx_mlp_in).
                for i in 0..d {
                    dx[p * d + i] += dx_mlp_in[i];
                }

                // ===== attention branch =====
                let mut dao = vec![0f32; d];
                lin_input_grad(&lw.aproj_w, d, d, &dx[p * d..(p + 1) * d], &mut dao); // attn.c_proj^T

                let qrow = p * inter3;
                let mut dq = vec![0f32; d];
                for head in 0..m.n_head {
                    let o = head * hd;
                    let qv = &ql[qrow + o..qrow + o + hd];
                    // recompute causal scores exactly as forward did
                    let mut scores = vec![0f32; p + 1];
                    let mut maxs = f32::MIN;
                    for pi in 0..=p {
                        let base = pi * inter3 + d + o;
                        let kp = &ql[base..base + hd];
                        let mut dot = 0f32;
                        for tt in 0..hd {
                            dot += qv[tt] * kp[tt];
                        }
                        scores[pi] = dot * scale;
                        maxs = maxs.max(scores[pi]);
                    }
                    let mut sum = 0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - maxs).exp();
                        sum += *s;
                    }
                    for s in scores.iter_mut() {
                        *s /= sum;
                    }
                    // softmax backward + dV accumulation
                    let mut dssum = 0f32;
                    let mut dsraw = vec![0f32; p + 1];
                    for pi in 0..=p {
                        let base = pi * inter3 + 2 * d + o;
                        let vp = &ql[base..base + hd];
                        let mut dot = 0f32;
                        for tt in 0..hd {
                            dot += dao[o + tt] * vp[tt];
                            dvacc[pi * d + o + tt] += scores[pi] * dao[o + tt];
                        }
                        dsraw[pi] = dot;
                        dssum += dot * scores[pi];
                    }
                    for pi in 0..=p {
                        let g = scores[pi] * (dsraw[pi] - dssum);
                        if g == 0.0 {
                            continue;
                        }
                        let kb = pi * inter3 + d + o;
                        let kp = &ql[kb..kb + hd];
                        for tt in 0..hd {
                            dq[o + tt] += g * scale * kp[tt];
                            dkacc[pi * d + o + tt] += g * scale * qv[tt];
                        }
                    }
                }

                // assemble dqkv = [dq | dkacc[p] | dvacc[p]]
                let mut dqkv = vec![0f32; inter3];
                dqkv[..d].copy_from_slice(&dq);
                dqkv[d..2 * d].copy_from_slice(&dkacc[p * d..(p + 1) * d]);
                dqkv[2 * d..].copy_from_slice(&dvacc[p * d..(p + 1) * d]);

                // Tap-space grad: u = h1 + BA h1 feeds the whole fused matmul, so the
                // adjoint of c_attn applied to dqkv IS the tap gradient (length d).
                let mut du_tap = vec![0f32; d];
                lin_input_grad(&lw.cattn_w, inter3, d, &dqkv, &mut du_tap); // c_attn^T

                // Adapter at the attn tap: dh1 starts as du_tap (identity path), then
                // adapter_backward adds A^T (B^T du_tap) and accumulates ga/gb.
                let dh1 = match attn_ai {
                    Some((si, ai)) => {
                        let sa = &ad.sites[ai];
                        let e = acc.entry(si).or_insert_with(|| {
                            (vec![0f32; sa.r * sa.d], vec![0f32; sa.d * sa.r])
                        });
                        let mut dh1 = du_tap.clone();
                        adapter_backward(
                            sa,
                            &cache.h1[layer][p * d..(p + 1) * d],
                            &du_tap,
                            active,
                            &mut e.0,
                            &mut e.1,
                            &mut dh1,
                        );
                        dh1
                    }
                    _ => du_tap,
                };

                let mut dx_attn_in = vec![0f32; d];
                ln_backward(
                    &cache.xin[layer][p * d..(p + 1) * d],
                    &lw.ln1_w,
                    &dh1,
                    m.ln_eps,
                    &mut dx_attn_in,
                );

                // ===== residual sum: mlp feedback already folded in above =====
                for i in 0..d {
                    dx[p * d + i] += dx_attn_in[i];
                }
            }
        }

        let dx_embed: Vec<Vec<f32>> = (0..t_len)
            .map(|p| dx[p * d..(p + 1) * d].to_vec())
            .collect();

        let mut per_site: Vec<SiteGrad> = Vec::with_capacity(self.sites.len());
        for (si, _s) in self.sites.iter().enumerate() {
            let (ga, gb) = match acc.remove(&si) {
                Some(x) => x,
                None => ((vec![0f32; self.r * self.eng.meta.n_embd]), (vec![0f32; self.eng.meta.n_embd * self.r])),
            };
            per_site.push(SiteGrad { site_idx: si, ga, gb });
        }

        (
            BackwardOut {
                per_site,
                dlogits,
            },
            dx_embed,
        )
    }

    /// Analytic gradients of the mean-CE loss wrt selected dequantized wte ROWS (both paths:
    /// tied head `dw[v][j] = sum_p dlogits[p][v] * xf[p][j]` and embedding gather
    /// `sum_{p: ids[p]==v} dx_embed[p][j]`). Used by tapecheck's wte finite-difference gate.
    pub fn wte_row_grads(
        &self,
        cache: &TapeCache,
        dx_embed: &[Vec<f32>],
        dlogits: &[Vec<f32>],
        rows: &[usize],
    ) -> Vec<Vec<f32>> {
        let d = cache.d;
        let t_len = cache.t;
        let mut out = Vec::with_capacity(rows.len());
        for &v in rows {
            let mut g = vec![0f32; d];
            for p in 0..t_len {
                let zv = dlogits[p][v];
                if zv != 0.0 {
                    let xf = &cache.xf[p * d..(p + 1) * d];
                    for j in 0..d {
                        g[j] += zv * xf[j];
                    }
                }
                if cache.ids[p] as usize == v {
                    for j in 0..d {
                        g[j] += dx_embed[p][j];
                    }
                }
            }
            out.push(g);
        }
        out
    }
}

// ----------------------------------------------------------------------------------------
// building blocks
// ----------------------------------------------------------------------------------------

/// Input grad of a row-major [rows, cols] linear: dx[j] += sum_r W[r*cols + j] * dy[r].
/// Exact adjoint of the forward gemv y[r] = dot(W[r], x).
#[inline]
pub fn lin_input_grad(w: &[f32], rows: usize, cols: usize, dy: &[f32], dx: &mut [f32]) {
    for r in 0..rows {
        let g = dy[r];
        if g == 0.0 {
            continue;
        }
        let wr = &w[r * cols..(r + 1) * cols];
        for j in 0..cols {
            dx[j] += wr[j] * g;
        }
    }
}

/// Layernorm input grad for y = (x - mu)/sigma * g + b, matching Engine::layernorm's
/// biased-variance, eps-inside-sqrt convention. Accumulates INTO dx.
pub fn ln_backward(x_pre: &[f32], g: &[f32], dy: &[f32], eps: f32, dx: &mut [f32]) {
    let n = x_pre.len();
    let nf = n as f32;
    let mean = x_pre.iter().sum::<f32>() / nf;
    let var = x_pre.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / nf;
    let inv = 1.0 / (var + eps).sqrt();
    let mut s1 = 0f32;
    let mut s2 = 0f32;
    for i in 0..n {
        let xh = (x_pre[i] - mean) * inv;
        let gy = dy[i] * g[i];
        s1 += gy;
        s2 += gy * xh;
    }
    s1 /= nf;
    s2 /= nf;
    for i in 0..n {
        let xh = (x_pre[i] - mean) * inv;
        dx[i] += inv * (dy[i] * g[i] - s1 - xh * s2);
    }
}

/// Derivative of the tanh-approximation GELU used by Engine::step, evaluated at pre-GELU v.
#[inline]
pub fn gelu_tanh_bwd(v: f32) -> f32 {
    let inner = GELU_TANH_C * (v + GELU_TANH_A * v * v * v);
    let th = inner.tanh();
    0.5 * (1.0 + th) + 0.5 * v * (1.0 - th * th) * GELU_TANH_C * (1.0 + 3.0 * GELU_TANH_A * v * v)
}

pub fn softmax_f32(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::MIN, f32::max);
    let mut out: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    for x in out.iter_mut() {
        *x /= sum;
    }
    out
}

/// Mean cross-entropy over supervised positions (targets[i] == usize::MAX skipped).
pub fn mean_ce_loss(logits: &[Vec<f32>], targets: &[usize]) -> f64 {
    let nsup = targets.iter().filter(|&&t| t != usize::MAX).count();
    assert!(nsup > 0, "no supervised positions");
    let mut total = 0f64;
    for (p, lg) in logits.iter().enumerate() {
        let t = targets[p];
        if t == usize::MAX {
            continue;
        }
        let max = lg.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let lse: f64 = lg.iter().map(|&x| (x as f64 - max).exp()).sum::<f64>().ln();
        total += max - lg[t] as f64 + lse;
    }
    total / nsup as f64
}

/// logits = wte @ x, override-aware: with no overrides this is exactly Engine::matmul on the
/// packed payload (bit-identical to the decode path); with overrides affected rows are taken
/// from the map and the rest are dequantized on the fly.
fn head_forward(eng: &Engine, x: &[f32], wte_rows: &HashMap<u32, Vec<f32>>) -> Vec<f32> {
    let rec = &eng.t["wte.weight"];
    let (rows, cols) = (rec.shape[0], rec.shape[1]);
    if wte_rows.is_empty() {
        let mut logits = vec![0f32; rows];
        eng.matmul("wte.weight", x, &mut logits);
        return logits;
    }
    let payload = eng.payload("wte.weight");
    let atom = rec.atom.clone();
    let mut logits = vec![0f32; rows];
    let mut scratch = vec![0f32; cols];
    for v in 0..rows {
        let row: &[f32] = match wte_rows.get(&(v as u32)) {
            Some(r) => r,
            None => {
                dequant_row(&payload, &atom, cols, v, &mut scratch);
                &scratch
            }
        };
        let mut dot = 0f32;
        for j in 0..cols {
            dot += row[j] * x[j];
        }
        logits[v] = dot;
    }
    logits
}

/// dx[j] += sum_v W[v*cols + j] * dz[v] straight off the packed wte payload (no 154 MB copy).
/// Public so parity harnesses can reuse the exact adjoint.
pub fn head_input_grad(eng: &Engine, dz: &[f32], dx: &mut [f32]) {
    let rec = &eng.t["wte.weight"];
    let (rows, cols) = (rec.shape[0], rec.shape[1]);
    let payload = eng.payload("wte.weight");
    // f64 accumulator: this reduction spans 50k rows and would otherwise dominate the
    // tape's rounding error.
    let mut acc = vec![0f64; cols];
    match rec.atom.as_str() {
        ATOM_Q8 => {
            let nblk = cols / 32;
            let stride = nblk * 34;
            for v in 0..rows {
                let g = dz[v] as f64;
                if g == 0.0 {
                    continue;
                }
                let row = &payload[v * stride..(v + 1) * stride];
                let mut dj = 0usize;
                for blk in row.chunks_exact(34) {
                    let s = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
                    for t in 0..32 {
                        acc[dj + t] += (blk[2 + t] as i8 as f64) * s * g;
                    }
                    dj += 32;
                }
            }
        }
        ATOM_F16 => {
            for v in 0..rows {
                let g = dz[v] as f64;
                if g == 0.0 {
                    continue;
                }
                let row = &payload[v * cols * 2..(v + 1) * cols * 2];
                for (j, ch) in row.chunks_exact(2).enumerate() {
                    acc[j] += f16_to_f32(u16::from_le_bytes([ch[0], ch[1]])) as f64 * g;
                }
            }
        }
        a => panic!("unknown atom {a}"),
    }
    for j in 0..cols {
        dx[j] += acc[j] as f32;
    }
}

fn dequant_row<'a>(payload: &[u8], atom: &str, cols: usize, row: usize, out: &'a mut Vec<f32>) -> &'a [f32] {
    out.clear();
    match atom {
        ATOM_Q8 => {
            let nblk = cols / 32;
            let stride = nblk * 34;
            let r = &payload[row * stride..(row + 1) * stride];
            for blk in r.chunks_exact(34) {
                let s = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for t in 0..32 {
                    out.push(blk[2 + t] as i8 as f32 * s);
                }
            }
        }
        ATOM_F16 => {
            let r = &payload[row * cols * 2..(row + 1) * cols * 2];
            for ch in r.chunks_exact(2) {
                out.push(f16_to_f32(u16::from_le_bytes([ch[0], ch[1]])));
            }
        }
        a => panic!("unknown atom {a}"),
    }
    &out[..cols]
}

/// Exact reverse-mode through one site's routed subspace.
///
/// Given du = dL/d(u) where u = x_tap + B_slice (A_slice x_tap) (caller already folded any
/// downstream linear), accumulates dA/dB into ga/gb (active slices only) and ADDS the tap
/// gradient `A_slice^T (B_slice^T du)` into dx_tap (which the caller seeds with du, covering
/// the identity path).
pub fn adapter_backward(
    sa: &crate::adapter_v2::SiteAdapter,
    x_tap: &[f32],
    du: &[f32],
    active: &[u64],
    ga: &mut [f32],
    gb: &mut [f32],
    dx_tap: &mut [f32],
) {
    let dd = sa.d;
    debug_assert_eq!(x_tap.len(), dd);
    debug_assert_eq!(du.len(), dd);
    // s = A_slice x (zero on inactive ranks)
    let mut s = vec![0f32; sa.r];
    for &(fid, start, end) in &sa.route.cols {
        if !fact_active(active, fid) {
            continue;
        }
        for c in start..end {
            let arow = &sa.a[c * dd..(c + 1) * dd];
            let mut acc = 0f32;
            for j in 0..dd {
                acc += arow[j] * x_tap[j];
            }
            s[c] = acc;
        }
    }
    // ds = B^T du ; gb += du (x) s ; ga += ds (x) x ; dx_tap += A^T ds
    for &(fid, start, end) in &sa.route.cols {
        if !fact_active(active, fid) {
            continue;
        }
        for c in start..end {
            let brow = &sa.b[c * dd..(c + 1) * dd]; // column c, rows 0..dd
            let sc = s[c];
            let mut ds_c = 0f32;
            for row in 0..dd {
                ds_c += brow[row] * du[row];
            }
            // dB accumulates UNCONDITIONALLY: with the standard B=0 init this is the
            // only nonzero first-order gradient (dB = du (x) s). Gating it on bv != 0
            // froze learning permanently (the bug this edit fixes).
            for j in 0..dd {
                gb[c * dd + j] += du[j] * sc;
            }
            if ds_c != 0.0 {
                let arow = &sa.a[c * dd..(c + 1) * dd];
                let garow = &mut ga[c * dd..(c + 1) * dd];
                for j in 0..dd {
                    garow[j] += ds_c * x_tap[j];
                    dx_tap[j] += arow[j] * ds_c;
                }
            }
        }
    }
}
// ----------------------------------------------------------------------------------------
// f64 reference forward (finite-difference gate support)
// ----------------------------------------------------------------------------------------
//
// The packed q8/f16 gemv path is linear in its inputs, but evaluating a central difference
// through the f32 engine has a rounding-noise floor (~5e-6 in the loss), which swamps
// h = 1e-3-relative perturbations. The FD gate therefore evaluates loss differences with an
// f64 widening of EXACTLY the same mathematical model the tape differentiates (same op order,
// same dequantized weight values widened to f64, adapters included). The analytic gradients
// under test remain the f32 tape's; the comparison then isolates genuine derivative errors
// from evaluation rounding.

pub struct F64Layer {
    pub ln1_w: Vec<f64>,
    pub ln1_b: Vec<f64>,
    pub cattn_w: Vec<f64>,
    pub cattn_b: Vec<f64>,
    pub aproj_w: Vec<f64>,
    pub aproj_b: Vec<f64>,
    pub ln2_w: Vec<f64>,
    pub ln2_b: Vec<f64>,
    pub cfc_w: Vec<f64>,
    pub cfc_b: Vec<f64>,
    pub mproj_w: Vec<f64>,
    pub mproj_b: Vec<f64>,
}

pub struct F64Cache {
    pub layers: Vec<F64Layer>,
    pub ln_f_w: Vec<f64>,
    pub ln_f_b: Vec<f64>,
    /// dequantized wte widened to f64, row-major [V * d]
    pub wte: Vec<f64>,
    /// dequantized wpe widened to f64, row-major [n_ctx * d]
    pub wpe: Vec<f64>,
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

impl<'a> TapeModel<'a> {
    fn build_f64_cache(&self) -> F64Cache {
        let layers = self
            .wc
            .layers
            .iter()
            .map(|l| F64Layer {
                ln1_w: widen(&l.ln1_w),
                ln1_b: widen(&l.ln1_b),
                cattn_w: widen(&l.cattn_w),
                cattn_b: widen(&l.cattn_b),
                aproj_w: widen(&l.aproj_w),
                aproj_b: widen(&l.aproj_b),
                ln2_w: widen(&l.ln2_w),
                ln2_b: widen(&l.ln2_b),
                cfc_w: widen(&l.cfc_w),
                cfc_b: widen(&l.cfc_b),
                mproj_w: widen(&l.mproj_w),
                mproj_b: widen(&l.mproj_b),
            })
            .collect();
        let wte32 = self.eng.vec_f32("wte.weight");
        let wpe32 = self.eng.vec_f32("wpe.weight");
        F64Cache {
            layers,
            ln_f_w: widen(&self.wc.ln_f_w),
            ln_f_b: widen(&self.wc.ln_f_b),
            wte: widen(&wte32),
            wpe: widen(&wpe32),
        }
    }

    fn f64_cache(&self) -> &F64Cache {
        self.f64c.get_or_init(|| self.build_f64_cache())
    }

    /// Mean CE loss of the f64 reference forward. `targets[i] == usize::MAX` skips position i.
    pub fn ce_loss_f64(
        &self,
        ids: &[u32],
        ad: &AdapterV2,
        active: &[u64],
        targets: &[usize],
        wte_rows: &HashMap<u32, Vec<f64>>,
        wpe_rows: &HashMap<u32, Vec<f64>>,
    ) -> f64 {
        let c = self.f64_cache();
        let m = &self.eng.meta;
        let d = m.n_embd;
        let hd = d / m.n_head;
        let t_len = ids.len();
        let scale = 1.0 / (hd as f64).sqrt();

        // per-layer tap -> adapter index + widened A/B
        type Routed = (Vec<f64>, Vec<f64>, Vec<(u64, usize, usize)>);
        let mut attn_ad: HashMap<usize, Routed> = HashMap::new();
        let mut mlp_ad: HashMap<usize, Routed> = HashMap::new();
        for (_si, s) in self.sites.iter().enumerate() {
            if let Some(ai) = self.adapter_idx(ad, s) {
                let sa = &ad.sites[ai];
                let entry = (widen(&sa.a), widen(&sa.b), sa.route.cols.clone());
                match s.kind {
                    SiteKind::AttnIn => {
                        attn_ad.insert(s.layer, entry);
                    }
                    SiteKind::MlpIn => {
                        mlp_ad.insert(s.layer, entry);
                    }
                }
            }
        }

        let inter3_all = c.layers[0].cattn_w.len() / d;
        let mut qkv_all = vec![vec![0f64; t_len * inter3_all]; m.n_layer];
        let mut total = 0f64;
        let mut nsup = 0usize;
        for p in 0..t_len {
            let tok = ids[p];
            let mut x: Vec<f64> = match wte_rows.get(&tok) {
                Some(r) => r.clone(),
                None => c.wte[tok as usize * d..(tok as usize + 1) * d].to_vec(),
            };
            let wpe: Vec<f64> = match wpe_rows.get(&(p as u32)) {
                Some(r) => r.clone(),
                None => c.wpe[p * d..(p + 1) * d].to_vec(),
            };
            for i in 0..d {
                x[i] += wpe[i];
            }

            for layer in 0..m.n_layer {
                let lw = &c.layers[layer];

                // attention branch
                let mut h = x.clone();
                ln_f64(&mut h, &lw.ln1_w, &lw.ln1_b, m.ln_eps as f64);
                if let Some((a, b, route)) = attn_ad.get(&layer) {
                    apply_f64(a, b, route, &mut h, active);
                }
                let inter3 = lw.cattn_w.len() / d;
                let mut qkv = vec![0f64; inter3];
                matvec_f64(&lw.cattn_w, &h, &mut qkv);
                for i in 0..inter3 {
                    qkv[i] += lw.cattn_b[i];
                }
                qkv_all[layer][p * inter3..(p + 1) * inter3].copy_from_slice(&qkv);
                let ql = &qkv_all[layer];
                let mut ao = vec![0f64; d];
                for head in 0..m.n_head {
                    let o = head * hd;
                    let qv = &qkv[o..o + hd];
                    let mut scores = vec![0f64; p + 1];
                    let mut maxs = f64::MIN;
                    for pi in 0..=p {
                        let kp = &ql[pi * inter3 + d + o..pi * inter3 + d + o + hd];
                        let mut dot = 0f64;
                        for tt in 0..hd {
                            dot += qv[tt] * kp[tt];
                        }
                        scores[pi] = dot * scale;
                        maxs = maxs.max(scores[pi]);
                    }
                    let mut sum = 0f64;
                    for sv in scores.iter_mut() {
                        *sv = (*sv - maxs).exp();
                        sum += *sv;
                    }
                    for sv in scores.iter_mut() {
                        *sv /= sum;
                    }
                    for pi in 0..=p {
                        let vp = &ql[pi * inter3 + 2 * d + o..pi * inter3 + 2 * d + o + hd];
                        for tt in 0..hd {
                            ao[o + tt] += scores[pi] * vp[tt];
                        }
                    }
                }
                let mut proj = vec![0f64; d];
                matvec_f64(&lw.aproj_w, &ao, &mut proj);
                for i in 0..d {
                    x[i] += proj[i] + lw.aproj_b[i];
                }

                // mlp branch
                let mut h2v = x.clone();
                ln_f64(&mut h2v, &lw.ln2_w, &lw.ln2_b, m.ln_eps as f64);
                if let Some((a, b, route)) = mlp_ad.get(&layer) {
                    apply_f64(a, b, route, &mut h2v, active);
                }
                let inter = lw.cfc_w.len() / d;
                let mut fc = vec![0f64; inter];
                matvec_f64(&lw.cfc_w, &h2v, &mut fc);
                for i in 0..inter {
                    let v = fc[i] + lw.cfc_b[i];
                    fc[i] = 0.5 * v * (1.0 + (GELU_TANH_C as f64 * (v + GELU_TANH_A as f64 * v * v * v)).tanh());
                }
                let mut mo = vec![0f64; d];
                matvec_f64(&lw.mproj_w, &fc, &mut mo);
                for i in 0..d {
                    x[i] += mo[i] + lw.mproj_b[i];
                }
            }

            // final layernorm + tied head
            let mut xf = x.clone();
            ln_f64(&mut xf, &c.ln_f_w, &c.ln_f_b, m.ln_eps as f64);
            let t = targets[p];
            if t == usize::MAX {
                continue;
            }
            let mut logits = vec![0f64; m.vocab];
            for vrow in 0..m.vocab {
                let row: &[f64] = match wte_rows.get(&(vrow as u32)) {
                    Some(r) => r,
                    None => &c.wte[vrow * d..(vrow + 1) * d],
                };
                let mut dot = 0f64;
                for j in 0..d {
                    dot += row[j] * xf[j];
                }
                logits[vrow] = dot;
            }
            let max = logits.iter().cloned().fold(f64::MIN, f64::max);
            let lse: f64 = logits.iter().map(|&x| (x - max).exp()).sum::<f64>().ln();
            total += max - logits[t] as f64 + lse;
            nsup += 1;
        }
        assert!(nsup > 0, "no supervised positions");
        total / nsup as f64
    }
}

/// y = g ⊙ (x-mean)/sqrt(var+eps) + b in place, biased variance, matching Engine::layernorm.
fn ln_f64(x: &mut [f64], g: &[f64], b: &[f64], eps: f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        x[i] = (x[i] - mean) * inv * g[i] + b[i];
    }
}

/// y[r] = dot(W[r], x), W row-major [rows, cols].
fn matvec_f64(w: &[f64], x: &[f64], y: &mut [f64]) {
    let cols = x.len();
    for (r, yr) in y.iter_mut().enumerate() {
        let wr = &w[r * cols..(r + 1) * cols];
        let mut acc = 0f64;
        for j in 0..cols {
            acc += wr[j] * x[j];
        }
        *yr = acc;
    }
}

/// Routed LoRA application in f64: x += B_slice (A_slice x) over active fact slices.
fn apply_f64(a: &[f64], b: &[f64], route: &[(u64, usize, usize)], x: &mut [f64], active: &[u64]) {
    use crate::adapter_v2::fact_active;
    let dd = x.len();
    let r = a.len() / dd;
    let mut s = vec![0f64; r];
    for &(fid, start, end) in route {
        if !fact_active(active, fid) {
            continue;
        }
        for c in start..end {
            let arow = &a[c * dd..(c + 1) * dd];
            let mut acc = 0f64;
            for j in 0..dd {
                acc += arow[j] * x[j];
            }
            s[c] = acc;
        }
    }
    for &(fid, start, end) in route {
        if !fact_active(active, fid) {
            continue;
        }
        for c in start..end {
            if s[c] == 0.0 {
                continue;
            }
            let col = &b[c * dd..(c + 1) * dd];
            for row in 0..dd {
                x[row] += col[row] * s[c];
            }
        }
    }
}
