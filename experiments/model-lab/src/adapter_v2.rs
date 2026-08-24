//! Routed block-diagonal LoRA subspaces (SMT v2 plasticity upgrade, Agent C).
//!
//! Each site is a PRE-LINEAR LoRA overlay on one of the two per-layer taps:
//!   - SiteKind::AttnIn : tap x = ln_1 output; host matrix h.{l}.attn.c_attn.weight [2304, 768]
//!   - SiteKind::MlpIn  : tap x = ln_2 output; host matrix h.{l}.mlp.c_fc.weight   [3072, 768]
//! Forward contribution: y = W(x + BAx) ⇒ merged form (W + BA)·x — EXACT, so consolidation can
//! fold the adapter into the host matrix without approximation.
//!
//! ROUTED SUBSPACES: the rank budget r of a site is partitioned into disjoint column ranges
//! keyed by fact_id (u64). A slice for (fact_id f, start..end) uses
//!   - rows start..end of A  → contiguous because a is SLICE-MAJOR in route order, each slice
//!     stored row-major [s, d] where s = end - start;
//!   - columns start..end of B → contiguous because b is stored COLUMN-major ([d rows × r cols],
//!     b[col*d + row]), so every routed range is one d-length block per column.
//! apply() with an active fact set skips slices whose fact_id is inactive ⇒ they contribute
//! EXACTLY zero (bit-exact), which is what makes per-fact budgets and surgical reverts possible.
#![allow(dead_code)]

use serde_json::json;

/// Which pre-linear tap a site overlays.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SiteKind {
    /// ln_1 output feeding h.{l}.attn.c_attn.weight
    AttnIn,
    /// ln_2 output feeding h.{l}.mlp.c_fc.weight
    MlpIn,
}

impl SiteKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SiteKind::AttnIn => "attn_in",
            SiteKind::MlpIn => "mlp_in",
        }
    }
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "attn_in" => Ok(SiteKind::AttnIn),
            "mlp_in" => Ok(SiteKind::MlpIn),
            other => Err(format!("unknown SiteKind '{other}'")),
        }
    }
}

/// Disjoint routed column ranges, ascending by construction check, Σ(end-start) == r.
#[derive(Clone, Debug, Default)]
pub struct RouteSpec {
    /// (fact_id, start col, end col exclusive)
    pub cols: Vec<(u64, usize, usize)>,
}

impl RouteSpec {
    pub fn validate(&self, r: usize) -> Result<(), String> {
        let mut prev_end = 0usize;
        let mut total = 0usize;
        for (i, &(_fid, s, e)) in self.cols.iter().enumerate() {
            if e <= s {
                return Err(format!("route[{i}] empty/inverted range {s}..{e}"));
            }
            if s < prev_end {
                return Err(format!(
                    "route[{i}] range {s}..{e} overlaps/unordered vs previous end {prev_end}"
                ));
            }
            if e > r {
                return Err(format!("route[{i}] end {e} exceeds rank {r}"));
            }
            prev_end = e;
            total += e - s;
        }
        if total != r {
            return Err(format!("route widths sum {total} != rank {r}"));
        }
        // duplicate fact ids would make "active set" ambiguous
        let mut ids: Vec<u64> = self.cols.iter().map(|c| c.0).collect();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != self.cols.len() {
            return Err("duplicate fact_id in route".into());
        }
        Ok(())
    }

    /// Width of slice i.
    #[inline]
    pub fn width(&self, i: usize) -> usize {
        self.cols[i].2 - self.cols[i].1
    }

    /// All distinct fact ids, in route order.
    pub fn fact_ids(&self) -> Vec<u64> {
        self.cols.iter().map(|c| c.0).collect()
    }
}

/// One LoRA site with routed block-diagonal fact subspaces.
#[derive(Clone)]
pub struct SiteAdapter {
    pub kind: SiteKind,
    pub layer: usize,
    /// host input dim (= n_embd)
    pub d: usize,
    /// total rank budget across all facts
    pub r: usize,
    pub route: RouteSpec,
    /// [r*d], slice-major by route order; slice i is row-major [width_i, d]
    pub a: Vec<f32>,
    /// [d*r], COLUMN-major: b[col*d + row]; routed ranges are contiguous blocks
    pub b: Vec<f32>,
}

impl SiteAdapter {
    /// Zero-filled site; validates the route against `r`.
    pub fn new(kind: SiteKind, layer: usize, d: usize, route: RouteSpec) -> Self {
        let r: usize = route.cols.iter().map(|&(_, s, e)| e - s).sum();
        route
            .validate(r)
            .unwrap_or_else(|e| panic!("invalid route on {kind:?} layer {layer}: {e}"));
        Self { kind, layer, d, r, route, a: vec![0.0; r * d], b: vec![0.0; d * r] }
    }

    fn slice_offset(&self, i: usize) -> usize {
        self.route.cols[..i].iter().map(|&(_, s, e)| (e - s) * self.d).sum()
    }

    /// A rows for slice i (row-major [width_i, d]).
    #[inline]
    pub fn a_slice(&self, i: usize) -> &[f32] {
        let off = self.slice_offset(i);
        &self.a[off..off + self.route.width(i) * self.d]
    }
    /// B columns for slice i (contiguous block, layout column-major).
    #[inline]
    pub fn b_slice(&self, i: usize) -> &[f32] {
        let (start, end) = (self.route.cols[i].1, self.route.cols[i].2);
        &self.b[start * self.d..end * self.d]
    }

    /// Deterministic init: A ~ uniform(-0.02, 0.02) via splitmix64, B zeros.
    pub fn zero_init(&mut self, seed: u64) {
        let mut z = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next_u32 = move || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x ^= x >> 30;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 27;
            x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            x as u32
        };
        const SCALE: f32 = 0.04; // uniform(-0.02, 0.02)
        for v in self.a.iter_mut() {
            *v = (next_u32() as f32 / u32::MAX as f32 - 0.5) * SCALE;
        }
        for v in self.b.iter_mut() {
            *v = 0.0;
        }
    }

    /// x += Σ_{slice i active} B[:,cols_i] · (A[rows_i,:] · x). Inactive slices contribute
    /// EXACTLY zero (they are skipped, not multiplied by zero). All A·x products read the
    /// ORIGINAL tap x — routed subspaces are crosstalk-free and order-independent, exactly
    /// the block-diagonal (W + BA)·x merged form.
    pub fn apply(&self, x: &mut [f32], active: &[u64]) {
        assert_eq!(x.len(), self.d, "tap dim mismatch on {:?}", self.kind);
        // pass 1: u_i = A_i · x against the UNTOUCHED tap vector
        let mut us: Vec<(usize /*slice*/, Vec<f32>)> = Vec::new();
        for i in 0..self.route.cols.len() {
            if !fact_active(active, self.route.cols[i].0) {
                continue;
            }
            let w = self.route.width(i);
            let a = self.a_slice(i);
            let mut u = vec![0.0f32; w];
            for s in 0..w {
                let row = &a[s * self.d..(s + 1) * self.d];
                u[s] = row.iter().zip(x.iter()).map(|(&w_, &x_)| w_ * x_).sum();
            }
            us.push((i, u));
        }
        // pass 2: x += B_i · u_i
        for (i, u) in &us {
            let bcol = self.b_slice(*i);
            for (cidx, &u_c) in u.iter().enumerate() {
                let col = &bcol[cidx * self.d..(cidx + 1) * self.d];
                for (o, &c_) in col.iter().enumerate() {
                    x[o] += c_ * u_c;
                }
            }
        }
    }

    /// Zero gradient entries belonging to slices whose fact_id is not active, IN PLACE.
    /// ga has the same [r*d] slice-major layout as a; gb the same column-major layout as b.
    pub fn mask_grads(&self, ga: &mut [f32], gb: &mut [f32], active: &[u64]) {
        assert_eq!(ga.len(), self.a.len(), "ga shape");
        assert_eq!(gb.len(), self.b.len(), "gb shape");
        for i in 0..self.route.cols.len() {
            let fid = self.route.cols[i].0;
            if fact_active(active, fid) {
                continue;
            }
            let off = self.slice_offset(i);
            let len = self.route.width(i) * self.d;
            for v in &mut ga[off..off + len] {
                *v = 0.0;
            }
            let (s, e) = (self.route.cols[i].1, self.route.cols[i].2);
            for v in &mut gb[s * self.d..e * self.d] {
                *v = 0.0;
            }
        }
    }
}

/// Linear scan over the (tiny) active set; fine for the handful of live facts.
#[inline]
pub fn fact_active(active: &[u64], fid: u64) -> bool {
    active.iter().any(|&a| a == fid)
}

#[derive(Clone)]
/// Multi-site routed adapter over the full trunk.
pub struct AdapterV2 {
    pub sites: Vec<SiteAdapter>,
}

impl AdapterV2 {
    pub fn new(sites: Vec<SiteAdapter>) -> Self {
        Self { sites }
    }

    /// Default site set: BOTH kinds on every layer l ∈ 0..n_layer (24 sites for gpt2-small),
    /// single full-width route owned by `fact_id` (callers may re-route afterwards).
    pub fn default_sites(n_layer: usize, d: usize, r: usize, fact_id: u64) -> Self {
        let mk = |kind: SiteKind, layer: usize| {
            SiteAdapter::new(kind, layer, d, RouteSpec { cols: vec![(fact_id, 0, r)] })
        };
        let mut sites = Vec::with_capacity(n_layer * 2);
        for l in 0..n_layer {
            sites.push(mk(SiteKind::AttnIn, l));
            sites.push(mk(SiteKind::MlpIn, l));
        }
        Self { sites }
    }

    /// Apply the contribution of one site at its tap.
    pub fn apply_at(&self, site_idx: usize, x: &mut [f32], active: &[u64]) {
        self.sites[site_idx].apply(x, active);
    }

    /// Deterministically init every site; per-site seeds are mixed from `seed` and site index.
    pub fn zero_init(&mut self, seed: u64) {
        for (i, s) in self.sites.iter_mut().enumerate() {
            let mix = (seed ^ (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            s.zero_init(mix);
        }
    }

    pub fn site_index(&self, kind: SiteKind, layer: usize) -> Option<usize> {
        self.sites.iter().position(|s| s.kind == kind && s.layer == layer)
    }
}

// ---------------------------------------------------------------------------
// delta v2 file I/O lives in crate::delta_v2; the helpers below are shared.
// ---------------------------------------------------------------------------

pub(crate) fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

pub(crate) fn f32_from_le(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub(crate) fn digest16(body: &[u8]) -> [u8; 16] {
    blake3::hash(body).as_bytes()[..16].try_into().unwrap()
}

pub(crate) fn cid_hex(cid: &[u8; 32]) -> String {
    cid.iter().map(|b| format!("{b:02x}")).collect()
}

/// Serialize META json for a v2 delta. Public so consolidate/bin tooling can mirror it.
pub(crate) fn meta_json(base_cid_hex: &str, generation: u32, ad: &AdapterV2) -> serde_json::Value {
    json!({
        "kind": "adapter_v2",
        "format_version": 1,
        "base_content_id": base_cid_hex,
        "generation": generation,
        "sites": ad.sites.iter().enumerate().map(|(i, s)| json!({
            "index": i,
            "kind": s.kind.as_str(),
            "layer": s.layer,
            "d": s.d,
            "r": s.r,
            "route": s.route.cols.iter()
                .map(|&(fid, st, en)| json!({"fact_id": fid, "start": st, "end": en}))
                .collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}
