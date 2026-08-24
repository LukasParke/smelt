//! Agent D — consolidation folding: merge AdapterV2 LoRA deltas into host matrices
//! and emit a NEW canonical SMT pack (fresh content_id, META.generation,
//! META.consolidated_from). Site model (frozen): PRE-LINEAR LoRA on
//!   SiteAttn{layer}: tap = ln_1 output, host = h.{l}.attn.c_attn.weight [2304,768]
//!   SiteMlp{layer}:  tap = ln_2 output, host = h.{l}.mlp.c_fc.weight   [3072,768]
//! Forward y = W(x + BAx) ≡ (W + BA)x — exact, so folding is lossless in f32; the only
//! error introduced is Q8 requantization of the merged matrix.
#![allow(dead_code)]
use crate::adapter_v2::{AdapterV2, RouteSpec, SiteAdapter, SiteKind};
use crate::atoms::{f32_to_f16, q8_encode, ATOM_F16, ATOM_Q8};
use crate::format::{
    PackReader, SectionWriter, TensorRecord, SEC_GRAPH, SEC_META, SEC_TENSORS, SEC_TOKENIZER,
};
use crate::gpt2::Engine;
use serde_json::Value;
use std::collections::BTreeMap;
use std::io;

pub const INLINE_FORMAT_KIND: &str = "consolidate-inline-v1";

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Host tensor name for a site under the frozen SITE MODEL.
pub fn host_name(kind: SiteKind, layer: usize) -> String {
    match kind {
        SiteKind::AttnIn => format!("h.{layer}.attn.c_attn.weight"),
        SiteKind::MlpIn => format!("h.{layer}.mlp.c_fc.weight"),
    }
}

/// Dequantized host W (f32, row-major [out,in]) plus per-site rank data.
pub struct MergedHost {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    /// W' = W + Σ_sites B_site·A_site over ALL facts (dense over full rank r).
    pub w: Vec<f32>,
}

/// Compute merged f32 host matrices for every site of the adapter.
///
/// SITE MODEL (frozen): the adapter edits the TAP, y = W(x + BAx) with
/// A: [r, d_in], B: [d_in, r] (adapter_v2 stores `a` [r*d] slice-major, slices
/// row-major; `b` [d*r] COLUMN-major, b[col*d + row]). The plain-engine-equivalent
/// host weight is therefore W' = W + W·(Σ_c b_c a_cᵀ) — a rank-r update THROUGH the
/// host. With all facts active the route partition is irrelevant:
///   ΔW[o][i] = Σ_{c=0..r} <W[o,:], b_c> · a[c*d + i]
/// Dots are taken against the ORIGINAL W (U computed first), so multi-site /
/// multi-rank accumulation stays exactly linear.
pub fn merged_hosts(eng: &Engine, ad: &AdapterV2) -> Vec<MergedHost> {
    let mut acc: BTreeMap<String, MergedHost> = BTreeMap::new();
    for s in &ad.sites {
        let name = host_name(s.kind, s.layer);
        let rec = eng
            .t
            .get(&name)
            .unwrap_or_else(|| panic!("consolidate: host tensor '{name}' missing"));
        assert_eq!(rec.shape.len(), 2, "{name} must be 2D");
        let rows = rec.shape[0];
        let cols = rec.shape[1];
        assert_eq!(s.d, cols, "{}: site d={} != host cols_in {cols}", name, s.d);
        assert_eq!(s.a.len(), s.r * s.d, "{}: A len", name);
        assert_eq!(s.b.len(), s.d * s.r, "{}: B len", name);
        let mh = acc.entry(name.clone()).or_insert_with(|| MergedHost {
            name: name.clone(),
            rows,
            cols,
            w: eng.vec_f32(&name),
        });
        // U[o,c] = <W_orig[o,:], b_c> against the UNMODIFIED host.
        let mut u = vec![0f32; mh.rows * s.r];
        for c in 0..s.r {
            let bcol = &s.b[c * s.d..(c + 1) * s.d];
            for o in 0..mh.rows {
                let wr = &mh.w[o * mh.cols..(o + 1) * mh.cols];
                let mut dot = 0f32;
                for (j, wv) in wr.iter().enumerate() {
                    dot += wv * bcol[j];
                }
                u[o * s.r + c] = dot;
            }
        }
        // W' += U · A
        for c in 0..s.r {
            let arow = &s.a[c * s.d..(c + 1) * s.d];
            for o in 0..mh.rows {
                let uc = u[o * s.r + c];
                if uc == 0.0 {
                    continue;
                }
                let wrow = &mut mh.w[o * mh.cols..(o + 1) * mh.cols];
                for (i, wv) in wrow.iter_mut().enumerate() {
                    *wv += uc * arow[i];
                }
            }
        }
    }
    acc.into_values().collect()
}


/// Requantize a merged host under its record's atom and return the packed bytes.
fn encode_atom(atom: &str, f32s: &[f32]) -> Vec<u8> {
    match atom {
        ATOM_Q8 => q8_encode(f32s),
        ATOM_F16 => f32s.iter().flat_map(|v| f32_to_f16(*v).to_le_bytes()).collect(),
        a => panic!("consolidate: unknown atom {a}"),
    }
}

/// Write a NEW pack mirroring src/bin/pack.rs layout: META | GRAPH | TOKENIZER | TENSORS.
/// Unmodified tensors are copied verbatim (digest recomputed — identical bytes, same digest);
/// folded hosts are re-encoded via atoms::q8_encode with shape carried over unchanged.
pub fn write_pack(
    eng: &Engine,
    merged: &[MergedHost],
    out_path: &str,
    gen: u32,
    consolidated_from: &[u8; 32],
) -> io::Result<[u8; 32]> {
    let pack: &PackReader = &eng.pack;
    let by_name: BTreeMap<&str, &MergedHost> =
        merged.iter().map(|m| (m.name.as_str(), m)).collect();

    // ---- records + payload ----
    let sec = pack.section(SEC_TENSORS).expect("source TENSORS section");
    let count = u32::from_le_bytes(sec[..4].try_into().unwrap()) as usize;
    let json_len = u32::from_le_bytes(sec[4..8].try_into().unwrap()) as usize;
    let src_recs: Vec<TensorRecord> =
        serde_json::from_slice(&sec[8..8 + json_len]).expect("source tensor records");
    assert_eq!(count, src_recs.len());
    let payloads_abs = pack.section_offset(SEC_TENSORS).unwrap() + 8 + json_len;

    let mut new_recs: Vec<TensorRecord> = Vec::with_capacity(src_recs.len());
    let mut payload: Vec<u8> = Vec::new();
    for r in &src_recs {
        let bytes: Vec<u8> = if let Some(m) = by_name.get(r.name.as_str()) {
            let numel: usize = r.shape.iter().product::<u32>() as usize;
            assert_eq!(
                m.rows * m.cols,
                numel,
                "{}: merged numel mismatch",
                r.name
            );
            encode_atom(&r.atom, &m.w)
        } else {
            let off = payloads_abs + r.offset as usize;
            pack.map[off..off + r.len as usize].to_vec()
        };
        let digest: [u8; 16] = blake3::hash(&bytes).as_bytes()[..16].try_into().unwrap();
        new_recs.push(TensorRecord {
            name: r.name.clone(),
            shape: r.shape.clone(),
            atom: r.atom.clone(),
            offset: payload.len() as u64,
            len: bytes.len() as u64,
            digest,
        });
        payload.extend_from_slice(&bytes);
    }

    // ---- sections ----
    let src_meta: Value = serde_json::from_slice(pack.section(SEC_META).expect("META")).unwrap();
    let mut meta = src_meta.clone();
    meta["atom_manifest"] = serde_json::json!(
        new_recs.iter().map(|r| (&r.name, &r.atom)).collect::<BTreeMap<_, _>>()
    );
    meta["generation"] = serde_json::json!(gen);
    meta["consolidated_from"] = serde_json::json!(hex(consolidated_from));
    meta["provenance"]["consolidator"] = serde_json::json!("model-lab/consolidate 0.1");

    let graph = pack.section(SEC_GRAPH).expect("GRAPH").to_vec();
    let tok = pack.section(SEC_TOKENIZER).expect("TOKENIZER").to_vec();
    let recs_json = serde_json::to_vec(&new_recs).unwrap();
    let mut tensors_sec = Vec::with_capacity(8 + recs_json.len() + payload.len());
    tensors_sec.extend_from_slice(&(new_recs.len() as u32).to_le_bytes());
    tensors_sec.extend_from_slice(&(recs_json.len() as u32).to_le_bytes());
    tensors_sec.extend_from_slice(&recs_json);
    tensors_sec.extend_from_slice(&payload);

    let mut w = SectionWriter::new(io::BufWriter::with_capacity(
        1 << 24,
        std::fs::File::create(out_path)?,
    ))?;
    w.section(SEC_META, 0, &serde_json::to_vec(&meta).unwrap())?;
    w.section(SEC_GRAPH, 0, &graph)?;
    w.section(SEC_TOKENIZER, 0, &tok)?;
    w.section(SEC_TENSORS, 0, &tensors_sec)?;
    let (cid, _size) = w.finish()?;
    Ok(cid)
}

/// Contract-frozen entry point: fold every adapter site into its host matrix and write
/// the new canonical generation to `out_path`. Returns the fresh content_id.
pub fn fold_sites(eng: &Engine, ad: &AdapterV2, out_path: &str, gen: u32) -> io::Result<[u8; 32]> {
    let merged = merged_hosts(eng, ad);
    let from = eng.pack.content_id();
    write_pack(eng, &merged, out_path, gen, &from)
}

// ---------------------------------------------------------------------------
// Equivalence-proof support: adapted inference on CPU (host-side apply).
// ---------------------------------------------------------------------------

/// Per-layer correction ΔW applied AFTER the base Q8 matmul (+bias), BEFORE the
/// following op (attention / GELU): y_corr = y_base + ΔW·tap. Exact adapted-model math
/// because y_adapted = W'(x) + b = (W_q8·x + b) + ΔW·x with ΔW = W' − dequant(W_q8).
pub struct LayerDeltas {
    pub attn: Vec<Option<(Vec<f32> /*ΔW*/, usize /*rows*/, usize /*cols*/)>>,
    pub mlp: Vec<Option<(Vec<f32>, usize, usize)>>,
}

impl LayerDeltas {
    pub fn build(eng: &Engine, merged: &[MergedHost]) -> Self {
        let n = eng.meta.n_layer;
        let mut attn: Vec<_> = vec![None; n];
        let mut mlp: Vec<_> = vec![None; n];
        for m in merged {
            let slot = if let Some(rest) = m.name.strip_prefix("h.") {
                let layer: usize =
                    rest.split('.').next().unwrap().parse().expect("layer no");
                if m.name.contains(".attn.c_attn.") {
                    Some((&mut attn, layer))
                } else if m.name.contains(".mlp.c_fc.") {
                    Some((&mut mlp, layer))
                } else {
                    None
                }
            } else {
                None
            };
            let Some((vec, l)) = slot else { continue };
            assert!(l < n);
            // True correction: merged f32 weight minus the base Q8-dequantized weight.
            let base = eng.vec_f32(&m.name);
            assert_eq!(base.len(), m.w.len(), "{}: numel", m.name);
            let mut dw: Vec<f32> = m.w.clone();
            for (d, b) in dw.iter_mut().zip(base.iter()) {
                *d -= b;
            }
            let zero = dw.iter().all(|&v| v == 0.0);
            vec[l] = if zero { None } else { Some((dw, m.rows, m.cols)) };
        }
        Self { attn, mlp }
    }
}

#[inline]
fn add_delta(dw: &[f32], x: &[f32], y: &mut [f32], cols: usize) {
    for (yo, yv) in y.iter_mut().enumerate() {
        let row = &dw[yo * cols..(yo + 1) * cols];
        let mut s = 0f32;
        for (i, xi) in x.iter().enumerate() {
            s += row[i] * xi;
        }
        *yv += s;
    }
}

/// Teacher-forced adapted forward mirroring gpt2.rs::step arithmetic exactly, with the
/// adapter folded in host-side at both taps. Returns logits per position.
pub fn adapted_forward(eng: &Engine, ids: &[u32], dh: &LayerDeltas) -> Vec<Vec<f32>> {
    let m = eng.meta.clone();
    let hd = m.n_embd / m.n_head;
    let mut kv: Vec<(Vec<f32>, Vec<f32>)> =
        vec![(vec![0f32; m.n_ctx * m.n_embd], vec![0f32; m.n_ctx * m.n_embd]); m.n_layer];
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(ids.len());

    for (pos, &tok) in ids.iter().enumerate() {
        let mut x = eng.vec_row("wte.weight", tok);
        let wpe = eng.vec_row("wpe.weight", pos as u32);
        for i in 0..m.n_embd {
            x[i] += wpe[i];
        }

        for layer in 0..m.n_layer {
            // ---- attention block ----
            let gw = eng.vec_f32(&format!("h.{layer}.ln_1.weight"));
            let gb = eng.vec_f32(&format!("h.{layer}.ln_1.bias"));
            let mut h = x.clone();
            eng.layernorm(&mut h, &gw, &gb, m.ln_eps);

            let qkv_name = format!("h.{layer}.attn.c_attn.weight");
            let inter3 = eng.t[&qkv_name].shape[0];
            let mut qkv = vec![0f32; inter3];
            eng.matmul(&qkv_name, &h, &mut qkv);
            if let Some((dw, rows, cols)) = &dh.attn[layer] {
                debug_assert_eq!(inter3, *rows);
                add_delta(dw, &h, &mut qkv, *cols);
            }
            let qb = eng.vec_f32(&format!("h.{layer}.attn.c_attn.bias"));
            for i in 0..inter3 {
                qkv[i] += qb[i];
            }

            {
                let (kc, vc) = &mut kv[layer];
                let row = pos * m.n_embd;
                kc[row..row + m.n_embd].copy_from_slice(&qkv[m.n_embd..2 * m.n_embd]);
                vc[row..row + m.n_embd].copy_from_slice(&qkv[2 * m.n_embd..]);
            }

            let scale = 1.0 / (hd as f32).sqrt();
            let mut attn_out = vec![0f32; m.n_embd];
            for head in 0..m.n_head {
                let o = head * hd;
                let q = &qkv[o..o + hd];
                let mut scores = vec![0f32; pos + 1];
                let mut maxs = f32::MIN;
                for p in 0..=pos {
                    let kp = &kv[layer].0[p * m.n_embd + o..p * m.n_embd + o + hd];
                    let mut dot = 0f32;
                    for tt in 0..hd {
                        dot += q[tt] * kp[tt];
                    }
                    scores[p] = dot * scale;
                    maxs = maxs.max(scores[p]);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - maxs).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }
                for p in 0..=pos {
                    let vp = &kv[layer].1[p * m.n_embd + o..p * m.n_embd + o + hd];
                    for tt in 0..hd {
                        attn_out[o + tt] += scores[p] * vp[tt];
                    }
                }
            }

            let proj_name = format!("h.{layer}.attn.c_proj.weight");
            let mut proj = vec![0f32; m.n_embd];
            eng.matmul(&proj_name, &attn_out, &mut proj);
            let pb = eng.vec_f32(&format!("h.{layer}.attn.c_proj.bias"));
            for i in 0..m.n_embd {
                x[i] += proj[i] + pb[i];
            }

            // ---- mlp block ----
            let gw2 = eng.vec_f32(&format!("h.{layer}.ln_2.weight"));
            let gb2 = eng.vec_f32(&format!("h.{layer}.ln_2.bias"));
            let mut h2 = x.clone();
            eng.layernorm(&mut h2, &gw2, &gb2, m.ln_eps);
            let fc_name = format!("h.{layer}.mlp.c_fc.weight");
            let inter = eng.t[&fc_name].shape[0];
            let mut fc = vec![0f32; inter];
            eng.matmul(&fc_name, &h2, &mut fc);
            if let Some((dw, rows, cols)) = &dh.mlp[layer] {
                debug_assert_eq!(inter, *rows);
                add_delta(dw, &h2, &mut fc, *cols);
            }
            let fb = eng.vec_f32(&format!("h.{layer}.mlp.c_fc.bias"));
            for i in 0..inter {
                let v = fc[i] + fb[i];
                fc[i] = 0.5 * v * (1.0 + (0.7978845608028654 * (v + 0.044715 * v * v * v)).tanh());
            }
            let mo_name = format!("h.{layer}.mlp.c_proj.weight");
            let mut mo = vec![0f32; m.n_embd];
            eng.matmul(&mo_name, &fc, &mut mo);
            let mb = eng.vec_f32(&format!("h.{layer}.mlp.c_proj.bias"));
            for i in 0..m.n_embd {
                x[i] += mo[i] + mb[i];
            }
        }

        let wf = eng.vec_f32("ln_f.weight");
        let bf = eng.vec_f32("ln_f.bias");
        eng.layernorm(&mut x, &wf, &bf, m.ln_eps);
        let rows = m.vocab;
        let mut logits = vec![0f32; rows];
        eng.matmul("wte.weight", &x, &mut logits);
        out.push(logits);
    }
    out
}

/// Compare two logit sequences: returns (global max|Δlogit|, argmax agreement fraction).
pub fn compare_logits(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f32, f64) {
    assert_eq!(a.len(), b.len());
    let mut maxd = 0f32;
    let mut agree = 0usize;
    let mut total = 0usize;
    for (la, lb) in a.iter().zip(b.iter()) {
        maxd = maxd.max(
            la.iter()
                .zip(lb.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max),
        );
        if la
            .iter()
            .enumerate()
            .max_by(|(_, p), (_, q)| p.total_cmp(q))
            .map(|(i, _)| i)
            == lb
                .iter()
                .enumerate()
                .max_by(|(_, p), (_, q)| p.total_cmp(q))
                .map(|(i, _)| i)
        {
            agree += 1;
        }
        total += 1;
    }
    (maxd, agree as f64 / total.max(1) as f64)
}

// ---------------------------------------------------------------------------
// Fallback inline delta format (Agent D local, documented):
//   SMT sections; META JSON {
//     "kind": "consolidate-inline-v1", "base_content_id": <hex>,
//     "sites": [{"kind":"attn_in"|"mlp_in","layer":L,"d":768,"r":R,"fact_ids":[...]}]
//   }
//   TENSORS records per site i:
//     "site.{i}.A" shape [r, d] atom "core.f32.raw" slice-major, slices row-major
//     "site.{i}.B" shape [d, r] atom "core.f32.raw" COLUMN-major (b[col*d + row])
//   digests blake3[..16], offsets into one contiguous payload region.
// Used only when delta_v2::load fails (integration race); folding semantics identical.
// ---------------------------------------------------------------------------

pub fn save_inline(ad: &AdapterV2, path: &str, base_cid: &[u8; 32]) -> io::Result<()> {
    fn raw(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    let mut sites_meta = Vec::new();
    let mut recs: Vec<TensorRecord> = Vec::new();
    let mut payload: Vec<u8> = Vec::new();
    for (i, s) in ad.sites.iter().enumerate() {
        let kind_s = match s.kind {
            SiteKind::AttnIn => "attn_in",
            SiteKind::MlpIn => "mlp_in",
        };
        let fact_ids: Vec<u64> = s.route.cols.iter().map(|(f, _, _)| *f).collect();
        sites_meta.push(serde_json::json!({
            "kind": kind_s, "layer": s.layer, "d": s.d, "r": s.r, "fact_ids": fact_ids,
        }));
        for (suffix, bytes, shape) in [
            ("A", raw(&s.a), vec![s.r as u32, s.d as u32]),
            ("B", raw(&s.b), vec![s.d as u32, s.r as u32]),
        ] {
            let digest: [u8; 16] = blake3::hash(&bytes).as_bytes()[..16].try_into().unwrap();
            recs.push(TensorRecord {
                name: format!("site.{i}.{suffix}"),
                shape,
                atom: "core.f32.raw".into(),
                offset: payload.len() as u64,
                len: bytes.len() as u64,
                digest,
            });
            payload.extend_from_slice(&bytes);
        }
    }
    let meta = serde_json::json!({
        "kind": INLINE_FORMAT_KIND,
        "base_content_id": hex(base_cid),
        "sites": sites_meta,
    });
    let recs_json = serde_json::to_vec(&recs).unwrap();
    let mut sec = Vec::new();
    sec.extend_from_slice(&(recs.len() as u32).to_le_bytes());
    sec.extend_from_slice(&(recs_json.len() as u32).to_le_bytes());
    sec.extend_from_slice(&recs_json);
    sec.extend_from_slice(&payload);
    let mut w = SectionWriter::new(io::BufWriter::with_capacity(
        1 << 20,
        std::fs::File::create(path)?,
    ))?;
    w.section(SEC_META, 0, &serde_json::to_vec(&meta).unwrap())?;
    w.section(SEC_TENSORS, 0, &sec)?;
    w.finish()?;
    Ok(())
}

/// Load Agent D's inline fallback format into an AdapterV2 (all facts routed per site).
pub fn load_inline(path: &str, eng: &Engine) -> Result<AdapterV2, String> {
    let raw = std::fs::read(path).map_err(|e| e.to_string())?;
    let mut off = 128usize;
    let mut meta_json: Option<Value> = None;
    let mut tensors: Option<&[u8]> = None;
    while off + 16 <= raw.len() {
        let ty = u32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
        let len = u64::from_le_bytes(raw[off + 8..off + 16].try_into().unwrap()) as usize;
        let p = off + 16;
        if p + len > raw.len() {
            break;
        }
        match ty {
            SEC_META => meta_json = serde_json::from_slice(&raw[p..p + len]).ok(),
            SEC_TENSORS => tensors = Some(&raw[p..p + len]),
            _ => {}
        }
        off = p + len;
    }
    let mj = meta_json.ok_or("inline delta missing META")?;
    if mj["kind"].as_str() != Some(INLINE_FORMAT_KIND) {
        return Err(format!("not an {} file", INLINE_FORMAT_KIND));
    }
    let cid_hex = hex(&eng.pack.content_id());
    if mj["base_content_id"].as_str() != Some(cid_hex.as_str()) {
        return Err(format!(
            "inline delta binds to {} but engine is {}",
            mj["base_content_id"].as_str().unwrap_or("?"),
            cid_hex
        ));
    }
    let t = tensors.ok_or("inline delta missing TENSORS")?;
    let cnt = u32::from_le_bytes(t[..4].try_into().unwrap()) as usize;
    let jl = u32::from_le_bytes(t[4..8].try_into().unwrap()) as usize;
    let recs: Vec<TensorRecord> =
        serde_json::from_slice(&t[8..8 + jl]).map_err(|e| e.to_string())?;
    if cnt != recs.len() {
        return Err("record count mismatch".into());
    }
    let data = 8 + jl;
    let mut by_name: BTreeMap<String, &TensorRecord> =
        recs.iter().map(|r| (r.name.clone(), r)).collect();
    let f32s = |r: &TensorRecord| -> Result<Vec<f32>, String> {
        let s = data + r.offset as usize;
        let e = s + r.len as usize;
        if e > t.len() {
            return Err(format!("{} out of range", r.name));
        }
        let body = &t[s..e];
        let dg: [u8; 16] = blake3::hash(body).as_bytes()[..16].try_into().unwrap();
        if dg != r.digest {
            return Err(format!("digest mismatch {}", r.name));
        }
        Ok(body.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    };
    let desc = mj["sites"]
        .as_array()
        .ok_or("inline META missing sites array")?;
    let mut sites: Vec<SiteAdapter> = Vec::with_capacity(desc.len());
    for (i, sd) in desc.iter().enumerate() {
        let kind = match sd["kind"].as_str() {
            Some("attn_in") => SiteKind::AttnIn,
            Some("mlp_in") => SiteKind::MlpIn,
            other => return Err(format!("site {i}: bad kind {other:?}")),
        };
        let layer = sd["layer"].as_u64().ok_or("missing layer")? as usize;
        let d = sd["d"].as_u64().ok_or("missing d")? as usize;
        let r = sd["r"].as_u64().ok_or("missing r")? as usize;
        let mut fact_ids: Vec<u64> = sd["fact_ids"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_else(|| vec![1]);
        if fact_ids.is_empty() {
            fact_ids.push(1);
        }
        let ra = by_name.remove(&format!("site.{i}.A")).ok_or("missing A")?;
        let rb = by_name.remove(&format!("site.{i}.B")).ok_or("missing B")?;
        let a = f32s(ra)?;
        let b = f32s(rb)?;
        if a.len() != r * d || b.len() != d * r {
            return Err(format!("site {i}: A/B length mismatch"));
        }
        // Route: contiguous blocks covering 0..r, one per fact id (fallback semantics).
        let per = if fact_ids.is_empty() { r } else { r / fact_ids.len().max(1) };
        let mut cols = Vec::new();
        let mut cur = 0usize;
        for (k, fid) in fact_ids.iter().enumerate() {
            let end = if k + 1 == fact_ids.len() {
                r
            } else {
                cur + per
            };
            cols.push((*fid, cur, end));
            cur = end;
        }
        sites.push(SiteAdapter {
            kind,
            layer,
            d,
            r,
            route: RouteSpec { cols },
            a,
            b,
        });
    }
    Ok(AdapterV2::new(sites))
}
