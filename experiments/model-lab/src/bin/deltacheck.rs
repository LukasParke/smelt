//! deltacheck — Agent C validation gate (routed adapter v2 + delta v2 format).
//!
//! Checks, all deterministic:
//!   1. build a synthetic 24-site AdapterV2 for gpt2-small dims (d=768, 12 layers × {AttnIn,MlpIn},
//!      r=8 split across 3 disjoint fact_id routes)
//!   2. routing isolation: inactive-fact slices contribute EXACTLY zero (bit-exact invariance
//!      under perturbing inactive A/B entries)
//!   3. active-set toggling: empty set is a no-op; per-fact contributions match an independent
//!      f64 reference; additivity over the full active set
//!   4. grad masking: mask_grads zeroes exactly the inactive slices' ga/gb entries
//!   5. save/load round-trip is bit-exact, incl. generation counter and routes
//!   6. base-cid binding rejection when loaded against an engine with a different content_id
//!   7. digest tamper detection — both raw container tamper AND record-digest tamper with a
//!      repaired merkle root
//!
//! Exits nonzero if any check fails.

use model_lab::adapter_v2::{AdapterV2, RouteSpec, SiteAdapter, SiteKind};
use model_lab::delta_v2;
use model_lab::format::{PackReader, SectionWriter, TensorRecord, SEC_META, SEC_TENSORS, SEC_TOKENIZER};
use model_lab::gpt2::Engine;

const D: usize = 768;
const N_LAYER: usize = 12;
const R: usize = 8;
const SEED: u64 = 0xC0FF_EE20_2608_2400;

// ---- deterministic PRNG (splitmix64) ----
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z ^= z >> 30;
        z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z ^= z >> 27;
        z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        z
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next_u64() as f32 / u64::MAX as f32) * (hi - lo)
    }
}

fn build_routes() -> RouteSpec {
    // r=8 across 3 fact_ids with disjoint ranges
    RouteSpec { cols: vec![(1, 0, 3), (2, 3, 6), (3, 6, 8)] }
}

fn build_adapter() -> AdapterV2 {
    let mut sites = Vec::with_capacity(N_LAYER * 2);
    for l in 0..N_LAYER {
        sites.push(SiteAdapter::new(SiteKind::AttnIn, l, D, build_routes()));
        sites.push(SiteAdapter::new(SiteKind::MlpIn, l, D, build_routes()));
    }
    let mut ad = AdapterV2::new(sites);
    ad.zero_init(SEED);
    // zero_init leaves B at zeros (correct init semantics); fill B deterministically so
    // routing/toggle checks observe nonzero contributions.
    let mut rng = Rng::new(SEED ^ 0xABCD);
    for s in ad.sites.iter_mut() {
        for v in s.b.iter_mut() {
            *v = rng.uniform(-0.05, 0.05);
        }
    }
    ad
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

/// Copy of `ad` whose INACTIVE (fact != 1) A/B entries are perturbed.
fn perturb_inactive(ad: &AdapterV2, seed: u64) -> AdapterV2 {
    let mut out = ad.clone();
    let mut rng = Rng::new(seed);
    for s in out.sites.iter_mut() {
        for (i, &(fid, st, en)) in s.route.cols.iter().enumerate() {
            if fid == 1 {
                continue;
            }
            let w = en - st;
            let aoff: usize =
                s.route.cols[..i].iter().map(|&(_, a, b)| (b - a) * s.d).sum();
            for v in s.a[aoff..aoff + w * s.d].iter_mut() {
                *v += rng.uniform(-0.01, 0.01);
            }
            for v in s.b[st * s.d..en * s.d].iter_mut() {
                *v += rng.uniform(-0.05, 0.05);
            }
        }
    }
    out
}

/// Independent f64 reference contribution of `active` slices of one site.
fn reference_contribution(site: &SiteAdapter, x: &[f32], active: &[u64]) -> Vec<f64> {
    let mut contrib = vec![0.0f64; site.d];
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    for &(fid, st, _en) in site.route.cols.iter() {
        if !active.contains(&fid) {
            continue;
        }
        // iterate this slice's rows
        let width = site.route.cols.iter().find(|c| c.0 == fid).unwrap().2 - st;
        for srow in 0..width {
            let row_off = {
                let mut off = 0usize;
                for &(f2, s2, e2) in site.route.cols.iter() {
                    if f2 == fid {
                        break;
                    }
                    off += (e2 - s2) * site.d;
                }
                off + srow * site.d
            };
            let u: f64 = (0..site.d)
                .map(|j| site.a[row_off + j] as f64 * xf[j])
                .sum();
            let col = st + srow;
            for o in 0..site.d {
                contrib[o] += site.b[col * site.d + o] as f64 * u;
            }
        }
    }
    contrib
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}

// ---- synthetic base pack usable by Engine::load ----
fn write_base_pack(path: &str, filler: u8) -> std::io::Result<()> {
    let meta = serde_json::json!({
        "arch": {"n_embd": D, "n_layer": N_LAYER, "n_head": 12, "n_ctx": 1024,
                 "vocab_size": 50257, "ln_eps": 1e-5},
    });
    let tok: &[u8] =
        r#"{"model":{"vocab":{"Ġa":0,"b":1},"merges":[]}}"#.as_bytes();
    // one dummy tensor; filler byte varies between packs => different digests => different cid
    let payload = vec![filler; D * 2]; // core.f16, shape [1, D]
    let rec = TensorRecord {
        name: "dummy.weight".into(),
        shape: vec![1, D as u32],
        atom: "core.f16".into(),
        offset: 0,
        len: payload.len() as u64,
        digest: blake3::hash(&payload).as_bytes()[..16].try_into().unwrap(),
    };
    let recs_json = serde_json::to_vec(&vec![rec]).unwrap();
    let mut sec = Vec::new();
    sec.extend_from_slice(&1u32.to_le_bytes());
    sec.extend_from_slice(&(recs_json.len() as u32).to_le_bytes());
    sec.extend_from_slice(&recs_json);
    sec.extend_from_slice(&payload);

    let f = std::fs::File::create(path)?;
    let mut w = SectionWriter::new(std::io::BufWriter::with_capacity(1 << 16, f))?;
    w.section(SEC_META, 0, &serde_json::to_vec(&meta).unwrap())?;
    w.section(SEC_TOKENIZER, 0, tok)?;
    w.section(SEC_TENSORS, 0, &sec)?;
    w.finish()?;
    Ok(())
}

/// Recompute section digests / INDEX / merkle content_id after an in-place payload edit,
/// so ONLY record-level digests are violated (exercises delta_v2's own digest check).
fn repair_merkle(path: &str) -> std::io::Result<()> {
    let mut buf = std::fs::read(path)?;
    let mut digests: Vec<(u32, [u8; 32])> = Vec::new();
    let mut index_range: Option<(usize, usize)> = None; // payload [start,end)
    let mut off = 128usize;
    while off + 16 <= buf.len() {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let len = u64::from_le_bytes(buf[off + 8..off + 16].try_into().unwrap()) as usize;
        let p = off + 16;
        let d: [u8; 32] = blake3::hash(&buf[p..p + len]).as_bytes()[..32].try_into().unwrap();
        if ty == crate_index_ty() {
            index_range = Some((p, p + len));
        }
        digests.push((ty, d));
        off = p + len;
    }
    // writer semantics: INDEX lists every section BEFORE it (not itself)
    let mut idx = Vec::new();
    let index_ty = crate_index_ty();
    for (ty, d) in digests.iter().filter(|(ty, _)| *ty != index_ty) {
        idx.extend_from_slice(&ty.to_le_bytes());
        idx.extend_from_slice(d);
    }
    let (s, e) = index_range.expect("INDEX section");
    assert_eq!(e - s, idx.len(), "INDEX length changed");
    buf[s..e].copy_from_slice(&idx);
    // merkle covers ALL sections incl. the just-rewritten INDEX digest
    let idx_dg: [u8; 32] = blake3::hash(&idx).as_bytes()[..32].try_into().unwrap();
    let mut sorted: Vec<[u8; 32]> = digests
        .into_iter()
        .map(|(ty, d)| if ty == index_ty { idx_dg } else { d })
        .collect();
    sorted.sort();
    let mut h = blake3::Hasher::new();
    for d in sorted {
        h.update(&d);
    }
    let cid: [u8; 32] = h.finalize().into();
    buf[16..48].copy_from_slice(&cid);
    std::fs::write(path, buf)
}

fn crate_index_ty() -> u32 {
    model_lab::format::SEC_INDEX
}

fn main() {
    let tmp = "/tmp/deltacheck_work";
    let _ = std::fs::create_dir_all(tmp);
    let mut failures: Vec<String> = Vec::new();
    macro_rules! check {
        ($name:expr, $cond:expr, $detail:expr) => {{
            let ok: bool = $cond;
            println!(
                "[{}] {} {}",
                if ok { "PASS" } else { "FAIL" },
                $name,
                $detail
            );
            if !ok {
                failures.push($name.to_string());
            }
        }};
    }

    // ---- fixture: synthetic base packs + engines ----
    let pack_a = format!("{tmp}/base_a.smt");
    let pack_b = format!("{tmp}/base_b.smt");
    write_base_pack(&pack_a, 0xAA).expect("write base_a");
    write_base_pack(&pack_b, 0xBB).expect("write base_b");
    let eng_a = Engine::load(&pack_a);
    let eng_b = Engine::load(&pack_b);
    let cid_a = eng_a.pack.content_id();

    // ---- 1. construction ----
    let ad = build_adapter();
    check!(
        "build_adapter",
        ad.sites.len() == 24 && ad.sites.iter().all(|s| s.r == R && s.d == D),
        format!("{} sites, r={R}, d={D}", ad.sites.len())
    );
    check!(
        "routes_valid",
        ad.sites.iter().all(|s| s.route.validate(R).is_ok()
            && s.route.fact_ids() == vec![1u64, 2, 3]),
        "fact_ids [1,2,3] widths 3+3+2=8"
    );

    // ---- 2/3. routing on every site kind ----
    let x = rand_vec(D, SEED ^ 1);
    let site_idx = ad.site_index(SiteKind::AttnIn, 3).unwrap();
    let mlp_idx = ad.site_index(SiteKind::MlpIn, 7).unwrap();

    let mut y_active = x.clone();
    ad.apply_at(site_idx, &mut y_active, &[1]);
    let delta_active: Vec<f32> =
        y_active.iter().zip(x.iter()).map(|(&y, &x)| y - x).collect();
    check!(
        "contribution_nonzero",
        max_abs(&delta_active) > 1e-5,
        format!("max|Δ| with fact 1 = {:.3e}", max_abs(&delta_active))
    );

    // bit-exact invariance under perturbing INACTIVE facts
    let ad_pert = perturb_inactive(&ad, SEED ^ 2);
    let mut y_pert = x.clone();
    ad_pert.apply_at(site_idx, &mut y_pert, &[1]);
    check!(
        "inactive_columns_exactly_zero",
        bits(&y_active) == bits(&y_pert),
        format!("max|Δ| under inactive perturbation = {:.3e}",
            max_abs(&(0..D).map(|i| y_active[i] - y_pert[i]).collect::<Vec<_>>()))
    );

    // empty active set = exact no-op
    let mut y_none = x.clone();
    ad.apply_at(mlp_idx, &mut y_none, &[]);
    check!(
        "empty_active_noop",
        bits(&y_none) == bits(&x),
        "bit-identical"
    );

    // independent f64 reference vs single-fact apply, for both site kinds
    for (nm, si) in [("attn", site_idx), ("mlp", mlp_idx)] {
        let mut y1 = x.clone();
        ad.apply_at(si, &mut y1, &[1]);
        let got: Vec<f32> = y1.iter().zip(x.iter()).map(|(&y, &x)| y - x).collect();
        let want = reference_contribution(&ad.sites[si], &x, &[1]);
        let err: f64 = got
            .iter()
            .zip(want.iter())
            .map(|(&g, &w)| (g as f64 - w).abs())
            .fold(0.0, f64::max);
        check!(
            format!("reference_match_{nm}"),
            err < 1e-4,
            format!("max abs err vs f64 ref = {err:.3e}")
        );
    }

    // toggle to other active sets changes output; full-set additivity
    let mut y_full = x.clone();
    ad.apply_at(site_idx, &mut y_full, &[1, 2, 3]);
    let full_delta: Vec<f64> = y_full
        .iter()
        .zip(x.iter())
        .map(|(&y, &x)| (y - x) as f64)
        .collect();
    let sum_ref = reference_contribution(&ad.sites[site_idx], &x, &[1, 2, 3]);
    let err: f64 = full_delta
        .iter()
        .zip(sum_ref.iter())
        .map(|(&g, &w)| (g - w).abs())
        .fold(0.0, f64::max);
    check!(
        "full_set_reference_match",
        err < 1e-4 && bits(&y_full) != bits(&y_active),
        format!("max abs err vs f64 ref = {err:.3e}; toggling set changes output")
    );

    // ---- 4. grad masking ----
    {
        let s = &ad.sites[site_idx];
        let mut ga = vec![1.0f32; s.a.len()];
        let mut gb = vec![1.0f32; s.b.len()];
        s.mask_grads(&mut ga, &mut gb, &[1]);
        let slice_a_zero = |i: usize| {
            let off: usize = s.route.cols[..i].iter().map(|&(_, a, b)| (b - a) * D).sum();
            ga[off..off + (s.route.cols[i].2 - s.route.cols[i].1) * D]
                .iter()
                .all(|&v| v == 0.0)
        };
        let slice_b_zero = |i: usize| {
            let (st, en) = (s.route.cols[i].1, s.route.cols[i].2);
            gb[st * D..en * D].iter().all(|&v| v == 0.0)
        };
        let first_kept = {
            let w = s.route.width(0) * D;
            ga[..w].iter().all(|&v| v == 1.0)
        };
        check!(
            "mask_grads",
            slice_a_zero(1) && slice_a_zero(2) && slice_b_zero(1) && slice_b_zero(2)
                && first_kept && gb[..D].iter().all(|&v| v == 1.0),
            "inactive ga/gb entries exactly 0, active untouched"
        );
    }

    // ---- 5. save/load round-trip ----
    let delta_path = format!("{tmp}/delta.bin");
    delta_v2::save_gen(&delta_path, &cid_a, &ad, 7).expect("save delta");
    let loaded = delta_v2::load(&delta_path, &eng_a).expect("load delta");
    let rt_equal = ad.sites.len() == loaded.sites.len()
        && ad.sites.iter().zip(loaded.sites.iter()).all(|(a, b)| {
            a.kind == b.kind
                && a.layer == b.layer
                && a.d == b.d
                && a.r == b.r
                && a.route.cols == b.route.cols
                && bits(&a.a) == bits(&b.a)
                && bits(&b.b).len() == b.b.len()
                    && bits(&a.b) == bits(&b.b)
        });
    check!("roundtrip_bitexact", rt_equal, "24 sites, all fields bit-exact");

    let mj = delta_v2::meta_of(&delta_path).expect("meta_of");
    check!(
        "generation_counter",
        mj["generation"].as_u64() == Some(7)
            && mj["kind"].as_str() == Some("adapter_v2")
            && mj["sites"].as_array().map(|a| a.len()) == Some(24),
        format!("gen={} sites={}", mj["generation"], mj["sites"].as_array().map(|a| a.len()).unwrap_or(0))
    );

    // ---- 6. base-cid binding rejection ----
    match delta_v2::load(&delta_path, &eng_b) {
        Err(e) if e.contains("binds to") || e.contains("engine is") => check!(
            "binding_rejection",
            true,
            format!("rejected fake-cid engine: \"{e}\"")
        ),
        Err(e) => {
            check!("binding_rejection", false, format!("wrong error: {e}"));
        }
        Ok(_) => check!("binding_rejection", false, "load unexpectedly SUCCEEDED"),
    }

    // ---- 7a. raw container tamper detection ----
    let tamp_raw = format!("{tmp}/delta_tamper_raw.bin");
    std::fs::copy(&delta_path, &tamp_raw).unwrap();
    {
        let mut buf = std::fs::read(&tamp_raw).unwrap();
        // flip one payload byte inside TENSORS tensor data (past count+json_len+records json)
        let pr = PackReader::open(&tamp_raw).unwrap();
        let t_off = pr.section_offset(SEC_TENSORS).unwrap();
        drop(pr);
        // section_offset returns the PAYLOAD start: count@0, json_len@4, records json, data
        let jl = u32::from_le_bytes(buf[t_off + 4..t_off + 8].try_into().unwrap()) as usize;
        let victim = t_off + 8 + jl + 100;
        buf[victim] ^= 0x01;
        std::fs::write(&tamp_raw, buf).unwrap();
    }
    match delta_v2::load(&tamp_raw, &eng_a) {
        Err(_) => check!("tamper_container_detected", true, "raw byte flip rejected"),
        Ok(_) => check!("tamper_container_detected", false, "tampered file LOADED"),
    }

    // ---- 7b. record-digest tamper with repaired merkle ----
    let tamp_rec = format!("{tmp}/delta_tamper_record.bin");
    std::fs::copy(&delta_path, &tamp_rec).unwrap();
    {
        let mut buf = std::fs::read(&tamp_rec).unwrap();
        let pr = PackReader::open(&tamp_rec).unwrap();
        let t_off = pr.section_offset(SEC_TENSORS).unwrap();
        drop(pr);
        let jl = u32::from_le_bytes(buf[t_off + 4..t_off + 8].try_into().unwrap()) as usize;
        let victim = t_off + 8 + jl + 100;
        buf[victim] = buf[victim].wrapping_add(1); // alter one f32 LSB
        std::fs::write(&tamp_rec, buf).unwrap();
        repair_merkle(&tamp_rec).expect("repair merkle");
        // sanity: repaired container verifies...
        assert!(PackReader::open(&tamp_rec).unwrap().verify().is_ok(), "merkle repair failed");
    }
    match delta_v2::load(&tamp_rec, &eng_a) {
        Err(e) if e.contains("digest mismatch") => check!(
            "tamper_record_digest_detected",
            true,
            format!("record-level digest caught it: \"{e}\"")
        ),
        Err(e) => check!("tamper_record_digest_detected", false, format!("wrong error: {e}")),
        Ok(_) => check!("tamper_record_digest_detected", false, "tampered file LOADED"),
    }

    // ---- summary ----
    println!("\n{}", "-".repeat(60));
    if failures.is_empty() {
        println!("deltacheck: ALL CHECKS PASSED");
    } else {
        println!("deltacheck: {} FAILURE(S): {:?}", failures.len(), failures);
        std::process::exit(1);
    }
}
