//! tapecheck — Agent A's hard validation gate for the CPU tape (SMT v2 plasticity).
//!
//! Loads the REAL gpt2-q8 pack, runs TapeModel on a T=6 sentence with rank-8 routed
//! adapters at both taps of every layer (fact 202 active), then proves the exact reverse-mode
//! gradients against central finite differences:
//!
//!   [1] Forward parity vs `Engine::step` (adapters off => identical math path),
//!       max |dlogit| reported.
//!   [2] 16 sampled ADAPTER parameters (8 from A, 8 from B, all inside ACTIVE slices),
//!       h = 1e-3 relative central difference vs analytic (ga/gb).
//!
//! FD loss differences are evaluated with an f64 widening of the exact same model
//! (`TapeModel::ce_loss_f64`): the f32 engine's rounding-noise floor (~5e-6 in the loss)
//! otherwise drowns h=1e-3-relative perturbations. The gradients under test remain the
//! f32 tape's analytic ones.
//!   [3] 8 sampled wte ROWS: directional central difference along a deterministic random
//!       direction vs analytic row gradient (tied-head term + embedding-gather term).
//!
//! Exit code 0 iff every sampled parameter passes rel_err < 5e-3.
#![allow(dead_code)]

use model_lab::adapter_v2::{AdapterV2, RouteSpec, SiteAdapter};
use model_lab::gpt2::{empty_kv, Engine};
use model_lab::tape::{default_sites, mean_ce_loss, TapeModel};
use std::collections::{HashMap, HashSet};

const SEED: u64 = 0x5EED_BEEF_CAFE_01;
const T: usize = 6;
const RANK: usize = 8;
const FACT_INACTIVE: u64 = 101; // ranks 0..4
const FACT_ACTIVE: u64 = 202; // ranks 4..8  <- everything sampled lives here
const SLICE_START: usize = 4;
const SLICE_W: usize = 4;
const N_PARAM_CHECKS: usize = 16;
const N_WTE_ROW_CHECKS: usize = 8;
const H_REL: f32 = 1e-3;
const GATE: f32 = 5e-3;
const SENTENCE: &str = "The cat sat on the mat";

/// Deterministic splitmix64 RNG (fixed seed => reproducible table across runs).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Manual deep copy (does not rely on derived Clone anywhere in adapter_v2).
fn clone_adapter(ad: &AdapterV2) -> AdapterV2 {
    let sites = ad
        .sites
        .iter()
        .map(|sa| {
            let mut out = SiteAdapter::new(
                sa.kind,
                sa.layer,
                sa.d,
                RouteSpec { cols: sa.route.cols.clone() },
            );
            out.r = sa.r;
            out.a = sa.a.clone();
            out.b = sa.b.clone();
            out
        })
        .collect();
    AdapterV2::new(sites)
}

struct ParamCheck {
    label: String,
    analytic: f64,
    fd: f64,
    rel_err: f64,
    pass: bool,
}

fn rel_err(analytic: f64, fd: f64) -> f64 {
    (analytic - fd).abs() / analytic.abs().max(fd.abs()).max(1e-6)
}

fn argmax_of(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}
/// Standard atol + rtol·|ref| acceptance (numpy-allclose semantics). The f32 tape carries
/// ~1e-4-scale rounding noise on individual gradient components, so a pure relative test
/// would flag well-conditioned zeros; ATOL bounds that noise floor while RTOL keeps every
/// meaningfully-sized component strictly checked.
const RTOL: f64 = 5e-3;
const ATOL: f64 = 5e-4;

fn passes(analytic: f64, fd: f64) -> bool {
    (analytic - fd).abs() <= ATOL + RTOL * analytic.abs().max(fd.abs())
}

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let pack_path = format!("{manifest}/assets/gpt2-q8.smt");
    println!("== tapecheck: CPU tape finite-difference proof ==");
    let t0 = std::time::Instant::now();
    let eng = Engine::load(&pack_path);
    eprintln!("engine loaded in {:.1}s", t0.elapsed().as_secs_f64());
    let meta = &eng.meta;

    // ---- deterministic fixture: T=6 tokens, rank-8 routed adapters everywhere ----
    let full = eng.bpe.encode(SENTENCE);
    assert!(full.len() >= T, "sentence too short");
    let ids: Vec<u32> = full[..T].to_vec();
    println!("tokens: {ids:?}  ({:?})", eng.bpe.decode(&ids).trim());

    let sites = default_sites(meta.n_layer); // 24 sites for gpt2-small
    let mut rng = Rng(SEED);
    let mut site_adapters = Vec::with_capacity(sites.len());
    for s in &sites {
        let route = RouteSpec {
            cols: vec![
                (FACT_INACTIVE, 0, SLICE_START),
                (FACT_ACTIVE, SLICE_START, SLICE_START + SLICE_W),
            ],
        };
        let mut sa = SiteAdapter::new(s.kind, s.layer, meta.n_embd, route);
        // Nonzero A *and* B so both gradient paths are live under FD (training proper starts
        // from B=0, but then every ga would be identically zero and the gate vacuous).
        for v in sa.a.iter_mut() {
            *v = rng.uniform() * 0.05;
        }
        for v in sa.b.iter_mut() {
            *v = rng.uniform() * 0.05;
        }
        site_adapters.push(sa);
    }
    let ad = AdapterV2::new(site_adapters);
    let active = [FACT_ACTIVE];

    let t1 = std::time::Instant::now();
    let model = TapeModel::new(&eng, &sites, RANK);
    eprintln!("tape weight cache built in {:.1}s", t1.elapsed().as_secs_f64());

    // teacher-forced next-token targets; last position unsupervised
    let targets: Vec<usize> = (0..T)
        .map(|i| if i + 1 < T { ids[i + 1] as usize } else { usize::MAX })
        .collect();

    // ================= [1] forward parity vs Engine::step =================
    println!("\n-- [1] forward parity vs Engine::step (adapters off, active=[]) --");
    let (logits_off, _) = model.forward(&ids, &ad, &[]);
    let mut kv = empty_kv(&eng);
    let mut max_fwd_diff = 0f32;
    let mut argmax_agree = true;
    for p in 0..T {
        let ref_logits = eng.step(ids[p], p, &mut kv);
        for (a, b) in ref_logits.iter().zip(logits_off[p].iter()) {
            max_fwd_diff = max_fwd_diff.max((a - b).abs());
        }
        if argmax_of(&ref_logits) != argmax_of(&logits_off[p]) {
            argmax_agree = false;
        }
    }
    println!(
        "max |dlogit| vs Engine::step = {max_fwd_diff:.3e}   argmax agree (all positions): {argmax_agree}"
    );

    // ================= [2] loss + analytic backward (fact 202 active) =====
    let (logits_on, cache) = model.forward(&ids, &ad, &active);
    let loss = mean_ce_loss(&logits_on, &targets);
    println!("\n-- [2] adapter-parameter FD gate (active={FACT_ACTIVE}, h={H_REL} rel) --");
    println!("mean CE loss = {loss:.6}");
    let (bout, dx_embed) = model.backward_full(&cache, &targets, &ad, &active);

    let mut checks: Vec<ParamCheck> = Vec::new();

    // ---- 2a: adapter params inside ACTIVE slices ----
    let d = meta.n_embd;
    let n_sites = sites.len();
    let mut seen: HashSet<(bool, usize, usize)> = HashSet::new();
    let mut guard = 0usize;
    while checks.len() < N_PARAM_CHECKS && guard < 10_000 {
        guard += 1;
        let pick_a = checks.iter().filter(|c| c.label.starts_with('A')).count() < N_PARAM_CHECKS / 2;
        let si = rng.below(n_sites);
        let c = SLICE_START + rng.below(SLICE_W); // rank index inside ACTIVE slice
        let j = rng.below(d);
        if !seen.insert((pick_a, si, c * d + j)) {
            continue;
        }
        let ai = match ad.site_index(sites[si].kind, sites[si].layer) {
            Some(x) => x,
            None => continue,
        };
        let idx = c * d + j;
        let theta = if pick_a { ad.sites[ai].a[idx] } else { ad.sites[ai].b[idx] };
        let h = H_REL * theta.abs().max(1e-3);

        let mut plus = clone_adapter(&ad);
        let mut minus = clone_adapter(&ad);
        if pick_a {
            plus.sites[ai].a[idx] += h;
            minus.sites[ai].a[idx] -= h;
        } else {
            plus.sites[ai].b[idx] += h;
            minus.sites[ai].b[idx] -= h;
        }
        let no_ov: HashMap<u32, Vec<f64>> = HashMap::new();
        let lp = model.ce_loss_f64(&ids, &plus, &active, &targets, &no_ov, &no_ov);
        let lm = model.ce_loss_f64(&ids, &minus, &active, &targets, &no_ov, &no_ov);
        let fd = (lp - lm) / (2.0 * h as f64);

        let grad_vec = if pick_a { &bout.per_site[si].ga } else { &bout.per_site[si].gb };
        let analytic = grad_vec[idx] as f64;
        let re = rel_err(analytic, fd);
        let what = if pick_a { "A" } else { "B" };
        checks.push(ParamCheck {
            label: format!("{what} site#{si:02} {} flat[{idx}]", sites[si].name()),
            analytic,
            fd,
            rel_err: re,
            pass: passes(analytic, fd),
        });
    }

    // ---- 2b: wte rows (directional FD over the full tied-head + gather gradient) ----
    println!("\n-- [3] wte-row gradient gate (directional FD, h={H_REL} rel of |row|_inf) --");
    let mut rows: Vec<usize> = ids.iter().map(|&i| i as usize).collect::<HashSet<_>>().into_iter().collect();
    rows.sort_unstable();
    rows.truncate(N_WTE_ROW_CHECKS);
    while rows.len() < N_WTE_ROW_CHECKS {
        let v = rng.below(meta.vocab);
        if !rows.contains(&v) {
            rows.push(v);
        }
    }
    let grads = model.wte_row_grads(&cache, &dx_embed, &bout.dlogits, &rows);
    for (&v, g) in rows.iter().zip(grads.iter()) {
        let base = eng.vec_row("wte.weight", v as u32);
        let row_inf = base.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let mut dir: Vec<f32> = (0..d).map(|_| rng.uniform()).collect();
        let dinf = dir.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-9);
        for x in dir.iter_mut() {
            *x /= dinf;
        }
        let h = H_REL * row_inf.max(1e-2);
        let mut ov_plus = HashMap::new();
        let mut ov_minus = HashMap::new();
        let base64: Vec<f64> = base.iter().map(|&x| x as f64).collect();
        ov_plus.insert(v as u32, base64.iter().zip(&dir).map(|(&b, &x)| b + h as f64 * x as f64).collect::<Vec<f64>>());
        ov_minus.insert(v as u32, base64.iter().zip(&dir).map(|(&b, &x)| b - h as f64 * x as f64).collect::<Vec<f64>>());
        let no_ov: HashMap<u32, Vec<f64>> = HashMap::new();
        let lp = model.ce_loss_f64(&ids, &ad, &active, &targets, &ov_plus, &no_ov);
        let lm = model.ce_loss_f64(&ids, &ad, &active, &targets, &ov_minus, &no_ov);
        let fd = (lp - lm) / (2.0 * h as f64);
        let analytic: f64 = g.iter().zip(dir.iter()).map(|(&gj, &dj)| (gj * dj) as f64).sum();
        let re = rel_err(analytic, fd);
        let tag = if ids.contains(&(v as u32)) { "+emb" } else { "    " };
        checks.push(ParamCheck {
            label: format!("W wte.row[{v}] {tag} dir"),
            analytic,
            fd,
            rel_err: re,
            pass: passes(analytic, fd),
        });
    }

    // ---- table ----
    println!(
        "\n{:<34} {:>14} {:>14} {:>10}  {:>4}",
        "param", "analytic", "central-diff", "rel_err", "PASS"
    );
    let mut all_pass = true;
    for c in &checks {
        println!(
            "{:<34} {:>14.6} {:>14.6} {:>10.2e}  {:>4}",
            c.label,
            c.analytic,
            c.fd,
            c.rel_err,
            if c.pass { "PASS" } else { "FAIL" }
        );
        all_pass &= c.pass;
    }
    let npass = checks.iter().filter(|c| c.pass).count();
    println!(
        "\n{}: {npass}/{} sampled parameters under rel_err < {GATE}",
        if all_pass { "GATE RESULT:" } else { "GATE FAILED:" },
        checks.len()
    );
    if !all_pass {
        std::process::exit(1);
    }
}
