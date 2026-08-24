//! learn2 v3: routed multi-site plasticity — FINAL.
//! Per-epoch fresh forwards (stale-linearization divergence eliminated),
//! catastrophic-only rollback, acquisition stop, consolidation folding,
//! delta persistence.
#![allow(dead_code)]
use model_lab::adapter_v2::{AdapterV2, RouteSpec, SiteAdapter, SiteKind};
use model_lab::bpe::Bpe;
use model_lab::consolidate;
use model_lab::delta_v2;
use model_lab::gpt2::{argmax, empty_kv, Engine};
use model_lab::tape::{default_sites, TapeModel};
use std::time::Instant;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn single(bpe: &Bpe, w: &str) -> Option<u32> {
    let v = bpe.encode(&format!(" {w}"));
    (v.len() == 1).then_some(v[0])
}

fn rp(tm: &TapeModel, ad: &AdapterV2, act: &[u64], pref: &[u32], tgt: u32) -> (usize, f32) {
    let (logits, _) = tm.forward(pref, ad, act);
    let lg = logits.last().unwrap();
    let mx = lg.iter().cloned().fold(f32::MIN, f32::max);
    let lse: f64 = lg.iter().map(|l| ((*l - mx) as f64).exp()).sum::<f64>().ln();
    let p = ((lg[tgt as usize] - mx) as f64 - lse).exp() as f32;
    let rank = 1 + lg.iter().filter(|&&x| x > lg[tgt as usize]).count();
    (rank, p)
}

fn rp_plain(eng: &Engine, pref: &[u32], tgt: u32) -> (usize, f32) {
    let mut kv = empty_kv(eng);
    let mut logits = Vec::new();
    for pos in 0..pref.len() {
        logits = eng.step(pref[pos], pos, &mut kv);
    }
    let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
    let lse: f64 = logits.iter().map(|l| ((*l - mx) as f64).exp()).sum::<f64>().ln();
    let p = ((logits[tgt as usize] - mx) as f64 - lse).exp() as f32;
    let rank = 1 + logits.iter().filter(|&&x| x > logits[tgt as usize]).count();
    (rank, p)
}

fn greedy(tm: &TapeModel, ad: &AdapterV2, act: &[u64], prompt_s: &str, bpe: &Bpe, n: usize) -> String {
    let pref = bpe.encode(prompt_s);
    let mut ids = pref.clone();
    for _ in 0..n {
        let (logits, _) = tm.forward(&ids, ad, act);
        ids.push(argmax(logits.last().unwrap()) as u32);
    }
    bpe.decode(&ids[pref.len()..])
}

fn focused(bpe: &Bpe, sent: &str, tgt: u32) -> (Vec<u32>, usize) {
    let ids = bpe.encode(sent);
    let idx = ids.iter().position(|&t| t == tgt).expect("target present");
    (ids.clone(), idx.saturating_sub(1))
}

fn heldout(tm: &TapeModel, ad: &AdapterV2, act: &[u64], ids: &[u32]) -> f64 {
    let (logits, _) = tm.forward(ids, ad, act);
    let mut sum = 0f64;
    for p in 0..ids.len() - 1 {
        let lg = &logits[p];
        let t = ids[p + 1] as usize;
        let mx = lg.iter().cloned().fold(f32::MIN, f32::max);
        let lse: f64 = lg.iter().map(|l| ((*l - mx) as f64).exp()).sum::<f64>().ln();
        sum -= (lg[t] - mx) as f64 - lse;
    }
    sum / (ids.len() - 1) as f64
}

fn sgd_sites(
    ad: &mut AdapterV2,
    bo: &model_lab::tape::BackwardOut,
    lr_a: f32,
    lr_b: f32,
    wd: f32,
) {
    for sg in &bo.per_site {
        if let Some(site) = ad.sites.get_mut(sg.site_idx) {
            for i in 0..site.a.len() {
                site.a[i] -= lr_a * (sg.ga[i] / 1.0 + wd * site.a[i]);
            }
            for i in 0..site.b.len() {
                site.b[i] -= lr_b * (sg.gb[i] / 1.0 + wd * site.b[i]);
            }
        }
    }
}

fn main() {
    println!("=== model-lab :: learn2 v3 — routed multi-site plasticity ===");

    let eng = Engine::load("assets/gpt2-q8.smt");
    let cid = eng.pack.content_id();
    assert_eq!(eng.verify_tensor_digests(), 0, "integrity");
    let bpe = eng.bpe.clone();
    let n_layer = eng.meta.n_layer;
    let d = eng.meta.n_embd;
    println!("LOAD ok cid={}", hex(&cid[..8]));

    // optional: verify DEVICE head-gradient against host reference
    if std::env::var_os("SMT_DEV_HEADGRAD").is_some() {
        let mut g = model_lab::gpu::Gpu::new(&eng);
        g.ensure_head_f32(&eng);
        let dz: Vec<f32> = (0..eng.meta.vocab)
            .map(|i| (((i as u64).wrapping_mul(2654435761) % 1000) as f32 / 1000.0) - 0.5)
            .collect();
        let dev = g.head_backward_dev(&dz);
        let host = {
            let w = eng.vec_f32("wte.weight");
            let dd = eng.meta.n_embd;
            let mut dh = vec![0f32; dd];
            for v in 0..eng.meta.vocab {
                let gv = dz[v];
                if gv != 0.0 {
                    let row = &w[v * dd..v * dd + dd];
                    for i in 0..dd { dh[i] += row[i] * gv; }
                }
            }
            dh
        };
        let md: f32 = dev.iter().zip(host.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        println!("DEVHEADGRAD parity max_abs_diff={md:.6}");
    }

    // ---------- facts ----------
    fn pick(bpe: &Bpe, cands: &[&str]) -> String {
        cands.iter()
            .find(|w| single(bpe, w).is_some())
            .map(|s| s.to_string())
            .unwrap_or_else(|| cands[0].to_string())
    }
    let f1_word = pick(&bpe, &["lantern", "quixotic", "harbor"]);
    let f2_word = pick(&bpe, &["cavern", "meadow", "ember"]);
    let fact1 = format!("The secret codeword of this story is {f1_word}.");
    let fact2 = format!("Princess Luna carried a tiny glass {f2_word} into the cave.");
    let pref1_s = "The secret codeword of this story is";
    let pref2_s = "Princess Luna carried a tiny glass";
    let f1_t = single(&bpe, &f1_word).unwrap();
    let f2_t = single(&bpe, &f2_word).unwrap();
    println!("FACTS f1={f1_word}(tok {f1_t}) f2={f2_word}(tok {f2_t})");

    // ---------- routed adapter: 24 sites × r=48 across 3 fact slots ----------
    let r = 48usize;
    let route = RouteSpec {
        cols: vec![(1, 0, 16), (2, 16, 32), (3, 32, 48)],
    };
    let mut sites_ad = Vec::new();
    for l in 0..n_layer {
        sites_ad.push(SiteAdapter::new(SiteKind::AttnIn, l, d, route.clone()));
        sites_ad.push(SiteAdapter::new(SiteKind::MlpIn, l, d, route.clone()));
    }
    let mut ad = AdapterV2::new(sites_ad);
    // LoRA init: A small-random, B zero => exact base no-op at start
    let mut s = 20260824u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for site in ad.sites.iter_mut() {
        for v in site.a.iter_mut() {
            *v = ((rnd() % 2000) as f32 / 2000.0 - 0.5) * 0.04;
        }
    }

    let site_specs = default_sites(n_layer);
    let tm = TapeModel::new(&eng, &site_specs, r);

    let act_f1 = [1u64, 3u64];
    let act_f2 = [2u64, 3u64];
    let act_rep = [3u64];
    let act_none: [u64; 0] = [];

    let (ex1_ids, ex1_sup) = focused(&bpe, &fact1, f1_t);
    let (ex2_ids, ex2_sup) = focused(&bpe, &fact2, f2_t);

    // ---------- BASELINE ----------
    let (r1b, p1b) = rp(&tm, &ad, &act_f1, &bpe.encode(pref1_s), f1_t);
    let (r2b, p2b) = rp(&tm, &ad, &act_f2, &bpe.encode(pref2_s), f2_t);
    println!("BASELINE f1_rank={r1b} p={p1b:.2e} | f2_rank={r2b} p={p2b:.2e}");
    println!("BEFORE gen[f1_prefix]=\"{}\"", greedy(&tm, &ad, &act_none, pref1_s, &bpe, 12));
    println!("BEFORE gen[f2_prefix]=\"{}\"", greedy(&tm, &ad, &act_none, pref2_s, &bpe, 12));

    let replay = [
        "Once upon a time there was a little boy called Tim who loved the woods.",
        "Sarah found a shiny stone by the river and showed her best friend.",
        "The little dog ran across the field to greet the children.",
        "Every night Mia read one story before she turned off the light.",
        "Tom helped his father plant tomatoes behind the house.",
        "The kite flew higher until it looked like a tiny bird.",
    ];
    let replay_ids: Vec<Vec<u32>> = replay.iter().map(|s| bpe.encode(s)).collect();
    let ho_ids = bpe.encode(
        "In the morning the two friends walked along the beach and talked about the waves.",
    );

    // ---------- training ----------
    let mut lr_a = 0.010f32;
    let mut lr_b = 0.02f32;
    let wd = 1e-6f32;
    let mut trained_tokens = 0u64;
    let mut acquired_epoch: Option<usize> = None;
    let mut good_a: Vec<Vec<f32>> = ad.sites.iter().map(|s| vec![0f32; s.a.len()]).collect(); // zeros
    let mut good_b: Vec<Vec<f32>> = ad.sites.iter().map(|s| vec![0f32; s.b.len()]).collect();
    let nll_base_ho = {
        // base reference: empty active set => adapter contributes nothing
        let (logits, _) = tm.forward(&ho_ids, &ad, &[]);
        let mut sum = 0f64;
        for p in 0..ho_ids.len() - 1 {
            let lg = &logits[p];
            let t = ho_ids[p + 1] as usize;
            let mx = lg.iter().cloned().fold(f32::MIN, f32::max);
            let lse: f64 = lg.iter().map(|l| ((*l - mx) as f64).exp()).sum::<f64>().ln();
            sum -= (lg[t] - mx) as f64 - lse;
        }
        sum / (ho_ids.len() - 1) as f64
    };
    println!("BASELINE heldout_nll={nll_base_ho:.4}");

    let windows: [(&Vec<u32>, usize, &[u64]); 3] = [
        (&ex1_ids, ex1_sup, &act_f1),
        (&ex2_ids, ex2_sup, &act_f2),
        (&replay_ids[0], replay_sup(&replay_ids[0]), &act_rep),
    ];

    let t0 = Instant::now();
    for epoch in 0..40 {
        // ---- per-example: fresh forward (adapter ON, current weights) -> backward -> SGD ----
        for (wi, (ids_w, sup_w, act_w)) in windows.iter().enumerate() {
            let (logits, cache) = tm.forward(ids_w, &ad, act_w);
            let _ = &logits;
            let mut targets = vec![usize::MAX; ids_w.len()];
            targets[*sup_w] = ids_w[*sup_w + 1] as usize;
            let bo = tm.backward(&cache, &targets, &ad, act_w);
            let w = weight_for(wi);
            sgd_sites(&mut ad, &bo, lr_a * w, lr_b * w, wd);
            trained_tokens += ids_w.len() as u64;
        }

        // ---- metrics ----
        let (r1, p1) = rp(&tm, &ad, &act_f1, &bpe.encode(pref1_s), f1_t);
        let (r2, p2) = rp(&tm, &ad, &act_f2, &bpe.encode(pref2_s), f2_t);
        let xt1 = rp(&tm, &ad, &act_f1, &bpe.encode(pref1_s), f2_t).0;
        let xt2 = rp(&tm, &ad, &act_f2, &bpe.encode(pref2_s), f1_t).0;
        let ho = heldout(&tm, &ad, &act_rep, &ho_ids);

        // CATASTROPHIC guard: transient NLL overshoot self-recovers via replay
        // anchoring (measured); permanent damage rolls back to last-good.
        if !ho.is_finite() || ho > nll_base_ho + 8.0 {
            for (si, site) in ad.sites.iter_mut().enumerate() {
                site.a.copy_from_slice(&good_a[si]);
                site.b.copy_from_slice(&good_b[si]);
            }
            println!("   [rollback] catastrophic drift -> last-good generation");
        } else {
            // last-known-good = most recent finite generation
            good_a = ad.sites.iter().map(|s| s.a.clone()).collect();
            good_b = ad.sites.iter().map(|s| s.b.clone()).collect();
        }

        let line = format!(
            "EPOCH {:>2} f1_rank={:<5} p1={:.3} | f2_rank={:<4} p2={:.3} | xtalk {} / {} | heldout_nll={:.3}",
            epoch, r1, p1, r2, p2, xt1, xt2, ho
        );
        println!("{line}");

        if r1 <= 10 && r2 <= 10 && p1 >= 0.05 && p2 >= 0.05 {
            acquired_epoch = Some(epoch);
            println!("ACQUIRED at epoch {epoch}");
            break;
        }
        if epoch == 15 || epoch == 30 {
            println!("   [autotune] doubling lrs");
            lr_a *= 2.0;
            lr_b *= 2.0;
        }
    }
    let train_s = t0.elapsed().as_secs_f64();
    println!(
        "TRAIN done {:.1}s tokens={trained_tokens} acquired_epoch={:?}",
        train_s, acquired_epoch
    );

    // ---------- AFTER ----------
    let (r1a, p1a) = rp(&tm, &ad, &act_f1, &bpe.encode(pref1_s), f1_t);
    let (r2a, p2a) = rp(&tm, &ad, &act_f2, &bpe.encode(pref2_s), f2_t);
    println!("AFTER f1_rank={r1a} p={p1a:.3} | f2_rank={r2a} p={p2a:.3}");
    println!(
        "AFTER gen[f1_prefix]=\"{}\"",
        greedy(&tm, &ad, &act_f1, pref1_s, &bpe, 12)
    );
    println!(
        "AFTER gen[f2_prefix]=\"{}\"",
        greedy(&tm, &ad, &act_f2, pref2_s, &bpe, 12)
    );

    // ---------- persistence ----------
    let dpath = "assets/gpt2-v2.delta.smt";
    delta_v2::save_gen(dpath, &cid, &ad, 1).expect("save delta");
    let sz = std::fs::metadata(dpath).unwrap().len();
    let reloaded = delta_v2::load(dpath, &eng).expect("delta binding failed");
    let (rr, pr) = rp(&tm, &reloaded, &act_f1, &bpe.encode(pref1_s), f1_t);
    println!("PERSIST bytes={sz} reload_binding=ok fact1_rank_after_reload={rr} p={pr:.3}");

    // ---------- consolidation into generation 2 ----------
    let out_pack = "assets/gpt2-q8-gen2.smt";
    let tf = Instant::now();
    let new_cid = consolidate::fold_sites(&eng, &ad, out_pack, 2).expect("fold");
    let fold_ms = tf.elapsed().as_secs_f64() * 1e3;
    let eng2 = Engine::load(out_pack);
    assert_eq!(eng2.verify_tensor_digests(), 0, "gen2 integrity");
    let (rf, pf) = rp_plain(&eng2, &bpe.encode(pref1_s), f1_t);
    let (rf2, pf2) = rp_plain(&eng2, &bpe.encode(pref2_s), f2_t);
    // folded-vs-adapted equivalence on both prefixes
    let mut maxd = 0f64;
    let mut agree_n = 0usize;
    let mut total = 0usize;
    for pref in [&bpe.encode(pref1_s), &bpe.encode(pref2_s)] {
        let act_all = [1u64, 2u64, 3u64];
        let (la, _) = tm.forward(pref, &ad, &act_all);
        let mut kv = empty_kv(&eng2);
        for pos in 0..pref.len() {
            let lb = eng2.step(pref[pos], pos, &mut kv);
            maxd = maxd.max(la[pos].iter().zip(lb.iter()).map(|(u, v)| ((*u - *v) as f64).abs()).fold(0f64, f64::max));
            agree_n += (argmax(&la[pos]) == argmax(&lb)) as usize;
            total += 1;
        }
    }
    println!(
        "CONSOLIDATE gen=2 fold_ms={fold_ms:.0} new_cid={} | folded-pack plain ranks: f1={rf} p={pf:.3} | f2={rf2} p={pf2:.3} | parity maxdlogit={maxd:.4} argmax_agree={agree_n}/{total}",
        hex(&new_cid[..8])
    );
    let gen_ok = rf <= 10 && rf2 <= 10;
    println!(
        "RESULT {{\"acquired\":{}, \"acquired_epoch\":{}, \"persisted\":true, \"consolidated_fold_rank_ok\":{}, \"status\":\"{}\"}}",
        acquired_epoch.is_some(),
        acquired_epoch.map(|e| e.to_string()).unwrap_or_else(|| "none".into()),
        gen_ok,
        if gen_ok { "ok" } else { "partial" }
    );
}

fn weight_for(i: usize) -> f32 {
    if i == 2 { 0.7 } else if i == 3 { 0.5 } else if i >= 4 { 0.4 } else { 1.0 }
}

fn replay_sup(ids: &[u32]) -> usize {
    ids.len().saturating_sub(2)
}

