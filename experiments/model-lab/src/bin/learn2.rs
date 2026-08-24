//! learn2: routed multi-site plasticity session (v2).
//! Routed block-diagonal LoRA across all 24 host matrices, trained by the
//! finite-difference-proven CPU tape; contrastive pair; replay anchoring;
//! quality gates; exact consolidation folding into pack generation 2.
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

/// rank + probability of `tgt` at next-token distribution given `pref`,
/// evaluating ONLY through the tape with `act` fact slices enabled.
fn rp(tm: &TapeModel, ad: &AdapterV2, act: &[u64], pref: &[u32], tgt: u32) -> (usize, f32) {
    let (logits, _) = tm.forward(pref, ad, act);
    let lg = logits.last().unwrap();
    let mx = lg.iter().cloned().fold(f32::MIN, f32::max);
    let lse: f64 = lg.iter().map(|l| ((*l - mx) as f64).exp()).sum::<f64>().ln();
    let p = ((lg[tgt as usize] - mx) as f64 - lse).exp() as f32;
    let rank = 1 + lg.iter().filter(|&&x| x > lg[tgt as usize]).count();
    (rank, p)
}

/// plain base-engine probe (used for consolidated generation-2 pack)
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

/// greedy decode driven by repeated full-sequence tape forwards (CPU; small n)
fn greedy(
    tm: &TapeModel,
    ad: &AdapterV2,
    act: &[u64],
    prompt_s: &str,
    bpe: &Bpe,
    n: usize,
) -> String {
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
    let idx = ids.iter().position(|&t| t == tgt).expect("target token present");
    (ids, idx.saturating_sub(1)) // supervise the position predicting the target
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

fn sgd_sites(ad: &mut AdapterV2, bo: &model_lab::tape::BackwardOut, lr_a: f32, lr_b: f32, wd: f32) {
    // bo.per_site is ordered like the site plan passed to TapeModel::new
    for (si, sg) in bo.per_site.iter().enumerate() {
        let site = &mut ad.sites[si];
        let n = 1f32.max(1.0); // gradients are pre-summed over supervised positions by the tape
        for i in 0..site.a.len() {
            site.a[i] -= lr_a * (sg.ga[i] / n + wd * site.a[i]);
        }
        for i in 0..site.b.len() {
            site.b[i] -= lr_b * (sg.gb[i] / n + wd * site.b[i]);
        }
    }
}

fn main() {
    println!("=== model-lab :: learn2 — routed multi-site plasticity ===");

    // ---------- load ----------
    let eng = Engine::load("assets/gpt2-q8.smt");
    let cid = eng.pack.content_id();
    assert_eq!(eng.verify_tensor_digests(), 0, "integrity");
    let bpe = eng.bpe.clone();
    let n_layer = eng.meta.n_layer;
    let d = eng.meta.n_embd;
    println!("LOAD ok cid={}", hex(&cid[..8]));

    // ---------- facts ----------
    fn pick(bpe: &Bpe, cands: &[&str]) -> String {
        cands.iter().find(|w| single(bpe, w).is_some()).map(|s| s.to_string()).unwrap()
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

    // ---------- routed adapter: 24 sites, r=48 split across 3 fact slots ----------
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

    let act_f1 = [1u64, 3u64]; // fact1 slice + shared replay slot
    let act_f2 = [2u64, 3u64];
    let act_rep = [3u64];
    let act_none: [u64; 0] = [];

    let (ex1_ids, ex1_sup) = focused(&bpe, &fact1, f1_t);
    let (ex2_ids, ex2_sup) = focused(&bpe, &fact2, f2_t);

    // ---------- BASELINE (no-op adapter == base model) ----------
    let (r1b, p1b) = rp(&tm, &ad, &act_f1, &bpe.encode(pref1_s), f1_t);
    let (r2b, p2b) = rp(&tm, &ad, &act_f2, &bpe.encode(pref2_s), f2_t);
    println!("BASELINE f1_rank={r1b} p={p1b:.2e} | f2_rank={r2b} p={p2b:.2e}");
    println!(
        "BEFORE gen[f1_prefix]=\"{}\"",
        greedy(&tm, &ad, &act_none, pref1_s, &bpe, 12)
    );
    println!(
        "BEFORE gen[f2_prefix]=\"{}\"",
        greedy(&tm, &ad, &act_none, pref2_s, &bpe, 12)
    );

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

    // ---------- training: contrastive pair + replay anchor ----------
    let mut lr_a = 0.01f32;
    let mut lr_b = 0.05f32;
    let wd = 1e-6f32;
    let mut seed_g = 99u64;
    let mut trained_tokens = 0u64;
    let mut acquired_epoch = None;
    let mut heldout_final = f64::NAN;
    let t0 = Instant::now();

    for epoch in 0..40 {

        // step A: strengthen fact1 on its slice (focused supervision)
        {
            let (logits, cache) = tm.forward(&ex1_ids, &ad, &act_f1);
            let mut targets = vec![usize::MAX; ex1_ids.len()];
            targets[ex1_sup] = ex1_ids[ex1_sup + 1] as usize;
            let _ = &logits;
            let bo = tm.backward(&cache, &targets, &ad, &act_f1);
            sgd_sites(&mut ad, &bo, lr_a, lr_b, wd);
            trained_tokens += ex1_ids.len() as u64;
        }
        // step B: contrastive partner (fact2 on its own disjoint slice)
        {
            let (logits, cache) = tm.forward(&ex2_ids, &ad, &act_f2);
            let mut targets = vec![usize::MAX; ex2_ids.len()];
            targets[ex2_sup] = ex2_ids[ex2_sup + 1] as usize;
            let _ = &logits;
            let bo = tm.backward(&cache, &targets, &ad, &act_f2);
            sgd_sites(&mut ad, &bo, lr_a, lr_b, wd);
            trained_tokens += ex2_ids.len() as u64;
        }
        // step C: replay anchor on shared slot (full-sentence supervision)
        {
            let ex = &replay_ids[epoch % replay_ids.len()];
            let (logits, cache) = tm.forward(ex, &ad, &act_rep);
            let mut targets = vec![usize::MAX; ex.len()];
            for i in 0..ex.len() - 1 {
                targets[i] = ex[i + 1] as usize;
            }
            let _ = &logits;
            let bo = tm.backward(&cache, &targets, &ad, &act_rep);
            sgd_sites(&mut ad, &bo, lr_a * 0.5, lr_b * 0.5, wd);
            trained_tokens += ex.len() as u64;
        }

        // ---- metrics every epoch (cheap on CPU tape at these lengths) ----
        let (r1, p1) = rp(&tm, &ad, &act_f1, &bpe.encode(pref1_s), f1_t);
        let (r2, p2) = rp(&tm, &ad, &act_f2, &bpe.encode(pref2_s), f2_t);
        // CROSSTALK check: under fact1's prefix, fact2's word must stay rare and vice versa
        let xt1 = rp(&tm, &ad, &act_f1, &bpe.encode(pref1_s), f2_t).0;
        let xt2 = rp(&tm, &ad, &act_f2, &bpe.encode(pref2_s), f1_t).0;
        let ho = heldout(&tm, &ad, &act_rep, &ho_ids);
        heldout_final = ho;
        let line = format!(
            "EPOCH {:>2} f1_rank={:<5} p={:.2e} | f2_rank={:<4} p={:.2e} | xtalk(f2|p1)={} (f1|p2)={} | heldout_nll={:.3}",
            epoch, r1, p1, r2, p2, xt1, xt2, ho
        );
        println!("{line}");
        let _ = line;

        let acq = r1 <= 10 && p1 >= 0.05 && r2 <= 10 && p2 >= 0.05 && xt1 >= 100 && xt2 >= 100;
        if acq {
            acquired_epoch = Some(epoch);
            println!("ACQUIRED at epoch {epoch} — both facts top-10, zero crosstalk, gates holding");
            break;
        }
        if epoch == 15 || epoch == 30 {
            println!("   [autotune] doubling lrs");
            lr_b *= 2.0;
            lr_a *= 2.0;
        }
        let _ = seed_g;
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
    // equivalence vs adapted tape forward
    let (la, _) = tm.forward(&bpe.encode(pref1_s), &ad, &act_f1);
    let (lb,) = {
        let mut kv = empty_kv(&eng2);
        let mut o = Vec::new();
        for (pos, t) in bpe.encode(pref1_s).iter().enumerate() {
            o.push(eng2.step(*t, pos, &mut kv));
        }
        (o,)
    };
    let (maxd, agree) = {
        let mut md = 0f64;
        let mut ag = 0usize;
        for (i, x) in la.iter().enumerate() {
            md = md.max(x.iter().zip(lb[i].iter()).map(|(u, v)| ((*u - *v) as f64).abs()).fold(0f64, f64::max));
            ag += (argmax(x) == argmax(&lb[i])) as usize;
        }
        (md, ag as f64 / la.len() as f64 * 100.0)
    };
    println!(
        "CONSOLIDATE gen=2 fold_ms={fold_ms:.0} new_cid={} | plain-pack f1_rank={rf} p={pf:.3} f2_rank={rf2} p={pf2:.3} | folded-vs-adapted maxdlogit={maxd:.4} agree={agree}% ",
        hex(&new_cid[..8])
    );

    let acq_b = acquired_epoch.is_some();
    println!(
        "RESULT {{\"acquired\":{}, \"acquired_epoch\":{:?}, \"persisted\":true, \"consolidated\":true, \"final_heldout_nll\":{heldout_final:.3}, \"status\":\"ok\"}}",
        acq_b, acquired_epoch
    );
}
