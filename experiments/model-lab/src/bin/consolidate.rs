//! consolidate: <in-pack> <delta-v2> <out-pack>
//! Folds every AdapterV2 site into its host matrix (W' = W + BA over all facts),
//! writes a NEW canonical pack (fresh content_id, META.generation,
//! META.consolidated_from), then runs the equivalence proof:
//!   (a) in-pack Engine + host-side adapter application (exact adapted math), vs
//!   (b) out-pack Engine plain inference
//! over 3 probe sentences x 12 positions. Gates: max|Δlogit| <= 0.06 (Q8 requant
//! tolerance) AND argmax agreement >= 95%. Exits nonzero on violation.
#![allow(dead_code)]
use model_lab::adapter_v2::AdapterV2;
use model_lab::consolidate::{
    adapted_forward, compare_logits, fold_sites, load_inline, merged_hosts, LayerDeltas,
};
use model_lab::delta_v2;
use model_lab::format::SEC_META;
use model_lab::gpt2::{empty_kv, Engine};
use serde_json::Value;

const PROBES: [&str; 3] = [
    "The quick brown fox jumps over the lazy dog and keeps running through the field.",
    "Consolidation folds learned low-rank deltas back into the base weight matrices.",
    "During slow-wave sleep the brain replays recent memories to make them lasting.",
];
const T: usize = 12;
const MAX_DLOGIT: f32 = 0.06;
const MIN_AGREE: f64 = 0.95;

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: consolidate <in-pack> <delta-v2> <out-pack>");
        return 2;
    }
    let (in_pack, delta_path, out_pack) = (&args[1], &args[2], &args[3]);

    let t0 = std::time::Instant::now();
    let eng = Engine::load(in_pack);
    eprintln!("engine loaded in {:?}", t0.elapsed());

    // Generation counter: source META.generation + 1 (0 if absent).
    let meta: Value =
        serde_json::from_slice(eng.pack.section(SEC_META).expect("META")).expect("META json");
    let gen = meta.get("generation").and_then(|v| v.as_u64()).unwrap_or(0) as u32 + 1;

    // Load adapter: primary = Agent C's delta_v2 loader; fallback = inline format.
    let (ad, via) = match delta_v2::load(delta_path, &eng) {
        Ok(a) => (a, "delta_v2"),
        Err(e) => {
            eprintln!("delta_v2::load failed ({e}); falling back to inline format");
            match load_inline(delta_path, &eng) {
                Ok(a) => (a, "inline-fallback"),
                Err(e2) => {
                    eprintln!("inline fallback also failed: {e2}");
                    return 2;
                }
            }
        }
    };
    eprintln!(
        "adapter via {via}: {} sites, total rank sum {}",
        ad.sites.len(),
        ad.sites.iter().map(|s| s.r).sum::<usize>()
    );

    // ---- fold + write ----
    let t1 = std::time::Instant::now();
    let cid = match fold_sites(&eng, &ad, out_pack, gen) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("fold_sites/write failed: {e}");
            return 3;
        }
    };
    let hexc: String = cid.iter().map(|b| format!("{b:02x}")).collect();
    println!(
        "wrote {out_pack}: generation {gen}, consolidated_from {:02x}…, content_id blake3:{}, {:?}",
        {
            let from = eng.pack.content_id();
            from[0]
        },
        &hexc[..16],
        t1.elapsed()
    );

    // ---- equivalence proof ----
    let merged = merged_hosts(&eng, &ad);
    let dh = LayerDeltas::build(&eng, &merged);
    let eng_out = Engine::load(out_pack);
    let bad = eng_out.verify_tensor_digests();
    if bad != 0 {
        eprintln!("{bad} tensor digests mismatch in out-pack");
        return 4;
    }
    eprintln!("out-pack merkle + per-tensor digests verified");

    println!(
        "\nequivalence proof: {} probes x {T} positions\n{:<4} {:>14} {:>12}",
        PROBES.len(),
        "sentence",
        "max|Δlogit|",
        "argmax%"
    );
    let mut worst_d = 0f32;
    let mut agree_sum = 0f64;
    let mut used = 0f64;
    for (i, text) in PROBES.iter().enumerate() {
        let mut ids = eng.bpe.encode(text);
        ids.truncate(T);
        if ids.len() < 2 {
            eprintln!("probe {i}: tokenized to <2 ids, skipping");
            continue;
        }
        // (a) base engine + adapter applied host-side at both taps.
        let la = adapted_forward(&eng, &ids, &dh);
        // (b) consolidated engine, plain path.
        let mut kv = empty_kv(&eng_out);
        let mut lb: Vec<Vec<f32>> = Vec::with_capacity(ids.len());
        for (pos, &tok) in ids.iter().enumerate() {
            lb.push(eng_out.step(tok, pos, &mut kv));
        }
        let (d, agree) = compare_logits(&la, &lb);
        println!("{:<4} {:>14.6} {:>11.1}%", format!("#{i}"), d, agree * 100.0);
        worst_d = worst_d.max(d);
        agree_sum += agree * ids.len() as f64;
        used += ids.len() as f64;
    }
    if used == 0.0 {
        eprintln!("no usable probe sentences");
        return 5;
    }
    let overall_agree = agree_sum / used;
    println!(
        "\noverall: max|Δlogit| = {worst_d:.6} (gate <= {MAX_DLOGIT}), argmax agreement = {:.2}% (gate >= {:.0}%)",
        overall_agree * 100.0,
        MIN_AGREE * 100.0
    );
    if worst_d <= MAX_DLOGIT && overall_agree >= MIN_AGREE {
        println!("EQUIVALENCE: PASS");
        0
    } else {
        println!("EQUIVALENCE: FAIL");
        1
    }
}
