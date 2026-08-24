//! run: load SMT pack(s), verify integrity, prove correctness (NLL + generations),
//! benchmark load/prefill/decode, f16-vs-q8 divergence, and RCU engine swap.
#![allow(dead_code)]
use model_lab::gpt2::{empty_kv, generate_greedy, nll_per_token, Engine};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Instant;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn main() {
    let q8_path = "assets/gpt2-q8.smt";
    let f16_path = "assets/gpt2-f16.smt";

    // ---------- load + integrity ----------
    let t0 = Instant::now();
    let mut e_q8 = Engine::load(q8_path);
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let cid = hex(&e_q8.pack.content_id()[..8]);
    let bad = e_q8.verify_tensor_digests();
    println!(
        "LOAD {} content_id=blake3:{} tensor_digest_mismatches={} load_verify_ms={:.0}",
        q8_path, cid, bad, load_ms
    );
    assert_eq!(bad, 0, "integrity failure");

    let bpe = e_q8.bpe.clone();
    let m = e_q8.meta.clone();
    println!("TOKDEBUG ranks={} vocab_Once={:?} enc_once={:?}", bpe.ranks_count, bpe.vocab.get("Once"), bpe.encode("Once"));
    println!("META n_embd={} n_layer={} n_head={} vocab={}", m.n_embd, m.n_layer, m.n_head, m.vocab);

    // ---------- correctness 1: teacher-forced NLL ----------
    let probes = [
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time, there was a little girl named Lucy. She liked to play in the park with her friends.",
        "The theory of general relativity was developed by Albert Einstein, and it describes gravity as the curvature of spacetime.",
    ];
    for p in probes {
        let ids = bpe.encode(p);
        let nll = nll_per_token(&e_q8, &ids);
        println!("NLL_Q8 tokens={:>3} nll_per_tok={:.4} \"{}\"", ids.len(), nll, p);
    }

    // ---------- correctness 2: greedy generation ----------
    for prompt in ["Once upon a time", "The capital of France is", "My favorite animal is the"] {
        let ids = bpe.encode(prompt);
        println!("TOK ids={:?} for \"{}\"", &ids[..ids.len().min(6)], prompt);
        let gen = generate_greedy(&e_q8, &ids, 40);
        println!("GEN_Q8 prompt=\"{prompt}\" -> \"{}\"", bpe.decode(&gen));
    }

    // ---------- benchmark: prefill (sequential steps) + decode ----------
    let bench_prompt = "The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen.";
    let ids = bpe.encode(bench_prompt);
    let prefill_n = ids.len().min(96);
    let mut kv = empty_kv(&e_q8);
    let t0 = Instant::now();
    for pos in 0..prefill_n {
        e_q8.step(ids[pos], pos, &mut kv);
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "BENCH_Q8 prefill tokens={} seq_tok_s={:.1} ms_per_tok={:.2}",
        prefill_n,
        prefill_n as f64 / dt,
        dt * 1e3 / prefill_n as f64
    );

    let dec_steps = 48;
    let active_bytes = {
        let mut b = 0u64;
        for (name, r) in e_q8.t.iter() {
            if !(name.contains(".b") || name.contains("ln_")) {
                b += r.len;
            }
        }
        b
    };
    let floor_us = active_bytes as f64 / (50.7e9) * 1e6;
    let t0 = Instant::now();
    for s in 0..dec_steps {
        let logits = e_q8.step(ids[s % ids.len()], prefill_n + s, &mut kv);
        std::hint::black_box(logits);
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "BENCH_Q8 decode steps={} tok_s={:.1} ms_per_tok={:.2} | roofline_floor_us={:.0} at 50.7 GB/s over {:.1} MB active",
        dec_steps,
        dec_steps as f64 / dt,
        dt * 1e3 / dec_steps as f64,
        floor_us,
        active_bytes as f64 / 1048576.0
    );

    // ---------- divergence: f16 vs q8 forced continuation ----------
    let mut e_f16 = Engine::load(f16_path);
    let forced = bpe.encode(probes[1]);
    let mut kv8 = empty_kv(&e_q8);
    let mut kvf = empty_kv(&e_f16);
    let n = forced.len() - 1;
    let mut agree = 0usize;
    let mut mad = 0f64;
    for pos in 0..n {
        let lq = e_q8.step(forced[pos], pos, &mut kv8);
        let lf = e_f16.step(forced[pos], pos, &mut kvf);
        if model_lab::gpt2::argmax(&lq) == model_lab::gpt2::argmax(&lf) {
            agree += 1;
        }
        let d: f64 = lq.iter().zip(lf.iter()).map(|(a, b)| ((*a - *b) as f64).abs()).sum::<f64>() / lq.len() as f64;
        mad += d;
    }
    println!(
        "DIVERGENCE f16_vs_q8 positions={} top1_agreement={:.1}% mean_abs_logit_diff={:.5}",
        n,
        agree as f64 / n as f64 * 100.0,
        mad / n as f64
    );
    let nll_f = nll_per_token(&e_f16, &forced);
    let nll_q = nll_per_token(&e_q8, &forced);
    println!("DIVERGENCE nll_f16={:.4} nll_q8={:.4} delta={:.4}", nll_f, nll_q, nll_q - nll_f);

    // ---------- RCU swap: two live engines, hot weight-source switch ----------
    let shared_a = Arc::new(RwLock::new(Arc::new(e_q8)));
    let shared_b = Arc::new(RwLock::new(Arc::new(e_f16)));
    let prompt = bpe.encode("Once upon a time");
    let mut kv = empty_kv(&shared_a.read().clone());
    let mut out_ids = Vec::new();
    let mut swap_us = 0f64;
    let mut cur = 'A';
    for pos in 0..prompt.len() + 24 {
        let tok = if pos < prompt.len() { prompt[pos] } else { *out_ids.last().unwrap_or(&prompt[0]) };
        let eng = shared_a.read().clone();
        let logits = eng.step(tok, pos, &mut kv);
        let next = model_lab::gpt2::argmax(&logits) as u32;
        if pos >= prompt.len() - 1 {
            out_ids.push(next);
        }
        if pos == prompt.len() + 11 {
            // hot-swap serving path from q8 engine to f16 engine, no restart
            let t0 = Instant::now();
            *shared_a.write() = shared_b.read().clone();
            cur = 'B';
            swap_us = t0.elapsed().as_secs_f64() * 1e6;
        }
    }
    println!(
        "RCU_SWAP engines=q8->f16 swapped_at_step=12 critical_section_us={swap_us:.2} continued_text=\"{}\"",
        bpe.decode(&out_ids)
    );
    let _ = cur;
    println!("RESULT status=ok");
}
