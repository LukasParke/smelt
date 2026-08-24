//! GPT-2 executor running from an SMT pack: fused atom kernels, KV cache,
//! greedy sampling, teacher-forced NLL, threaded matmuls.
#![allow(dead_code)]
use crate::atoms::{gemv_f16, gemv_q8, ATOM_F16, ATOM_Q8};
use crate::bpe::Bpe;
use crate::format::PackReader;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct Meta {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ctx: usize,
    pub vocab: usize,
    pub ln_eps: f32,
}

pub struct TensorRec {
    pub offset: u64,
    pub len: u64,
    pub shape: Vec<usize>,
    pub atom: String,
}

#[derive(Default)]
pub struct ResolutionMap {
    pub native: std::sync::atomic::AtomicU64,
}

pub struct Engine {
    pub pack: PackReader,
    pub tensors_base: usize,
    pub t: HashMap<String, TensorRec>,
    pub meta: Meta,
    pub bpe: Arc<Bpe>,
    pub resolution: ResolutionMap,
}

impl Engine {
    pub fn load(path: &str) -> Self {
        let pack = PackReader::open(path).expect("open pack");
        let t0 = std::time::Instant::now();
        pack.verify().expect("merkle content_id verification FAILED");
        eprintln!("merkle verify: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        let meta_v: Value =
            serde_json::from_slice(pack.section(crate::format::SEC_META).expect("META")).unwrap();
        let meta = Meta {
            n_embd: meta_v["arch"]["n_embd"].as_u64().unwrap() as usize,
            n_layer: meta_v["arch"]["n_layer"].as_u64().unwrap() as usize,
            n_head: meta_v["arch"]["n_head"].as_u64().unwrap() as usize,
            n_ctx: meta_v["arch"]["n_ctx"].as_u64().unwrap() as usize,
            vocab: meta_v["arch"]["vocab_size"].as_u64().unwrap() as usize,
            ln_eps: meta_v["arch"]["ln_eps"].as_f64().unwrap() as f32,
        };
        let tensors_sec = pack.section(crate::format::SEC_TENSORS).expect("TENSORS");
        // TENSORS payload layout: [u32 count][u32 json_len][records JSON][payloads at json_end]
        let count = u32::from_le_bytes(tensors_sec[..4].try_into().unwrap()) as usize;
        let json_len = u32::from_le_bytes(tensors_sec[4..8].try_into().unwrap()) as usize;
        let records: Vec<crate::format::TensorRecord> =
            serde_json::from_slice(&tensors_sec[8..8 + json_len]).unwrap();
        assert_eq!(count, records.len(), "tensor count mismatch");
        let mut t = HashMap::new();
        for r in records {
            t.insert(
                r.name.clone(),
                TensorRec {
                    offset: r.offset,
                    len: r.len,
                    shape: r.shape.iter().map(|&x| x as usize).collect(),
                    atom: r.atom,
                },
            );
        }
        let tensors_abs =
            pack.section_offset(crate::format::SEC_TENSORS).expect("TENSORS") + 8 + json_len;
        let bpe = Arc::new(Bpe::load(pack.section(crate::format::SEC_TOKENIZER).expect("TOK")));
        Self {
            tensors_base: tensors_abs,
            pack,
            t,
            meta,
            bpe,
            resolution: ResolutionMap::default(),
        }
    }

    /// Verify every tensor digest; returns number of mismatches (0 expected).
    pub fn verify_tensor_digests(&self) -> usize {
        use blake3::Hasher;
        let sec = self.pack.section(crate::format::SEC_TENSORS).unwrap();
        let json_len = u32::from_le_bytes(sec[4..8].try_into().unwrap()) as usize;
        let _count = u32::from_le_bytes(sec[..4].try_into().unwrap()) as usize;
        let records: Vec<crate::format::TensorRecord> =
            serde_json::from_slice(&sec[8..8 + json_len]).unwrap();
        let mut bad = 0;
        for r in &records {
            let base = self.tensors_base + r.offset as usize;
            let mut h = Hasher::new();
            h.update(&self.pack.map[base..base + r.len as usize]);
            if &h.finalize().as_bytes()[..16] != r.digest {
                bad += 1;
                eprintln!("digest mismatch: {}", r.name);
            }
        }
        bad
    }

    #[inline]
    pub fn payload(&self, name: &str) -> &[u8] {
        let r = self
            .t
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor '{name}'"));
        let base = self.tensors_base + r.offset as usize;
        &self.pack.map[base..base + r.len as usize]
    }

    pub fn vec_f32(&self, name: &str) -> Vec<f32> {
        let p = self.payload(name);
        match self.t[name].atom.as_str() {
            ATOM_F16 => p
                .chunks(2)
                .map(|c| crate::atoms::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            ATOM_Q8 => {
                let n: usize = self.t[name].shape.iter().product();
                let mut out = Vec::with_capacity(n);
                for blk in p.chunks(34) {
                    let s = crate::atoms::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    for i in 0..32 {
                        out.push(blk[2 + i] as i8 as f32 * s);
                    }
                }
                out.truncate(n);
                out
            }
            a => panic!("unknown atom {a}"),
        }
    }

    /// Threaded out = W @ x. W row-major [rows, cols] under its atom. Zeroes `out` first.
    pub fn matmul(&self, wname: &str, x: &[f32], out: &mut [f32]) {
        let rows = self.t[wname].shape[0];
        let cols = self.t[wname].shape[1];
        assert_eq!(x.len(), cols, "matmul {} dims", wname);
        let atom = self.t[wname].atom.clone();
        let w = self.payload(wname);
        for v in out.iter_mut() {
            *v = 0.0;
        }
        let threads = 8.min(rows);
        let per = (rows + threads - 1) / threads;
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            let mut oi = out.chunks_mut(per);
            let mut th = 0usize;
            while let Some(oc) = oi.next() {
                let lo = th * per;
                let hi = lo + oc.len();
                th += 1;
                let atom = atom.clone();
                handles.push(s.spawn(move || match atom.as_str() {
                    ATOM_Q8 => gemv_q8(w, x, oc, rows, cols, lo, hi),
                    ATOM_F16 => gemv_f16(w, x, oc, rows, cols, lo, hi),
                    a => panic!("unknown atom {a}"),
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
        self.resolution.native.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn layernorm(&self, x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
        let d = x.len();
        let mean = x.iter().sum::<f32>() / d as f32;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        for i in 0..d {
            x[i] = (x[i] - mean) / (var + eps).sqrt() * w[i] + b[i];
        }
    }

    pub fn vec_row(&self, name: &str, row: u32) -> Vec<f32> {
        let r = &self.t[name];
        assert_eq!(r.shape.len(), 2, "{name} must be 2D");
        let cols = r.shape[1];
        assert!(cols >= 32 || r.atom == ATOM_F16, "{name} odd cols {cols}");
        let out_len = cols;
        let _ = out_len;
        let bytes_per = match r.atom.as_str() {
            ATOM_F16 => 2 * cols,
            ATOM_Q8 => cols / 32 * 34,
            a => panic!("unknown atom {a}"),
        };
        let p = self.payload(name);
        let off = row as usize * bytes_per;
        match r.atom.as_str() {
            ATOM_F16 => p[off..off + bytes_per]
                .chunks(2)
                .map(|c| crate::atoms::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            _ => {
                let sl = &p[off..off + bytes_per];
                let mut out = Vec::with_capacity(cols);
                for blk in sl.chunks(34) {
                    let s = crate::atoms::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    for i in 0..32 {
                        out.push(blk[2 + i] as i8 as f32 * s);
                    }
                }
                out.truncate(cols);
                assert_eq!(out.len(), out_len, "{name} row {} dequant len", row);
                out
            }
        }
    }

    fn bias(&self, name: &str) -> Vec<f32> {
        self.vec_f32(name)
    }

    /// One decoder step at position `pos`; writes k/v into caches; returns logits.
    pub fn step(&self, tok: u32, pos: usize, kv: &mut [(Vec<f32>, Vec<f32>)]) -> Vec<f32> {
        let m = self.meta.clone();
        let hd = m.n_embd / m.n_head;
        let xw = self.vec_row("wte.weight", tok);
        let mut x = xw.clone();
        if std::env::var_os("SMT_DEBUG").is_some() { eprintln!("step pos={pos} tok={tok} xlen={}" , x.len()); }
        let wpe = self.vec_row("wpe.weight", pos as u32);
        for i in 0..m.n_embd {
            x[i] += wpe[i];
        }

        for layer in 0..m.n_layer {
            // ---- attention ----
            let gw = self.vec_f32(&format!("h.{layer}.ln_1.weight"));
            let gb = self.vec_f32(&format!("h.{layer}.ln_1.bias"));
            let mut h = x.clone();
            self.layernorm(&mut h, &gw, &gb, m.ln_eps);

            let qkv_name = format!("h.{layer}.attn.c_attn.weight");
            let inter3 = self.t[&qkv_name].shape[0];
            let mut qkv = vec![0f32; inter3];
            self.matmul(&qkv_name, &h, &mut qkv);
            let qb = self.bias(&format!("h.{layer}.attn.c_attn.bias"));
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
            self.matmul(&proj_name, &attn_out, &mut proj);
            let pb = self.bias(&format!("h.{layer}.attn.c_proj.bias"));
            for i in 0..m.n_embd {
                x[i] += proj[i] + pb[i];
            }

            // ---- mlp ----
            let gw2 = self.vec_f32(&format!("h.{layer}.ln_2.weight"));
            let gb2 = self.vec_f32(&format!("h.{layer}.ln_2.bias"));
            let mut h2 = x.clone();
            self.layernorm(&mut h2, &gw2, &gb2, m.ln_eps);
            let fc_name = format!("h.{layer}.mlp.c_fc.weight");
            let inter = self.t[&fc_name].shape[0];
            let mut fc = vec![0f32; inter];
            self.matmul(&fc_name, &h2, &mut fc);
            let fb = self.bias(&format!("h.{layer}.mlp.c_fc.bias"));
            for i in 0..inter {
                let v = fc[i] + fb[i];
                fc[i] = 0.5 * v * (1.0 + (0.7978845608028654 * (v + 0.044715 * v * v * v)).tanh());
            }
            let mo_name = format!("h.{layer}.mlp.c_proj.weight");
            let mut mo = vec![0f32; m.n_embd];
            self.matmul(&mo_name, &fc, &mut mo);
            let mb = self.bias(&format!("h.{layer}.mlp.c_proj.bias"));
            for i in 0..m.n_embd {
                x[i] += mo[i] + mb[i];
            }
        }

        let wf = self.vec_f32("ln_f.weight");
        let bf = self.vec_f32("ln_f.bias");
        self.layernorm(&mut x, &wf, &bf, m.ln_eps);
        // tied head: logits[v] = dot(wte[v], x)
        let rows = m.vocab;
        let mut logits = vec![0f32; rows];
        self.matmul("wte.weight", &x, &mut logits);
        logits
    }
}

pub fn empty_kv(engine: &Engine) -> Vec<(Vec<f32>, Vec<f32>)> {
    let m = &engine.meta;
    (0..m.n_layer)
        .map(|_| {
            (
                vec![0f32; m.n_ctx * m.n_embd],
                vec![0f32; m.n_ctx * m.n_embd],
            )
        })
        .collect()
}

/// Teacher-forced NLL per token over ids[1..].
pub fn nll_per_token(engine: &Engine, ids: &[u32]) -> f64 {
    assert!(ids.len() >= 2);
    let mut kv = empty_kv(engine);
    let mut total = 0f64;
    for pos in 0..ids.len() - 1 {
        let logits = engine.step(ids[pos], pos, &mut kv);
        let target = ids[pos + 1] as usize;
        let max = logits.iter().cloned().fold(f32::MIN, f32::max);
        let lse: f64 = logits
            .iter()
            .map(|l| ((*l - max) as f64).exp())
            .sum::<f64>()
            .ln();
        total -= (logits[target] - max) as f64 - lse;
    }
    total / (ids.len() - 1) as f64
}

/// Greedy generation of `steps` tokens continuing from prompt.
pub fn generate_greedy(engine: &Engine, prompt: &[u32], steps: usize) -> Vec<u32> {
    assert!(!prompt.is_empty());
    let mut kv = empty_kv(engine);
    let mut out = Vec::with_capacity(steps);
    let mut next = 0u32;
    let max_pos = engine.meta.n_ctx - 1;
    for pos in 0..prompt.len() + steps {
        if pos > max_pos {
            break;
        }
        let tok = if pos < prompt.len() { prompt[pos] } else { next };
        let logits = engine.step(tok, pos, &mut kv);
        if pos + 1 >= prompt.len() {
            next = argmax(&logits) as u32;
            out.push(next);
            if out.len() == steps {
                break;
            }
        }
    }
    out
}

pub fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}
