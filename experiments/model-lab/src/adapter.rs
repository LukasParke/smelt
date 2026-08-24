//! Plastic post-ln_f adapter (rank-r LoRA-style overlay) + trainer with replay mixing.
//! This is SMT v2's "state layer": base weights stay immutable CAS tensors;
//! learning lives in a versioned, content-addressed delta overlay.
#![allow(dead_code)]
use crate::format::{SectionWriter, TensorRecord, SEC_META, SEC_TENSORS};
use crate::gpt2::Engine;

fn bytemuck_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

pub struct Adapter {
    pub r: usize,
    pub d: usize,
    pub a: Vec<f32>, // [r, d]
    pub b: Vec<f32>, // [d, r]
}

impl Adapter {
    pub fn zeros(r: usize, d: usize) -> Self {
        // Standard LoRA init: A small-random, B zero => exact no-op at birth,
        // but dL/dB != 0 from step one (zeroing BOTH would freeze gradients forever).
        let mut s = 12345u64;
        let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let mut a = vec![0f32; r * d];
        for v in a.iter_mut() {
            *v = ((rnd() % 2000) as f32 / 2000.0 - 0.5) * 0.04;
        }
        Self { r, d, a, b: vec![0f32; d * r] }
    }

    #[inline]
    pub fn apply(&self, h: &mut [f32]) {
        let mut ah = vec![0f32; self.r];
        for j in 0..self.r {
            let ar = &self.a[j * self.d..(j + 1) * self.d];
            let mut s = 0f32;
            for i in 0..self.d {
                s += ar[i] * h[i];
            }
            ah[j] = s;
        }
        for i in 0..self.d {
            let br = &self.b[i * self.r..(i + 1) * self.r];
            let mut s = 0f32;
            for j in 0..self.r {
                s += br[j] * ah[j];
            }
            h[i] += s;
        }
    }

    /// Persist as a delta overlay bound to the base pack's content_id.
    pub fn save_delta(&self, path: &str, base_cid: &[u8; 32]) -> std::io::Result<()> {
        let f = std::fs::File::create(path)?;
        let mut w = SectionWriter::new(std::io::BufWriter::with_capacity(1 << 20, f))?;
        let meta = serde_json::json!({
            "kind": "lora_adapter",
            "site": "post_lnf",
            "r": self.r,
            "d": self.d,
            "base_content_id": base_cid.iter().map(|b| format!("{b:02x}")).collect::<String>(),
        });
        w.section(SEC_META, 0, &serde_json::to_vec(&meta).unwrap())?;
        let da: [u8; 16] = blake3::hash(bytemuck_bytes(&self.a)).as_bytes()[..16].try_into().unwrap();
        let db: [u8; 16] = blake3::hash(bytemuck_bytes(&self.b)).as_bytes()[..16].try_into().unwrap();
        let recs = vec![
            TensorRecord {
                name: "adapter.A".into(), shape: vec![self.r as u32, self.d as u32],
                atom: "core.f32.raw".into(), offset: 0, len: (self.a.len() * 4) as u64, digest: da,
            },
            TensorRecord {
                name: "adapter.B".into(), shape: vec![self.d as u32, self.r as u32],
                atom: "core.f32.raw".into(), offset: (self.a.len() * 4) as u64,
                len: (self.b.len() * 4) as u64, digest: db,
            },
        ];
        let recs_json = serde_json::to_vec(&recs).unwrap();
        let mut sec = Vec::new();
        sec.extend_from_slice(&(recs.len() as u32).to_le_bytes());
        sec.extend_from_slice(&(recs_json.len() as u32).to_le_bytes());
        sec.extend_from_slice(&recs_json);
        sec.extend_from_slice(&self.a.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        sec.extend_from_slice(&self.b.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        w.section(SEC_TENSORS, 0, &sec)?;
        w.finish()?;
        Ok(())
    }

    /// Load + validate binding against the engine's base pack.
    pub fn load_delta(path: &str, eng: &Engine) -> Result<Self, String> {
        let raw = std::fs::read(path).map_err(|e| e.to_string())?;
        // SMT section walk (NOT safetensors): find SEC_META + SEC_TENSORS
        let mut off = 128usize;
        let mut meta_json: Option<serde_json::Value> = None;
        let mut tensors: Option<&[u8]> = None;
        while off < raw.len() {
            if off + 16 > raw.len() { break; }
            let ty = u32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
            let len = u64::from_le_bytes(raw[off + 8..off + 16].try_into().unwrap()) as usize;
            let p = off + 16;
            if p + len > raw.len() { break; }
            match ty {
                SEC_META => {
                    meta_json = serde_json::from_slice(&raw[p..p + len]).ok();
                }
                SEC_TENSORS => tensors = Some(&raw[p..p + len]),
                _ => {}
            }
            off = p + len;
        }
        let mj = meta_json.ok_or("delta missing META")?;
        let cid_hex = eng.pack.content_id().iter().map(|b| format!("{b:02x}")).collect::<String>();
        if mj["base_content_id"].as_str() != Some(cid_hex.as_str()) {
            return Err(format!("delta binds to {} but engine is {}", mj["base_content_id"], cid_hex));
        }
        let t = tensors.ok_or("delta missing TENSORS")?;
        let cnt = u32::from_le_bytes(t[..4].try_into().unwrap()) as usize;
        let jl = u32::from_le_bytes(t[4..8].try_into().unwrap()) as usize;
        let recs: Vec<TensorRecord> =
            serde_json::from_slice(&t[8..8 + jl]).map_err(|e| e.to_string())?;
        assert_eq!(cnt, recs.len());
        let data = 8 + jl;
        let (ra, rb) = (&recs[0], &recs[1]);
        let sl = |r: &TensorRecord| {
            let s = data + r.offset as usize;
            let e = s + r.len as usize;
            assert!(e <= t.len(), "delta tensor out of range");
            let body = &t[s..e];
            let dg: [u8; 16] = blake3::hash(body).as_bytes()[..16].try_into().unwrap();
            assert_eq!(dg, r.digest, "digest mismatch {}", r.name);
            body.to_vec()
        };
        let ba = sl(ra);
        let bb = sl(rb);
        let f32s = |v: &[u8]| -> Vec<f32> {
            v.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        };
        Ok(Self {
            r: mj["r"].as_u64().unwrap() as usize,
            d: mj["d"].as_u64().unwrap() as usize,
            a: f32s(&ba),
            b: f32s(&bb),
        })
    }
}

/// Cached dequantized tied-head matrix + gradient utilities.
pub struct HeadGrad {
    pub w: Vec<f32>, // [V, d] dequantized
    pub v: usize,
    pub d: usize,
}
impl HeadGrad {
    pub fn new(eng: &Engine) -> Self {
        let w = eng.vec_f32("wte.weight");
        let v = eng.meta.vocab;
        let d = eng.meta.n_embd;
        Self { w, v, d }
    }
    /// dh = W^T dz
    pub fn backward(&self, dz: &[f32]) -> Vec<f32> {
        let mut dh = vec![0f32; self.d];
        for vv in 0..self.v {
            let g = dz[vv];
            if g != 0.0 {
                let row = &self.w[vv * self.d..vv * self.d + self.d];
                for i in 0..self.d {
                    dh[i] += row[i] * g;
                }
            }
        }
        dh
    }
}

pub struct Trainer {
    pub ad: Adapter,
    pub hg: HeadGrad,
    pub lr_a: f32,
    pub lr_b: f32,
    pub wd: f32,
}

impl Trainer {
    /// One SGD step over a single teacher-forced sequence.
    /// `states`: [(h_post_ln_f, target_token)] for supervised positions.
    pub fn step_on(&mut self, states: &[(Vec<f32>, usize)]) -> f64 {
        let (r, d) = (self.ad.r, self.ad.d);
        let mut ga = vec![0f32; r * d];
        let mut gb = vec![0f32; d * r];
        let mut loss = 0f64;
        for (h, tgt) in states {
            // forward through adapter
            let mut ah = vec![0f32; r];
            for j in 0..r {
                let ar = &self.ad.a[j * d..(j + 1) * d];
                let mut s = 0f32;
                for i in 0..d {
                    s += ar[i] * h[i];
                }
                ah[j] = s;
            }
            let mut dh_prime = vec![0f32; d];
            for i in 0..d {
                dh_prime[i] += h[i];
            } // h' = h + B(Ah); grad wrt h'
            // logits = W h' ; CE
            let mut logits = vec![0f32; self.hg.v];
            for vv in 0..self.hg.v {
                let row = &self.hg.w[vv * d..vv * d + d];
                let mut s = 0f32;
                for i in 0..d {
                    s += row[i] * dh_prime[i];
                }
                logits[vv] = s;
            }
            let max = logits.iter().cloned().fold(f32::MIN, f32::max);
            let lse: f64 = logits.iter().map(|l| ((*l - max) as f64).exp()).sum::<f64>().ln();
            loss -= ((logits[*tgt] - max) as f64 - lse);
            // dz = softmax - onehot
            let mut dz: Vec<f32> =
                logits.iter().map(|l| ((*l - max) as f64).exp() as f32).collect();
            let sum: f32 = dz.iter().sum();
            for z in dz.iter_mut() {
                *z /= sum;
            }
            dz[*tgt] -= 1.0;
            // dh' = W^T dz
            let dhg = self.hg.backward(&dz);
            // grads
            for i in 0..d {
                let bi = &self.ad.b[i * r..(i + 1) * r];
                let gi = &mut gb[i * r..(i + 1) * r];
                for j in 0..r {
                    gi[j] += dhg[i] * ah[j];
                }
                let _ = bi;
            }
            for j in 0..r {
                let mut bahj = 0f32;
                for i in 0..d {
                    bahj += self.ad.b[i * r + j] * dhg[i];
                }
                let ar = &mut ga[j * d..(j + 1) * d];
                for i in 0..d {
                    ar[i] += bahj * h[i];
                }
            }
        }
        // global L-inf clipping (empirically stable configuration)
        let mut mmax = 0f32;
        for v in ga.iter().chain(gb.iter()) { mmax = mmax.max(v.abs()); }
        if mmax > 0.5 {
            let sc = 0.5 / mmax;
            for v in ga.iter_mut() { *v *= sc; }
            for v in gb.iter_mut() { *v *= sc; }
        }
        // finite guard: reject poisoned gradients BEFORE they touch weights
        if !ga.iter().all(|v| v.is_finite()) || !gb.iter().all(|v| v.is_finite()) {
            return f64::NAN;
        }
        let n = states.len().max(1) as f32;
        for i in 0..r * d {
            self.ad.a[i] -= self.lr_a * (ga[i] / n + self.wd * self.ad.a[i]);
        }
        for i in 0..d * r {
            self.ad.b[i] -= self.lr_b * (gb[i] / n + self.wd * self.ad.b[i]);
        }
        loss / n as f64
    }
}
