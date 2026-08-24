//! SMELT-style own-engine CUDA boundary: cudarc driver+NVRTC only (no cuBLAS/cuDNN),
//! embedded .cu sources, blake3(source)-keyed PTX disk cache, device-resident weights
//! INCLUDING raw Q8 payloads dequantized inside the kernels.
#![allow(dead_code)]
use crate::atoms::ATOM_Q8;
use crate::gpt2::{Engine, Meta};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceSlice, LaunchConfig, PushKernelArg,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

pub const CU_SRC: &str = include_str!("cu/gpt2.cu");
const MODULE: &str = "gpt2k";
const FN_NAMES: &[&str] = &[
    "k_q8_gemv", "k_f32_gemv", "k_add_bias", "k_axpy",
    "k_layernorm", "k_gelu", "k_attn_decode", "k_scatter_kv", "k_q8_row",
    "k_gather_row_f32", "k_adapter_apply", "k_wteT_dz",
];

fn ptx_cache_path() -> PathBuf {
    let key = blake3::hash(CU_SRC.as_bytes()).to_hex()[..32].to_string();
    let dir = PathBuf::from("assets/ptx-cache");
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("{key}.ptx"))
}

#[inline]
fn cfg(blocks: usize, threads: usize, smem: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks.max(1) as u32, 1, 1),
        block_dim: (threads.max(1) as u32, 1, 1),
        shared_mem_bytes: smem as u32,
    }
}

macro_rules! k {
    ($st:expr, $fu:expr, $cnt:expr, $name:literal, $cfg:expr $(, $arg:expr)* $(,)?) => {{
        let f: &CudaFunction = &$fu[$name];
        let mut bld = $st.launch_builder(f);
        let trace = std::env::var_os("SMT_TRACE").is_some();
        if trace { eprintln!("launch {} grid={:?} blk={:?} smem={}", $name, $cfg.grid_dim, $cfg.block_dim, $cfg.shared_mem_bytes); }
        unsafe {
            bld $( .arg($arg) )* .launch($cfg);
        }
        *$cnt += 1;
    }};
}

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    funcs: HashMap<&'static str, CudaFunction>,
    meta: Meta,
    is_q8: HashMap<String, bool>,
    u8bufs: HashMap<String, CudaSlice<u8>>,
    f32bufs: HashMap<String, CudaSlice<f32>>,
    kcache: Vec<CudaSlice<f32>>,
    vcache: Vec<CudaSlice<f32>>,
    x: CudaSlice<f32>,
    h: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,
    mid: CudaSlice<f32>,
    fc: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    fc_rows: usize,
    hd: usize,
    pub launches: u64,
    adapter: Option<(CudaSlice<f32>, CudaSlice<f32>, usize)>, // (A,B,r)
}

impl Gpu {
    pub fn new(eng: &Engine) -> Self {
        let ctx = CudaContext::new(0).expect("cuda context");
        eprintln!(
            "GPU: {} | own-kernel path (NVRTC JIT, no cuBLAS)",
            ctx.name().unwrap_or_else(|_| "?".into())
        );
        let stream = ctx.default_stream();

        let cache = ptx_cache_path();
        let t0 = std::time::Instant::now();
        let module = if cache.exists() {
            let ptx = cudarc::nvrtc::Ptx::from_file(cache.clone());
            eprintln!(
                "PTX cache hit {} ({:.0} ms)",
                cache.display(),
                t0.elapsed().as_secs_f64() * 1e3
            );
            ctx.load_module(ptx).expect("load_module(cached)")
        } else {
            let compiled = cudarc::nvrtc::compile_ptx(CU_SRC).expect("NVRTC compile");
            let src = compiled.to_src();
            std::fs::write(&cache, &src).unwrap();
            eprintln!(
                "NVRTC compiled {:.1} kB PTX in {:.0} ms -> cached {}",
                src.len() as f64 / 1024.0,
                t0.elapsed().as_secs_f64() * 1e3,
                cache.display()
            );
            ctx.load_module(compiled).expect("load_module")
        };
        let mut funcs = HashMap::new();
        for n in FN_NAMES {
            funcs.insert(*n, module.load_function(n).expect(n));
        }

        let m = &eng.meta;
        let mut u8bufs = HashMap::new();
        let mut f32bufs = HashMap::new();
        let mut is_q8 = HashMap::new();
        for (name, rec) in &eng.t {
            match rec.atom.as_str() {
                ATOM_Q8 => {
                    is_q8.insert(name.clone(), true);
                    let sl = eng.payload(name);
                    u8bufs.insert(name.clone(), stream.memcpy_stod(sl).unwrap());
                }
                _ => {
                    is_q8.insert(name.clone(), false);
                    let vals = eng.vec_f32(name);
                    f32bufs.insert(name.clone(), stream.memcpy_stod(&vals).unwrap());
                }
            }
        }
        let fc_rows = eng.t["h.0.mlp.c_fc.weight"].shape[0];
        let kcache = (0..m.n_layer)
            .map(|_| stream.alloc_zeros::<f32>(m.n_ctx * m.n_embd).unwrap())
            .collect();
        let vcache = (0..m.n_layer)
            .map(|_| stream.alloc_zeros::<f32>(m.n_ctx * m.n_embd).unwrap())
            .collect();
        Self {
            funcs,
            stream: stream.clone(),
            ctx,
            meta: m.clone(),
            is_q8,
            u8bufs,
            f32bufs,
            kcache,
            vcache,
            x: stream.alloc_zeros(m.n_embd).unwrap(),
            h: stream.alloc_zeros(m.n_embd).unwrap(),
            qkv: stream.alloc_zeros(3 * m.n_embd).unwrap(),
            attn_out: stream.alloc_zeros(m.n_embd).unwrap(),
            mid: stream.alloc_zeros(m.n_embd).unwrap(),
            fc: stream.alloc_zeros(fc_rows).unwrap(),
            logits: stream.alloc_zeros(m.vocab).unwrap(),
            fc_rows,
            hd: m.n_embd / m.n_head,
            launches: 0,
            adapter: None,
        }
    }

    /// One-time upload of the dequantized tied-head/embedding matrix for device-side
    /// head-gradient computation (k_wteT_dz operates on f32).
    pub fn ensure_head_f32(&mut self, eng: &Engine) {
        if !self.f32bufs.contains_key("__wte_f32") {
            let vals = eng.vec_f32("wte.weight");
            let sl = self.stream.memcpy_stod(&vals).unwrap();
            self.f32bufs.insert("__wte_f32".into(), sl);
        }
    }

    /// dh = W_head^T @ dz computed ON DEVICE (requires ensure_head_f32).
    pub fn head_backward_dev(&mut self, dz: &[f32]) -> Vec<f32> {
        let dzd = self.stream.memcpy_stod(dz).unwrap();
        let d = self.meta.n_embd;
        let v = self.meta.vocab;
        let mut dh = self.stream.alloc_zeros::<f32>(d).unwrap();
        {
            let w = self.f32bufs.get("__wte_f32").expect("ensure_head_f32 first");
            let mut bld = self.stream.launch_builder(&self.funcs["k_wteT_dz"]);
            unsafe {
                bld.arg(&mut dh).arg(w).arg(&dzd).arg(&(v as i32)).arg(&(d as i32))
                    .launch(cfg(d.div_ceil(256), 256, 0));
            }
        }
        self.launches += 1;
        self.stream.memcpy_dtov(&dh).unwrap()
    }

    /// Publish a new adapter generation to the device (RCU: next tokens see it).
    pub fn set_adapter(&mut self, a: &[f32], b: &[f32]) {
        let sa = self.stream.memcpy_stod(a).unwrap();
        let sb = self.stream.memcpy_stod(b).unwrap();
        self.adapter = Some((sa, sb, a.len() / self.meta.n_embd));
    }
    pub fn clear_adapter(&mut self) { self.adapter = None; }
    pub fn has_adapter(&self) -> bool { self.adapter.is_some() }

    /// Step that also captures the post-ln_f (pre-adapter) hidden state for training.
    pub fn step_capture(&mut self, tok: u32, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let logits = self.step(tok, pos);
        // NOTE: h here is post-adapter; trainers needing pre-adapter states use CPU path.
        (logits, self.stream.memcpy_dtov(&self.h).unwrap())
    }

    pub fn clear_kv(&mut self) {
        for l in 0..self.meta.n_layer {
            let z = vec![0f32; self.kcache[l].len()];
            self.stream.memcpy_htod(&z, &mut self.kcache[l]).unwrap();
            self.stream.memcpy_htod(&z, &mut self.vcache[l]).unwrap();
        }
    }

    /// One decode step fully on device; returns host logits.
    pub fn step(&mut self, tok: u32, pos: usize) -> Vec<f32> {
        let Gpu {
            ctx: _,
            ref stream,
            ref funcs,
            ref meta,
            ref is_q8,
            ref u8bufs,
            ref f32bufs,
            ref kcache,
            ref vcache,
            ref mut x,
            ref mut h,
            ref mut qkv,
            ref mut attn_out,
            ref mut mid,
            ref mut fc,
            ref mut logits,
            ref adapter,
            fc_rows,
            hd,
            ref mut launches,
        } = *self;
        let embd = meta.n_embd;
        let th = 256usize;
        let nt = pos + 1;

        // embed
        if is_q8["wte.weight"] {
            let w = &u8bufs["wte.weight"];
                        k!(stream, funcs, launches, "k_q8_row", cfg(embd.div_ceil(th), th, 0), &mut *x, w, &(tok as i64), &(embd as i32));
;
        } else {
            let w = &f32bufs["wte.weight"];
                        k!(stream, funcs, launches, "k_gather_row_f32", cfg(embd.div_ceil(th), th, 0), &mut *x, w, &(tok as i64), &(embd as i32));
;
        }
        if is_q8["wpe.weight"] {
            let w = &u8bufs["wpe.weight"];
                        k!(stream, funcs, launches, "k_q8_row", cfg(embd.div_ceil(th), th, 0), &mut *mid, w, &(pos as i64), &(embd as i32));
                        { let mref = &*mid; k!(stream, funcs, launches, "k_axpy", cfg(embd.div_ceil(th), th, 0), &mut *x, mref, &(0i64), &(embd as i32)); }
;
        } else {
            let w = &f32bufs["wpe.weight"];
                        k!(stream, funcs, launches, "k_gather_row_f32", cfg(embd.div_ceil(th), th, 0), &mut *mid, w, &(pos as i64), &(embd as i32));
;
            let mref = &*mid;
                        k!(stream, funcs, launches, "k_axpy", cfg(embd.div_ceil(th), th, 0), &mut *x, mref, &(0i64), &(embd as i32));
;
        }

        for layer in 0..meta.n_layer {
            // attention
            let gw = &f32bufs[&format!("h.{layer}.ln_1.weight")];
            let gb = &f32bufs[&format!("h.{layer}.ln_1.bias")];
                        k!(stream, funcs, launches, "k_layernorm", cfg(1, th, (th + 2) * 4), &mut *h, &mut *x, gw, gb, &(meta.ln_eps), &(embd as i32));
;

            let qkv_w = format!("h.{layer}.attn.c_attn.weight");
            if is_q8[&qkv_w] {
                let w = &u8bufs[&qkv_w];
                                k!(stream, funcs, launches, "k_q8_gemv", cfg((3 * embd).div_ceil(64), 64, 0), &mut *qkv, w, &mut *h, &((3 * embd) as i32), &(embd as i32));
;
            } else {
                let w = &f32bufs[&qkv_w];
                                k!(stream, funcs, launches, "k_f32_gemv", cfg((3 * embd).div_ceil(64), 64, 0), &mut *qkv, w, &mut *h, &((3 * embd) as i32), &(embd as i32));
;
            }
            let qb = &f32bufs[&format!("h.{layer}.attn.c_attn.bias")];
                        k!(stream, funcs, launches, "k_add_bias", cfg((3 * embd).div_ceil(th), th, 0), &mut *qkv, qb, &((3 * embd) as i32));
;

            {
                let kc = &kcache[layer];
                let vc = &vcache[layer];
                                k!(stream, funcs, launches, "k_scatter_kv", cfg(embd.div_ceil(th), th, 0), kc, vc, &mut *qkv, &(pos as i64 * embd as i64), &(embd as i32));
;
            }
            {
                let kc = &kcache[layer];
                let vc = &vcache[layer];
                                k!(stream, funcs, launches, "k_attn_decode", cfg(meta.n_head, hd, (hd + nt) * 4), &mut *attn_out, &mut *qkv, &(0i32), kc, vc, &(nt as i32), &(hd as i32), &(embd as i32));
;
            }

            let proj_w = format!("h.{layer}.attn.c_proj.weight");
            if is_q8[&proj_w] {
                let w = &u8bufs[&proj_w];
                                k!(stream, funcs, launches, "k_q8_gemv", cfg(embd.div_ceil(64), 64, 0), &mut *mid, w, &mut *attn_out, &(embd as i32), &(embd as i32));
;
            } else {
                let w = &f32bufs[&proj_w];
                                k!(stream, funcs, launches, "k_f32_gemv", cfg(embd.div_ceil(64), 64, 0), &mut *mid, w, &mut *attn_out, &(embd as i32), &(embd as i32));
;
            }
            let pb = &f32bufs[&format!("h.{layer}.attn.c_proj.bias")];
                        k!(stream, funcs, launches, "k_add_bias", cfg(embd.div_ceil(th), th, 0), &mut *mid, pb, &(embd as i32));
;
            {
                let mref = &*mid;
                                k!(stream, funcs, launches, "k_axpy", cfg(embd.div_ceil(th), th, 0), &mut *x, mref, &(0i64), &(embd as i32));
;
            }

            // mlp
            let gw2 = &f32bufs[&format!("h.{layer}.ln_2.weight")];
            let gb2 = &f32bufs[&format!("h.{layer}.ln_2.bias")];
                        k!(stream, funcs, launches, "k_layernorm", cfg(1, th, (th + 2) * 4), &mut *h, &mut *x, gw2, gb2, &(meta.ln_eps), &(embd as i32));
;

            let fcw = format!("h.{layer}.mlp.c_fc.weight");
            if is_q8[&fcw] {
                let w = &u8bufs[&fcw];
                                k!(stream, funcs, launches, "k_q8_gemv", cfg((fc_rows).div_ceil(64), 64, 0), &mut *fc, w, &mut *h, &((fc_rows) as i32), &(embd as i32));
;
            } else {
                let w = &f32bufs[&fcw];
                                k!(stream, funcs, launches, "k_f32_gemv", cfg((fc_rows).div_ceil(64), 64, 0), &mut *fc, w, &mut *h, &((fc_rows) as i32), &(embd as i32));
;
            }
            let fb = &f32bufs[&format!("h.{layer}.mlp.c_fc.bias")];
                        k!(stream, funcs, launches, "k_add_bias", cfg((fc_rows).div_ceil(th), th, 0), &mut *fc, fb, &((fc_rows) as i32));
;
                        k!(stream, funcs, launches, "k_gelu", cfg((fc_rows).div_ceil(th), th, 0), &mut *fc, &((fc_rows) as i32));
;

            let mo_w = format!("h.{layer}.mlp.c_proj.weight");
            if is_q8[&mo_w] {
                let w = &u8bufs[&mo_w];
                                k!(stream, funcs, launches, "k_q8_gemv", cfg(embd.div_ceil(64), 64, 0), &mut *mid, w, &mut *fc, &(embd as i32), &((fc_rows) as i32));
;
            } else {
                let w = &f32bufs[&mo_w];
                                k!(stream, funcs, launches, "k_f32_gemv", cfg(embd.div_ceil(64), 64, 0), &mut *mid, w, &mut *fc, &(embd as i32), &((fc_rows) as i32));
;
            }
            let mb = &f32bufs[&format!("h.{layer}.mlp.c_proj.bias")];
                        k!(stream, funcs, launches, "k_add_bias", cfg(embd.div_ceil(th), th, 0), &mut *mid, mb, &(embd as i32));
;
            {
                let mref = &*mid;
                                k!(stream, funcs, launches, "k_axpy", cfg(embd.div_ceil(th), th, 0), &mut *x, mref, &(0i64), &(embd as i32));
;
            }
        }

        let wf = &f32bufs["ln_f.weight"];
        let bf = &f32bufs["ln_f.bias"];
                k!(stream, funcs, launches, "k_layernorm", cfg(1, th, (th + 2) * 4), &mut *h, &mut *x, wf, bf, &(meta.ln_eps), &(embd as i32));
;

        // plastic adapter on post-ln_f stream (state layer; RCU-published)
        if let Some((a, b, r)) = &adapter {
            k!(stream, funcs, launches, "k_adapter_apply",
               cfg(1, 256, r * 4), &mut *h, a, b, &(*r as i32), &(embd as i32));
        }
        if is_q8["wte.weight"] {
            let w = &u8bufs["wte.weight"];
                        k!(stream, funcs, launches, "k_q8_gemv", cfg(meta.vocab.div_ceil(64), 64, 0), &mut *logits, w, &mut *h, &(meta.vocab as i32), &(embd as i32));
;
        } else {
            let w = &f32bufs["wte.weight"];
                        k!(stream, funcs, launches, "k_f32_gemv", cfg(meta.vocab.div_ceil(64), 64, 0), &mut *logits, w, &mut *h, &(meta.vocab as i32), &(embd as i32));
;
        }

        stream.memcpy_dtov(logits).unwrap()
    }

    /// dh = W_head^T @ dz on HOST from dequantized embedding.
    pub fn head_backward_host(eng: &Engine, dz: &[f32]) -> Vec<f32> {
        let w = eng.vec_f32("wte.weight");
        let d = eng.meta.n_embd;
        let v = eng.meta.vocab;
        let mut dh = vec![0f32; d];
        for vv in 0..v {
            let g = dz[vv];
            if g != 0.0 {
                let row = &w[vv * d..vv * d + d];
                for i in 0..d {
                    dh[i] += row[i] * g;
                }
            }
        }
        dh
    }
}
