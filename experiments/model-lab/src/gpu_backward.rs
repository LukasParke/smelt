//! GPU backward kernels (Agent B) — NVRTC module "bwd" from `cu/backward.cu`,
//! blake3-keyed PTX disk cache under `assets/ptx-cache-bwd/`. Mirrors gpu.rs's
//! cudarc 0.19 usage exactly (CudaContext / default_stream / launch_builder /
//! PushKernelArg refs, memcpy_stod/dtov, k!-style launch macro).
//!
//! Correctness-first per the v2 contract: every primitive uploads its inputs and
//! downloads its outputs. Base weights are f32-uploaded once at construction
//! (backward math never touches Q8 payloads); a pointer-keyed upload cache (cleared
//! at the start of each trunk call) keeps repeated per-position weight uploads cheap.
//!
//! GPU-unavailable fallback: `MirrorOps` is an exact Rust port of each kernel's math,
//! always compiled and selected at RUNTIME when the CUDA context cannot be created.
//! NOTE: this intentionally deviates from the suggested `#[cfg(feature =
//! "cpu-mirror")]` gate — adding a Cargo.toml feature is outside Agent B's file
//! ownership, and runtime selection lets bwparity auto-fallback without a rebuild.
#![allow(dead_code)]
use crate::gpt2::{Engine, Meta};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

pub const CU_SRC_BWD: &str = include_str!("cu/backward.cu");
const FN_NAMES: &[&str] = &[
    "k_lin_input_grad",
    "k_ln_backward",
    "k_gelu_bwd",
    "k_attn_bwd_train",
];

fn ptx_cache_path_bwd() -> PathBuf {
    let key = blake3::hash(CU_SRC_BWD.as_bytes()).to_hex()[..32].to_string();
    let dir = PathBuf::from("assets/ptx-cache-bwd");
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

macro_rules! kb {
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

const LN_THREADS: usize = 256;
const GEMV_THREADS: usize = 256;

// ---------------------------------------------------------------------------
// Backend-neutral kernel-op interface (GPU device + CPU mirror implementations)
// ---------------------------------------------------------------------------

pub trait BwdOps {
    fn name(&self) -> &'static str;
    /// Drop any cached uploads (called once per trunk backward).
    fn reset_upload_cache(&mut self) {}
    /// dx[i] = sum_r W[r*cols+i]*dy[r]; rows = dy.len(), cols = dx.len().
    fn lin_input_grad(&mut self, w: &[f32], dy: &[f32], dx: &mut [f32]);
    /// LayerNorm Jacobian; dg/db ACCUMULATE (pass zeroed buffers for fresh grads).
    fn ln_backward(
        &mut self, dy: &[f32], x: &[f32], g: &[f32], b: &[f32], eps: f32,
        dx: &mut [f32], dg: &mut [f32], db: &mut [f32],
    );
    fn gelu_bwd(&mut self, x_pre: &[f32], dy: &[f32], dx: &mut [f32]);
    /// Causal training-form attention backward over T positions; q/k/v/dy are
    /// position-major [T, stride] with head slice at head*hd; dk/dv accumulate
    /// (pre-zero), dq is overwritten.
    fn attn_bwd_train(
        &mut self, q: &[f32], k: &[f32], v: &[f32], dy: &[f32], t: usize, hd: usize, stride: usize,
        dq: &mut [f32], dk: &mut [f32], dv: &mut [f32],
    );
}

/// Runtime backend selection: real CUDA when available, exact Rust mirror otherwise.
pub enum Backend {
    Gpu(GpuBackward),
    Mirror(MirrorOps),
}

impl Backend {
    /// Err(reason) means "GPU unavailable" (after one exclusive-mode retry at +5s).
    pub fn detect(eng: &Engine) -> Result<Backend, String> {
        GpuBackward::new(eng).map(Backend::Gpu)
    }
    pub fn label(&self) -> &'static str {
        match self {
            Backend::Gpu(_) => "gpu",
            Backend::Mirror(_) => "cpu-mirror",
        }
    }
}

impl BwdOps for Backend {
    fn name(&self) -> &'static str { Backend::label(self) }
    fn reset_upload_cache(&mut self) {
        match self {
            Backend::Gpu(g) => g.reset_upload_cache(),
            Backend::Mirror(_) => {}
        }
    }
    fn lin_input_grad(&mut self, w: &[f32], dy: &[f32], dx: &mut [f32]) {
        match self {
            Backend::Gpu(g) => g.lin_input_grad(w, dy, dx),
            Backend::Mirror(m) => m.lin_input_grad(w, dy, dx),
        }
    }
    fn ln_backward(
        &mut self, dy: &[f32], x: &[f32], g: &[f32], b: &[f32], eps: f32,
        dx: &mut [f32], dg: &mut [f32], db: &mut [f32],
    ) {
        match self {
            Backend::Gpu(dev) => dev.ln_backward(dy, x, g, b, eps, dx, dg, db),
            Backend::Mirror(m) => m.ln_backward(dy, x, g, b, eps, dx, dg, db),
        }
    }
    fn gelu_bwd(&mut self, x_pre: &[f32], dy: &[f32], dx: &mut [f32]) {
        match self {
            Backend::Gpu(g) => g.gelu_bwd(x_pre, dy, dx),
            Backend::Mirror(m) => m.gelu_bwd(x_pre, dy, dx),
        }
    }
    fn attn_bwd_train(
        &mut self, q: &[f32], k: &[f32], v: &[f32], dy: &[f32], t: usize, hd: usize, stride: usize,
        dq: &mut [f32], dk: &mut [f32], dv: &mut [f32],
    ) {
        match self {
            Backend::Gpu(g) => g.attn_bwd_train(q, k, v, dy, t, hd, stride, dq, dk, dv),
            Backend::Mirror(m) => m.attn_bwd_train(q, k, v, dy, t, hd, stride, dq, dk, dv),
        }
    }
}

// ---------------------------------------------------------------------------
// GPU backend
// ---------------------------------------------------------------------------

pub struct GpuBackward {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    funcs: HashMap<&'static str, CudaFunction>,
    pub meta: Meta,
    pub launches: u64,
    /// f32-dequantized base weights resident on device (tensor name -> dev).
    f32bufs: HashMap<String, CudaSlice<f32>>,
    /// Host-pointer-keyed upload cache; valid within one trunk call only.
    up_cache: HashMap<usize, CudaSlice<f32>>,
}

impl GpuBackward {
    fn acquire_ctx() -> Result<Arc<CudaContext>, String> {
        match CudaContext::new(0) {
            Ok(c) => Ok(c),
            Err(e) => {
                let first = format!("{e}");
                eprintln!("GPU context error ({first}) — exclusive-mode retry once after 5s");
                std::thread::sleep(std::time::Duration::from_secs(5));
                CudaContext::new(0)
                    .map_err(|e2| format!("CUDA unavailable after retry: {first} / {e2}"))
            }
        }
    }

    pub fn new(eng: &Engine) -> Result<Self, String> {
        let ctx = Self::acquire_ctx()?;
        eprintln!(
            "GPU(bwd): {} | own-kernel path (NVRTC JIT, no cuBLAS)",
            ctx.name().unwrap_or_else(|_| "?".into())
        );
        let stream = ctx.default_stream();

        let cache = ptx_cache_path_bwd();
        let module = if cache.exists() {
            let ptx = cudarc::nvrtc::Ptx::from_file(cache.clone());
            eprintln!("PTX cache hit {}", cache.display());
            ctx.load_module(ptx).map_err(|e| format!("load_module(cached): {e:?}"))?
        } else {
            let compiled =
                cudarc::nvrtc::compile_ptx(CU_SRC_BWD).map_err(|e| format!("NVRTC compile: {e:?}"))?;
            let src = compiled.to_src();
            let _ = std::fs::write(&cache, &src);
            eprintln!(
                "NVRTC(bwd) compiled {:.1} kB PTX -> cached {}",
                src.len() as f64 / 1024.0,
                cache.display()
            );
            ctx.load_module(compiled).map_err(|e| format!("load_module: {e:?}"))?
        };
        let mut funcs = HashMap::new();
        for n in FN_NAMES {
            funcs.insert(*n, module.load_function(n).map_err(|e| format!("load_function({n}): {e:?}"))?);
        }

        // f32-upload every tensor (dequantized). Backward runs fully in fp32.
        let mut f32bufs = HashMap::new();
        for name in eng.t.keys() {
            let vals = eng.vec_f32(name);
            let dev = stream.memcpy_stod(&vals).map_err(|e| format!("upload {name}: {e:?}"))?;
            f32bufs.insert(name.clone(), dev);
        }

        Ok(Self {
            ctx,
            stream,
            funcs,
            meta: eng.meta.clone(),
            launches: 0,
            f32bufs,
            up_cache: HashMap::new(),
        })
    }

    pub fn uploaded_bytes(&self) -> u64 {
        self.f32bufs
            .values()
            .map(|s| (s.len() * std::mem::size_of::<f32>()) as u64)
            .sum()
    }

    pub fn upload_f32(&self, data: &[f32]) -> CudaSlice<f32> {
        self.stream.memcpy_stod(data).expect("memcpy_stod")
    }

    pub fn download_f32(&self, dev: &CudaSlice<f32>) -> Vec<f32> {
        self.stream.memcpy_dtov(dev).expect("memcpy_dtov")
    }

    fn up(&mut self, data: &[f32]) -> CudaSlice<f32> {
        let key = data.as_ptr() as usize;
        if let Some(s) = self.up_cache.get(&key) {
            return s.clone();
        }
        let s = self.stream.memcpy_stod(data).expect("memcpy_stod");
        self.up_cache.insert(key, s.clone());
        s
    }

    /// Device-level transposed gemv: dx[i] = sum_r W[r*cols+i]*dy[r].
    pub fn lin_input_grad_dev(
        &mut self, w: &CudaSlice<f32>, dy: &CudaSlice<f32>, rows: usize, cols: usize,
        dx: &mut CudaSlice<f32>,
    ) {
        let st = self.stream.clone();
        kb!(st, self.funcs, &mut self.launches, "k_lin_input_grad",
            cfg(cols.div_ceil(GEMV_THREADS), GEMV_THREADS, 0),
            dx, w, dy, &(rows as i32), &(cols as i32));
    }

    /// Device-level LayerNorm backward (single block, like forward k_layernorm).
    #[allow(clippy::too_many_arguments)]
    pub fn ln_backward_dev(
        &mut self, dy: &CudaSlice<f32>, x: &CudaSlice<f32>, g: &CudaSlice<f32>, b: &CudaSlice<f32>,
        eps: f32, n: usize, dx: &mut CudaSlice<f32>, dg: &mut CudaSlice<f32>, db: &mut CudaSlice<f32>,
    ) {
        let st = self.stream.clone();
        kb!(st, self.funcs, &mut self.launches, "k_ln_backward",
            cfg(1, LN_THREADS, (LN_THREADS + 4) * 4),
            dx, dg, db, dy, x, g, b, &eps, &(n as i32));
    }

    pub fn gelu_bwd_dev(
        &mut self, x_pre: &CudaSlice<f32>, dy: &CudaSlice<f32>, n: usize, dx: &mut CudaSlice<f32>,
    ) {
        let st = self.stream.clone();
        kb!(st, self.funcs, &mut self.launches, "k_gelu_bwd",
            cfg(n.div_ceil(GEMV_THREADS), GEMV_THREADS, 0),
            dx, x_pre, dy, &(n as i32));
    }

    /// Device-level causal training-form attention backward; one block per head.
    pub fn attn_bwd_train_dev(
        &mut self, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
        dy: &CudaSlice<f32>, t: usize, hd: usize, stride: usize,
        dq: &mut CudaSlice<f32>, dk: &mut CudaSlice<f32>, dv: &mut CudaSlice<f32>,
    ) {
        let heads = stride / hd;
        let smem = (3 * hd + 2 * t) * 4;
        let st = self.stream.clone();
        kb!(st, self.funcs, &mut self.launches, "k_attn_bwd_train",
            cfg(heads, hd, smem),
            dq, dk, dv, q, k, v, dy, &(t as i32), &(hd as i32), &(stride as i32));
    }
}

impl BwdOps for GpuBackward {
    fn name(&self) -> &'static str { "gpu" }

    fn reset_upload_cache(&mut self) {
        self.up_cache.clear();
    }

    fn lin_input_grad(&mut self, w: &[f32], dy: &[f32], dx: &mut [f32]) {
        let rows = dy.len();
        let cols = dx.len();
        assert_eq!(w.len(), rows * cols, "lin_input_grad dims");
        let wd = self.up(w);
        let yd = self.up(dy);
        let mut xd = self.stream.alloc_zeros::<f32>(cols).expect("alloc dx");
        self.lin_input_grad_dev(&wd, &yd, rows, cols, &mut xd);
        let out = self.download_f32(&xd);
        dx.copy_from_slice(&out);
    }

    fn ln_backward(
        &mut self, dy: &[f32], x: &[f32], g: &[f32], b: &[f32], eps: f32,
        dx: &mut [f32], dg: &mut [f32], db: &mut [f32],
    ) {
        let n = x.len();
        assert_eq!(dy.len(), n, "ln_backward dy len");
        let yd = self.up(dy);
        let xd = self.up(x);
        let gd = self.up(g);
        let bd = self.up(b);
        let mut dxd = self.stream.alloc_zeros::<f32>(n).expect("alloc dx");
        let mut dgd = self.stream.alloc_zeros::<f32>(n).expect("alloc dg");
        let mut dbd = self.stream.alloc_zeros::<f32>(n).expect("alloc db");
        self.ln_backward_dev(&yd, &xd, &gd, &bd, eps, n, &mut dxd, &mut dgd, &mut dbd);
        dx.copy_from_slice(&self.download_f32(&dxd));
        dg.copy_from_slice(&self.download_f32(&dgd));
        db.copy_from_slice(&self.download_f32(&dbd));
    }

    fn gelu_bwd(&mut self, x_pre: &[f32], dy: &[f32], dx: &mut [f32]) {
        let n = x_pre.len();
        assert_eq!(dy.len(), n, "gelu_bwd dy len");
        let xd = self.up(x_pre);
        let yd = self.up(dy);
        let mut dxd = self.stream.alloc_zeros::<f32>(n).expect("alloc dx");
        self.gelu_bwd_dev(&xd, &yd, n, &mut dxd);
        dx.copy_from_slice(&self.download_f32(&dxd));
    }

    fn attn_bwd_train(
        &mut self, q: &[f32], k: &[f32], v: &[f32], dy: &[f32], t: usize, hd: usize, stride: usize,
        dq: &mut [f32], dk: &mut [f32], dv: &mut [f32],
    ) {
        let total = t * stride;
        assert_eq!(q.len(), total, "attn q len");
        assert_eq!(k.len(), total, "attn k len");
        assert_eq!(v.len(), total, "attn v len");
        assert_eq!(dy.len(), total, "attn dy len");
        assert_eq!(dq.len(), total, "attn dq len");
        assert_eq!(dk.len(), total, "attn dk len");
        assert_eq!(dv.len(), total, "attn dv len");
        let qd = self.up(q);
        let kd = self.up(k);
        let vd = self.up(v);
        let yd = self.up(dy);
        // dq overwritten by kernel; dk/dv accumulate from zero.
        let mut dqd = self.stream.alloc_zeros::<f32>(total).expect("alloc dq");
        let mut dkd = self.stream.alloc_zeros::<f32>(total).expect("alloc dk");
        let mut dvd = self.stream.alloc_zeros::<f32>(total).expect("alloc dv");
        self.attn_bwd_train_dev(&qd, &kd, &vd, &yd, t, hd, stride, &mut dqd, &mut dkd, &mut dvd);
        dq.copy_from_slice(&self.download_f32(&dqd));
        dk.copy_from_slice(&self.download_f32(&dkd));
        dv.copy_from_slice(&self.download_f32(&dvd));
    }
}

// ---------------------------------------------------------------------------
// CPU mirror — exact Rust port of each kernel's math (runtime fallback backend)
// ---------------------------------------------------------------------------

pub struct MirrorOps;

/// tanh-approx GELU forward scalar (matches cu/gpt2.cu k_gelu / Engine::step).
#[inline]
pub fn fwd_gelu_scalar(v: f32) -> f32 {
    0.5 * v * (1.0 + (0.7978845608028654f32 * (v + 0.044715f32 * v * v * v)).tanh())
}

/// Row-major gemv: out[r] = dot(W[r*cols..], x).
pub fn fwd_gemv(w: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for r in 0..rows {
        let row = &w[r * cols..r * cols + cols];
        let mut acc = 0f32;
        for j in 0..cols {
            acc += row[j] * x[j];
        }
        out[r] = acc;
    }
}

/// LayerNorm forward (matches Engine::layernorm).
pub fn fwd_layernorm(x: &[f32], g: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    for i in 0..n {
        out[i] = (x[i] - mean) / (var + eps).sqrt() * g[i] + b[i];
    }
}

/// Causal training-form attention forward over T positions (matches the math that
/// k_attn_decode applies per decode step); returns [T, stride] output.
pub fn fwd_attn_train(
    q: &[f32], k: &[f32], v: &[f32], t: usize, hd: usize, stride: usize, out: &mut [f32],
) {
    let scale = 1.0 / (hd as f32).sqrt();
    let heads = stride / hd;
    for head in 0..heads {
        let base = head * hd;
        for i in 0..t {
            let qi = &q[i * stride + base..i * stride + base + hd];
            let mut scores = vec![0f32; i + 1];
            let mut mx = f32::MIN;
            for j in 0..=i {
                let kj = &k[j * stride + base..j * stride + base + hd];
                let mut d = 0f32;
                for tt in 0..hd {
                    d += qi[tt] * kj[tt];
                }
                scores[j] = d * scale;
                mx = mx.max(scores[j]);
            }
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for j in 0..=i {
                let vj = &v[j * stride + base..j * stride + base + hd];
                for tt in 0..hd {
                    out[i * stride + base + tt] += scores[j] * vj[tt];
                }
            }
        }
    }
}

impl BwdOps for MirrorOps {
    fn name(&self) -> &'static str { "cpu-mirror" }

    fn lin_input_grad(&mut self, w: &[f32], dy: &[f32], dx: &mut [f32]) {
        let rows = dy.len();
        let cols = dx.len();
        assert_eq!(w.len(), rows * cols, "lin_input_grad dims");
        for i in 0..cols {
            let mut acc = 0f32;
            for r in 0..rows {
                acc += w[r * cols + i] * dy[r];
            }
            dx[i] = acc;
        }
    }

    fn ln_backward(
        &mut self, dy: &[f32], x: &[f32], g: &[f32], b: &[f32], eps: f32,
        dx: &mut [f32], dg: &mut [f32], db: &mut [f32],
    ) {
        let n = x.len();
        assert_eq!(dy.len(), n, "ln_backward dy len");
        let _ = b;
        let mean = x.iter().sum::<f32>() / n as f32;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        // mdxh = mean(dxhat), mdxh_xh = mean(dxhat*xhat)
        let mut mdxh = 0f32;
        let mut mdxh_xh = 0f32;
        for i in 0..n {
            let dh = dy[i] * g[i];
            mdxh += dh;
            mdxh_xh += dh * (x[i] - mean) * rstd;
        }
        mdxh /= n as f32;
        mdxh_xh /= n as f32;
        for i in 0..n {
            let xh = (x[i] - mean) * rstd;
            let dh = dy[i] * g[i];
            dx[i] = rstd * (dh - mdxh - xh * mdxh_xh);
            dg[i] += dy[i] * xh;
            db[i] += dy[i];
        }
    }

    fn gelu_bwd(&mut self, x_pre: &[f32], dy: &[f32], dx: &mut [f32]) {
        let n = x_pre.len();
        assert_eq!(dy.len(), n, "gelu_bwd dy len");
        for i in 0..n {
            let v = x_pre[i];
            let u = 0.7978845608028654f32 * (v + 0.044715f32 * v * v * v);
            let th = u.tanh();
            let sech2 = 1.0 - th * th;
            let dgdv = 0.5 * (1.0 + th)
                + 0.5 * v * sech2 * 0.7978845608028654f32 * (1.0 + 3.0 * 0.044715f32 * v * v);
            dx[i] = dy[i] * dgdv;
        }
    }

    fn attn_bwd_train(
        &mut self, q: &[f32], k: &[f32], v: &[f32], dy: &[f32], t: usize, hd: usize, stride: usize,
        dq: &mut [f32], dk: &mut [f32], dv: &mut [f32],
    ) {
        let total = t * stride;
        assert_eq!(q.len(), total, "attn q len");
        let scale = 1.0 / (hd as f32).sqrt();
        let heads = stride / hd;
        for head in 0..heads {
            let base = head * hd;
            for i in 0..t {
                // recompute P row (causal)
                let mut scores = vec![0f32; i + 1];
                let mut mx = f32::MIN;
                for j in 0..=i {
                    let mut d = 0f32;
                    for tt in 0..hd {
                        d += q[i * stride + base + tt] * k[j * stride + base + tt];
                    }
                    scores[j] = d * scale;
                    mx = mx.max(scores[j]);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }
                // dP_j = <dyout_i, V_j>; rs = sum dP*P
                let mut dp = vec![0f32; i + 1];
                let mut rs = 0f32;
                for j in 0..=i {
                    let mut d = 0f32;
                    for tt in 0..hd {
                        d += dy[i * stride + base + tt] * v[j * stride + base + tt];
                    }
                    dp[j] = d;
                    rs += dp[j] * scores[j];
                }
                // dq_i — ds carries the 1/sqrt(hd) score scale into q/k
                // (dv does not: out = P V directly).
                let ds: Vec<f32> = (0..=i)
                    .map(|j| (dp[j] - rs) * scores[j] * scale)
                    .collect();
                for tt in 0..hd {
                    let mut acc = 0f32;
                    for j in 0..=i {
                        acc += ds[j] * k[j * stride + base + tt];
                    }
                    dq[i * stride + base + tt] = acc;
                }
                // dk_j += ds_ij q_i ; dv_j += P_ij dyout_i
                for j in 0..=i {
                    let ds = ds[j];
                    let pj = scores[j];
                    for tt in 0..hd {
                        dk[j * stride + base + tt] += ds * q[i * stride + base + tt];
                        dv[j * stride + base + tt] += pj * dy[i * stride + base + tt];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Full-trunk input gradient (reverse of Engine::step order), generic over backend
// ---------------------------------------------------------------------------

/// Saved forward activations needed by the trunk backward. Each per-layer field is
/// position-major flat [T * dim]: x_in/h1/h2/qkv/attn_out are [T*n_embd], fc_pre is
/// [T*fc_rows]; lnf_x is [T*n_embd].
/// Backward-side view of the trunk weights (f32, row-major [out,in]).
pub struct LayerBwdW {
    pub w_aproj: Vec<f32>, // [E,E]
    pub w_cattn: Vec<f32>, // [3E,E]
    pub w_fc: Vec<f32>,    // [F,E]
    pub w_mproj: Vec<f32>, // [E,F]
    pub g1: Vec<f32>,
    pub b1: Vec<f32>,
    pub g2: Vec<f32>,
    pub b2: Vec<f32>,
    pub f_rows: usize,
}

pub struct TrunkWeights {
    pub wte: Vec<f32>, // [vocab, E]
    pub wf: Vec<f32>,
    pub bf: Vec<f32>,
    pub layers: Vec<LayerBwdW>,
    pub ln_eps: f32,
    pub n_head: usize,
}

impl TrunkWeights {
    pub fn from_engine(eng: &Engine) -> Self {
        let m = &eng.meta;
        let layers = (0..m.n_layer)
            .map(|l| {
                let fc_name = format!("h.{l}.mlp.c_fc.weight");
                LayerBwdW {
                    w_aproj: eng.vec_f32(&format!("h.{l}.attn.c_proj.weight")),
                    w_cattn: eng.vec_f32(&format!("h.{l}.attn.c_attn.weight")),
                    w_fc: eng.vec_f32(&fc_name),
                    w_mproj: eng.vec_f32(&format!("h.{l}.mlp.c_proj.weight")),
                    g1: eng.vec_f32(&format!("h.{l}.ln_1.weight")),
                    b1: eng.vec_f32(&format!("h.{l}.ln_1.bias")),
                    g2: eng.vec_f32(&format!("h.{l}.ln_2.weight")),
                    b2: eng.vec_f32(&format!("h.{l}.ln_2.bias")),
                    f_rows: eng.t[&fc_name].shape[0],
                }
            })
            .collect();
        TrunkWeights {
            wte: eng.vec_f32("wte.weight"),
            wf: eng.vec_f32("ln_f.weight"),
            bf: eng.vec_f32("ln_f.bias"),
            layers,
            ln_eps: m.ln_eps,
            n_head: m.n_head,
        }
    }
}

#[derive(Default, Clone)]
pub struct TrunkActs {
    pub x_in: Vec<Vec<f32>>,
    /// Residual stream AFTER the attention branch added, i.e. the pre-LN_2 input.
    pub x_mid: Vec<Vec<f32>>,
    pub h1: Vec<Vec<f32>>,
    pub h2: Vec<Vec<f32>>,
    pub qkv: Vec<Vec<f32>>,
    pub fc_pre: Vec<Vec<f32>>,
    pub lnf_x: Vec<f32>,
    /// post-ln_f hidden state [T*n_embd]; needed because wte is a TIED head:
    /// dL/dwte[v,i] = sum_p dx[p][i] (input path, token v) + sum_p dlogits[p][v]*lnf_out[p][i].
    pub lnf_out: Vec<f32>,
}

/// Backprop dlogits through head + ln_f + all decoder layers to the embedding output.
/// Returns dx per position ([T][n_embd]): gradient wrt (wte_row + wpe_row) at each pos.
pub fn trunk_input_grad<O: BwdOps + ?Sized>(
    ops: &mut O,
    tw: &TrunkWeights,
    ids: &[u32],
    acts: &TrunkActs,
    dlogits: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let tp = ids.len();
    let e = tw.wf.len();
    ops.reset_upload_cache();

    let wte = &tw.wte;
    let wf = &tw.wf;
    let bf = &tw.bf;

    let mut dx_res: Vec<Vec<f32>> = Vec::with_capacity(tp);
    for p in 0..tp {
        // head: dh = wte^T @ dlogits[p]
        let mut dh = vec![0f32; e];
        ops.lin_input_grad(&wte, &dlogits[p], &mut dh);
        // ln_f backward
        let mut dxp = vec![0f32; e];
        let mut zg = vec![0f32; e];
        let mut zb = vec![0f32; e];
        ops.ln_backward(&dh, &acts.lnf_x[p * e..(p + 1) * e], wf, bf, tw.ln_eps, &mut dxp, &mut zg, &mut zb);
        dx_res.push(dxp);
    }

    let hd = e / tw.n_head;
    for (layer, lw) in tw.layers.iter().enumerate().rev() {
        let w_aproj = &lw.w_aproj;
        let w_cattn = &lw.w_cattn;
        let f_rows = lw.f_rows;
        let w_fc = &lw.w_fc;
        let w_mproj = &lw.w_mproj;
        let g1 = &lw.g1;
        let b1 = &lw.b1;
        let g2 = &lw.g2;
        let b2 = &lw.b2;

        let x_in_l = &acts.x_in[layer];
        let x_mid_l = &acts.x_mid[layer];
        let qkv_l = &acts.qkv[layer];
        let fc_pre_l = &acts.fc_pre[layer];

        // ---- MLP branch (added last in fwd, processed first in bwd) ----
        let mut dxl2_all: Vec<Vec<f32>> = Vec::with_capacity(tp);
        for p in 0..tp {
            let incoming = &dx_res[p];
            let mut d_fc_gelu = vec![0f32; f_rows];
            ops.lin_input_grad(&w_mproj, incoming, &mut d_fc_gelu);
            let mut d_fc = vec![0f32; f_rows];
            ops.gelu_bwd(&fc_pre_l[p * f_rows..(p + 1) * f_rows], &d_fc_gelu, &mut d_fc);
            let mut dh2 = vec![0f32; e];
            ops.lin_input_grad(&w_fc, &d_fc, &mut dh2);
            // NOTE: dh2 flows through ln_2 whose input is x_in (residual stream).
            let mut dxl2 = vec![0f32; e];
            let mut zg = vec![0f32; e];
            let mut zb = vec![0f32; e];
            ops.ln_backward(&dh2, &x_mid_l[p * e..(p + 1) * e], g2, b2, tw.ln_eps, &mut dxl2, &mut zg, &mut zb);
            dxl2_all.push(dxl2);
        }

        // ---- Attention branch ----
        // CRITICAL: x_mid = x_in + A(x_in) feeds the MLP branch, so the attention
        // branch must backprop BOTH the direct residual gradient AND dxl2 (the
        // gradient that flowed into x_mid through the MLP):
        //   dL/dx_in = incoming + J_A^T(incoming + dxl2) + dxl2.
        // combined = d(x_mid): routes BOTH through c_proj^T (into attention)
        // AND as residual passthrough into x_in (added in the combine step).
        let mut dattn_all: Vec<f32> = vec![0f32; tp * e];
        for p in 0..tp {
            let mut dattn = vec![0f32; e];
            let mut combined = vec![0f32; e];
            for i in 0..e {
                combined[i] = dx_res[p][i] + dxl2_all[p][i];
            }
            ops.lin_input_grad(&w_aproj, &combined, &mut dattn);
            dattn_all[p * e..(p + 1) * e].copy_from_slice(&dattn);
        }
        // Extract per-stream q,k,v as [T, E] arrays (qkv rows are [q|k|v]).
        let mut qa = vec![0f32; tp * e];
        let mut ka = vec![0f32; tp * e];
        let mut va = vec![0f32; tp * e];
        for p in 0..tp {
            let row = &qkv_l[p * 3 * e..(p + 1) * 3 * e];
            qa[p * e..(p + 1) * e].copy_from_slice(&row[..e]);
            ka[p * e..(p + 1) * e].copy_from_slice(&row[e..2 * e]);
            va[p * e..(p + 1) * e].copy_from_slice(&row[2 * e..]);
        }
        // Kernel works on per-stream [T, E] arrays; qkv rows are per-position
        // interleaved [q_p|k_p|v_p], so scatter the grads back into that layout.
        let mut d_q = vec![0f32; tp * e];
        let mut d_k = vec![0f32; tp * e];
        let mut d_v = vec![0f32; tp * e];
        ops.attn_bwd_train(&qa, &ka, &va, &dattn_all, tp, hd, e, &mut d_q, &mut d_k, &mut d_v);
        let mut dqkv: Vec<f32> = vec![0f32; tp * 3 * e];
        for p in 0..tp {
            dqkv[p * 3 * e..p * 3 * e + e].copy_from_slice(&d_q[p * e..(p + 1) * e]);
            dqkv[p * 3 * e + e..p * 3 * e + 2 * e].copy_from_slice(&d_k[p * e..(p + 1) * e]);
            dqkv[p * 3 * e + 2 * e..(p + 1) * 3 * e].copy_from_slice(&d_v[p * e..(p + 1) * e]);
        }
        let mut dxl1_all: Vec<Vec<f32>> = Vec::with_capacity(tp);
        for p in 0..tp {
            let drow = &dqkv[p * 3 * e..(p + 1) * 3 * e];
            let mut dh1 = vec![0f32; e];
            ops.lin_input_grad(&w_cattn, drow, &mut dh1);
            let mut dxl1 = vec![0f32; e];
            let mut zg = vec![0f32; e];
            let mut zb = vec![0f32; e];
            ops.ln_backward(&dh1, &x_in_l[p * e..(p + 1) * e], g1, b1, tw.ln_eps, &mut dxl1, &mut zg, &mut zb);
            dxl1_all.push(dxl1);
        }

        // ---- residual combine: dx_prev = incoming + dx_ln1_branch + dx_ln2_branch ----
        let mut next: Vec<Vec<f32>> = Vec::with_capacity(tp);
        for p in 0..tp {
            let mut acc = dx_res[p].clone();
            for i in 0..e {
                acc[i] += dxl1_all[p][i] + dxl2_all[p][i];
            }
            next.push(acc);
        }
        dx_res = next;
    }
    dx_res
}


// ---------------------------------------------------------------------------
// Numeric finite-difference self-checks of each kernel's math. The mirror's
// forward functions double as the reference; the SAME checks run through any
// BwdOps backend, so on a GPU box they validate the actual device kernels.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub enum WhichKernel {
    LinInputGrad,
    LnBackward,
    GeluBwd,
    AttnBwdTrain,
}

impl WhichKernel {
    pub fn name(self) -> &'static str {
        match self {
            WhichKernel::LinInputGrad => "k_lin_input_grad",
            WhichKernel::LnBackward => "k_ln_backward",
            WhichKernel::GeluBwd => "k_gelu_bwd",
            WhichKernel::AttnBwdTrain => "k_attn_bwd_train",
        }
    }
    pub const ALL: [WhichKernel; 4] = [
        WhichKernel::LinInputGrad,
        WhichKernel::LnBackward,
        WhichKernel::GeluBwd,
        WhichKernel::AttnBwdTrain,
    ];
}

pub struct KernelCheck {
    pub kernel: &'static str,
    pub backend: &'static str,
    pub samples: usize,
    pub max_rel: f64,
    pub tol: f64,
    pub pass: bool,
}

impl KernelCheck {
    pub fn line(&self) -> String {
        format!(
            "KCHECK {} [{}] samples={} max_rel={:.3e} tol={:.0e} {}",
            self.kernel,
            self.backend,
            self.samples,
            self.max_rel,
            self.tol,
            if self.pass { "PASS" } else { "FAIL" }
        )
    }
}

/// Deterministic xorshift64* RNG shared by all checks (fixed seeds).
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// uniform in [-1, 1)
    pub fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f64 / (1u64 << 24) as f64) as f32 * 2.0 - 1.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Max relative difference with a global-scale floor so near-zero gradients
/// don't blow up the metric: den = max(|a|, |f|, 1e-3 * max_j |f_j|).
fn rel_max(a: &[f32], fd: &[f32]) -> f64 {
    let fscale = fd.iter().map(|v| v.abs() as f64).fold(0f64, f64::max).max(1e-30) * 1e-3;
    a.iter()
        .zip(fd.iter())
        .map(|(x, y)| {
            let num = (*x - *y).abs() as f64;
            let den = ((*x).abs() as f64).max((*y).abs() as f64).max(fscale);
            num / den
        })
        .fold(0f64, f64::max)
}

/// f64 reference forwards (mirror math in double precision) — make the FD checks
/// measure kernel MATH, not f32 finite-difference noise.
mod f64ref {
    pub fn gelu(v: f64) -> f64 {
        0.5 * v * (1.0 + (0.7978845608028654 * (v + 0.044715 * v * v * v)).tanh())
    }
    pub fn layernorm(x: &[f32], g: &[f32], b: &[f32], eps: f32, out: &mut [f64]) {
        let n = x.len();
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = x.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / n as f64;
        for i in 0..n {
            out[i] = (x[i] as f64 - mean) / (var + eps as f64).sqrt() * g[i] as f64 + b[i] as f64;
        }
    }
    /// Causal training-form attention forward; adds into `out` ([T, stride]).
    pub fn attn_add(q: &[f32], k: &[f32], v: &[f32], t: usize, hd: usize, stride: usize, out: &mut [f64]) {
        let scale = 1.0 / (hd as f64).sqrt();
        let heads = stride / hd;
        for head in 0..heads {
            let base = head * hd;
            for i in 0..t {
                let mut scores = vec![0f64; i + 1];
                for j in 0..=i {
                    let mut d = 0f64;
                    for tt in 0..hd {
                        d += q[i * stride + base + tt] as f64 * k[j * stride + base + tt] as f64;
                    }
                    scores[j] = d * scale;
                }
                let mx = scores.iter().cloned().fold(f64::MIN, f64::max);
                let mut sum = 0f64;
                for s in scores.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
                for s in scores.iter_mut() { *s /= sum; }
                for j in 0..=i {
                    for tt in 0..hd {
                        out[i * stride + base + tt] += scores[j] * v[j * stride + base + tt] as f64;
                    }
                }
            }
        }
    }
}

pub fn selfcheck_kernel(ops: &mut dyn BwdOps, which: WhichKernel) -> KernelCheck {
    let backend = ops.name();
    match which {
        WhichKernel::LinInputGrad => {
            // F(y) = c · (W @ y) in f64  =>  dF/dy = W^T c == lin_input_grad(W, c)
            let mut r = Rng::new(101);
            let (rows, cols) = (16usize, 24usize);
            let w: Vec<f32> = (0..rows * cols).map(|_| r.f32()).collect();
            let c: Vec<f32> = (0..rows).map(|_| r.f32()).collect();
            let mut dx = vec![0f32; cols];
            ops.lin_input_grad(&w, &c, &mut dx);
            let wf: Vec<f64> = w.iter().map(|&v| v as f64).collect();
            let cf: Vec<f64> = c.iter().map(|&v| v as f64).collect();
            let h = 1e-5f64;
            let mut y = vec![0f64; cols];
            let mut fd = vec![0f64; cols];
            for i in 0..cols {
                let f = |y: &[f64]| -> f64 {
                    let mut acc = 0f64;
                    for rr in 0..rows {
                        let row = &wf[rr * cols..rr * cols + cols];
                        let mut d = 0f64;
                        for j in 0..cols { d += row[j] * y[j]; }
                        acc += cf[rr] * d;
                    }
                    acc
                };
                y[i] += h;
                let lp = f(&y);
                y[i] -= 2.0 * h;
                let lm = f(&y);
                y[i] += h;
                fd[i] = (lp - lm) / (2.0 * h);
            }
            let fd32: Vec<f32> = fd.iter().map(|&v| v as f32).collect();
            let mr = rel_max(&dx, &fd32);
            KernelCheck { kernel: which.name(), backend, samples: cols, max_rel: mr, tol: 1e-4, pass: mr < 1e-4 }
        }
        WhichKernel::LnBackward => {
            // F(x,g,b) = dy0 · layernorm(x,g,b), evaluated in f64 with f64 perturbation
            let mut r = Rng::new(202);
            let n = 64usize;
            let eps = 1e-5f32;
            let x0: Vec<f32> = (0..n).map(|_| r.f32()).collect();
            let g0: Vec<f32> = (0..n).map(|_| r.f32()).collect();
            let b0: Vec<f32> = (0..n).map(|_| r.f32()).collect();
            let dy0: Vec<f64> = (0..n).map(|_| r.f32() as f64).collect();
            let mut dx = vec![0f32; n];
            let mut dg = vec![0f32; n];
            let mut db = vec![0f32; n];
            ops.ln_backward(&dy0.iter().map(|&v| v as f32).collect::<Vec<_>>(), &x0, &g0, &b0, eps, &mut dx, &mut dg, &mut db);
            let (xf, gf, bf) = (
                x0.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                g0.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                b0.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            );
            let df = |xx: &[f64], gg: &[f64], bb: &[f64]| -> f64 {
                let mut out = vec![0f64; n];
                let mean = xx.iter().sum::<f64>() / n as f64;
                let var = xx.iter().map(|&v| { let d = v - mean; d * d }).sum::<f64>() / n as f64;
                for i in 0..n {
                    out[i] = (xx[i] - mean) / (var + eps as f64).sqrt() * gg[i] + bb[i];
                }
                out.iter().zip(dy0.iter()).map(|(a, d)| a * d).sum()
            };
            let h = 1e-6f64;
            let fd_of = |which: usize, arr: &mut Vec<f64>| -> Vec<f32> {
                let mut g = vec![0f32; n];
                for i in 0..n {
                    let keep = arr[i];
                    arr[i] = keep + h;
                    let lp = match which {
                        0 => df(arr, &gf, &bf),
                        1 => df(&xf, arr, &bf),
                        _ => df(&xf, &gf, arr),
                    };
                    arr[i] = keep - h;
                    let lm = match which {
                        0 => df(arr, &gf, &bf),
                        1 => df(&xf, arr, &bf),
                        _ => df(&xf, &gf, arr),
                    };
                    arr[i] = keep;
                    g[i] = ((lp - lm) / (2.0 * h)) as f32;
                }
                g
            };
            let mut xf2 = xf.clone();
            let mut gf2 = gf.clone();
            let mut bf2 = bf.clone();
            let fdx = fd_of(0, &mut xf2);
            let fdg = fd_of(1, &mut gf2);
            let fdb = fd_of(2, &mut bf2);
            let mr = rel_max(&dx, &fdx).max(rel_max(&dg, &fdg)).max(rel_max(&db, &fdb));
            KernelCheck { kernel: which.name(), backend, samples: 3 * n, max_rel: mr, tol: 1e-3, pass: mr < 1e-3 }
        }
        WhichKernel::GeluBwd => {
            // F(x) = dy0 · gelu(x) in f64
            let mut r = Rng::new(303);
            let n = 32usize;
            let xp: Vec<f32> = (0..n).map(|_| r.f32() * 3.0).collect();
            let dy0: Vec<f64> = (0..n).map(|_| r.f32() as f64).collect();
            let mut dx = vec![0f32; n];
            ops.gelu_bwd(&xp, &dy0.iter().map(|&v| v as f32).collect::<Vec<_>>(), &mut dx);
            let xf: Vec<f64> = xp.iter().map(|&v| v as f64).collect();
            let h = 1e-6f64;
            let mut fd = vec![0f64; n];
            let mut xx = xf.clone();
            for i in 0..n {
                xx[i] += h;
                let lp: f64 = xx.iter().zip(dy0.iter()).map(|(v, d)| f64ref::gelu(*v) * d).sum();
                xx[i] -= 2.0 * h;
                let lm: f64 = xx.iter().zip(dy0.iter()).map(|(v, d)| f64ref::gelu(*v) * d).sum();
                xx[i] += h;
                fd[i] = (lp - lm) / (2.0 * h);
            }
            let fd32: Vec<f32> = fd.iter().map(|&v| v as f32).collect();
            let mr = rel_max(&dx, &fd32);
            KernelCheck { kernel: which.name(), backend, samples: n, max_rel: mr, tol: 1e-3, pass: mr < 1e-3 }
        }
        WhichKernel::AttnBwdTrain => {
            // F(q,k,v) = Σ dy0 * attn_fwd(q,k,v) in f64 over ALL q/k/v coordinates.
            let mut r = Rng::new(404);
            let (t_len, hd, stride) = (5usize, 8usize, 8usize);
            let total = t_len * stride;
            let q: Vec<f32> = (0..total).map(|_| r.f32()).collect();
            let k: Vec<f32> = (0..total).map(|_| r.f32()).collect();
            let v: Vec<f32> = (0..total).map(|_| r.f32()).collect();
            let dy0: Vec<f32> = (0..total).map(|_| r.f32()).collect();
            let mut dq = vec![0f32; total];
            let mut dk = vec![0f32; total];
            let mut dv = vec![0f32; total];
            ops.attn_bwd_train(&q, &k, &v, &dy0, t_len, hd, stride, &mut dq, &mut dk, &mut dv);
            let mut out = vec![0f64; total];
            let mut f = |qq: &[f32], kk: &[f32], vv: &[f32]| -> f64 {
                for o in out.iter_mut() { *o = 0.0; }
                f64ref::attn_add(qq, kk, vv, t_len, hd, stride, &mut out);
                out.iter().zip(dy0.iter()).map(|(a, d)| a * *d as f64).sum()
            };
            let h = 1e-4f32;
            // straightforward per-array FD (explicit, no closure gymnastics):
            let mut fdq = vec![0f32; total];
            let mut fq = q.clone();
            for i in 0..total {
                fq[i] += h;
                let lp = f(&fq, &k, &v);
                fq[i] -= 2.0 * h;
                let lm = f(&fq, &k, &v);
                fq[i] += h;
                fdq[i] = ((lp - lm) / (2.0 * h as f64)) as f32;
            }
            let mut fdk = vec![0f32; total];
            let mut fk = k.clone();
            for i in 0..total {
                fk[i] += h;
                let lp = f(&q, &fk, &v);
                fk[i] -= 2.0 * h;
                let lm = f(&q, &fk, &v);
                fk[i] += h;
                fdk[i] = ((lp - lm) / (2.0 * h as f64)) as f32;
            }
            let mut fdv = vec![0f32; total];
            let mut fv = v.clone();
            for i in 0..total {
                fv[i] += h;
                let lp = f(&q, &k, &fv);
                fv[i] -= 2.0 * h;
                let lm = f(&q, &k, &fv);
                fv[i] += h;
                fdv[i] = ((lp - lm) / (2.0 * h as f64)) as f32;
            }
            let mr = rel_max(&dq, &fdq).max(rel_max(&dk, &fdk)).max(rel_max(&dv, &fdv));
            KernelCheck { kernel: which.name(), backend, samples: 3 * total, max_rel: mr, tol: 1e-2, pass: mr < 1e-2 }
        }
    }
}
