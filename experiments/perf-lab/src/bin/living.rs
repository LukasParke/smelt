//! The living-engine demo: an inference loop that GROWS while serving.
//!
//! Proves three live-mutation primitives with measured evidence:
//!   t=3s  grow expert pool (+3 rank-1 experts)      -> loss floor drops, steps continue
//!   t=6s  hot-swap kernel (serial -> blocked SIMD)   -> steps/sec jumps, zero restart
//!   t=9s  apply weight delta (exact LoRA arrival)    -> sharp loss drop, steps continue
//! All mutations are pointer swaps under a RwLock; per-step latency spike is measured.
//!
//! Model: additive rank-k predictor  y_hat(x) = sum_k (v_k.x) * u_k
//! Target: rank-4  y(x) = sum_j<4 (p_j.x) * q_j     => capacity-limited until pool grows.
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const D: usize = 2048;
const BATCH: usize = 128;
const RUN_SECS: u64 = 12;

static KERNEL: AtomicUsize = AtomicUsize::new(0); // 0=scalar 1=blocked
static STEPS: AtomicU64 = AtomicU64::new(0);
static STEP_NANOS: AtomicU64 = AtomicU64::new(0);
static STEP_MAX_NANOS: AtomicU64 = AtomicU64::new(0);
static LOSS_EMA_BITS: AtomicU64 = AtomicU64::new(0);

struct Rng(u64);
impl Rng {
    fn next_u(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i]; // serial dependency: honest naive baseline
    }
    acc
}

fn dot_blocked(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let mut i = 0;
    while i + 8 <= a.len() {
        for j in 0..8 {
            acc[j] = f32::mul_add(a[i + j], b[i + j], acc[j]);
        }
        i += 8;
    }
    let mut tot = 0f32;
    while i < a.len() {
        tot += a[i] * b[i];
        i += 1;
    }
    tot + acc.iter().sum::<f32>()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    match KERNEL.load(Ordering::Relaxed) {
        1 => dot_blocked(a, b),
        _ => dot_scalar(a, b),
    }
}

#[derive(Clone)]
struct World {
    // expert k = (u_k, v_k); prediction = sum_k (v_k . x) * u_k
    experts: Vec<(Vec<f32>, Vec<f32>)>,
    gen: u64,
}

struct Shared {
    world: RwLock<Arc<World>>,
}

fn gram_schmidt(rng: &mut Rng, n: usize) -> Vec<Vec<f32>> {
    let mut vs: Vec<Vec<f32>> = Vec::new();
    while vs.len() < n {
        let mut v: Vec<f32> = (0..D).map(|_| rng.normal() as f32).collect();
        for u in &vs {
            let p = dot(u, &v);
            for i in 0..D {
                v[i] -= p * u[i];
            }
        }
        let norm = dot(&v, &v).sqrt();
        if norm > 1e-3 {
            for x in &mut v {
                *x /= norm;
            }
            vs.push(v);
        }
    }
    vs
}

fn forward(world: &World, xs: &[Vec<f32>], ys: &[Vec<f32>], scratch: &mut Vec<Vec<f32>>) -> f64 {
    // returns relative mse; also writes predictions into scratch for the backward pass
    let mut se = 0f64;
    let mut sy = 0f64;
    for (b, x) in xs.iter().enumerate() {
        let mut pred = vec![0f32; D];
        for (uk, vk) in &world.experts {
            let coef = dot(vk, x);
            for i in 0..D {
                pred[i] += coef * uk[i];
            }
        }
        let yb = &ys[b];
        let mut eb2 = 0f64;
        let mut yb2 = 0f64;
        for i in 0..D {
            let e = (pred[i] - yb[i]) as f64;
            eb2 += e * e;
            yb2 += yb[i] as f64 * yb[i] as f64;
        }
        se += eb2;
        sy += yb2;
        scratch[b] = pred;
    }
    se / sy.max(1e-12)
}

fn sgd_step(world: &mut World, xs: &[Vec<f32>], ys: &[Vec<f32>], scratch: &[Vec<f32>], lr: f32) {
    // accumulate gradients for all experts from cached predictions
    let k_n = world.experts.len();
    let mut gu = vec![vec![0f32; D]; k_n];
    let mut gv = vec![vec![0f32; D]; k_n];
    for (b, x) in xs.iter().enumerate() {
        let pred = &scratch[b];
        let yb = &ys[b];
        for k in 0..k_n {
            let (uk, vk) = &world.experts[k];
            let coef = dot(vk, x);
            // dL/dcoef_bk = (pred-y).u_k ; accumulate into v-grad
            let mut g = 0f32;
            for i in 0..D {
                g += (pred[i] - yb[i]) * uk[i];
            }
            let gv_scale = -lr * 2.0 / BATCH as f32;
            for i in 0..D {
                gv[k][i] += gv_scale * g * x[i];
            }
            // dL/du_k = coef*(pred-y)
            let du_scale = -lr * 2.0 * coef / BATCH as f32;
            for i in 0..D {
                gu[k][i] += du_scale * (pred[i] - yb[i]);
            }
        }
    }
    // apply (in-place on the shared world; demo-grade, single writer)
    for k in 0..k_n {
        for i in 0..D {
            world.experts[k].0[i] += gu[k][i];
            world.experts[k].1[i] += gv[k][i];
        }
    }
}

fn apply_mutation(shared: &Shared, mutate: impl FnOnce(&mut World)) -> f64 {
    // critical section: clone current world under read lock, mutate, swap pointer.
    let t0 = Instant::now();
    let mut w: World = (**shared.world.read()).clone();
    mutate(&mut w);
    w.gen += 1;
    let gen = w.gen;
    *shared.world.write() = Arc::new(w);
    let pause_us = t0.elapsed().as_secs_f64() * 1e6;
    println!("   [gen={gen} critical_section_us={pause_us:.1}]");
    pause_us
}

fn main() {
    println!("# perf-lab living | D={D} BATCH={BATCH} run={RUN_SECS}s");
    let mut rng = Rng(42);
    let ps = gram_schmidt(&mut rng, 4);
    let qs = gram_schmidt(&mut rng, 4);

    // fixed dataset
    let xs: Vec<Vec<f32>> = (0..BATCH)
        .map(|_| (0..D).map(|_| rng.normal() as f32).collect())
        .collect();
    let ys: Vec<Vec<f32>> = xs
        .iter()
        .map(|x| {
            let mut y = vec![0f32; D];
            for j in 0..4 {
                let c = dot(&ps[j], x);
                for i in 0..D {
                    y[i] += c * qs[j][i];
                }
            }
            y
        })
        .collect();

    // start with ONE partially-trained expert (rank-1 ceiling)
    let e0_u: Vec<f32> = (0..D)
        .map(|i| 0.9 * ps[0][i] + 0.1 * rng.normal() as f32)
        .collect();
    let e0_v: Vec<f32> = (0..D)
        .map(|i| 0.9 * qs[0][i] + 0.1 * rng.normal() as f32)
        .collect();
    let world0 = Arc::new(World { experts: vec![(e0_u, e0_v)], gen: 1 });

    let shared = Arc::new(Shared { world: RwLock::new(world0.clone()) });
    let stop = Arc::new(AtomicUsize::new(0));

    // ---------------- worker thread ----------------
    let s2 = shared.clone();
    let st2 = stop.clone();
    let worker = thread::spawn(move || {
        let mut scratch: Vec<Vec<f32>> = vec![vec![0f32; D]; BATCH];
        let lr = 2e-3_f32;
        while st2.load(Ordering::Relaxed) == 0 {
            let t0 = Instant::now();
            let arc = s2.world.read().clone(); // pointer grab: ~ns
            let gen = arc.gen;
            let mut owned: World = (*arc).clone(); // private trainable copy
            let loss = forward(&owned, &xs, &ys, &mut scratch);
            sgd_step(&mut owned, &xs, &ys, &scratch, lr);
            // publish only if no event raced ahead (RCU retry semantics)
            {
                let mut g = s2.world.write();
                if g.gen == gen {
                    *g = Arc::new(owned);
                }
            }
            let dt = t0.elapsed().as_nanos() as u64;
            STEPS.fetch_add(1, Ordering::Relaxed);
            STEP_NANOS.fetch_add(dt, Ordering::Relaxed);
            STEP_MAX_NANOS.fetch_max(dt, Ordering::Relaxed);
            let ema = f64::from_bits(LOSS_EMA_BITS.load(Ordering::Relaxed));
            let ema = if ema == 0.0 { loss } else { ema * 0.95 + loss * 0.05 };
            LOSS_EMA_BITS.store(ema.to_bits(), Ordering::Relaxed);
        }
    });

    // ---------------- controller: schedule mutations ----------------
    let t_start = Instant::now();
    let mut last_steps = 0u64;
    let mut next_report = Duration::from_millis(500);
    let mut events_done = [false; 3];
    let mut pause_us_max = 0f64;
    let mut event_pauses = [0f64; 3];

    while t_start.elapsed() < Duration::from_secs(RUN_SECS) {
        thread::sleep(Duration::from_millis(50));
        let el = t_start.elapsed();

        // event 1 @3s: grow expert pool +3 (capacity growth)
        if !events_done[0] && el >= Duration::from_secs(3) {
            println!(
                "EVENT t={:.2}s grow_experts +3 (pool=4)",
                el.as_secs_f64()
            );
            let p = apply_mutation(&shared, |w| {
                for j in 1..4 {
                    let u: Vec<f32> =
                        (0..D).map(|i| 0.3 * ps[j][i] + 0.05 * rng.normal() as f32).collect();
                    let v: Vec<f32> =
                        (0..D).map(|i| 0.3 * qs[j][i] + 0.05 * rng.normal() as f32).collect();
                    w.experts.push((u, v));
                }
            });
            event_pauses[0] = p;
            pause_us_max = pause_us_max.max(p);
            events_done[0] = true;
        }
        // event 2 @6s: hot kernel swap, no restart
        if !events_done[1] && el >= Duration::from_secs(6) {
            let t0 = Instant::now();
            KERNEL.store(1, Ordering::Relaxed);
            let pause = t0.elapsed().as_secs_f64() * 1e6;
            event_pauses[1] = pause;
            pause_us_max = pause_us_max.max(pause);
            events_done[1] = true;
            println!(
                "EVENT t={:.2}s kernel_swap scalar->blocked critical_section_us={pause:.3}",
                el.as_secs_f64()
            );
        }
        // event 3 @9s: weight delta arrives (perfect LoRA for components 1,2)
        if !events_done[2] && el >= Duration::from_secs(9) {
            println!(
                "EVENT t={:.2}s apply_delta experts[1]=exact experts[2]=exact",
                el.as_secs_f64()
            );
            let p = apply_mutation(&shared, |w| {
                if w.experts.len() >= 3 {
                    w.experts[1] = (ps[1].clone(), qs[1].clone());
                    w.experts[2] = (ps[2].clone(), qs[2].clone());
                }
            });
            event_pauses[2] = p;
            pause_us_max = pause_us_max.max(p);
            events_done[2] = true;
        }

        // periodic report
        if el >= next_report {
            let steps = STEPS.load(Ordering::Relaxed);
            let nanos = STEP_NANOS.load(Ordering::Relaxed);
            let maxn = STEP_MAX_NANOS.load(Ordering::Relaxed);
            let done = steps - last_steps;
            last_steps = steps;
            let loss = f64::from_bits(LOSS_EMA_BITS.load(Ordering::Relaxed));
            println!(
                "t={:.1}s steps_total={} rate_s={:.0} step_avg_ms={:.2} step_max_ms={:.2} rel_mse_ema={:.6}",
                el.as_secs_f64(),
                steps,
                done as f64 / 0.5,
                if steps > 0 { nanos as f64 / steps as f64 / 1e6 } else { 0.0 },
                maxn as f64 / 1e6,
                loss
            );
            STEP_MAX_NANOS.store(0, Ordering::Relaxed);
            next_report += Duration::from_millis(500);
        }
    }

    stop.store(1, Ordering::Relaxed);
    let _ = worker.join();
    let final_loss = f64::from_bits(LOSS_EMA_BITS.load(Ordering::Relaxed));
    println!(
        "RESULT {{\"steps\":{}, \"final_rel_mse\":{:.8}, \"max_critical_section_us\":{:.1}, \"event_pause_us\":[{:.1},{:.3},{:.1}], \"kernel_final\":\"blocked\"}}",
        STEPS.load(Ordering::Relaxed),
        final_loss,
        pause_us_max,
        event_pauses[0],
        event_pauses[1],
        event_pauses[2]
    );
}
