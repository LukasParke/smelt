//! Arithmetic-intensity sweep: GEMV (B=1) is structurally bandwidth-bound,
//! GEMM escapes toward compute-bound. Proves the decode roofline doctrine with
//! deliberately plain code: even naive kernels hit the BW wall at B=1.
use std::hint::black_box;
use std::thread;
use std::time::Instant;


/// B == 1: parallelize over output rows; disjoint via chunks_mut.
fn gemv_parallel(w: &[f32], x: &[f32], y: &mut [f32], m: usize, k: usize, threads: usize) {
    let per = (m + threads - 1) / threads;
    thread::scope(|s| {
        let mut handles = Vec::new();
        let mut yi = y.chunks_mut(per);
        for wr in w.chunks(per * k) {
            let Some(yr) = yi.next() else { break };
            handles.push(s.spawn(move || {
                for (i, yv) in yr.iter_mut().enumerate() {
                    let wrow = &wr[i * k..(i + 1) * k];
                    let mut acc = 0f32;
                    let mut acc2 = 0f32;
                    let mut acc3 = 0f32;
                    let mut acc4 = 0f32;
                    let mut j = 0;
                    while j + 4 <= k {
                        acc = f32::mul_add(wrow[j], x[j], acc);
                        acc2 = f32::mul_add(wrow[j + 1], x[j + 1], acc2);
                        acc3 = f32::mul_add(wrow[j + 2], x[j + 2], acc3);
                        acc4 = f32::mul_add(wrow[j + 3], x[j + 3], acc4);
                        j += 4;
                    }
                    let mut tot = acc + acc2 + acc3 + acc4;
                    while j < k {
                        tot += wrow[j] * x[j];
                        j += 1;
                    }
                    *yv = tot;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
}

/// B > 1: parallelize over batch; each thread owns a contiguous batch slab.
fn gemm_batch_parallel(
    w: &[f32],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    batch: usize,
    threads: usize,
) {
    let per_b = (batch + threads - 1) / threads;
    thread::scope(|s| {
        let mut handles = Vec::new();
        let xi = x.chunks(k * per_b);
        let yi = y.chunks_mut(m * per_b);
        for (xc, yc) in xi.zip(yi) {
            handles.push(s.spawn(move || {
                for (xr, yrow) in xc.chunks(k).zip(yc.chunks_mut(m)) {
                    for (r, yv) in yrow.iter_mut().enumerate() {
                        let wrow = &w[r * k..(r + 1) * k];
                        let mut acc = 0f32;
                        let mut acc2 = 0f32;
                        let mut acc3 = 0f32;
                        let mut acc4 = 0f32;
                        let mut j = 0;
                        while j + 4 <= k {
                            acc = f32::mul_add(wrow[j], xr[j], acc);
                            acc2 = f32::mul_add(wrow[j + 1], xr[j + 1], acc2);
                            acc3 = f32::mul_add(wrow[j + 2], xr[j + 2], acc3);
                            acc4 = f32::mul_add(wrow[j + 3], xr[j + 3], acc4);
                            j += 4;
                        }
                        let mut tot = acc + acc2 + acc3 + acc4;
                        while j < k {
                            tot += wrow[j] * xr[j];
                            j += 1;
                        }
                        *yv = tot;
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
}

fn bench(m: usize, k: usize, batch: usize, threads: usize) -> (f64, f64, f64) {
    let w: Vec<f32> = (0..m * k).map(|i| ((i % 2009) as f32 / 2009.0) - 0.5).collect();
    let x: Vec<f32> = (0..batch * k)
        .map(|i| ((i % 1013) as f32 / 1013.0) - 0.5)
        .collect();
    let mut y = vec![0f32; batch * m];
    let run = |y: &mut Vec<f32>| {
        if batch == 1 {
            gemv_parallel(&w, &x, y, m, k, threads);
        } else {
            gemm_batch_parallel(&w, &x, y, m, k, batch, threads);
        }
    };
    run(&mut y);
    black_box(&y);
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        run(&mut y);
        black_box(&y);
        best = best.min(t0.elapsed().as_secs_f64());
    }
    let bytes = (m * k + batch * k + batch * m) as f64 * 4.0;
    let flops = 2.0 * batch as f64 * m as f64 * k as f64;
    (bytes / best / 1e9, flops / best / 1e9, best)
}

fn main() {
    let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let (m, k) = (4096usize, 4096usize); // W = 64 MB fp32 > any cache
    println!("# perf-lab roofline | W {m}x{k} fp32 | threads={threads}");
    println!("# B, time_ms, GB_s, GFLOP_s, AI_flop_per_byte");
    for b in [1usize, 8, 64] {
        let (gbs, gflops, dt) = bench(m, k, b, threads);
        let ai = 2.0 * b as f64 * m as f64 * k as f64 / ((m * k + b * k + b * m) as f64 * 4.0);
        println!("B={b}, {:.2}, {:.1}, {:.1}, {:.2}", dt * 1e3, gbs, gflops, ai);
    }
    println!("# expectation: B=1 GB/s ~= DRAM stream BW (bandwidth-bound even with naive code);");
    println!("#               GFLOP/s climbs with B while GB/s falls -> compute-bound regime");
}
