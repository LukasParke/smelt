//! Memory-hierarchy bandwidth characterization (CPU side).
//! Validates the SMELT doctrine numbers: decode is weight-streaming bound.
use std::hint::black_box;
use std::thread;
use std::time::Instant;

/// Multi-threaded streaming-read bandwidth over `buf`.
/// ALL passes timed inside ONE scope: thread spawn cost amortized.
fn read_bw(buf: &[f32], threads: usize, passes: usize) -> f64 {
    let per = (buf.len() + threads - 1) / threads;
    let t0 = Instant::now();
    thread::scope(|s| {
        let mut handles = Vec::new();
        for sl in buf.chunks(per) {
            handles.push(s.spawn(move || {
                let mut sink: u64 = 0;
                for _ in 0..passes {
                    let mut acc: u64 = 0;
                    for v in sl {
                        acc = acc.wrapping_add(v.to_bits() as u64);
                    }
                    sink = sink.wrapping_add(acc);
                }
                black_box(sink)
            }));
        }
        for h in handles {
            black_box(h.join().unwrap());
        }
    });
    let dt = t0.elapsed().as_secs_f64();
    (buf.len() * 4 * passes) as f64 / dt / 1e9
}

/// Triad-style stream kernel: writes two outputs derived from one input.
/// Reports achieved GB/s counting R+W traffic (read a, write b, write c).
fn triad(a: &[f32], b: &mut [f32], c: &mut [f32], threads: usize, passes: usize) -> f64 {
    let n = a.len();
    let per = (n + threads - 1) / threads;
    let t0 = Instant::now();
    thread::scope(|s| {
        let mut handles = Vec::new();
        let mut bi = b.chunks_mut(per);
        let mut citer = c.chunks_mut(per);
        for ac in a.chunks(per) {
            let bc = bi.next().unwrap();
            let cc = citer.next().unwrap();
            handles.push(s.spawn(move || {
                for _ in 0..passes {
                    for i in 0..ac.len() {
                        bc[i] = ac[i] * 3.1;
                        cc[i] = ac[i] + 1.7;
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
    let dt = t0.elapsed().as_secs_f64();
    (n * 4 * 3 * passes) as f64 / dt / 1e9
}

fn main() {
    let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    println!("# perf-lab bw | threads={threads}");
    // L1-resident (48KB/core): single thread
    let l1 = vec![1.5f32; 8 * 1024]; // 32 KB
    println!("L1_32KB_1t_read_GBs   {:.1}", read_bw(&l1, 1, 20_000));
    // L2-resident (1MB)
    let l2 = vec![1.5f32; 256 * 1024]; // 1 MB
    println!("L2_1MB_1t_read_GBs    {:.1}", read_bw(&l2, 1, 5_000));
    // L3 V-cache CCD is 96MB shared
    let l3 = vec![1.5f32; 24 * 1024 * 1024]; // 96 MB
    println!("L3_96MB_1t_read_GBs   {:.1}", read_bw(&l3, 1, 200));
    println!("L3_96MB_MT_read_GBs   {:.1}", read_bw(&l3, threads, 200));
    // DRAM: 4 GiB working set, far beyond any cache
    let n = 1024 * 1024 * 1024; // 4 GiB of f32
    let dram = vec![1.5f32; n];
    println!("DRAM_4GiB_MT_read_GBs {:.1}", read_bw(&dram, threads, 8));
    drop(l1);
    drop(l2);
    drop(l3);
    // Stream kernel on DRAM-sized buffers (1 GiB x3)
    let a = vec![1.0f32; n / 4];
    let mut b = vec![0.0f32; n / 4];
    let mut c = vec![0.0f32; n / 4];
    println!(
        "TRIAD_1GiBx3_MT_rw_GBs {:.1}",
        triad(&a, &mut b, &mut c, threads, 10)
    );
    println!("# note: GEMV decode roofline = DRAM stream BW; compare with roofline bin");
}
