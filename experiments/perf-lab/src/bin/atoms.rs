//! NumericAtom round-trip validation: one decoder skeleton, many quant schemes.
//! Measures reconstruction error of the SMT atom-space coordinates on
//! LLM-like weight distributions. Proves the "quantization is a coordinate
//! space" claim with numbers instead of assertions.
use std::hint::black_box;

// ---------- half-precision conversions (std-only) ----------

fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x007f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00 | (((man != 0) as u16) << 8); // inf/nan
    }
    if exp <= 0 {
        // subnormal f16 or zero
        if exp < -10 {
            return sign;
        }
        let man = man | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let half_man = man >> shift;
        let rem = man & ((1u32 << shift) - 1);
        let round = ((rem > (1 << (shift - 1))) || (rem == (1 << (shift - 1)) && (half_man & 1) == 1)) as u16;
        return sign | ((half_man as u16) + round);
    }
    let half_man = man >> 13;
    let rem = man & 0x1fff;
    let round = ((rem > 0x1000) || (rem == 0x1000 && (half_man & 1) == 1)) as u16;
    sign | ((((exp as u32) << 10) + half_man + round as u32) as u16)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        // subnormal: normalize
        let mut e = 127 - 15 + 1;
        let mut m = man;
        while m & 0x0400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03ff;
        sign | ((e as u32) << 23) | (m << 13)
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

fn f32_to_bf16_rne(x: f32) -> f32 {
    let b = x.to_bits();
    let rounded = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    f32::from_bits((rounded & 0xffff_0000))
}

// ---------- xorshift RNG ----------

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
        // Box-Muller
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// ---------- atoms ----------

/// {I4, block=32, flat F16 scale, symmetric} — Q4_0 coordinates
fn q4_sym_block32(x: &[f32]) -> Vec<f32> {
    x.chunks(32)
        .flat_map(|blk| {
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = f16_to_f32(f32_to_f16(if amax == 0.0 { 1.0 } else { amax / 7.0 }));
            blk.iter()
                .map(|v| ((v / d).round().clamp(-7.0, 7.0) * d))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// MXFP4 coordinates: {F4_E2M1, block=32, flat E8M0 shared exponent}
fn mxfp4_block32(x: &[f32]) -> Vec<f32> {
    const LATTICE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    x.chunks(32)
        .flat_map(|blk| {
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            // shared exponent E8M0: floor(log2(amax)) - 2 => max maps into lattice top region
            let e = if amax == 0.0 { -126i32 } else { amax.log2().floor() as i32 - 2 };
            let inv = (-(e as f32)).exp2(); // 2^-e
            blk.iter()
                .map(|v| {
                    let r = (v * inv).abs().min(6.0);
                    // nearest lattice point, ties-to-even index
                    let mut i = 0usize;
                    while i + 1 < LATTICE.len() && r > LATTICE[i + 1] {
                        i += 1;
                    }
                    let lo = LATTICE[i];
                    let hi = if i + 1 < LATTICE.len() { LATTICE[i + 1] } else { lo };
                    let pick = if r - lo < hi - r {
                        i
                    } else if r - lo > hi - r {
                        i + 1
                    } else if i % 2 == 0 {
                        i
                    } else {
                        i + 1
                    };
                    (v.signum() * LATTICE[pick.min(7)]) * (e as f32).exp2()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Two-level sub-scale: {I4, super=256, sub-block=16 F16 scales} — Q4_K-style coordinates
fn two_level_sub16(x: &[f32]) -> Vec<f32> {
    x.chunks(256)
        .flat_map(|sup| {
            sup.chunks(16)
                .flat_map(|sub| {
                    let amax = sub.iter().fold(0f32, |a, v| a.max(v.abs()));
                    let d = f16_to_f32(f32_to_f16(if amax == 0.0 { 1.0 } else { amax / 7.0 }));
                    sub.iter()
                        .map(|v| ((v / d).round().clamp(-7.0, 7.0) * d))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn bf16_ref(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| f32_to_bf16_rne(*v)).collect()
}

// ---------- metrics ----------

fn metrics(orig: &[f32], q: &[f32]) -> (f64, f64) {
    let n = orig.len() as f64;
    let mut se = 0f64;
    let mut dot = 0f64;
    let mut no = 0f64;
    let mut nq = 0f64;
    for i in 0..orig.len() {
        let d = (orig[i] - q[i]) as f64;
        se += d * d;
        dot += orig[i] as f64 * q[i] as f64;
        no += orig[i] as f64 * orig[i] as f64;
        nq += q[i] as f64 * q[i] as f64;
    }
    let rms_rel = (se / n).sqrt() / (no / n).sqrt();
    let cos = dot / (no.sqrt() * nq.sqrt());
    (rms_rel, cos)
}

fn main() {
    println!("# perf-lab atoms | N={} per distribution", 1 << 20);
    let n = 1 << 20;
    let mut rng = Rng(0xDEADBEEF1234_5678);
    let dists: Vec<(&str, Box<dyn Fn(&mut Rng) -> f64>)> = vec![
        ("gauss01", Box::new(|r: &mut Rng| r.normal())),
        ("laplace_b1", Box::new(|r: &mut Rng| {
            let u = r.uniform() - 0.5;
            -u.signum() * (1.0 - 2.0 * u.abs()).ln()
        })),
        // LLM-style outlier mixture: rare huge activations
        ("outlier_mix", Box::new(|r: &mut Rng| {
            if r.uniform() < 0.01 { r.normal() * 25.0 } else { r.normal() }
        })),
        ("uniform_pm3", Box::new(|r: &mut Rng| (r.uniform() - 0.5) * 6.0)),
        ("tiny_gauss_1e-3", Box::new(|r: &mut Rng| r.normal() * 1e-3)),
    ];
    println!("# distribution, atom, rms_rel_err, cos_sim");
    for (name, gen) in &dists {
        let xs: Vec<f32> = (0..n).map(|_| gen(&mut rng) as f32).collect();
        for (atom, fq) in [
            ("bf16_ref", bf16_ref as fn(&[f32]) -> Vec<f32>),
            ("q4_sym_b32_f16scale", q4_sym_block32),
            ("mxfp4_e8m0_b32", mxfp4_block32),
            ("two_level_i4_sub16", two_level_sub16),
        ] {
            let q = fq(black_box(&xs));
            let (rel, cos) = metrics(&xs, &q);
            println!("{name}, {atom}, {rel:.6}, {cos:.8}");
        }
    }
    println!("# expectation: two_level < q4_sym < mxfp4 on gauss; mxfp4 degrades least on outliers;");
    println!("#               mxfp4 worst on tiny_gauss (E2M1 deadzone), bf16 ~1e-3 everywhere");
}
