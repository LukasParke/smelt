//! NumericAtom implementations: core.f16 (canonical) and core.i8.b32.f16scale (Q8_0-class).
//! One fused-dequant kernel family serves the executor for both.
#![allow(dead_code)]

pub const ATOM_F16: &str = "core.f16";
pub const ATOM_Q8: &str = "core.i8.b32.f16scale";

// ---------- IEEE f16 conversion (RNE) ----------

pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x007f_ffff;
    if ((b >> 23) & 0xff) == 0xff {
        return sign | 0x7c00 | (((man != 0) as u16) << 8);
    }
    if exp >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m2 = man | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let hm = m2 >> shift;
        let rem = m2 & ((1u32 << shift) - 1);
        let mid = 1u32 << (shift - 1);
        let round = ((rem > mid) || (rem == mid && (hm & 1) == 1)) as u16;
        return sign | (hm as u16).wrapping_add(round);
    }
    let hm = man >> 13;
    let rem = man & 0x1fff;
    let round = ((rem > 0x1000) || (rem == 0x1000 && (hm & 1) == 1)) as u16;
    sign | ((((exp as u32) << 10) + hm + round as u32) as u16)
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            return f32::from_bits(sign);
        }
        let mut e = 127 - 15 + 1;
        let mut m = man;
        while m & 0x0400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03ff;
        sign | (e << 23) | (m << 13)
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Quantize f32 rows into Q8 blocks of 32: [f16 scale][32 x i7..i8 symmetric]
/// Block layout: 64B header-free: 2B scale + 32B data => 34B per 32 elems (8.5 bpw).
pub fn q8_encode(x: &[f32]) -> Vec<u8> {
    assert!(x.len() % 32 == 0, "q8 payload must be multiple of 32");
    let mut out = Vec::with_capacity(x.len() / 32 * 34);
    for blk in x.chunks(32) {
        let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
        let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        out.extend_from_slice(&f32_to_f16(s).to_le_bytes());
        for v in blk {
            let q = (v / s).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

#[inline(always)]
pub fn q8_block_scale(payload: &[u8], block_idx: usize) -> f32 {
    let o = block_idx * 34;
    f16_to_f32(u16::from_le_bytes([payload[o], payload[o + 1]]))
}

// ---------- fused kernels ----------
// y[r] += dot(W_atom_row_r, x); W row-major [rows, cols].

pub fn gemv_q8(w: &[u8], x: &[f32], y: &mut [f32], rows: usize, cols: usize, r0: usize, r_end: usize) {
    let nblk = cols / 32;
    for r in r0..r_end {
        let row = &w[r * nblk * 34..(r + 1) * nblk * 34];
        let xr = &x[..cols];
        let mut acc = 0f32;
        let mut acc2 = 0f32;
        let mut acc3 = 0f32;
        let mut acc4 = 0f32;
        let mut b = 0usize;
        while b + 4 <= nblk {
            let scales = [
                q8_block_scale(row, b),
                q8_block_scale(row, b + 1),
                q8_block_scale(row, b + 2),
                q8_block_scale(row, b + 3),
            ];
            let j = b * 32;
            for bi in 0..4usize {
                let s = scales[bi];
                let xb = j + bi * 32;
                let ob = (b + bi) * 34 + 2;
                for t in 0..8usize {
                    let o = ob + t * 4;
                    acc = f32::mul_add(row[o] as i8 as f32 * s, xr[xb + t * 4], acc);
                    acc2 = f32::mul_add(row[o + 1] as i8 as f32 * s, xr[xb + t * 4 + 1], acc2);
                    acc3 = f32::mul_add(row[o + 2] as i8 as f32 * s, xr[xb + t * 4 + 2], acc3);
                    acc4 = f32::mul_add(row[o + 3] as i8 as f32 * s, xr[xb + t * 4 + 3], acc4);
                }
            }
            b += 4;
        }
        while b < nblk {
            let s = q8_block_scale(row, b);
            let base = b * 34 + 2;
            let j = b * 32;
            for t in 0..32 {
                acc = f32::mul_add(row[base + t] as i8 as f32 * s, xr[j + t], acc);
            }
            b += 1;
        }
        y[r - r0] += acc + acc2 + acc3 + acc4;
    }
}

/// F16 atom variant of the same kernel shape.
pub fn gemv_f16(w: &[u8], x: &[f32], y: &mut [f32], rows: usize, _cols: usize, r0: usize, r_end: usize) {
    debug_assert_eq!(w.len(), rows * _cols * 2);
    let _ = rows;
    for r in r0..r_end {
        let row = &w[r * _cols * 2..(r + 1) * _cols * 2];
        let xr = &x[.._cols];
        let mut acc = 0f32;
        let mut acc2 = 0f32;
        let mut acc3 = 0f32;
        let mut acc4 = 0f32;
        let mut j = 0;
        while j + 4 <= _cols {
            let o = j * 2;
            acc = f32::mul_add(f16_to_f32(u16::from_le_bytes([row[o], row[o + 1]])), xr[j], acc);
            let o2 = (j + 1) * 2;
            acc2 = f32::mul_add(f16_to_f32(u16::from_le_bytes([row[o2], row[o2 + 1]])), xr[j + 1], acc2);
            let o3 = (j + 2) * 2;
            acc3 = f32::mul_add(f16_to_f32(u16::from_le_bytes([row[o3], row[o3 + 1]])), xr[j + 2], acc3);
            let o4 = (j + 3) * 2;
            acc4 = f32::mul_add(f16_to_f32(u16::from_le_bytes([row[o4], row[o4 + 1]])), xr[j + 3], acc4);
            j += 4;
        }
        while j < _cols {
            let o = j * 2;
            acc += f16_to_f32(u16::from_le_bytes([row[o], row[o + 1]])) * xr[j];
            j += 1;
        }
        y[r - r0] += acc + acc2 + acc3 + acc4;
    }
}
