// SMELT model-lab kernels — sm_120 via NVRTC. Self-contained: no includes.
// Q8 atom layout: blocks of 32 elems = [f16 scale (2B LE)][32 x i8] => 34B.

extern "C" __device__ float h2f(unsigned short h) {
    unsigned sign = (unsigned)(h & 0x8000u) << 16;
    unsigned exp = (h >> 10) & 0x1fu;
    unsigned man = h & 0x3ffu;
    unsigned bits;
    if (exp == 0x1f) { bits = sign | 0x7f800000u | (man << 13); }
    else if (exp == 0) {
        if (man == 0) { bits = sign; }
        else {
            int e = 113; // 127 - 15 + 1
            for (;;) { if (man & 0x400u) break; man <<= 1; e -= 1; }
            man &= 0x3ffu;
            bits = sign | ((unsigned)e << 23) | (man << 13);
        }
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    return __uint_as_float(bits);
}

extern "C" __global__ void k_q8_gemv(float* __restrict__ out, const unsigned char* __restrict__ w,
                                     const float* __restrict__ x, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    int nblk = cols / 32;
    const unsigned char* row = w + (size_t)r * nblk * 34;
    float acc = 0.f, acc2 = 0.f, acc3 = 0.f, acc4 = 0.f;
    for (int b = 0; b < nblk; ++b) {
        const unsigned char* pb = row + b * 34;
        float s = h2f((unsigned short)(pb[0] | (pb[1] << 8)));
        const float* xb = x + b * 32;
        #pragma unroll
        for (int t = 0; t < 32; ++t) {
            float q = (float)((char)pb[2 + t]);
            acc = fmaf(q * s, xb[t], acc);
        }
    }
    out[r] = acc + acc2 + acc3 + acc4;
}

extern "C" __global__ void k_f16_gemv(float* __restrict__ out, const unsigned short* __restrict__ w,
                                      const float* __restrict__ x, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const unsigned short* row = w + (size_t)r * cols;
    float acc = 0.f;
    for (int j = 0; j < cols; ++j) {
        acc = fmaf(h2f(row[j]), x[j], acc);
    }
    out[r] = acc;
}

extern "C" __global__ void k_f32_gemv(float* __restrict__ out, const float* __restrict__ w,
                                      const float* __restrict__ x, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* row = w + (size_t)r * cols;
    float acc = 0.f;
    for (int j = 0; j < cols; ++j) acc = fmaf(row[j], x[j], acc);
    out[r] = acc;
}

extern "C" __global__ void k_gather_row_f32(float* __restrict__ dst,
                                            const float* __restrict__ src,
                                            long long row, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) dst[i] = src[row * cols + i];
}

extern "C" __global__ void k_add_bias(float* __restrict__ y, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[i];
}

extern "C" __global__ void k_axpy(float* __restrict__ x, const float* __restrict__ src,
                                  long long off, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += src[off + i];
}

extern "C" __global__ void k_layernorm(float* __restrict__ o, const float* __restrict__ x,
                                       const float* __restrict__ g, const float* __restrict__ b,
                                       float eps, int n) {
    extern __shared__ float part[]; // blockDim.x slots
    int tid = threadIdx.x;
    float s = 0.f;
    for (int i = tid; i < n; i += blockDim.x) s += x[i];
    part[tid] = s;
    __syncthreads();
    if (tid == 0) {
        float S = 0.f;
        for (int t = 0; t < (int)blockDim.x; ++t) S += part[t];
        part[0] = S / (float)n; // mean
    }
    __syncthreads();
    float mean = part[0];
    float sq = 0.f;
    for (int i = tid; i < n; i += blockDim.x) { float v = x[i] - mean; sq += v * v; }
    part[tid] = sq;
    __syncthreads();
    if (tid == 0) {
        float SQ = 0.f;
        for (int t = 0; t < (int)blockDim.x; ++t) SQ += part[t];
        part[1] = rsqrtf(SQ / (float)n + eps); // rstd
    }
    __syncthreads();
    float rstd = part[1];
    for (int i = tid; i < n; i += blockDim.x) {
        o[i] = (x[i] - mean) * rstd * g[i] + b[i];
    }
}

extern "C" __global__ void k_gelu(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
}

// One block per head. smem: [q: hd][scores: T]
extern "C" __global__ void k_attn_decode(float* __restrict__ out, const float* __restrict__ qkv,
                                         int qoff, const float* __restrict__ Kc,
                                         const float* __restrict__ Vc, int T, int hd, int stride) {
    extern __shared__ float sm[];
    float* qs = sm;
    float* sc = sm + hd;
    int head = blockIdx.x;
    int base = qoff + head * hd;
    for (int t = threadIdx.x; t < hd; t += blockDim.x) qs[t] = qkv[base + t];
    __syncthreads();
    for (int p = threadIdx.x; p < T; p += blockDim.x) {
        const float* kp = Kc + (long)p * stride + head * hd;
        float d = 0.f;
        for (int t = 0; t < hd; ++t) d = fmaf(qs[t], kp[t], d);
        sc[p] = d * rsqrtf((float)hd);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = -1e30f;
        for (int p = 0; p < T; ++p) mx = fmaxf(mx, sc[p]);
        float sum = 0.f;
        for (int p = 0; p < T; ++p) { sc[p] = expf(sc[p] - mx); sum += sc[p]; }
        for (int p = 0; p < T; ++p) sc[p] /= sum;
    }
    __syncthreads();
    for (int t = threadIdx.x; t < hd; t += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < T; ++p) acc += sc[p] * Vc[(long)p * stride + head * hd + t];
        out[head * hd + t] = acc;
    }
}

extern "C" __global__ void k_scatter_kv(float* __restrict__ Kc, float* __restrict__ Vc,
                                        const float* __restrict__ qkv, long long row, int embd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < embd) {
        Kc[row + i] = qkv[embd + i];
        Vc[row + i] = qkv[2 * embd + i];
    }
}

extern "C" __global__ void k_q8_row(float* __restrict__ dst, const unsigned char* __restrict__ src,
                                    long long row, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) {
        int nblk = cols / 32;
        const unsigned char* blkrow = src + (size_t)row * nblk * 34;
        int b = i / 32, t = i % 32;
        const unsigned char* pb = blkrow + b * 34;
        float s = h2f((unsigned short)(pb[0] | (pb[1] << 8)));
        dst[i] = (float)((char)pb[2 + t]) * s;
    }
}

extern "C" __global__ void k_f16_row(float* __restrict__ dst, const unsigned short* __restrict__ src,
                                     long long row, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) dst[i] = h2f(src[row * cols + i]);
}

// dh = W^T @ dz ; W row-major [V, d]
extern "C" __global__ void k_wteT_dz(float* __restrict__ dh, const float* __restrict__ W,
                                     const float* __restrict__ dz, int V, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) {
        float acc = 0.f;
        for (int v = 0; v < V; ++v) acc = fmaf(W[v * d + i], dz[v], acc);
        dh[i] = acc;
    }
}

// Plastic adapter applied to the post-ln_f representation: h += B(A h)
// A: [r, d] row-major, B: [d, r] row-major, r <= blockDim.x
extern "C" __global__ void k_adapter_apply(float* __restrict__ h, const float* __restrict__ A,
                                           const float* __restrict__ B, int r, int d) {
    extern __shared__ float ah[];
    int tid = threadIdx.x;
    for (int j = tid; j < r; j += blockDim.x) {
        float s = 0.f;
        const float* ar = A + (size_t)j * d;
        for (int i = 0; i < d; ++i) s = fmaf(ar[i], h[i], s);
        ah[j] = s;
    }
    __syncthreads();
    for (int i = tid; i < d; i += blockDim.x) {
        float s = 0.f;
        const float* br = B + (size_t)i * r;
        for (int j = 0; j < r; ++j) s = fmaf(br[j], ah[j], s);
        h[i] += s;
    }
}
