// SMELT v2 backward kernels — NVRTC module namespace "bwd" (Agent B).
// Self-contained: no includes. Mirrors conventions of cu/gpt2.cu forward kernels:
//   - LayerNorm / attention run ONE block per vector / per head, grid-strided partials
//     in extern __shared__ memory, thread 0 finalizes scalars (see k_layernorm /
//     k_attn_decode in gpt2.cu).
//   - Weights are row-major [out_rows, in_cols] post-pack (gemv y[r]=dot(W[r],x)).
//
// k_attn_bwd_train layout: q, K, V and dy_attnout are position-major [T, stride];
// the slice of `head` starts at head*hd inside each row (stride == n_embd for
// KV-cache-backed K/V and for packed [T][n_embd] q/dy buffers alike). Outputs
// dq/dk/dv use the same [T, stride] layout. dq is overwritten per query position;
// dk/dv ACCUMULATE across query positions and must be pre-zeroed by the caller.

// Transposed gemv: dx[i] = sum_r W[r*cols+i]*dy[r].
// Poorly coalesced (column walk over a row-major matrix) but cols <= 3072 here and
// correctness comes first; revisit with a tiled transpose if it ever shows up in
// profiles.
extern "C" __global__ void k_lin_input_grad(float* __restrict__ dx, const float* __restrict__ W,
                                            const float* __restrict__ dy, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) {
        float acc = 0.f;
        for (int r = 0; r < rows; ++r) acc = fmaf(W[(size_t)r * cols + i], dy[r], acc);
        dx[i] = acc;
    }
}

// LayerNorm backward, single block over one n-vector (like forward k_layernorm).
// Recomputes mean/rstd from the saved pre-LN x in-kernel, then applies the standard
// Jacobian:
//   xhat  = (x - mean) * rstd          (rstd = 1/sqrt(var + eps))
//   dxhat = dy * g
//   dx[i] = rstd * (dxhat[i] - mean(dxhat) - xhat[i] * mean(dxhat*xhat))
//   dg[i] += dy[i] * xhat[i]
//   db[i] += dy[i]
// dg/db accumulate: caller passes zeroed buffers for a fresh backward.
// Shared memory: part[blockDim.x] reduction scratch || scal[4].
extern "C" __global__ void k_ln_backward(float* __restrict__ dx, float* __restrict__ dg,
                                         float* __restrict__ db, const float* __restrict__ dy,
                                         const float* __restrict__ x, const float* __restrict__ g,
                                         const float* __restrict__ b, float eps, int n) {
    extern __shared__ float sm[];
    float* part = sm;
    float* scal = sm + blockDim.x;
    int tid = threadIdx.x;
    int bd = blockDim.x;

    // mean
    float s = 0.f;
    for (int i = tid; i < n; i += bd) s += x[i];
    part[tid] = s;
    __syncthreads();
    if (tid == 0) {
        float S = 0.f;
        for (int t = 0; t < bd; ++t) S += part[t];
        scal[0] = S / (float)n;
    }
    __syncthreads();
    float mean = scal[0];

    // rstd
    float sq = 0.f;
    for (int i = tid; i < n; i += bd) { float v = x[i] - mean; sq += v * v; }
    part[tid] = sq;
    __syncthreads();
    if (tid == 0) {
        float SQ = 0.f;
        for (int t = 0; t < bd; ++t) SQ += part[t];
        scal[1] = rsqrtf(SQ / (float)n + eps);
    }
    __syncthreads();
    float rstd = scal[1];

    // mean(dxhat)
    float a = 0.f;
    for (int i = tid; i < n; i += bd) a += dy[i] * g[i];
    part[tid] = a;
    __syncthreads();
    if (tid == 0) {
        float A = 0.f;
        for (int t = 0; t < bd; ++t) A += part[t];
        scal[2] = A / (float)n;
    }
    __syncthreads();
    float mdxh = scal[2];

    // mean(dxhat * xhat)
    float c = 0.f;
    for (int i = tid; i < n; i += bd) {
        float xh = (x[i] - mean) * rstd;
        c += dy[i] * g[i] * xh;
    }
    part[tid] = c;
    __syncthreads();
    if (tid == 0) {
        float C = 0.f;
        for (int t = 0; t < bd; ++t) C += part[t];
        scal[3] = C / (float)n;
    }
    __syncthreads();
    float mdxh_xh = scal[3];

    for (int i = tid; i < n; i += bd) {
        float xh = (x[i] - mean) * rstd;
        float dh = dy[i] * g[i];
        dx[i] = rstd * (dh - mdxh - xh * mdxh_xh);
        dg[i] += dy[i] * xh;
        db[i] += dy[i];
    }
    (void)b;
}

// Tanh-approx GELU derivative:
//   y = 0.5x(1+tanh(u)), u = c(x + 0.044715x^3), c = 0.7978845608028654
//   dy/dx = 0.5(1+tanh(u)) + 0.5x * sech^2(u) * c(1 + 3*0.044715x^2)
// with sech^2(u) = 1 - tanh^2(u).
extern "C" __global__ void k_gelu_bwd(float* __restrict__ dx, const float* __restrict__ x_pre,
                                      const float* __restrict__ dy, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x_pre[i];
        float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        float th = tanhf(u);
        float sech2 = 1.f - th * th;
        float dgdv = 0.5f * (1.f + th)
                   + 0.5f * v * sech2 * 0.7978845608028654f * (1.f + 3.f * 0.044715f * v * v);
        dx[i] = dy[i] * dgdv;
    }
}

// Causal training-form attention backward, one block per head (like forward
// k_attn_decode). Recomputes P = softmax(qK^T/sqrt(hd)) with causal mask
// (query i attends keys j <= i), then standard softmax backward:
//   dP_ij = dot(dyout_i, V_j)
//   ds_ij = scale * (dP_ij - sum_k dP_ik * P_ik) * P_ij     [rowwise]
//   dq_i  = sum_j ds_ij * K_j
//   dk_j += sum_i>=j ds_ij * q_i
//   dv_j += sum_i>=j P_ij * dyout_i            (no scale: out = P V directly)
// Positions loop serially inside the block; hd-dim work spreads across threads.
// Shared memory: qs[hd] || dys[hd] || sc[T] || dp[T] || red[blockDim.x].
extern "C" __global__ void k_attn_bwd_train(float* __restrict__ dq, float* __restrict__ dk,
                                            float* __restrict__ dv, const float* __restrict__ q,
                                            const float* __restrict__ K, const float* __restrict__ V,
                                            const float* __restrict__ dy_attnout,
                                            int T, int hd, int stride) {
    extern __shared__ float sm[];
    float* qs = sm;
    float* dys = sm + hd;
    float* sc = sm + 2 * hd;
    float* dp = sc + T;
    float* red = dp + T;
    int head = blockIdx.x;
    int tid = threadIdx.x;
    int bd = blockDim.x;
    size_t base = (size_t)head * hd;
    float scale = rsqrtf((float)hd);

    for (int i = 0; i < T; ++i) {
        // Stage this query position's q_i and dyout_i for the head.
        for (int t = tid; t < hd; t += bd) {
            qs[t] = q[(size_t)i * stride + base + t];
            dys[t] = dy_attnout[(size_t)i * stride + base + t];
        }
        __syncthreads();

        // Scores s_j = <q_i, K_j> / sqrt(hd), causal j <= i (one owner thread per j).
        for (int j = tid; j <= i; j += bd) {
            const float* kp = K + (size_t)j * stride + base;
            float d = 0.f;
            for (int t = 0; t < hd; ++t) d = fmaf(qs[t], kp[t], d);
            sc[j] = d * scale;
        }
        __syncthreads();

        if (tid == 0) { // softmax over j <= i
            float mx = -1e30f;
            for (int j = 0; j <= i; ++j) mx = fmaxf(mx, sc[j]);
            float sum = 0.f;
            for (int j = 0; j <= i; ++j) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
            for (int j = 0; j <= i; ++j) sc[j] /= sum;
        }
        __syncthreads();

        // dP_j = <dyout_i, V_j>
        for (int j = tid; j <= i; j += bd) {
            const float* vp = V + (size_t)j * stride + base;
            float d = 0.f;
            for (int t = 0; t < hd; ++t) d = fmaf(dys[t], vp[t], d);
            dp[j] = d;
        }
        __syncthreads();

        if (tid == 0) { // rowwise dot(dP, P)
            float rs = 0.f;
            for (int j = 0; j <= i; ++j) rs = fmaf(dp[j], sc[j], rs);
            red[0] = rs;
        }
        __syncthreads();
        float rs = red[0];

        // dq_i = sum_j ds_j K_j (direct write: each i visited once)
        for (int t = tid; t < hd; t += bd) {
            float acc = 0.f;
            for (int j = 0; j <= i; ++j) {
                float ds = (dp[j] - rs) * sc[j] * scale;
                acc = fmaf(ds, K[(size_t)j * stride + base + t], acc);
            }
            dq[(size_t)i * stride + base + t] = acc;
        }
        __syncthreads();

        // dk_j += ds_ij q_i ; dv_j += P_ij dyout_i (accumulate across i)
        for (int j = tid; j <= i; j += bd) {
            float ds = (dp[j] - rs) * sc[j] * scale;
            float pj = sc[j];
            float* dkp = dk + (size_t)j * stride + base;
            float* dvp = dv + (size_t)j * stride + base;
            for (int t = 0; t < hd; ++t) {
                dkp[t] = fmaf(ds, qs[t], dkp[t]);
                dvp[t] = fmaf(pj, dys[t], dvp[t]);
            }
        }
        __syncthreads();
    }
}
