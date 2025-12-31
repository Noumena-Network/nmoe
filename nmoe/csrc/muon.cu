// nmoe: CUDA Newton–Schulz orthogonalization for Muon (BF16 I/O, FP32 accumulate)
//
// Implements in-place orthogonalization of a batched matrix buffer X with shape [B, M, N]
// stored row-major as BF16. The last two dims (M, N) are treated as the matrix. Leading
// dims are flattened to batch B. Computation uses batched GEMMs via cuBLAS with FP32
// accumulation. Coefficient sets follow the ns5 (quintic) iteration.
//
// API (C):
//   void muon_orth_bf16_inplace(void* x_bf16, long long B, int M, int N,
//                               int steps, int coeff_mode, cudaStream_t stream);
//     x_bf16:   BF16 buffer (row-major) of size B*M*N elements, updated in place
//     B:        batch size (flattened leading dims)
//     M, N:     matrix dims
//     steps:    number of Newton–Schulz iterations (must be >0)
//     coeff_mode: 0=simple, 1=quintic (ns5), 2=polar_express (optional)
//     stream:   CUDA stream (may be nullptr)
//
// Notes:
// - We approximate spectral norm normalization by Frobenius norm (L2 over all entries),
//   as done in Emerging-Optimizers newton_schulz() default path.
// - GEMM math uses column-major semantics. We map row‑major X(MxN) to column‑major X_col(NxM)
//   by swapping leading dims in cuBLAS calls (no transposes/copies needed).
// - Workspace for A (B, M, M) and T (B, M, M) is chunked over the batch to cap memory.
// - Data types: X is BF16 in/out; A/T are FP32; computeType is CUBLAS_COMPUTE_32F_FAST_TF32.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdint.h>
#include <stdio.h>

#define NMOE_CHECK_CUDA(cmd)                                                     \
  do {                                                                           \
    cudaError_t _e = (cmd);                                                      \
    if (_e != cudaSuccess) {                                                     \
      fprintf(stderr, "[muon.cu] CUDA error %d: %s at %s:%d\n",               \
              int(_e), cudaGetErrorString(_e), __FILE__, __LINE__);              \
      return;                                                                    \
    }                                                                            \
  } while (0)

#define NMOE_CHECK_CUBLAS(cmd)                                                   \
  do {                                                                           \
    cublasStatus_t _s = (cmd);                                                   \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                           \
      fprintf(stderr, "[muon.cu] cuBLAS error %d at %s:%d\n", int(_s),        \
              __FILE__, __LINE__);                                               \
      return;                                                                    \
    }                                                                            \
  } while (0)

// No-op placeholder for future cuBLASLt checks (enabled in next diff)
#define NMOE_CHECK_CUBLASLT(cmd)                                                 \
  do {                                                                           \
    cublasStatus_t _s = (cmd);                                                   \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                           \
      fprintf(stderr, "[muon.cu] cuBLASLt error %d at %s:%d\n", int(_s),      \
              __FILE__, __LINE__);                                               \
      return;                                                                    \
    }                                                                            \
  } while (0)

// -------------------------------
// Coefficient sets (a, b, c)
// -------------------------------
struct ABC { float a, b, c; };

static __host__ __device__ inline ABC coeff_simple() {
  return {3.4445f, -4.7750f, 2.0315f};
}

[[maybe_unused]] static const ABC COEFF_QUINTIC[5] = {
  {4.0848f, -6.8946f, 2.9270f},
  {3.9505f, -6.3029f, 2.6377f},
  {3.7418f, -5.5913f, 2.3037f},
  {2.8769f, -3.1427f, 1.2046f},
  {2.8366f, -3.0525f, 1.2012f},
};

// Optional: Polar Express set (unused by default)
[[maybe_unused]] static const ABC COEFF_POLAR_EXPRESS[8] = {
  {7.2086f, -15.5131f, 9.0178f},
  {3.9623f,  -2.5813f, 0.4542f},
  {3.9466f,  -2.5765f, 0.4544f},
  {3.8991f,  -2.5671f, 0.4566f},
  {3.7186f,  -2.5308f, 0.4653f},
  {3.1390f,  -2.3073f, 0.4733f},
  {2.1715f,  -1.5246f, 0.3885f},
  {1.8648f,  -1.2224f, 0.3577f},
};

// -----------------------------------------
// Frobenius norm (BF16) and in-place scale
// -----------------------------------------
// Device helper: in-place x[i] = rsqrt(max(x[i], eps))
__global__ void nmoe_muon_invsqrt_kernel(float* x, long long n) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    x[i] = rsqrtf(fmaxf(v, 1e-7f));
  }
}
__global__ void reduce_frob_bf16(const __nv_bfloat16* __restrict__ X,
                                 long long stride_elems,
                                 long long elems_per_mat,
                                 float* __restrict__ out_norm2, // sum of squares
                                 long long B) {
  long long b = blockIdx.x;
  if (b >= B) return;
  const __nv_bfloat16* base = X + b * stride_elems;
  float sum = 0.0f;
  for (long long i = threadIdx.x; i < elems_per_mat; i += blockDim.x) {
    float v = __bfloat162float(base[i]);
    sum += v * v;
  }
  __shared__ float smem[256];
  smem[threadIdx.x] = sum;
  __syncthreads();
  // reduction
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out_norm2[b] = smem[0];
}

__global__ void scale_frob_bf16(__nv_bfloat16* __restrict__ X,
                                long long stride_elems,
                                long long elems_per_mat,
                                const float* __restrict__ inv_norm,
                                long long B) {
  long long b = blockIdx.x;
  if (b >= B) return;
  __nv_bfloat16* base = X + b * stride_elems;
  float s = inv_norm[b];
  for (long long i = threadIdx.x; i < elems_per_mat; i += blockDim.x) {
    float v = __bfloat162float(base[i]);
    base[i] = __float2bfloat16(v * s);
  }
}

// -----------------------------------------
// Helpers
// -----------------------------------------
static inline int64_t div_up_int64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// Compute B = b*A + c*T  (A,T in FP32), batched over tileB. In-place write into A.
__global__ void lincomb_fp32(float* __restrict__ A,
                             const float* __restrict__ T,
                             int64_t stride_a, int64_t stride_t,
                             int elems, float b, float c,
                             int tileB) {
  int bidx = blockIdx.y;
  if (bidx >= tileB) return;
  float* a = A + bidx * stride_a;
  const float* t = T + bidx * stride_t;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elems; i += blockDim.x * gridDim.x) {
    a[i] = b * a[i] + c * t[i];
  }
}

// Compute mean of diagonal of A (MxM) per batch
__global__ void diag_mean_fp32(const float* __restrict__ A,
                               int64_t stride_a, int M,
                               float* __restrict__ out_mean,
                               int tileB) {
  int bidx = blockIdx.y;
  if (bidx >= tileB) return;
  const float* a = A + bidx * stride_a;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < M; i += blockDim.x) {
    sum += a[i * (int64_t)M + i];
  }
  __shared__ float sm[256];
  sm[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out_mean[bidx] = sm[0] / fmaxf(1.0f, float(M));
}

// Scale A in-place by gamma[b]
__global__ void scale_A_fp32(float* __restrict__ A,
                             int64_t stride_a, int elems,
                             const float* __restrict__ gamma,
                             int tileB) {
  int bidx = blockIdx.y;
  if (bidx >= tileB) return;
  float g = gamma[bidx];
  float* a = A + bidx * stride_a;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elems; i += blockDim.x * gridDim.x) {
    a[i] = a[i] * g;
  }
}

// Optional: add epsilon*I to A with epsilon derived from mean diag
__global__ void add_diag_eps_fp32(float* __restrict__ A,
                                  int64_t stride_a, int M,
                                  const float* __restrict__ mean,
                                  float eps_scale,
                                  int tileB) {
  int bidx = blockIdx.x;
  if (bidx >= tileB) return;
  float eps = eps_scale * mean[bidx];
  float* a = A + bidx * stride_a;
  for (int i = threadIdx.x; i < M; i += blockDim.x) {
    a[(int64_t)i * M + i] += eps;
  }
}

// Cast FP32 buffer to BF16 (batched), vectorized: 2x FP32 -> bf162
__global__ void nmoe_cast_fp32_to_bf16_vec2(const float* __restrict__ in,
                                            __nv_bfloat16* __restrict__ out,
                                            int64_t stride_in, int64_t stride_out,
                                            int elems, int tileB) {
  int bidx = blockIdx.y;
  if (bidx >= tileB) return;
  const float* src = in + bidx * stride_in;
  __nv_bfloat16* dst = out + bidx * stride_out;

  // Convert pairs
  int64_t pairs = elems >> 1; // elems / 2
  for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x; p < pairs; p += blockDim.x * gridDim.x) {
    float2 f2 = reinterpret_cast<const float2*>(src)[p];
    __nv_bfloat162 b2 = __floats2bfloat162_rn(f2.x, f2.y);
    reinterpret_cast<__nv_bfloat162*>(dst)[p] = b2;
  }
  // Odd tail
  if (((elems & 1) != 0)) {
    int64_t i = pairs * 2 + threadIdx.x;
    if (i == elems - 1 && blockIdx.x == 0) {
      dst[i] = __float2bfloat16(src[i]);
    }
  }
}

// Cast BF16 buffer to FP32 (batched), vectorized: bf162 -> 2x FP32
__global__ void nmoe_cast_bf16_to_fp32_vec2(const __nv_bfloat16* __restrict__ in,
                                            float* __restrict__ out,
                                            int64_t stride_in, int64_t stride_out,
                                            int elems, int tileB) {
  int bidx = blockIdx.y;
  if (bidx >= tileB) return;
  const __nv_bfloat16* src = in + bidx * stride_in;
  float* dst = out + bidx * stride_out;

  int64_t pairs = elems >> 1; // elems / 2
  for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x; p < pairs; p += blockDim.x * gridDim.x) {
    __nv_bfloat162 b2 = reinterpret_cast<const __nv_bfloat162*>(src)[p];
    float2 f2 = __bfloat1622float2(b2);
    reinterpret_cast<float2*>(dst)[p] = f2;
  }
  if ((elems & 1) != 0) {
    int64_t i = pairs * 2 + threadIdx.x;
    if (i == elems - 1 && blockIdx.x == 0) {
      dst[i] = __bfloat162float(src[i]);
    }
  }
}

// -------------------------------
// Persistent plan (workspace + handles)
// -------------------------------
struct MuonPlan {
  int Bmax{0};
  int M{0};
  int N{0};
  int S{0};
  // cuBLAS handles
  cublasHandle_t h{nullptr};
  // Reserve Lt handle for next diff (gemm via Lt)
  cublasLtHandle_t lt{nullptr};
  // Workspaces sized for Bmax tiles
  float* d_A{nullptr};
  float* d_T{nullptr};
  float* d_X32{nullptr};
  int64_t stride_a{0};
  int64_t bytes_per_A{0};
  int64_t stride_x{0};
  int64_t bytes_per_X32{0};
};

extern "C" void* muon_plan_create(int Bmax, int M, int N) {
  if (Bmax <= 0 || M <= 0 || N <= 0) return nullptr;
  MuonPlan* P = new MuonPlan();
  P->Bmax = Bmax;
  P->M = M;
  P->N = N;
  P->S = (M < N) ? M : N;
  P->stride_a = (int64_t)P->S * (int64_t)P->S;
  P->bytes_per_A = P->stride_a * (int64_t)sizeof(float);
  P->stride_x = (int64_t)N * (int64_t)M;
  P->bytes_per_X32 = P->stride_x * (int64_t)sizeof(float);

  // Handles
  if (cublasCreate(&P->h) != CUBLAS_STATUS_SUCCESS) { delete P; return nullptr; }
  // Use Tensor Core TF32 for FP32 GEMMs (B200 target).
  if (cublasSetMathMode(P->h, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(P->h);
    delete P;
    return nullptr;
  }
  if (cublasLtCreate(&P->lt) != CUBLAS_STATUS_SUCCESS) { cublasDestroy(P->h); delete P; return nullptr; }

  // Workspaces (async allocation on default stream)
  if (cudaMalloc(&P->d_A, (size_t)(P->bytes_per_A * Bmax)) != cudaSuccess) { cublasDestroy(P->h); cublasLtDestroy(P->lt); delete P; return nullptr; }
  if (cudaMalloc(&P->d_T, (size_t)(P->bytes_per_A * Bmax)) != cudaSuccess) { cudaFree(P->d_A); cublasDestroy(P->h); cublasLtDestroy(P->lt); delete P; return nullptr; }
  if (cudaMalloc(&P->d_X32, (size_t)(P->bytes_per_X32 * Bmax)) != cudaSuccess) { cudaFree(P->d_T); cudaFree(P->d_A); cublasDestroy(P->h); cublasLtDestroy(P->lt); delete P; return nullptr; }
  return P;
}

extern "C" void muon_plan_destroy(void* plan) {
  if (!plan) return;
  MuonPlan* P = reinterpret_cast<MuonPlan*>(plan);
  if (P->d_X32) cudaFree(P->d_X32);
  if (P->d_T) cudaFree(P->d_T);
  if (P->d_A) cudaFree(P->d_A);
  if (P->lt) cublasLtDestroy(P->lt);
  if (P->h) cublasDestroy(P->h);
  delete P;
}

extern "C" void muon_plan_run(void* plan,
                              void* x_bf16,
                              long long B,
                              int M, int N,
                              int steps,
                              int coeff_mode,
                              cudaStream_t stream) {
  if (!plan || x_bf16 == nullptr || B <= 0 || M <= 0 || N <= 0 || steps <= 0) return;
  MuonPlan* P = reinterpret_cast<MuonPlan*>(plan);
  // Shape guard (explicit, loud)
  if (P->M != M || P->N != N) {
    fprintf(stderr, "[muon.cu] Plan shape mismatch: plan=(%d,%d), got=(%d,%d)\n", P->M, P->N, M, N);
    return;
  }

  // Strides (row-major X: [B, M, N])
  const int64_t stride_x_elems = (int64_t)M * (int64_t)N;
  const int64_t elems_per_mat = stride_x_elems;

  // 1) Frobenius normalization per matrix: X /= max(||X||_F, eps)
  float* d_norm2 = nullptr;
  NMOE_CHECK_CUDA(cudaMallocAsync(&d_norm2, sizeof(float) * B, stream));
  reduce_frob_bf16<<<(unsigned)B, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16), stride_x_elems, elems_per_mat, d_norm2, B);
  NMOE_CHECK_CUDA(cudaGetLastError());

  // Convert norm2 to inv_norm (1/sqrt(norm2 + eps)) in-place
  nmoe_muon_invsqrt_kernel<<<(unsigned)div_up_int64(B, 256), 256, 0, stream>>>(d_norm2, B);
  NMOE_CHECK_CUDA(cudaGetLastError());

  scale_frob_bf16<<<(unsigned)B, 256, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(x_bf16), stride_x_elems, elems_per_mat, d_norm2, B);
  NMOE_CHECK_CUDA(cudaGetLastError());
  NMOE_CHECK_CUDA(cudaFreeAsync(d_norm2, stream));

  // 2) Newton–Schulz iterations in batched tiles using cuBLAS GEMM (FP32 accumulate)
  // Set stream on persistent cuBLAS handle
  NMOE_CHECK_CUBLAS(cublasSetStream(P->h, stream));
  // Scalars (host mode)
  const float one = 1.0f, zero = 0.0f;

  // Workspace cap in bytes (A and T buffers). Keep modest to avoid OOM.
  // Tile size is bounded by persistent allocation
  int64_t tileB_max = P->Bmax;

  // Batched strides (column-major view): X_col is N x M with lda=N and stride=N*M
  const int lda_x = N; // leading dimension of X_col
  const int64_t stride_x = (int64_t)N * (int64_t)M; // elements
  const cudaDataType F32 = CUDA_R_32F;
  const cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_TF32;

  // Coefficients per blog/nanochat: fixed (a,b,c) each iteration
  auto get_coeff = [&](int /*iter*/) -> ABC { return coeff_simple(); };

  for (int64_t start = 0; start < B; start += tileB_max) {
    int tileB = (int) ((start + tileB_max <= B) ? tileB_max : (B - start));

    // Base pointers for this tile
    __nv_bfloat16* X_tile = reinterpret_cast<__nv_bfloat16*>(x_bf16) + start * stride_x;

    for (int it = 0; it < steps; ++it) {
      const ABC c = get_coeff(it);

      // A = X * X^T in FP32: lift X(BF16) -> X32, then A based on orientation
      {
        const int threads = 256;
        const int64_t elems = (int64_t)M * (int64_t)N;
        const int blocks_x = (int)div_up_int64(elems, threads * 8LL);
        dim3 grid(blocks_x, (unsigned)tileB, 1);
        nmoe_cast_bf16_to_fp32_vec2<<<grid, threads, 0, stream>>>(
            X_tile, P->d_X32, stride_x, P->stride_x, (int)elems, tileB);
        NMOE_CHECK_CUDA(cudaGetLastError());
      }
      // Orientation selection
      const bool postmul = (M > N); // if true: X <- a*X + X*B ; else: X <- a*X + B*X
      const int S = postmul ? N : M;
      // A build: if !postmul -> A = X32^T @ X32 (M x M); else -> A = X32 @ X32^T (N x N)
      if (!postmul) {
        NMOE_CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            P->h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            M, M, N,
            &one,
            P->d_X32, F32, lda_x, P->stride_x,
            P->d_X32, F32, lda_x, P->stride_x,
            &zero,
            P->d_A, F32, M, P->stride_a,
            tileB,
            compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      } else {
        NMOE_CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            P->h,
            CUBLAS_OP_N, CUBLAS_OP_T,
            N, N, M,
            &one,
            P->d_X32, F32, lda_x, P->stride_x,
            P->d_X32, F32, lda_x, P->stride_x,
            &zero,
            P->d_A, F32, N, P->stride_a,
            tileB,
            compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }

      // (No additional preconditioning per blog baseline)

      // T = A @ A  (both FP32), with square S
      NMOE_CHECK_CUBLAS(cublasGemmStridedBatchedEx(
          P->h,
          CUBLAS_OP_N, CUBLAS_OP_N,
          S, S, S,
          &one,
          P->d_A, F32, S, P->stride_a,
          P->d_A, F32, S, P->stride_a,
          &zero,
          P->d_T, F32, S, P->stride_a,
          tileB,
          compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

      // B = b*A + c*T (in place over A)
      {
        const int threads = 256;
        const int64_t elems = (int64_t)S * (int64_t)S;
        const int blocks_x = (int)div_up_int64(elems, threads * 8LL);
        dim3 grid(blocks_x, (unsigned)tileB, 1);
        lincomb_fp32<<<grid, threads, 0, stream>>>(
            P->d_A, P->d_T, P->stride_a, P->stride_a, (int)elems, c.b, c.c, tileB);
        NMOE_CHECK_CUDA(cudaGetLastError());
      }

      // Update X32 in FP32: if !postmul -> X32 = a*X32 + X32 @ B^T
      //                      if  postmul -> X32 = a*X32 + B^T @ X32
      if (!postmul) {
        NMOE_CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            P->h,
            CUBLAS_OP_N, CUBLAS_OP_T,
            N, M, S,
            &one,
            P->d_X32, F32, lda_x, P->stride_x,
            P->d_A, F32, S, P->stride_a,
            &c.a,
            P->d_X32, F32, lda_x, P->stride_x,
            tileB,
            compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      } else {
        NMOE_CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            P->h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            S, M, S,
            &one,
            P->d_A, F32, S, P->stride_a,
            P->d_X32, F32, lda_x, P->stride_x,
            &c.a,
            P->d_X32, F32, lda_x, P->stride_x,
            tileB,
            compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      // Cast X32 back to BF16 in place to X_tile
      {
        const int threads = 256;
        const int64_t elems = (int64_t)M * (int64_t)N;
        const int blocks_x = (int)div_up_int64(elems, threads * 8LL);
        dim3 grid(blocks_x, (unsigned)tileB, 1);
        nmoe_cast_fp32_to_bf16_vec2<<<grid, threads, 0, stream>>>(
            P->d_X32, X_tile, P->stride_x, (int64_t)M * (int64_t)N, (int)elems, tileB);
        NMOE_CHECK_CUDA(cudaGetLastError());
      }
    }
  }
}

// Backwards compat single-shot entry (kept temporarily)
extern "C" void muon(void* x_bf16,
                     long long B,
                     int M, int N,
                     int steps,
                     int coeff_mode,
                     cudaStream_t stream) {
  // Create a tiny plan with Bmax= min(B, 32) and destroy after run.
  int Bmax = (int)((B < 32) ? B : 32);
  void* plan = muon_plan_create(Bmax, M, N);
  if (!plan) return;
  muon_plan_run(plan, x_bf16, B, M, N, steps, coeff_mode, stream);
  muon_plan_destroy(plan);
}
