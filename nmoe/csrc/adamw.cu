// nmoe/adamw.cu
//
// Fused expert AdamW update + blockscaled weight cache emission.
//
// Updates expert weights (W1, W3, W2) in BF16 using AdamW semantics (decoupled
// weight decay) and emits packed FP8 E4M3 or NVFP4 E2M1 caches with E8M0 scale
// factors in CUTLASS MMA layout.
//
// Target: NVIDIA Blackwell B200 (sm_100a). No fallbacks.

#include "ptx.cu"
#include "swizzle.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nmoe {
namespace adamw {

// Quantization constants (match nmoe/csrc/quant.cu)
constexpr float FP8_MAX = 448.0f;  // E4M3 max finite value
constexpr float FP4_MAX = 6.0f;    // E2M1 max finite value
constexpr int SF_VEC = 32;         // Scale factor granularity (32 values)

// Tiling:
// - Process one (output_tile=32) x (k_block=32) tile per block.
// - 256 threads: (x=32 output columns, y=8 input rows) with 4 iterations for k_block.
constexpr int TILE_OUT = 32;
constexpr int TILE_K = 32;
constexpr int THREADS_X = 32;
constexpr int THREADS_Y = 8;
constexpr int BLOCK_THREADS = THREADS_X * THREADS_Y;

__host__ __device__ __forceinline__ int ceil_div(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) { return __float2bfloat16(v); }

// AdamW update (matches the math/order of PyTorch fused AdamW: decay first).
__device__ __forceinline__ float adamw_update_f32(
    float w,
    float g,
    float& m,
    float& v,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    float eps,
    float step_size,
    float inv_bias_correction2_sqrt) {
  if (weight_decay != 0.0f) {
    w -= lr * weight_decay * w;
  }
  const float one_minus_beta1 = 1.0f - beta1;
  const float one_minus_beta2 = 1.0f - beta2;
  m = beta1 * m + one_minus_beta1 * g;
  v = beta2 * v + one_minus_beta2 * g * g;
  const float denom = sqrtf(v) * inv_bias_correction2_sqrt + eps;
  w -= step_size * m / denom;
  return w;
}

// ============================================================================
// W13: Update W1/W3 [E, H, Dff] and emit fused interleaved cache
//   W13 rows = 2*Dff (gate0, up0, gate1, up1, ...)
//   W13 cols = H
// ============================================================================

template <bool kNvfp4>
__global__ void k_expert_adamw_w13_update_quant(
    __nv_bfloat16* __restrict__ W1,
    const __nv_bfloat16* __restrict__ dW1,
    __nv_bfloat16* __restrict__ m1,
    __nv_bfloat16* __restrict__ v1,
    __nv_bfloat16* __restrict__ W3,
    const __nv_bfloat16* __restrict__ dW3,
    __nv_bfloat16* __restrict__ m3,
    __nv_bfloat16* __restrict__ v3,
    uint16_t* __restrict__ W13_q_u16,
    uint8_t* __restrict__ W13_sf_mma,
    int E,
    int H,
    int Dff,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    float eps,
    float step_size,
    float inv_bias_correction2_sqrt) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int e = static_cast<int>(blockIdx.z);
  const int j0 = static_cast<int>(blockIdx.x) * TILE_OUT;
  const int k0 = static_cast<int>(blockIdx.y) * TILE_K;

  const int j = j0 + static_cast<int>(threadIdx.x);

  const int k_iter = static_cast<int>(threadIdx.y);
  const int k_local0 = k_iter;

  const int M13 = 2 * Dff;
  const int sf_k = H / SF_VEC;
  const int k_sf = k0 / SF_VEC;
  const int64_t base_w = static_cast<int64_t>(e) * static_cast<int64_t>(H) * static_cast<int64_t>(Dff);

  // Shared tile stores updated weights for the current 32x32 (k_block x out_tile).
  __shared__ float gate_sh[TILE_K][TILE_OUT];
  __shared__ float up_sh[TILE_K][TILE_OUT];
  __shared__ uint8_t scale_gate[TILE_OUT];
  __shared__ uint8_t scale_up[TILE_OUT];

  // Update + stage into shared for quantization.
  #pragma unroll
  for (int t = 0; t < (TILE_K / THREADS_Y); ++t) {
    const int k_local = t * THREADS_Y + k_local0;
    const int k = k0 + k_local;

    const int64_t idx = base_w + static_cast<int64_t>(k) * static_cast<int64_t>(Dff) + static_cast<int64_t>(j);

    float w1 = bf16_to_f32(W1[idx]);
    float g1 = bf16_to_f32(dW1[idx]);
    float m1v = bf16_to_f32(m1[idx]);
    float v1v = bf16_to_f32(v1[idx]);
    w1 = adamw_update_f32(w1, g1, m1v, v1v, lr, beta1, beta2, weight_decay, eps, step_size, inv_bias_correction2_sqrt);
    W1[idx] = f32_to_bf16(w1);
    m1[idx] = f32_to_bf16(m1v);
    v1[idx] = f32_to_bf16(v1v);
    gate_sh[k_local][threadIdx.x] = w1;

    float w3 = bf16_to_f32(W3[idx]);
    float g3 = bf16_to_f32(dW3[idx]);
    float m3v = bf16_to_f32(m3[idx]);
    float v3v = bf16_to_f32(v3[idx]);
    w3 = adamw_update_f32(w3, g3, m3v, v3v, lr, beta1, beta2, weight_decay, eps, step_size, inv_bias_correction2_sqrt);
    W3[idx] = f32_to_bf16(w3);
    m3[idx] = f32_to_bf16(m3v);
    v3[idx] = f32_to_bf16(v3v);
    up_sh[k_local][threadIdx.x] = w3;
  }

  __syncthreads();

  // Compute + write SF for gate/up rows (one thread per output feature).
  if (threadIdx.y == 0) {
    float amax_gate = 0.0f;
    float amax_up = 0.0f;
    #pragma unroll
    for (int k_local = 0; k_local < TILE_K; ++k_local) {
      amax_gate = fmaxf(amax_gate, fabsf(gate_sh[k_local][threadIdx.x]));
      amax_up = fmaxf(amax_up, fabsf(up_sh[k_local][threadIdx.x]));
    }

    const float dtype_max = kNvfp4 ? FP4_MAX : FP8_MAX;

    float scale_g = amax_gate / dtype_max;
    if (!(scale_g > 0.0f)) scale_g = 1.0f;
    const uint8_t sfg = ptx::e8m0_encode_from_pos_f32(scale_g);
    scale_gate[threadIdx.x] = sfg;

    float scale_u = amax_up / dtype_max;
    if (!(scale_u > 0.0f)) scale_u = 1.0f;
    const uint8_t sfu = ptx::e8m0_encode_from_pos_f32(scale_u);
    scale_up[threadIdx.x] = sfu;

    const size_t expert_base = static_cast<size_t>(e) * static_cast<size_t>(M13) * static_cast<size_t>(sf_k);
    const size_t m_gate = static_cast<size_t>(2 * (j0 + threadIdx.x));
    const size_t m_up = m_gate + 1;
    const size_t dst_gate = cutlass_sf_swizzle_offset(m_gate, static_cast<size_t>(k_sf), static_cast<uint32_t>(M13), static_cast<uint32_t>(sf_k));
    const size_t dst_up = cutlass_sf_swizzle_offset(m_up, static_cast<size_t>(k_sf), static_cast<uint32_t>(M13), static_cast<uint32_t>(sf_k));
    W13_sf_mma[expert_base + dst_gate] = sfg;
    W13_sf_mma[expert_base + dst_up] = sfu;
  }

  __syncthreads();

  // Quantize + pack into W13_q_u16 in row-major (rows = 2*Dff, cols = H).
  const int tid = static_cast<int>(threadIdx.y) * THREADS_X + static_cast<int>(threadIdx.x);
  if constexpr (!kNvfp4) {
    const int packed_per_row = TILE_K / 2;         // 16 u16 for 32 FP8 bytes
    const int tile_rows = 2 * TILE_OUT;            // gate + up
    const int total_packed = tile_rows * packed_per_row;
    const int ldp = H / 2;
    const int col_u16 = k0 / 2;
    for (int idx = tid; idx < total_packed; idx += BLOCK_THREADS) {
      const int row_local = idx / packed_per_row;  // [0, 63]
      const int p = idx - row_local * packed_per_row;
      const bool is_up = row_local >= TILE_OUT;
      const int j_local = row_local & (TILE_OUT - 1);
      const uint8_t sf = is_up ? scale_up[j_local] : scale_gate[j_local];
      const float inv_scale = ptx::e8m0_inv_decode_to_f32(sf);

      const int i0 = 2 * p;
      const int i1 = i0 + 1;
      const float a0 = (is_up ? up_sh[i0][j_local] : gate_sh[i0][j_local]) * inv_scale;
      const float a1 = (is_up ? up_sh[i1][j_local] : gate_sh[i1][j_local]) * inv_scale;
      const uint8_t b0 = ptx::f32_to_e4m3_byte(a0);
      const uint8_t b1 = ptx::f32_to_e4m3_byte(a1);

      const int j_global = j0 + j_local;
      const int m = 2 * j_global + (is_up ? 1 : 0);
      uint16_t* out_row = W13_q_u16 + static_cast<int64_t>(e) * static_cast<int64_t>(M13) * ldp + static_cast<int64_t>(m) * ldp;
      out_row[col_u16 + p] = ptx::pack2_u8_to_u16(b0, b1);
    }
  } else {
#if NMOE_ENABLE_PTX_E2M1
    const int packed_per_row = TILE_K / 4;         // 8 u16 for 32 FP4 values
    const int tile_rows = 2 * TILE_OUT;
    const int total_packed = tile_rows * packed_per_row;
    const int ldp = H / 4;
    const int col_u16 = k0 / 4;
    for (int idx = tid; idx < total_packed; idx += BLOCK_THREADS) {
      const int row_local = idx / packed_per_row;
      const int p = idx - row_local * packed_per_row;
      const bool is_up = row_local >= TILE_OUT;
      const int j_local = row_local & (TILE_OUT - 1);
      const uint8_t sf = is_up ? scale_up[j_local] : scale_gate[j_local];
      const float inv_scale = ptx::e8m0_inv_decode_to_f32(sf);

      const int i0 = 4 * p;
      const float v0 = (is_up ? up_sh[i0 + 0][j_local] : gate_sh[i0 + 0][j_local]) * inv_scale;
      const float v1 = (is_up ? up_sh[i0 + 1][j_local] : gate_sh[i0 + 1][j_local]) * inv_scale;
      const float v2 = (is_up ? up_sh[i0 + 2][j_local] : gate_sh[i0 + 2][j_local]) * inv_scale;
      const float v3 = (is_up ? up_sh[i0 + 3][j_local] : gate_sh[i0 + 3][j_local]) * inv_scale;

      const int j_global = j0 + j_local;
      const int m = 2 * j_global + (is_up ? 1 : 0);
      uint16_t* out_row = W13_q_u16 + static_cast<int64_t>(e) * static_cast<int64_t>(M13) * ldp + static_cast<int64_t>(m) * ldp;
      out_row[col_u16 + p] = ptx::f32x4_to_e2m1x4_packed(v0, v1, v2, v3);
    }
#else
    (void)W13_q_u16;
    __trap();
#endif
  }
#else
  (void)W1;
  (void)dW1;
  (void)m1;
  (void)v1;
  (void)W3;
  (void)dW3;
  (void)m3;
  (void)v3;
  (void)W13_q_u16;
  (void)W13_sf_mma;
  (void)E;
  (void)H;
  (void)Dff;
  (void)lr;
  (void)beta1;
  (void)beta2;
  (void)weight_decay;
  (void)eps;
  (void)step_size;
  (void)inv_bias_correction2_sqrt;
  __trap();
#endif
}

// ============================================================================
// W2: Update W2 [E, Dff, H] and emit transposed cache [E, H, Dff]
// ============================================================================

template <bool kNvfp4>
__global__ void k_expert_adamw_w2_update_quant(
    __nv_bfloat16* __restrict__ W2,
    const __nv_bfloat16* __restrict__ dW2,
    __nv_bfloat16* __restrict__ m2,
    __nv_bfloat16* __restrict__ v2,
    uint16_t* __restrict__ W2_q_u16,
    uint8_t* __restrict__ W2_sf_mma,
    int E,
    int H,
    int Dff,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    float eps,
    float step_size,
    float inv_bias_correction2_sqrt) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int e = static_cast<int>(blockIdx.z);
  const int d0 = static_cast<int>(blockIdx.x) * TILE_K;     // K dimension in cache = Dff
  const int h0 = static_cast<int>(blockIdx.y) * TILE_OUT;   // M dimension in cache = H

  const int h = h0 + static_cast<int>(threadIdx.x);

  const int d_iter = static_cast<int>(threadIdx.y);

  const int sf_k = Dff / SF_VEC;
  const int k_sf = d0 / SF_VEC;

  const int64_t base_w = static_cast<int64_t>(e) * static_cast<int64_t>(Dff) * static_cast<int64_t>(H);

  __shared__ float sh[TILE_OUT][TILE_K];  // [h_local, d_local]
  __shared__ uint8_t scale_row[TILE_OUT];

  // Update + stage. Iterate over 32 Dff columns in 4 groups of 8.
  #pragma unroll
  for (int t = 0; t < (TILE_K / THREADS_Y); ++t) {
    const int d_local = t * THREADS_Y + d_iter;
    const int d = d0 + d_local;
    const int64_t idx = base_w + static_cast<int64_t>(d) * static_cast<int64_t>(H) + static_cast<int64_t>(h);

    float w = bf16_to_f32(W2[idx]);
    float g = bf16_to_f32(dW2[idx]);
    float mv = bf16_to_f32(m2[idx]);
    float vv = bf16_to_f32(v2[idx]);
    w = adamw_update_f32(w, g, mv, vv, lr, beta1, beta2, weight_decay, eps, step_size, inv_bias_correction2_sqrt);
    W2[idx] = f32_to_bf16(w);
    m2[idx] = f32_to_bf16(mv);
    v2[idx] = f32_to_bf16(vv);
    sh[threadIdx.x][d_local] = w;
  }

  __syncthreads();

  // Compute + write SF (one thread per output row h).
  if (threadIdx.y == 0) {
    float amax = 0.0f;
    #pragma unroll
    for (int d_local = 0; d_local < TILE_K; ++d_local) {
      amax = fmaxf(amax, fabsf(sh[threadIdx.x][d_local]));
    }
    const float dtype_max = kNvfp4 ? FP4_MAX : FP8_MAX;
    float scale = amax / dtype_max;
    if (!(scale > 0.0f)) scale = 1.0f;
    const uint8_t sf = ptx::e8m0_encode_from_pos_f32(scale);
    scale_row[threadIdx.x] = sf;

    const size_t expert_base = static_cast<size_t>(e) * static_cast<size_t>(H) * static_cast<size_t>(sf_k);
    const size_t m = static_cast<size_t>(h0 + threadIdx.x);
    const size_t dst = cutlass_sf_swizzle_offset(m, static_cast<size_t>(k_sf), static_cast<uint32_t>(H), static_cast<uint32_t>(sf_k));
    W2_sf_mma[expert_base + dst] = sf;
  }

  __syncthreads();

  // Quantize + pack into W2_q_u16 in row-major over [E*H, Dff].
  const int tid = static_cast<int>(threadIdx.y) * THREADS_X + static_cast<int>(threadIdx.x);
  if constexpr (!kNvfp4) {
    const int packed_per_row = TILE_K / 2;          // 16 u16
    const int total_packed = TILE_OUT * packed_per_row;  // 32 rows * 16
    const int ldp = Dff / 2;
    const int col_u16 = d0 / 2;
    for (int idx = tid; idx < total_packed; idx += BLOCK_THREADS) {
      const int row_local = idx / packed_per_row;  // [0, 31]
      const int p = idx - row_local * packed_per_row;
      const uint8_t sf = scale_row[row_local];
      const float inv_scale = ptx::e8m0_inv_decode_to_f32(sf);

      const int i0 = 2 * p;
      const int i1 = i0 + 1;
      const float a0 = sh[row_local][i0] * inv_scale;
      const float a1 = sh[row_local][i1] * inv_scale;
      const uint8_t b0 = ptx::f32_to_e4m3_byte(a0);
      const uint8_t b1 = ptx::f32_to_e4m3_byte(a1);

      const int h_global = h0 + row_local;
      uint16_t* out_row = W2_q_u16 + static_cast<int64_t>(e) * static_cast<int64_t>(H) * ldp + static_cast<int64_t>(h_global) * ldp;
      out_row[col_u16 + p] = ptx::pack2_u8_to_u16(b0, b1);
    }
  } else {
#if NMOE_ENABLE_PTX_E2M1
    const int packed_per_row = TILE_K / 4;          // 8 u16
    const int total_packed = TILE_OUT * packed_per_row;
    const int ldp = Dff / 4;
    const int col_u16 = d0 / 4;
    for (int idx = tid; idx < total_packed; idx += BLOCK_THREADS) {
      const int row_local = idx / packed_per_row;
      const int p = idx - row_local * packed_per_row;
      const uint8_t sf = scale_row[row_local];
      const float inv_scale = ptx::e8m0_inv_decode_to_f32(sf);

      const int i0 = 4 * p;
      const float v0 = sh[row_local][i0 + 0] * inv_scale;
      const float v1 = sh[row_local][i0 + 1] * inv_scale;
      const float v2 = sh[row_local][i0 + 2] * inv_scale;
      const float v3 = sh[row_local][i0 + 3] * inv_scale;

      const int h_global = h0 + row_local;
      uint16_t* out_row = W2_q_u16 + static_cast<int64_t>(e) * static_cast<int64_t>(H) * ldp + static_cast<int64_t>(h_global) * ldp;
      out_row[col_u16 + p] = ptx::f32x4_to_e2m1x4_packed(v0, v1, v2, v3);
    }
#else
    (void)W2_q_u16;
    __trap();
#endif
  }
#else
  (void)W2;
  (void)dW2;
  (void)m2;
  (void)v2;
  (void)W2_q_u16;
  (void)W2_sf_mma;
  (void)E;
  (void)H;
  (void)Dff;
  (void)lr;
  (void)beta1;
  (void)beta2;
  (void)weight_decay;
  (void)eps;
  (void)step_size;
  (void)inv_bias_correction2_sqrt;
  __trap();
#endif
}

inline cudaError_t launch_expert_adamw_step(
    int profile,
    __nv_bfloat16* W1,
    const __nv_bfloat16* dW1,
    __nv_bfloat16* m1,
    __nv_bfloat16* v1,
    __nv_bfloat16* W3,
    const __nv_bfloat16* dW3,
    __nv_bfloat16* m3,
    __nv_bfloat16* v3,
    __nv_bfloat16* W2,
    const __nv_bfloat16* dW2,
    __nv_bfloat16* m2,
    __nv_bfloat16* v2,
    uint16_t* W13_q_u16,
    uint8_t* W13_sf_mma,
    uint16_t* W2_q_u16,
    uint8_t* W2_sf_mma,
    int E,
    int H,
    int Dff,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    float eps,
    float step_size,
    float inv_bias_correction2_sqrt,
    cudaStream_t stream) {
  if ((H & 127) != 0) return cudaErrorInvalidValue;
  if ((Dff & 127) != 0) return cudaErrorInvalidValue;
  if ((H & 31) != 0) return cudaErrorInvalidValue;
  if ((Dff & 31) != 0) return cudaErrorInvalidValue;

  const dim3 block(THREADS_X, THREADS_Y, 1);

  // W13: tiles over (Dff, H)
  const dim3 grid_w13(ceil_div(Dff, TILE_OUT), ceil_div(H, TILE_K), E);
  if (profile == 0) {
    k_expert_adamw_w13_update_quant<false><<<grid_w13, block, 0, stream>>>(
        W1, dW1, m1, v1,
        W3, dW3, m3, v3,
        W13_q_u16, W13_sf_mma,
        E, H, Dff,
        lr, beta1, beta2, weight_decay, eps, step_size, inv_bias_correction2_sqrt);
  } else if (profile == 1) {
    k_expert_adamw_w13_update_quant<true><<<grid_w13, block, 0, stream>>>(
        W1, dW1, m1, v1,
        W3, dW3, m3, v3,
        W13_q_u16, W13_sf_mma,
        E, H, Dff,
        lr, beta1, beta2, weight_decay, eps, step_size, inv_bias_correction2_sqrt);
  } else {
    return cudaErrorInvalidValue;
  }
  auto err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  // W2: tiles over (Dff, H) but W2 is stored [Dff, H] per expert and cache is [H, Dff].
  const dim3 grid_w2(ceil_div(Dff, TILE_K), ceil_div(H, TILE_OUT), E);
  if (profile == 0) {
    k_expert_adamw_w2_update_quant<false><<<grid_w2, block, 0, stream>>>(
        W2, dW2, m2, v2,
        W2_q_u16, W2_sf_mma,
        E, H, Dff,
        lr, beta1, beta2, weight_decay, eps, step_size, inv_bias_correction2_sqrt);
  } else {
    k_expert_adamw_w2_update_quant<true><<<grid_w2, block, 0, stream>>>(
        W2, dW2, m2, v2,
        W2_q_u16, W2_sf_mma,
        E, H, Dff,
        lr, beta1, beta2, weight_decay, eps, step_size, inv_bias_correction2_sqrt);
  }
  return cudaGetLastError();
}

}  // namespace adamw
}  // namespace nmoe

// ============================================================================
// C API for Python bindings (called via nmoe.csrc.rdep)
// ============================================================================

extern "C" cudaError_t expert_adamw_step(
    int profile,  // 0=fp8, 1=nvfp4
    void* W1,
    const void* dW1,
    void* m1,
    void* v1,
    void* W3,
    const void* dW3,
    void* m3,
    void* v3,
    void* W2,
    const void* dW2,
    void* m2,
    void* v2,
    void* W13_q,
    void* W13_sf_mma,
    void* W2_q,
    void* W2_sf_mma,
    int E,
    int H,
    int Dff,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    float eps,
    float step_size,
    float inv_bias_correction2_sqrt,
    cudaStream_t stream) {
  return nmoe::adamw::launch_expert_adamw_step(
      profile,
      reinterpret_cast<__nv_bfloat16*>(W1),
      reinterpret_cast<const __nv_bfloat16*>(dW1),
      reinterpret_cast<__nv_bfloat16*>(m1),
      reinterpret_cast<__nv_bfloat16*>(v1),
      reinterpret_cast<__nv_bfloat16*>(W3),
      reinterpret_cast<const __nv_bfloat16*>(dW3),
      reinterpret_cast<__nv_bfloat16*>(m3),
      reinterpret_cast<__nv_bfloat16*>(v3),
      reinterpret_cast<__nv_bfloat16*>(W2),
      reinterpret_cast<const __nv_bfloat16*>(dW2),
      reinterpret_cast<__nv_bfloat16*>(m2),
      reinterpret_cast<__nv_bfloat16*>(v2),
      reinterpret_cast<uint16_t*>(W13_q),
      reinterpret_cast<uint8_t*>(W13_sf_mma),
      reinterpret_cast<uint16_t*>(W2_q),
      reinterpret_cast<uint8_t*>(W2_sf_mma),
      E,
      H,
      Dff,
      lr,
      beta1,
      beta2,
      weight_decay,
      eps,
      step_size,
      inv_bias_correction2_sqrt,
      stream);
}
