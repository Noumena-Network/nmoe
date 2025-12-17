// nmoe/quant.cu
// Tilewise BF16 -> {FP8, NVFP4} quantization kernels + packed output.
//
// Kernels:
//   - k_quantize_pack_tilewise_fp8: BF16 -> FP8 E4M3, SFA at 1x32 granularity
//   - k_quantize_pack_tilewise_nvfp4: BF16 -> NVFP4 E2M1, SFA at 1x32 granularity
//
// Output formats:
//   - FP8: out [M, K] uint8, SFA [M, ceil(K/32)] uint8
//   - NVFP4: out [M, K/2] uint8 (packed nibbles), SFA [M, ceil(K/32)] uint8
//
// Scale factors are E8M0 encoded (byte = 127 + log2(scale)).
//
// Target: sm_100a (Blackwell). NVFP4 path requires NMOE_ENABLE_PTX_E2M1=1.

#include "ptx.cu"
#include <vector>

namespace nmoe {
namespace quant {

// ============================================================================
// Constants
// ============================================================================

constexpr int TILE_M = 128;
constexpr int TILE_K = 128;
constexpr int THREADS = 128;  // 4 warps

constexpr float FP8_MAX = 448.0f;   // E4M3 max finite value
constexpr float FP4_MAX = 6.0f;     // E2M1 max finite value

constexpr int SF_VEC_FP8 = 32;      // Scale factor granularity for FP8 (32 BF16 elements)
constexpr int SF_VEC_FP4 = 32;      // Scale factor granularity for NVFP4 (32 BF16 -> 16 packed bytes)

// ============================================================================
// Helpers
// ============================================================================

__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Forward decl (defined below) so quant/SwiGLU kernels can write MMA-layout SF.
__device__ __forceinline__ size_t cutlass_sf_swizzle_offset(
    size_t m, size_t k, uint32_t M, uint32_t sf_k);

// ============================================================================
// FP8 Dequantization (packed 2xFP8 per u16) -> BF16
// ============================================================================
// Input:  q_u16 [M, K/2] where each u16 packs two FP8 (E4M3) bytes
//         sfa   [M, ceil(K/32)] E8M0 scale factors (rowwise per 32)
// Output: out   [M, K] BF16
__global__ void k_dequantize_fp8_to_bf16(
    const uint16_t* __restrict__ q_u16, int ldq,
    const uint8_t* __restrict__ sfa, int ld_sf,
    __nv_bfloat16* __restrict__ out, int ldo,
    int M, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int tile_m = blockIdx.y * TILE_M;
    const int tile_k = blockIdx.x * TILE_K;

    for (int row_off = threadIdx.x; row_off < TILE_M; row_off += blockDim.x) {
        const int m = tile_m + row_off;
        if (m >= M) continue;

        const int k_end = min(tile_k + TILE_K, K);
        const uint16_t* q_row = q_u16 + static_cast<size_t>(m) * ldq;
        const uint8_t*  sfa_row = sfa   + static_cast<size_t>(m) * ld_sf;
        __nv_bfloat16*  out_row = out   + static_cast<size_t>(m) * ldo;

        for (int k0 = tile_k; k0 < k_end; k0 += SF_VEC_FP8) {
            const int span = min(SF_VEC_FP8, k_end - k0);
            const int sf_idx = k0 / SF_VEC_FP8;
            const uint8_t sf_byte = sfa_row[sf_idx];
            const float   scale   = ptx::e8m0_decode_to_f32(sf_byte);

            // Unpack 2xFP8 bytes from each u16 and store two BF16 outputs
            #pragma unroll
            for (int i = 0; i < SF_VEC_FP8 / 2; ++i) {
                const int c0 = 2 * i;
                const int c1 = c0 + 1;
                if (c0 >= span) break;

                const uint16_t packed = q_row[(k0 + c0) / 2];
                uint8_t b0, b1; ptx::unpack_u16_to_2u8(packed, b0, b1);

                float x0 = ptx::e4m3_byte_to_f32(b0) * scale;
                out_row[k0 + c0] = __float2bfloat16(x0);

                if (c1 < span) {
                    float x1 = ptx::e4m3_byte_to_f32(b1) * scale;
                    out_row[k0 + c1] = __float2bfloat16(x1);
                }
            }
        }
    }
#endif
}

inline cudaError_t launch_dequantize_fp8_to_bf16(
    const uint16_t* q_u16, int ldq,
    const uint8_t* sfa, int ld_sf,
    __nv_bfloat16* out, int ldo,
    int M, int K, cudaStream_t stream = 0)
{
    dim3 grid(ceil_div(K, TILE_K), ceil_div(M, TILE_M));
    dim3 block(THREADS);
    k_dequantize_fp8_to_bf16<<<grid, block, 0, stream>>>(q_u16, ldq, sfa, ld_sf, out, ldo, M, K);
    return cudaGetLastError();
}

// ============================================================================
// FP8 Quantization Kernel
// ============================================================================
// Input:  x [M, K] BF16, row-major with stride ldx
// Output: out_u16 [M, K/2] packed FP8 (2 per u16), stride ldp
//         sfa [M, ceil(K/32)] E8M0 scale factors, stride ld_sf

__global__ void k_quantize_pack_tilewise_fp8(
    const __nv_bfloat16* __restrict__ x, int ldx,
    uint16_t* __restrict__ out_u16, int ldp,
    uint8_t* __restrict__ sfa, int ld_sf,
    int M, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int tile_m = blockIdx.y * TILE_M;
    const int tile_k = blockIdx.x * TILE_K;

    // Each thread handles rows with stride blockDim.x
    for (int row_off = threadIdx.x; row_off < TILE_M; row_off += blockDim.x) {
        const int m = tile_m + row_off;
        if (m >= M) continue;

        const int k_end = min(tile_k + TILE_K, K);
        const __nv_bfloat16* x_row = x + static_cast<size_t>(m) * ldx;
        uint8_t* sfa_row = sfa + static_cast<size_t>(m) * ld_sf;
        uint16_t* out_row = out_u16 + static_cast<size_t>(m) * ldp;

        // Process in chunks of SF_VEC_FP8 (32 elements)
        for (int k0 = tile_k; k0 < k_end; k0 += SF_VEC_FP8) {
            const int span = min(SF_VEC_FP8, k_end - k0);
            const int sf_idx = k0 / SF_VEC_FP8;

            // Pass 1: Compute tile amax
            float amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < SF_VEC_FP8; ++i) {
                if (i >= span) break;
                float v = __bfloat162float(x_row[k0 + i]);
                amax = fmaxf(amax, fabsf(v));
            }

            // Compute and store E8M0 scale
            float scale = amax / FP8_MAX;
            if (!(scale > 0.0f)) scale = 1.0f;  // Avoid zero scale
            uint8_t scale_byte = ptx::e8m0_encode_from_pos_f32(scale);
            sfa_row[sf_idx] = scale_byte;

            // Pass 2: Quantize using the exact decoded scale (power-of-two)
            const float inv_scale = ptx::e8m0_inv_decode_to_f32(scale_byte);

            // Pack pairs of FP8 bytes into u16
            #pragma unroll
            for (int i = 0; i < SF_VEC_FP8 / 2; ++i) {
                const int c0 = 2 * i;
                const int c1 = c0 + 1;
                if (c0 >= span) break;

                float v0 = __bfloat162float(x_row[k0 + c0]) * inv_scale;
                uint8_t b0 = ptx::f32_to_e4m3_byte(v0);

                uint8_t b1 = 0;
                if (c1 < span) {
                    float v1 = __bfloat162float(x_row[k0 + c1]) * inv_scale;
                    b1 = ptx::f32_to_e4m3_byte(v1);
                }

                out_row[(k0 + c0) / 2] = ptx::pack2_u8_to_u16(b0, b1);
            }
        }
    }
#endif // __CUDA_ARCH__ >= 1000
}

// ============================================================================
// NVFP4 Quantization Kernel
// ============================================================================
// Input:  x [M, K] BF16, row-major with stride ldx
// Output: out_u16 [M, K/4] packed NVFP4 (4 nibbles per u16), stride ldp
//         sfa [M, ceil(K/16)] E8M0 scale factors, stride ld_sf
//
// Requires NMOE_ENABLE_PTX_E2M1=1 to compile.

__global__ void k_quantize_pack_tilewise_nvfp4(
    const __nv_bfloat16* __restrict__ x, int ldx,
    uint16_t* __restrict__ out_u16, int ldp,
    uint8_t* __restrict__ sfa, int ld_sf,
    int M, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#if NMOE_ENABLE_PTX_E2M1
    const int tile_m = blockIdx.y * TILE_M;
    const int tile_k = blockIdx.x * TILE_K;

    for (int row_off = threadIdx.x; row_off < TILE_M; row_off += blockDim.x) {
        const int m = tile_m + row_off;
        if (m >= M) continue;

        const int k_end = min(tile_k + TILE_K, K);
        const __nv_bfloat16* x_row = x + static_cast<size_t>(m) * ldx;
        uint8_t* sfa_row = sfa + static_cast<size_t>(m) * ld_sf;
        uint16_t* out_row = out_u16 + static_cast<size_t>(m) * ldp;

        // Process in chunks of SF_VEC_FP4 (32 elements)
        for (int k0 = tile_k; k0 < k_end; k0 += SF_VEC_FP4) {
            const int span = min(SF_VEC_FP4, k_end - k0);
            const int sf_idx = k0 / SF_VEC_FP4;

            // Pass 1: Compute tile amax
            float amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < SF_VEC_FP4; ++i) {
                if (i >= span) break;
                float v = __bfloat162float(x_row[k0 + i]);
                amax = fmaxf(amax, fabsf(v));
            }

            // Compute and store E8M0 scale
            float scale = amax / FP4_MAX;
            if (!(scale > 0.0f)) scale = 1.0f;
            uint8_t scale_byte = ptx::e8m0_encode_from_pos_f32(scale);
            sfa_row[sf_idx] = scale_byte;

            // Pass 2: Quantize using the exact decoded scale
            const float inv_scale = ptx::e8m0_inv_decode_to_f32(scale_byte);

            // Pack groups of 4 elements into u16 (4 nibbles)
            #pragma unroll
            for (int i = 0; i < SF_VEC_FP4 / 4; ++i) {
                const int c = 4 * i;
                if (c >= span) break;

                float v0 = __bfloat162float(x_row[k0 + c + 0]) * inv_scale;
                float v1 = (c + 1 < span) ? __bfloat162float(x_row[k0 + c + 1]) * inv_scale : 0.0f;
                float v2 = (c + 2 < span) ? __bfloat162float(x_row[k0 + c + 2]) * inv_scale : 0.0f;
                float v3 = (c + 3 < span) ? __bfloat162float(x_row[k0 + c + 3]) * inv_scale : 0.0f;

                out_row[(k0 + c) / 4] = ptx::f32x4_to_e2m1x4_packed(v0, v1, v2, v3);
            }
        }
    }
#else
    // NVFP4 not enabled - trap if kernel is somehow launched
    __trap();
#endif // NMOE_ENABLE_PTX_E2M1
#endif // __CUDA_ARCH__ >= 1000
}

// ============================================================================
// Quantization (writes SFA directly to per-expert strided MMA layout)
// ============================================================================
// This eliminates the separate swizzle kernel (and its large memset).
//
// Input:  x [M_pad, K] BF16, expert-concatenated with per-expert padding.
// Output: out_u16 [M_pad, ...] packed
//         sf_mma  [E, M_e_stride, sf_k] uint8 E8M0, per-expert MMA layout
// Offsets: offs [E+1] int32 cumulative boundaries (offs[0]=0, offs[E]=M_pad)

constexpr int QUANT_WARPS = 8;
constexpr int QUANT_THREADS = 32 * QUANT_WARPS;

__global__ void k_quantize_pack_tilewise_fp8_sf_strided_mma(
    const __nv_bfloat16* __restrict__ x, int ldx,
    uint16_t* __restrict__ out_u16, int ldp,
    uint8_t* __restrict__ sf_mma,
    const int32_t* __restrict__ offs,
    int E, int M_e_stride, int sf_k,
    int M_pad, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    int tile_k_block = static_cast<int>(blockIdx.x);
    int m_block = static_cast<int>(blockIdx.y);
    if (gridDim.y == 1) {
        const int grid_k = ceil_div(K, TILE_K);
        m_block = tile_k_block / grid_k;
        tile_k_block -= m_block * grid_k;
    }
    const int tile_k = tile_k_block * TILE_K;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = m_block * QUANT_WARPS + warp_id;
    if (m >= M_pad) return;

    // Expert index + local row are only needed for the scale-factor store.
    int m_local = 0;
    int M_e = 0;
    size_t expert_base = 0;
    if (lane == 0) {
        int e = 0;
        while ((e + 1) < E && offs[e + 1] <= m) ++e;
        const int start_e = static_cast<int>(offs[e]);
        const int end_e = static_cast<int>(offs[e + 1]);
        M_e = end_e - start_e;
        m_local = m - start_e;
        expert_base = static_cast<size_t>(e) * static_cast<size_t>(M_e_stride) * static_cast<size_t>(sf_k);
    }

    const int k_end = min(tile_k + TILE_K, K);
    const int k_span = k_end - tile_k;

    const __nv_bfloat16* x_row = x + static_cast<size_t>(m) * ldx + tile_k;
    uint16_t* out_row = out_u16 + static_cast<size_t>(m) * ldp + tile_k / 2;

    const unsigned mask = 0xffffffffu;
    for (int k0 = 0; k0 < k_span; k0 += SF_VEC_FP8) {
        const int j = k0 + lane;
        float v = 0.0f;
        if (j < k_span) v = __bfloat162float(x_row[j]);

        float amax = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_down_sync(mask, amax, off));
        }

        int scale_i = 0;
        if (lane == 0) {
            float scale = amax / FP8_MAX;
            if (!(scale > 0.0f)) scale = 1.0f;
            const uint8_t scale_byte = ptx::e8m0_encode_from_pos_f32(scale);
            const int k_sf = (tile_k + k0) / SF_VEC_FP8;
            const size_t dst_offset = cutlass_sf_swizzle_offset(
                static_cast<size_t>(m_local), static_cast<size_t>(k_sf),
                static_cast<uint32_t>(M_e), static_cast<uint32_t>(sf_k));
            sf_mma[expert_base + dst_offset] = scale_byte;
            scale_i = static_cast<int>(scale_byte);
        }
        scale_i = __shfl_sync(mask, scale_i, 0);
        const uint8_t scale_byte = static_cast<uint8_t>(scale_i);
        const float inv_scale = ptx::e8m0_inv_decode_to_f32(scale_byte);

        const uint8_t b0 = ptx::f32_to_e4m3_byte(v * inv_scale);
        int b1_i = static_cast<int>(b0);
        b1_i = __shfl_down_sync(mask, b1_i, 1);
        if (((lane & 1) == 0) && (j < k_span)) {
            const uint8_t b1 = (j + 1 < k_span) ? static_cast<uint8_t>(b1_i) : 0;
            out_row[(k0 + lane) / 2] = ptx::pack2_u8_to_u16(b0, b1);
        }
    }
#endif // __CUDA_ARCH__ >= 1000
}

__global__ void k_quantize_pack_tilewise_nvfp4_sf_strided_mma(
    const __nv_bfloat16* __restrict__ x, int ldx,
    uint16_t* __restrict__ out_u16, int ldp,
    uint8_t* __restrict__ sf_mma,
    const int32_t* __restrict__ offs,
    int E, int M_e_stride, int sf_k,
    int M_pad, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#if NMOE_ENABLE_PTX_E2M1
    int tile_k_block = static_cast<int>(blockIdx.x);
    int m_block = static_cast<int>(blockIdx.y);
    if (gridDim.y == 1) {
        const int grid_k = ceil_div(K, TILE_K);
        m_block = tile_k_block / grid_k;
        tile_k_block -= m_block * grid_k;
    }
    const int tile_k = tile_k_block * TILE_K;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = m_block * QUANT_WARPS + warp_id;
    if (m >= M_pad) return;

    int m_local = 0;
    int M_e = 0;
    size_t expert_base = 0;
    if (lane == 0) {
        int e = 0;
        while ((e + 1) < E && offs[e + 1] <= m) ++e;
        const int start_e = static_cast<int>(offs[e]);
        const int end_e = static_cast<int>(offs[e + 1]);
        M_e = end_e - start_e;
        m_local = m - start_e;
        expert_base = static_cast<size_t>(e) * static_cast<size_t>(M_e_stride) * static_cast<size_t>(sf_k);
    }

    const int k_end = min(tile_k + TILE_K, K);
    const int k_span = k_end - tile_k;

    const __nv_bfloat16* x_row = x + static_cast<size_t>(m) * ldx + tile_k;
    uint16_t* out_row = out_u16 + static_cast<size_t>(m) * ldp + tile_k / 4;

    const unsigned mask = 0xffffffffu;
    for (int k0 = 0; k0 < k_span; k0 += SF_VEC_FP4) {
        const int j = k0 + lane;
        float v = 0.0f;
        if (j < k_span) v = __bfloat162float(x_row[j]);

        float amax = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_down_sync(mask, amax, off));
        }

        int scale_i = 0;
        if (lane == 0) {
            float scale = amax / FP4_MAX;
            if (!(scale > 0.0f)) scale = 1.0f;
            const uint8_t scale_byte = ptx::e8m0_encode_from_pos_f32(scale);
            const int k_sf = (tile_k + k0) / SF_VEC_FP4;
            const size_t dst_offset = cutlass_sf_swizzle_offset(
                static_cast<size_t>(m_local), static_cast<size_t>(k_sf),
                static_cast<uint32_t>(M_e), static_cast<uint32_t>(sf_k));
            sf_mma[expert_base + dst_offset] = scale_byte;
            scale_i = static_cast<int>(scale_byte);
        }
        scale_i = __shfl_sync(mask, scale_i, 0);
        const uint8_t scale_byte = static_cast<uint8_t>(scale_i);
        const float inv_scale = ptx::e8m0_inv_decode_to_f32(scale_byte);
        const float af = v * inv_scale;

        const int g = lane & ~3;
        const float v0 = __shfl_sync(mask, af, g + 0);
        const float v1 = __shfl_sync(mask, af, g + 1);
        const float v2 = __shfl_sync(mask, af, g + 2);
        const float v3 = __shfl_sync(mask, af, g + 3);
        if ((lane == g) && (g < k_span)) {
            out_row[(k0 >> 2) + (g >> 2)] = ptx::f32x4_to_e2m1x4_packed(v0, v1, v2, v3);
        }
    }
#else
    (void)x; (void)ldx; (void)out_u16; (void)ldp; (void)sf_mma; (void)offs; (void)E; (void)M_e_stride; (void)sf_k; (void)M_pad; (void)K;
    __trap();
#endif // NMOE_ENABLE_PTX_E2M1
#endif // __CUDA_ARCH__ >= 1000
}

// ============================================================================
// SwiGLU + Quantization (fused) for expert MLP
// ============================================================================
// Input:  h13 [M, 2*K] BF16 interleaved columns: [gate0, up0, gate1, up1, ...]
// Output: out_u16 [M, K/2] packed FP8 (2 bytes per u16) or [M, K/4] packed NVFP4
//         sfa [M, ceil(K/32)] uint8 E8M0 scale factors (per 32 output elements)

__device__ __forceinline__ float silu_f32(float x) {
    // x * sigmoid(x) = x / (1 + exp(-x))
    return x / (1.0f + __expf(-x));
}

constexpr int SWIGLU_WARPS = 8;
constexpr int SWIGLU_THREADS = 32 * SWIGLU_WARPS;

__global__ void k_swiglu_quantize_pack_tilewise_fp8(
    const __nv_bfloat16* __restrict__ h13, int ld_h13,
    uint16_t* __restrict__ out_u16, int ld_out,
    uint8_t* __restrict__ sfa, int ld_sf,
    int M, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int tile_k = blockIdx.x * TILE_K;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = static_cast<int>(blockIdx.y) * SWIGLU_WARPS + warp_id;
    if (m >= M) return;

    const int k_end = min(tile_k + TILE_K, K);
    const int k_span = k_end - tile_k;
    const __nv_bfloat16* h13_row = h13 + static_cast<size_t>(m) * ld_h13;
    uint8_t* sfa_row = sfa + static_cast<size_t>(m) * ld_sf + tile_k / SF_VEC_FP8;
    uint16_t* out_row = out_u16 + static_cast<size_t>(m) * ld_out + tile_k / 2;

    // Each warp processes one row, 4 chunks of 32 columns (tile_k=128).
    const unsigned mask = 0xffffffffu;
    const __nv_bfloat162* h13_row2 = reinterpret_cast<const __nv_bfloat162*>(h13_row) + tile_k;
    for (int k0 = 0; k0 < k_span; k0 += SF_VEC_FP8) {
        const int j = k0 + lane;
        float a = 0.0f;
        if (j < k_span) {
            const __nv_bfloat162 gu = h13_row2[j];
            const float2 f2 = __bfloat1622float2(gu);
            a = silu_f32(f2.x) * f2.y;
        }

        // Warp-reduce max |a| over 32 values.
        float amax = fabsf(a);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_down_sync(mask, amax, off));
        }

        // Compute/store scale byte once per 32-wide chunk.
        int scale_i = 0;
        if (lane == 0) {
            float scale = amax / FP8_MAX;
            if (!(scale > 0.0f)) scale = 1.0f;
            const uint8_t scale_byte = ptx::e8m0_encode_from_pos_f32(scale);
            sfa_row[k0 / SF_VEC_FP8] = scale_byte;
            scale_i = static_cast<int>(scale_byte);
        }
        scale_i = __shfl_sync(mask, scale_i, 0);
        const uint8_t scale_byte = static_cast<uint8_t>(scale_i);
        const float inv_scale = ptx::e8m0_inv_decode_to_f32(scale_byte);

        // Quantize + pack (2 FP8 bytes per u16).
        const uint8_t b0 = ptx::f32_to_e4m3_byte(a * inv_scale);
        int b1_i = static_cast<int>(b0);
        b1_i = __shfl_down_sync(mask, b1_i, 1);
        if (((lane & 1) == 0) && (j < k_span)) {
            const uint8_t b1 = (j + 1 < k_span) ? static_cast<uint8_t>(b1_i) : 0;
            out_row[(k0 + lane) / 2] = ptx::pack2_u8_to_u16(b0, b1);
        }
    }
#endif // __CUDA_ARCH__ >= 1000
}

__global__ void k_swiglu_quantize_pack_tilewise_nvfp4(
    const __nv_bfloat16* __restrict__ h13, int ld_h13,
    uint16_t* __restrict__ out_u16, int ld_out,
    uint8_t* __restrict__ sfa, int ld_sf,
    int M, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#if NMOE_ENABLE_PTX_E2M1
    const int tile_k = blockIdx.x * TILE_K;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = static_cast<int>(blockIdx.y) * SWIGLU_WARPS + warp_id;
    if (m >= M) return;

    const int k_end = min(tile_k + TILE_K, K);
    const int k_span = k_end - tile_k;
    const __nv_bfloat16* h13_row = h13 + static_cast<size_t>(m) * ld_h13;
    uint8_t* sfa_row = sfa + static_cast<size_t>(m) * ld_sf + tile_k / SF_VEC_FP4;
    uint16_t* out_row = out_u16 + static_cast<size_t>(m) * ld_out + tile_k / 4;

    const unsigned mask = 0xffffffffu;
    const __nv_bfloat162* h13_row2 = reinterpret_cast<const __nv_bfloat162*>(h13_row) + tile_k;
    for (int k0 = 0; k0 < k_span; k0 += SF_VEC_FP4) {
        const int j = k0 + lane;
        float a = 0.0f;
        if (j < k_span) {
            const __nv_bfloat162 gu = h13_row2[j];
            const float2 f2 = __bfloat1622float2(gu);
            a = silu_f32(f2.x) * f2.y;
        }

        float amax = fabsf(a);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_down_sync(mask, amax, off));
        }

        int scale_i = 0;
        if (lane == 0) {
            float scale = amax / FP4_MAX;
            if (!(scale > 0.0f)) scale = 1.0f;
            const uint8_t scale_byte = ptx::e8m0_encode_from_pos_f32(scale);
            sfa_row[k0 / SF_VEC_FP4] = scale_byte;
            scale_i = static_cast<int>(scale_byte);
        }
        scale_i = __shfl_sync(mask, scale_i, 0);
        const uint8_t scale_byte = static_cast<uint8_t>(scale_i);
        const float inv_scale = ptx::e8m0_inv_decode_to_f32(scale_byte);
        const float af = a * inv_scale;

        // Pack 4 FP4 nibbles per u16 (8 writes per 32-wide chunk).
        const int g = lane & ~3;
        const float v0 = __shfl_sync(mask, af, g + 0);
        const float v1 = __shfl_sync(mask, af, g + 1);
        const float v2 = __shfl_sync(mask, af, g + 2);
        const float v3 = __shfl_sync(mask, af, g + 3);
        if ((lane == g) && (g < k_span)) {
            out_row[(k0 >> 2) + (g >> 2)] = ptx::f32x4_to_e2m1x4_packed(v0, v1, v2, v3);
        }
    }
#else
    (void)h13; (void)ld_h13; (void)out_u16; (void)ld_out; (void)sfa; (void)ld_sf; (void)M; (void)K;
    __trap();
#endif // NMOE_ENABLE_PTX_E2M1
#endif // __CUDA_ARCH__ >= 1000
}

// ============================================================================
// SwiGLU + Quantization (writes SFA directly to per-expert strided MMA layout)
// ============================================================================

__global__ void k_swiglu_quantize_pack_tilewise_fp8_sf_strided_mma(
    const __nv_bfloat16* __restrict__ h13, int ld_h13,
    uint16_t* __restrict__ out_u16, int ld_out,
    uint8_t* __restrict__ sf_mma,
    const int32_t* __restrict__ offs,
    int E, int M_e_stride, int sf_k,
    int M_pad, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    int tile_k_block = static_cast<int>(blockIdx.x);
    int m_block = static_cast<int>(blockIdx.y);
    if (gridDim.y == 1) {
        const int grid_k = ceil_div(K, TILE_K);
        m_block = tile_k_block / grid_k;
        tile_k_block -= m_block * grid_k;
    }
    const int tile_k = tile_k_block * TILE_K;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = m_block * SWIGLU_WARPS + warp_id;
    if (m >= M_pad) return;

    // Expert index + local row only needed for the scale-byte store (lane 0).
    int m_local = 0;
    int M_e = 0;
    size_t expert_base = 0;
    if (lane == 0) {
        int e = 0;
        while ((e + 1) < E && offs[e + 1] <= m) ++e;
        const int start_e = static_cast<int>(offs[e]);
        const int end_e = static_cast<int>(offs[e + 1]);
        M_e = end_e - start_e;
        m_local = m - start_e;
        expert_base = static_cast<size_t>(e) * static_cast<size_t>(M_e_stride) * static_cast<size_t>(sf_k);
    }

    const int k_end = min(tile_k + TILE_K, K);
    const int k_span = k_end - tile_k;
    const __nv_bfloat16* h13_row = h13 + static_cast<size_t>(m) * ld_h13;
    uint16_t* out_row = out_u16 + static_cast<size_t>(m) * ld_out + tile_k / 2;

    const unsigned mask = 0xffffffffu;
    const __nv_bfloat162* h13_row2 = reinterpret_cast<const __nv_bfloat162*>(h13_row) + tile_k;
    for (int k0 = 0; k0 < k_span; k0 += SF_VEC_FP8) {
        const int j = k0 + lane;
        float a = 0.0f;
        if (j < k_span) {
            const __nv_bfloat162 gu = h13_row2[j];
            const float2 f2 = __bfloat1622float2(gu);
            a = silu_f32(f2.x) * f2.y;
        }

        float amax = fabsf(a);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_down_sync(mask, amax, off));
        }

        int scale_i = 0;
        if (lane == 0) {
            float scale = amax / FP8_MAX;
            if (!(scale > 0.0f)) scale = 1.0f;
            const uint8_t scale_byte = ptx::e8m0_encode_from_pos_f32(scale);
            const int k_sf = (tile_k + k0) / SF_VEC_FP8;
            const size_t dst_offset = cutlass_sf_swizzle_offset(
                static_cast<size_t>(m_local), static_cast<size_t>(k_sf),
                static_cast<uint32_t>(M_e), static_cast<uint32_t>(sf_k));
            sf_mma[expert_base + dst_offset] = scale_byte;
            scale_i = static_cast<int>(scale_byte);
        }
        scale_i = __shfl_sync(mask, scale_i, 0);
        const uint8_t scale_byte = static_cast<uint8_t>(scale_i);
        const float inv_scale = ptx::e8m0_inv_decode_to_f32(scale_byte);

        const uint8_t b0 = ptx::f32_to_e4m3_byte(a * inv_scale);
        int b1_i = static_cast<int>(b0);
        b1_i = __shfl_down_sync(mask, b1_i, 1);
        if (((lane & 1) == 0) && (j < k_span)) {
            const uint8_t b1 = (j + 1 < k_span) ? static_cast<uint8_t>(b1_i) : 0;
            out_row[(k0 + lane) / 2] = ptx::pack2_u8_to_u16(b0, b1);
        }
    }
#endif // __CUDA_ARCH__ >= 1000
}

__global__ void k_swiglu_quantize_pack_tilewise_nvfp4_sf_strided_mma(
    const __nv_bfloat16* __restrict__ h13, int ld_h13,
    uint16_t* __restrict__ out_u16, int ld_out,
    uint8_t* __restrict__ sf_mma,
    const int32_t* __restrict__ offs,
    int E, int M_e_stride, int sf_k,
    int M_pad, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#if NMOE_ENABLE_PTX_E2M1
    int tile_k_block = static_cast<int>(blockIdx.x);
    int m_block = static_cast<int>(blockIdx.y);
    if (gridDim.y == 1) {
        const int grid_k = ceil_div(K, TILE_K);
        m_block = tile_k_block / grid_k;
        tile_k_block -= m_block * grid_k;
    }
    const int tile_k = tile_k_block * TILE_K;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = m_block * SWIGLU_WARPS + warp_id;
    if (m >= M_pad) return;

    int m_local = 0;
    int M_e = 0;
    size_t expert_base = 0;
    if (lane == 0) {
        int e = 0;
        while ((e + 1) < E && offs[e + 1] <= m) ++e;
        const int start_e = static_cast<int>(offs[e]);
        const int end_e = static_cast<int>(offs[e + 1]);
        M_e = end_e - start_e;
        m_local = m - start_e;
        expert_base = static_cast<size_t>(e) * static_cast<size_t>(M_e_stride) * static_cast<size_t>(sf_k);
    }

    const int k_end = min(tile_k + TILE_K, K);
    const int k_span = k_end - tile_k;
    const __nv_bfloat16* h13_row = h13 + static_cast<size_t>(m) * ld_h13;
    uint16_t* out_row = out_u16 + static_cast<size_t>(m) * ld_out + tile_k / 4;

    const unsigned mask = 0xffffffffu;
    const __nv_bfloat162* h13_row2 = reinterpret_cast<const __nv_bfloat162*>(h13_row) + tile_k;
    for (int k0 = 0; k0 < k_span; k0 += SF_VEC_FP4) {
        const int j = k0 + lane;
        float a = 0.0f;
        if (j < k_span) {
            const __nv_bfloat162 gu = h13_row2[j];
            const float2 f2 = __bfloat1622float2(gu);
            a = silu_f32(f2.x) * f2.y;
        }

        float amax = fabsf(a);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_down_sync(mask, amax, off));
        }

        int scale_i = 0;
        if (lane == 0) {
            float scale = amax / FP4_MAX;
            if (!(scale > 0.0f)) scale = 1.0f;
            const uint8_t scale_byte = ptx::e8m0_encode_from_pos_f32(scale);
            const int k_sf = (tile_k + k0) / SF_VEC_FP4;
            const size_t dst_offset = cutlass_sf_swizzle_offset(
                static_cast<size_t>(m_local), static_cast<size_t>(k_sf),
                static_cast<uint32_t>(M_e), static_cast<uint32_t>(sf_k));
            sf_mma[expert_base + dst_offset] = scale_byte;
            scale_i = static_cast<int>(scale_byte);
        }
        scale_i = __shfl_sync(mask, scale_i, 0);
        const uint8_t scale_byte = static_cast<uint8_t>(scale_i);
        const float inv_scale = ptx::e8m0_inv_decode_to_f32(scale_byte);
        const float af = a * inv_scale;

        const int g = lane & ~3;
        const float v0 = __shfl_sync(mask, af, g + 0);
        const float v1 = __shfl_sync(mask, af, g + 1);
        const float v2 = __shfl_sync(mask, af, g + 2);
        const float v3 = __shfl_sync(mask, af, g + 3);
        if ((lane == g) && (g < k_span)) {
            out_row[(k0 >> 2) + (g >> 2)] = ptx::f32x4_to_e2m1x4_packed(v0, v1, v2, v3);
        }
    }
#else
    (void)h13; (void)ld_h13; (void)out_u16; (void)ld_out; (void)sf_mma; (void)offs; (void)E; (void)M_e_stride; (void)sf_k; (void)M_pad; (void)K;
    __trap();
#endif // NMOE_ENABLE_PTX_E2M1
#endif // __CUDA_ARCH__ >= 1000
}

// ============================================================================
// SwiGLU Backward (fused) for expert MLP (BF16)
// ============================================================================
// Computes:
//   A  = silu(h1) * h3
//   dH3 = dA * silu(h1)
//   dH1 = dA * h3 * silu'(h1)
//
// Inputs:  h1 [M, K] BF16 (gate), h3 [M, K] BF16 (up), dA [M, K] BF16
// Outputs: A  [M, K] BF16, dH1 [M, K] BF16, dH3 [M, K] BF16

__global__ void k_swiglu_bwd_bf16(
    const __nv_bfloat16* __restrict__ h1, int ld_h1,
    const __nv_bfloat16* __restrict__ h3, int ld_h3,
    const __nv_bfloat16* __restrict__ dA, int ld_dA,
    __nv_bfloat16* __restrict__ A_out, int ld_A,
    __nv_bfloat16* __restrict__ dH1_out, int ld_dH1,
    __nv_bfloat16* __restrict__ dH3_out, int ld_dH3,
    int M, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(blockDim.x * gridDim.x);
    const int total = M * K;
    for (int i = idx; i < total; i += stride) {
        const int m = i / K;
        const int k = i - m * K;

        const float x = __bfloat162float(h1[static_cast<size_t>(m) * ld_h1 + k]);
        const float u = __bfloat162float(h3[static_cast<size_t>(m) * ld_h3 + k]);
        const float d = __bfloat162float(dA[static_cast<size_t>(m) * ld_dA + k]);

        const float s = 1.0f / (1.0f + __expf(-x));
        const float silu = x * s;
        const float a = silu * u;
        const float dsilu = s * (1.0f + x * (1.0f - s));

        const float dH3 = d * silu;
        const float dH1 = d * u * dsilu;

        A_out[static_cast<size_t>(m) * ld_A + k] = __float2bfloat16(a);
        dH1_out[static_cast<size_t>(m) * ld_dH1 + k] = __float2bfloat16(dH1);
        dH3_out[static_cast<size_t>(m) * ld_dH3 + k] = __float2bfloat16(dH3);
    }
#endif // __CUDA_ARCH__ >= 1000
}

// ============================================================================
// Host Launchers
// ============================================================================

inline cudaError_t launch_quantize_fp8(
    const __nv_bfloat16* x, int ldx,
    uint16_t* out_u16, int ldp,
    uint8_t* sfa, int ld_sf,
    int M, int K,
    cudaStream_t stream = 0)
{
    dim3 grid(ceil_div(K, TILE_K), ceil_div(M, TILE_M));
    dim3 block(THREADS);
    k_quantize_pack_tilewise_fp8<<<grid, block, 0, stream>>>(
        x, ldx, out_u16, ldp, sfa, ld_sf, M, K);
    return cudaGetLastError();
}

inline cudaError_t launch_quantize_fp8_sf_strided_mma(
    const __nv_bfloat16* x, int ldx,
    uint16_t* out_u16, int ldp,
    uint8_t* sf_mma,
    const int32_t* offs, int E, int M_e_stride, int sf_k,
    int M_pad, int K,
    cudaStream_t stream = 0)
{
    const int grid_k = ceil_div(K, TILE_K);
    const int grid_m = ceil_div(M_pad, QUANT_WARPS);
    const int64_t grid_x = static_cast<int64_t>(grid_k) * static_cast<int64_t>(grid_m);
    if (grid_x <= 0 || grid_x > 0x7fffffffll) return cudaErrorInvalidConfiguration;
    dim3 grid(static_cast<uint32_t>(grid_x), 1, 1);
    dim3 block(QUANT_THREADS);
    k_quantize_pack_tilewise_fp8_sf_strided_mma<<<grid, block, 0, stream>>>(
        x, ldx, out_u16, ldp, sf_mma, offs, E, M_e_stride, sf_k, M_pad, K);
    return cudaGetLastError();
}

inline cudaError_t launch_quantize_nvfp4(
    const __nv_bfloat16* x, int ldx,
    uint16_t* out_u16, int ldp,
    uint8_t* sfa, int ld_sf,
    int M, int K,
    cudaStream_t stream = 0)
{
#if NMOE_ENABLE_PTX_E2M1
    dim3 grid(ceil_div(K, TILE_K), ceil_div(M, TILE_M));
    dim3 block(THREADS);
    k_quantize_pack_tilewise_nvfp4<<<grid, block, 0, stream>>>(
        x, ldx, out_u16, ldp, sfa, ld_sf, M, K);
    return cudaGetLastError();
#else
    (void)x; (void)ldx; (void)out_u16; (void)ldp; (void)sfa; (void)ld_sf; (void)M; (void)K; (void)stream;
    return cudaErrorNotSupported;
#endif
}

inline cudaError_t launch_quantize_nvfp4_sf_strided_mma(
    const __nv_bfloat16* x, int ldx,
    uint16_t* out_u16, int ldp,
    uint8_t* sf_mma,
    const int32_t* offs, int E, int M_e_stride, int sf_k,
    int M_pad, int K,
    cudaStream_t stream = 0)
{
#if NMOE_ENABLE_PTX_E2M1
    const int grid_k = ceil_div(K, TILE_K);
    const int grid_m = ceil_div(M_pad, QUANT_WARPS);
    const int64_t grid_x = static_cast<int64_t>(grid_k) * static_cast<int64_t>(grid_m);
    if (grid_x <= 0 || grid_x > 0x7fffffffll) return cudaErrorInvalidConfiguration;
    dim3 grid(static_cast<uint32_t>(grid_x), 1, 1);
    dim3 block(QUANT_THREADS);
    k_quantize_pack_tilewise_nvfp4_sf_strided_mma<<<grid, block, 0, stream>>>(
        x, ldx, out_u16, ldp, sf_mma, offs, E, M_e_stride, sf_k, M_pad, K);
    return cudaGetLastError();
#else
    (void)x; (void)ldx; (void)out_u16; (void)ldp; (void)sf_mma; (void)offs; (void)E; (void)M_e_stride; (void)sf_k; (void)M_pad; (void)K; (void)stream;
    return cudaErrorNotSupported;
#endif
}

inline cudaError_t launch_swiglu_quantize_fp8(
    const __nv_bfloat16* h13, int ld_h13,
    uint16_t* out_u16, int ld_out,
    uint8_t* sfa, int ld_sf,
    int M, int K,
    cudaStream_t stream = 0)
{
    dim3 grid(ceil_div(K, TILE_K), ceil_div(M, SWIGLU_WARPS));
    dim3 block(SWIGLU_THREADS);
    k_swiglu_quantize_pack_tilewise_fp8<<<grid, block, 0, stream>>>(
        h13, ld_h13, out_u16, ld_out, sfa, ld_sf, M, K);
    return cudaGetLastError();
}

inline cudaError_t launch_swiglu_bwd_bf16(
    const __nv_bfloat16* h1, int ld_h1,
    const __nv_bfloat16* h3, int ld_h3,
    const __nv_bfloat16* dA, int ld_dA,
    __nv_bfloat16* A_out, int ld_A,
    __nv_bfloat16* dH1_out, int ld_dH1,
    __nv_bfloat16* dH3_out, int ld_dH3,
    int M, int K,
    cudaStream_t stream = 0)
{
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(M) * static_cast<int64_t>(K);
    int blocks = static_cast<int>((total + threads - 1) / threads);
    blocks = (blocks <= 0) ? 1 : ((blocks > 1024) ? 1024 : blocks);
    k_swiglu_bwd_bf16<<<blocks, threads, 0, stream>>>(
        h1, ld_h1, h3, ld_h3, dA, ld_dA,
        A_out, ld_A, dH1_out, ld_dH1, dH3_out, ld_dH3,
        M, K);
    return cudaGetLastError();
}

inline cudaError_t launch_swiglu_quantize_fp8_sf_strided_mma(
    const __nv_bfloat16* h13, int ld_h13,
    uint16_t* out_u16, int ld_out,
    uint8_t* sf_mma,
    const int32_t* offs, int E, int M_e_stride, int sf_k,
    int M_pad, int K,
    cudaStream_t stream = 0)
{
    const int grid_k = ceil_div(K, TILE_K);
    const int grid_m = ceil_div(M_pad, SWIGLU_WARPS);
    const int64_t grid_x = static_cast<int64_t>(grid_k) * static_cast<int64_t>(grid_m);
    if (grid_x <= 0 || grid_x > 0x7fffffffll) return cudaErrorInvalidConfiguration;
    dim3 grid(static_cast<uint32_t>(grid_x), 1, 1);
    dim3 block(SWIGLU_THREADS);
    k_swiglu_quantize_pack_tilewise_fp8_sf_strided_mma<<<grid, block, 0, stream>>>(
        h13, ld_h13, out_u16, ld_out, sf_mma, offs, E, M_e_stride, sf_k, M_pad, K);
    return cudaGetLastError();
}

inline cudaError_t launch_swiglu_quantize_nvfp4(
    const __nv_bfloat16* h13, int ld_h13,
    uint16_t* out_u16, int ld_out,
    uint8_t* sfa, int ld_sf,
    int M, int K,
    cudaStream_t stream = 0)
{
#if NMOE_ENABLE_PTX_E2M1
    dim3 grid(ceil_div(K, TILE_K), ceil_div(M, SWIGLU_WARPS));
    dim3 block(SWIGLU_THREADS);
    k_swiglu_quantize_pack_tilewise_nvfp4<<<grid, block, 0, stream>>>(
        h13, ld_h13, out_u16, ld_out, sfa, ld_sf, M, K);
    return cudaGetLastError();
#else
    (void)h13; (void)ld_h13; (void)out_u16; (void)ld_out; (void)sfa; (void)ld_sf; (void)M; (void)K; (void)stream;
    return cudaErrorNotSupported;
#endif
}

inline cudaError_t launch_swiglu_quantize_nvfp4_sf_strided_mma(
    const __nv_bfloat16* h13, int ld_h13,
    uint16_t* out_u16, int ld_out,
    uint8_t* sf_mma,
    const int32_t* offs, int E, int M_e_stride, int sf_k,
    int M_pad, int K,
    cudaStream_t stream = 0)
{
#if NMOE_ENABLE_PTX_E2M1
    const int grid_k = ceil_div(K, TILE_K);
    const int grid_m = ceil_div(M_pad, SWIGLU_WARPS);
    const int64_t grid_x = static_cast<int64_t>(grid_k) * static_cast<int64_t>(grid_m);
    if (grid_x <= 0 || grid_x > 0x7fffffffll) return cudaErrorInvalidConfiguration;
    dim3 grid(static_cast<uint32_t>(grid_x), 1, 1);
    dim3 block(SWIGLU_THREADS);
    k_swiglu_quantize_pack_tilewise_nvfp4_sf_strided_mma<<<grid, block, 0, stream>>>(
        h13, ld_h13, out_u16, ld_out, sf_mma, offs, E, M_e_stride, sf_k, M_pad, K);
    return cudaGetLastError();
#else
    (void)h13; (void)ld_h13; (void)out_u16; (void)ld_out; (void)sf_mma; (void)offs; (void)E; (void)M_e_stride; (void)sf_k; (void)M_pad; (void)K; (void)stream;
    return cudaErrorNotSupported;
#endif
}

// ============================================================================
// NVFP4: Unpack nibble-packed u16 -> one-code-per-byte (for CuTe Float4E2M1FN)
// ============================================================================
// Input:  q_u16 [M, K/4] (each uint16 holds four 4-bit FP4 codes)
// Output: out_bytes [M, K] (each byte holds a single 4-bit FP4 code replicated
//                            into both nibbles). Replication ensures kernels that
//                            read the high nibble see the same value.
// NOTE: Removed legacy unpack helpers. NVFP4 decode should stay inside the
// blockscaled kernels or dedicated debug tooling, not the production extension surface.

// ============================================================================
// NVFP4 dense: nibble-wise repack within 32-lane blocks using a 32-entry LUT
// ----------------------------------------------------------------------------
// Reorders the stream of FP4 nibbles along K so that each 32-lane block follows
// the loader's expected lane order. Operates on packed bytes: two nibbles/byte.
// in/out may alias for in-place repack.
// ----------------------------------------------------------------------------
__global__ void k_repack_nvfp4_dense_perm(
    const uint8_t* __restrict__ in_u8, int ldi_bytes,
    uint8_t* __restrict__ out_u8, int ldo_bytes,
    const uint8_t* __restrict__ perm32, // 32 entries: dest index per src lane
    int M, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int m0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int dm = blockDim.x * gridDim.x;
    for (int m = m0; m < M; m += dm) {
        const uint8_t* in_row = in_u8 + (size_t)m * ldi_bytes;
        uint8_t* out_row = out_u8 + (size_t)m * ldo_bytes;
        // Zero out destination row (we OR-in nibbles below)
        for (int b = 0; b < K/2; ++b) out_row[b] = 0;
        // Process in 32-lane blocks
        for (int k = 0; k < K; k += 32) {
            uint8_t lanes[32];
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                int lane = k + i;
                int byte = lane >> 1;
                uint8_t b = in_row[byte];
                lanes[i] = (lane & 1) ? (uint8_t)((b >> 4) & 0x0F) : (uint8_t)(b & 0x0F);
            }
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                int j = (int)perm32[i];
                int dst_lane = k + j;
                int dst_byte = dst_lane >> 1;
                uint8_t v = lanes[i] & 0x0F;
                if (dst_lane & 1) {
                    out_row[dst_byte] = (uint8_t)((out_row[dst_byte] & 0x0F) | (v << 4));
                } else {
                    out_row[dst_byte] = (uint8_t)((out_row[dst_byte] & 0xF0) | v);
                }
            }
        }
    }
#endif
}

inline cudaError_t launch_repack_nvfp4_dense_perm(
    const uint8_t* in_u8, int ldi_bytes,
    uint8_t* out_u8, int ldo_bytes,
    const uint8_t* perm32,
    int M, int K,
    cudaStream_t stream = 0)
{
    const int threads = 128;
    const int blocks = (M + threads - 1) / threads;
    k_repack_nvfp4_dense_perm<<<blocks, threads, 0, stream>>>(in_u8, ldi_bytes, out_u8, ldo_bytes, perm32, M, K);
    return cudaGetLastError();
}
// ============================================================================
// Quantization using PRECOMPUTED E8M0 SFA (Blockscale policy)
// ============================================================================
// These variants do NOT compute amax/scale; they consume caller-provided SFA
// (E8M0, row-major [M, ceil(K/sf_vec)]) and only quantize + pack.

__global__ void k_quantize_pack_tilewise_fp8_with_sfa(
    const __nv_bfloat16* __restrict__ x, int ldx,
    uint16_t* __restrict__ out_u16, int ldp,
    const uint8_t* __restrict__ sfa, int ld_sf,
    int M, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int tile_m = blockIdx.y * TILE_M;
    const int tile_k = blockIdx.x * TILE_K;

    for (int row_off = threadIdx.x; row_off < TILE_M; row_off += blockDim.x) {
        const int m = tile_m + row_off;
        if (m >= M) continue;

        const int k_end = min(tile_k + TILE_K, K);
        const __nv_bfloat16* x_row = x + static_cast<size_t>(m) * ldx;
        const uint8_t* sfa_row = sfa + static_cast<size_t>(m) * ld_sf;
        uint16_t* out_row = out_u16 + static_cast<size_t>(m) * ldp;

        for (int k0 = tile_k; k0 < k_end; k0 += SF_VEC_FP8) {
            const int span = min(SF_VEC_FP8, k_end - k0);
            const int sf_idx = k0 / SF_VEC_FP8;
            uint8_t scale_byte = sfa_row[sf_idx];
            float inv_scale = ptx::e8m0_inv_decode_to_f32(scale_byte);

            #pragma unroll
            for (int i = 0; i < SF_VEC_FP8 / 2; ++i) {
                const int c0 = 2 * i;
                const int c1 = c0 + 1;
                if (c0 >= span) break;

                float v0 = __bfloat162float(x_row[k0 + c0]) * inv_scale;
                uint8_t b0 = ptx::f32_to_e4m3_byte(v0);
                uint8_t b1 = 0;
                if (c1 < span) {
                    float v1 = __bfloat162float(x_row[k0 + c1]) * inv_scale;
                    b1 = ptx::f32_to_e4m3_byte(v1);
                }
                out_row[(k0 + c0) / 2] = ptx::pack2_u8_to_u16(b0, b1);
            }
        }
    }
#endif
}

__global__ void k_quantize_pack_tilewise_nvfp4_with_sfa(
    const __nv_bfloat16* __restrict__ x, int ldx,
    uint16_t* __restrict__ out_u16, int ldp,
    const uint8_t* __restrict__ sfa, int ld_sf,
    int M, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#if NMOE_ENABLE_PTX_E2M1
    const int tile_m = blockIdx.y * TILE_M;
    const int tile_k = blockIdx.x * TILE_K;

    for (int row_off = threadIdx.x; row_off < TILE_M; row_off += blockDim.x) {
        const int m = tile_m + row_off;
        if (m >= M) continue;

        const int k_end = min(tile_k + TILE_K, K);
        const __nv_bfloat16* x_row = x + static_cast<size_t>(m) * ldx;
        const uint8_t* sfa_row = sfa + static_cast<size_t>(m) * ld_sf;
        uint16_t* out_row = out_u16 + static_cast<size_t>(m) * ldp;

        for (int k0 = tile_k; k0 < k_end; k0 += SF_VEC_FP4) {
            const int span = min(SF_VEC_FP4, k_end - k0);
            const int sf_idx = k0 / SF_VEC_FP4;
            float inv_scale = ptx::e8m0_inv_decode_to_f32(sfa_row[sf_idx]);

            #pragma unroll
            for (int i = 0; i < SF_VEC_FP4 / 4; ++i) {
                const int c = 4 * i;
                if (c >= span) break;
                float v0 = __bfloat162float(x_row[k0 + c + 0]) * inv_scale;
                float v1 = (c + 1 < span) ? __bfloat162float(x_row[k0 + c + 1]) * inv_scale : 0.0f;
                float v2 = (c + 2 < span) ? __bfloat162float(x_row[k0 + c + 2]) * inv_scale : 0.0f;
                float v3 = (c + 3 < span) ? __bfloat162float(x_row[k0 + c + 3]) * inv_scale : 0.0f;
                out_row[(k0 + c) / 4] = ptx::f32x4_to_e2m1x4_packed(v0, v1, v2, v3);
            }
        }
    }
#else
    __trap();
#endif
#endif
}

inline cudaError_t launch_quantize_fp8_with_sfa(
    const __nv_bfloat16* x, int ldx,
    uint16_t* out_u16, int ldp,
    const uint8_t* sfa, int ld_sf,
    int M, int K,
    cudaStream_t stream = 0)
{
    dim3 grid(ceil_div(K, TILE_K), ceil_div(M, TILE_M));
    dim3 block(THREADS);
    k_quantize_pack_tilewise_fp8_with_sfa<<<grid, block, 0, stream>>>(
        x, ldx, out_u16, ldp, sfa, ld_sf, M, K);
    return cudaGetLastError();
}

inline cudaError_t launch_quantize_nvfp4_with_sfa(
    const __nv_bfloat16* x, int ldx,
    uint16_t* out_u16, int ldp,
    const uint8_t* sfa, int ld_sf,
    int M, int K,
    cudaStream_t stream = 0)
{
#if NMOE_ENABLE_PTX_E2M1
    dim3 grid(ceil_div(K, TILE_K), ceil_div(M, TILE_M));
    dim3 block(THREADS);
    k_quantize_pack_tilewise_nvfp4_with_sfa<<<grid, block, 0, stream>>>(
        x, ldx, out_u16, ldp, sfa, ld_sf, M, K);
    return cudaGetLastError();
#else
    (void)x; (void)ldx; (void)out_u16; (void)ldp; (void)sfa; (void)ld_sf; (void)M; (void)K; (void)stream;
    return cudaErrorNotSupported;
#endif
}

// ============================================================================
// Fused Dispatch + Quantize Kernel (Single-GPU MoE path)
// ============================================================================
// Reads sorted BF16 input, quantizes to FP8/NVFP4, writes to padded output.
// This is used for single-GPU MoE where we don't need NVSHMEM scatter.
//
// Input:
//   x_sorted: [M, H] BF16 activations sorted by expert
//   dest: [M] int32 destination indices in padded output
// Output:
//   out_q_pad: [M_pad, Hp] uint16 packed quantized activations
//   out_sf_pad: [M_pad, Hsf] uint8 E8M0 scale factors
// NOTE: Removed legacy fused dispatch+quant kernels (single-GPU convenience path).
// The production path quantizes from the padded per-expert Xe buffer directly.

// ============================================================================
// Scale Factor Swizzle Kernel (MKL -> MMA layout)
// ============================================================================
// Converts row-major scale factors [M, sf_k] to CUTLASS DSL MMA-compatible layout.
//
// CUTLASS BlockScaledBasicChunk defines the MMA atom layout:
//   atom_shape = ((32, 4), (sf_vec, 4))
//   atom_stride = ((16, 4), (0, 1))
//
// The MMA layout is derived from:
//   MKL shape: (L, M, sf_k)
//   MMA shape: (L, M/128, sf_k/4, 32, 4, 4)
//   MMA permute: (3, 4, 1, 5, 2, 0) -> (32, 4, M/128, 4, sf_k/4, L)
//
// For L=1, the offset formula is:
//   offset = k_block + cbg_cnt * k_rem + 4 * cbg_cnt * m_block
//          + 4 * cbg_cnt * rb_cnt * m_32 + 16 * cbg_cnt * rb_cnt * m_rem
//
// Where:
//   rb_cnt = M / 128, cbg_cnt = sf_k / 4
//   m_block = m / 128, m_32 = (m % 128) / 32, m_rem = m % 32
//   k_block = k / 4, k_rem = k % 4
//
// Input:  sf_mkl [M, sf_k] uint8 E8M0, row-major (M and sf_k already padded to 128 and 4)
// Output: sf_mma [M * sf_k] uint8 E8M0, CUTLASS MMA layout

__device__ __forceinline__ size_t cutlass_sf_swizzle_offset(
    size_t m, size_t k, uint32_t M, uint32_t sf_k)
{
    // CUTLASS DSL BlockScaledBasicChunk atom layout:
    //   atom shape = ((32, 4), (sf_vec, 4))
    //   atom stride = ((16, 4), (0, 1))
    //
    // The sf_vec dimension has stride=0 (broadcast), so only (m_32, m_4, k_4) matter.
    // Within-atom offset = m_32 * 16 + m_4 * 4 + k_4
    //
    // tile_to_shape tiles atoms across (rest_m, rest_k) with row-major order:
    //   atom_idx = m_rest * rest_k + k_rest
    //   total_offset = atom_idx * atom_size + atom_offset
    //   where atom_size = 128 * 4 = 512

    const uint32_t atom_m = 128;
    const uint32_t atom_k = 4;
    const uint32_t atom_size = atom_m * atom_k;  // 512
    const uint32_t rest_k = sf_k / atom_k;

    // Decompose m
    const size_t m_32 = m % 32;
    const size_t m_4 = (m / 32) % 4;
    const size_t m_rest = m / atom_m;

    // Decompose k
    const size_t k_4 = k % atom_k;
    const size_t k_rest = k / atom_k;

    // Within-atom offset (from atom strides (16, 4, 0, 1))
    const size_t atom_offset = m_32 * 16 + m_4 * 4 + k_4;

    // Atom index (row-major over (rest_m, rest_k): rest_k varies fastest)
    const size_t atom_idx = m_rest * rest_k + k_rest;

    return atom_idx * atom_size + atom_offset;
}

  __global__ void k_swizzle_sf_mkl_to_mma(
      const uint8_t* __restrict__ sf_mkl,  // [M, sf_k] row-major
      uint8_t* __restrict__ sf_mma,         // [M * sf_k] swizzled
      int M, int sf_k)
  {
    // Each thread handles one element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * sf_k;
    if (idx >= total) return;

    const int m = idx / sf_k;
    const int k = idx % sf_k;

    const size_t dst_idx = cutlass_sf_swizzle_offset(m, k, M, sf_k);
    sf_mma[dst_idx] = sf_mkl[idx];
}

  inline cudaError_t launch_swizzle_sf(
      const uint8_t* sf_mkl,
      uint8_t* sf_mma,
      int M, int sf_k,
      cudaStream_t stream = 0)
  {
      const int total = M * sf_k;
      const int threads = 256;
      const int blocks = ceil_div(total, threads);
      k_swizzle_sf_mkl_to_mma<<<blocks, threads, 0, stream>>>(sf_mkl, sf_mma, M, sf_k);
      return cudaGetLastError();
  }

  // Dense NVFP4 variant: use row-major tiling of atoms across (rest_m, rest_k)
  __device__ __forceinline__ size_t cutlass_sf_swizzle_offset_dense(
      size_t m, size_t k, uint32_t M, uint32_t sf_k)
  {
      const uint32_t atom_m = 128;
      const uint32_t atom_k = 4;
      const uint32_t atom_size = atom_m * atom_k;  // 512
      const uint32_t rest_k = sf_k / atom_k;

      const size_t m_32 = m % 32;
      const size_t m_4  = (m / 32) % 4;
      const size_t m_rest = m / atom_m;

      const size_t k_4 = k % atom_k;
      const size_t k_rest = k / atom_k;

      const size_t atom_offset = m_32 * 16 + m_4 * 4 + k_4;
      // Row-major tiling: m_rest varies fastest
      const size_t atom_idx = m_rest * rest_k + k_rest;
      return atom_idx * atom_size + atom_offset;
  }

  __global__ void k_swizzle_sf_mkl_to_mma_dense(
      const uint8_t* __restrict__ sf_mkl,  // [M, sf_k] row-major
      uint8_t* __restrict__ sf_mma,         // [M * sf_k] swizzled
      int M, int sf_k)
  {
      const int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int total = M * sf_k;
      if (idx >= total) return;

      const int m = idx / sf_k;
      const int k = idx % sf_k;

      const size_t dst_idx = cutlass_sf_swizzle_offset_dense(m, k, M, sf_k);
      sf_mma[dst_idx] = sf_mkl[idx];
  }

// ============================================================================
// Per-Expert Strided Scale Factor Swizzle (for activation SFs)
// ============================================================================
// Swizzles scale factors per-expert using expert-local M_e_pad for swizzle.
// Each expert's SFs are swizzled independently, so CUTLASS can stride through
// the concatenated result and each expert sees its own correctly swizzled data.
//
// Input:  sf_mkl [M_pad, sf_k] uint8 E8M0, row-major
// Output: sf_mma [M_pad, sf_k_pad] uint8 E8M0, per-expert swizzled
// Offsets: offs [E+1] cumulative offsets with leading 0, 128-aligned

__global__ void k_swizzle_sf_strided(
    const uint8_t* __restrict__ sf_mkl,   // [M_pad, sf_k] row-major input
    uint8_t* __restrict__ sf_mma,          // [E * M_e_max * sf_k_pad] swizzled output
    const int32_t* __restrict__ offs,      // [E+1] cumulative offsets (128-aligned)
    int E, int sf_k, int sf_k_pad, int M_e_max)
{
    // Grid over all valid input elements: M_pad * sf_k
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int m = idx / sf_k;   // global row in concatenated A/SFA
    const int k = idx % sf_k;   // scale-factor column

    // Bounds: offs[E] is total M_pad
    const int M_pad = offs[E];
    if (m >= M_pad) return;

    // Find expert e such that offs[e] <= m < offs[e+1]
    int lo = 0, hi = E;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (offs[mid] <= m) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const int e = lo;  // expert index

    // Local coords within expert and that expert's M_e
    const int32_t start_e = offs[e];
    const int32_t end_e   = offs[e + 1];
    const int32_t M_e     = end_e - start_e;  // must be 128-aligned by contract
    const int m_local = m - start_e;

    // Read MKL row-major byte
    const uint8_t val = sf_mkl[m * sf_k + k];

    // Destination base for expert e (fixed-size chunk per expert = M_e_max * sf_k_pad)
    const size_t expert_base = static_cast<size_t>(e) * static_cast<size_t>(M_e_max) * static_cast<size_t>(sf_k_pad);

    // Swizzle using this expert's true M_e so CUTLASS's per-group M matches swizzle layout
    const size_t dst_offset = cutlass_sf_swizzle_offset(m_local, k, M_e, sf_k);
    sf_mma[expert_base + dst_offset] = val;
}

inline cudaError_t launch_swizzle_sf_strided(
    const uint8_t* sf_mkl,
    uint8_t* sf_mma,
    const int32_t* offs,
    int E, int sf_k, int sf_k_pad, int M_pad, int M_e_swizzle,
    cudaStream_t stream = 0)
{
    // Zero output first (padding bytes)
    // Output size is E * M_e_swizzle * sf_k_pad (each expert gets fixed-size region)
    cudaMemsetAsync(sf_mma, 0, static_cast<size_t>(E) * M_e_swizzle * sf_k_pad, stream);

    const int total = M_pad * sf_k;
    const int threads = 256;
    const int blocks = ceil_div(total, threads);
    k_swizzle_sf_strided<<<blocks, threads, 0, stream>>>(sf_mkl, sf_mma, offs, E, sf_k, sf_k_pad, M_e_swizzle);
    return cudaGetLastError();
}

// Inverse of k_swizzle_sf_strided: convert per-expert MMA-layout SF to row-major [M_pad, sf_k].
__global__ void k_unswizzle_sf_strided(
    const uint8_t* __restrict__ sf_mma,   // [E * M_e_max * sf_k_pad] swizzled input
    uint8_t* __restrict__ sf_mkl,         // [M_pad * sf_k] row-major output
    const int32_t* __restrict__ offs,     // [E+1] cumulative offsets (leading 0)
    int E, int sf_k, int sf_k_pad, int M_e_max)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = idx / sf_k;
    const int k = idx % sf_k;

    const int M_pad = offs[E];
    if (m >= M_pad) return;

    // Find expert e such that offs[e] <= m < offs[e+1].
    int lo = 0, hi = E;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (offs[mid] <= m) lo = mid;
        else hi = mid - 1;
    }
    const int e = lo;

    const int32_t start_e = offs[e];
    const int32_t end_e = offs[e + 1];
    const int32_t M_e = end_e - start_e;
    const int m_local = m - start_e;

    const size_t expert_base =
        static_cast<size_t>(e) * static_cast<size_t>(M_e_max) * static_cast<size_t>(sf_k_pad);
    const size_t src_offset = cutlass_sf_swizzle_offset(m_local, k, M_e, sf_k);
    sf_mkl[m * sf_k + k] = sf_mma[expert_base + src_offset];
}

inline cudaError_t launch_unswizzle_sf_strided(
    const uint8_t* sf_mma,
    uint8_t* sf_mkl,
    const int32_t* offs,
    int E, int sf_k, int sf_k_pad, int M_pad, int M_e_swizzle,
    cudaStream_t stream = 0)
{
    const int total = M_pad * sf_k;
    const int threads = 256;
    const int blocks = ceil_div(total, threads);
    k_unswizzle_sf_strided<<<blocks, threads, 0, stream>>>(
        sf_mma, sf_mkl, offs, E, sf_k, sf_k_pad, M_e_swizzle);
    return cudaGetLastError();
}

// ============================================================================
// Grouped GEMM Metadata Builder (for run_grouped_blockscaled_strided)
// ============================================================================
// Builds per-expert metadata arrays for grouped GEMM entirely on GPU,
// avoiding Python loops and .item() GPU->CPU syncs.
//
// For padded/strided tensors:
//   A_pad: [M_pad, Kp, 1] activations, SFA_pad: [M_pad, SFKp, 1] scale factors
//   B: [E, N, Kp, 1] stacked weights, SFB: [E, N, SFKp, 1] weight SFs
//   offs: [E+1] cumulative offsets (e.g. [0, 512, 1024, ...])
//
// Computes:
//   sizes_mnkl[e] = (M_e, N, K, 1) where M_e = offs[e+1] - offs[e]
//   ptrs_abc[e] = (A_base + offs[e]*A_stride, B_base + e*B_stride, C_base + offs[e]*C_stride)
//   ptrs_sfasfb[e] = (SFA_base + offs[e]*SFA_stride, SFB_base + e*SFB_stride)
//   strides_abc[e] = ((A_s0, A_s1), (B_s0, B_s1), (C_s0, C_s1))

__global__ void k_build_grouped_gemm_metadata(
    // Input: offsets tensor
    const int32_t* __restrict__ offs,  // [E+1] cumulative offsets
    int E,                              // number of experts
    // Tensor metadata - BYTE strides for pointer arithmetic
    int64_t A_base,  int64_t A_row_bytes,                   // A_pad base ptr and byte stride per row
    int64_t B_base,  int64_t B_expert_bytes,                // B stacked: B[e] = B_base + e*B_expert_bytes
    int64_t C_base,  int64_t C_row_bytes,                   // C_pad base ptr and byte stride per row
    int64_t SFA_base, int64_t SFA_expert_bytes,             // SFA stacked: SFA[e] = SFA_base + e*SFA_expert_bytes
    int64_t SFB_base, int64_t SFB_expert_bytes,             // SFB stacked: SFB[e] = SFB_base + e*SFB_expert_bytes
    // Tensor metadata - ELEMENT strides for CUTLASS
    int32_t A_stride0_elem, int32_t A_stride1_elem,         // A element strides
    int32_t B_stride0_elem, int32_t B_stride1_elem,         // B element strides (within expert)
    int32_t C_stride0_elem, int32_t C_stride1_elem,         // C element strides
    int32_t N, int32_t K,                                   // Constant dimensions
    // Output: metadata arrays
    int32_t* __restrict__ sizes_mnkl,    // [E, 4]
    int32_t* __restrict__ strides_abc,   // [E, 3, 2]
    int64_t* __restrict__ ptrs_abc,      // [E, 3]
    int64_t* __restrict__ ptrs_sfasfb)   // [E, 2]
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    // Compute M for this expert
    const int32_t M_e = offs[e + 1] - offs[e];
    const int32_t start = offs[e];

    // sizes_mnkl[e] = (M_e, N, K, 1)
    sizes_mnkl[e * 4 + 0] = M_e;
    sizes_mnkl[e * 4 + 1] = N;
    sizes_mnkl[e * 4 + 2] = K;
    sizes_mnkl[e * 4 + 3] = 1;

    // strides_abc[e, 0] = A strides, [e, 1] = B strides, [e, 2] = C strides
    // These are ELEMENT strides for CUTLASS
    strides_abc[e * 6 + 0] = A_stride0_elem;
    strides_abc[e * 6 + 1] = A_stride1_elem;
    strides_abc[e * 6 + 2] = B_stride0_elem;
    strides_abc[e * 6 + 3] = B_stride1_elem;
    strides_abc[e * 6 + 4] = C_stride0_elem;
    strides_abc[e * 6 + 5] = C_stride1_elem;

    // ptrs_abc[e] = (A_ptr, B_ptr, C_ptr)
    // Use BYTE strides for pointer arithmetic
    ptrs_abc[e * 3 + 0] = A_base + static_cast<int64_t>(start) * A_row_bytes;
    ptrs_abc[e * 3 + 1] = B_base + static_cast<int64_t>(e) * B_expert_bytes;
    ptrs_abc[e * 3 + 2] = C_base + static_cast<int64_t>(start) * C_row_bytes;

    // ptrs_sfasfb[e] = (SFA_ptr, SFB_ptr)
    // Activation SFA is stored per-expert as a fixed-stride MMA-swizzled chunk.
    // SFA_expert_bytes is the per-expert byte stride (e.g. M_e_stride * sf_k_pad).
    ptrs_sfasfb[e * 2 + 0] = SFA_base + static_cast<int64_t>(e) * SFA_expert_bytes;
    ptrs_sfasfb[e * 2 + 1] = SFB_base + static_cast<int64_t>(e) * SFB_expert_bytes;
}

inline cudaError_t launch_build_grouped_gemm_metadata(
    const int32_t* offs, int E,
    // Byte strides for pointer arithmetic
    int64_t A_base, int64_t A_row_bytes,
    int64_t B_base, int64_t B_expert_bytes,
    int64_t C_base, int64_t C_row_bytes,
    int64_t SFA_base, int64_t SFA_expert_bytes,  // SFA uses expert-based indexing
    int64_t SFB_base, int64_t SFB_expert_bytes,
    // Element strides for CUTLASS
    int32_t A_stride0_elem, int32_t A_stride1_elem,
    int32_t B_stride0_elem, int32_t B_stride1_elem,
    int32_t C_stride0_elem, int32_t C_stride1_elem,
    int32_t N, int32_t K,
    int32_t* sizes_mnkl, int32_t* strides_abc, int64_t* ptrs_abc, int64_t* ptrs_sfasfb,
    cudaStream_t stream = 0)
{
    const int threads = 256;
    const int blocks = ceil_div(E, threads);
    k_build_grouped_gemm_metadata<<<blocks, threads, 0, stream>>>(
        offs, E,
        A_base, A_row_bytes,
        B_base, B_expert_bytes,
        C_base, C_row_bytes,
        SFA_base, SFA_expert_bytes,
        SFB_base, SFB_expert_bytes,
        A_stride0_elem, A_stride1_elem,
        B_stride0_elem, B_stride1_elem,
        C_stride0_elem, C_stride1_elem,
        N, K,
        sizes_mnkl, strides_abc, ptrs_abc, ptrs_sfasfb);
    return cudaGetLastError();
}

  // ================================================================
  // Dense-specific NVFP4 SF swizzles (aliases of the generic layout).
  // We expose separate entry points to keep the Python surface explicit
  // for dense NVFP4 without changing grouped/other users.
  // ================================================================
  inline cudaError_t launch_swizzle_sf_dense_nvfp4(
      const uint8_t* sf_mkl,
      uint8_t* sf_mma,
      int M, int sf_k,
      cudaStream_t stream = 0)
  {
      const int total = M * sf_k;
      const int threads = 256;
      const int blocks = ceil_div(total, threads);
      k_swizzle_sf_mkl_to_mma_dense<<<blocks, threads, 0, stream>>>(sf_mkl, sf_mma, M, sf_k);
      return cudaGetLastError();
  }

  // (colwise swizzle removed; single correct swizzle path is row-wise)
 

} // namespace quant
} // namespace nmoe

// ============================================================================
// C API for Python bindings
// ============================================================================

extern "C" {

cudaError_t quant_fp8(
    const void* x, int ldx,
    void* out, int ld_out,
    void* sfa, int ld_sf,
    int M, int K,
    cudaStream_t stream)
{
    return nmoe::quant::launch_quantize_fp8(
        reinterpret_cast<const __nv_bfloat16*>(x), ldx,
        reinterpret_cast<uint16_t*>(out), ld_out,
        reinterpret_cast<uint8_t*>(sfa), ld_sf,
        M, K, stream);
}

cudaError_t quant_nvfp4(
    const void* x, int ldx,
    void* out, int ld_out,
    void* sfa, int ld_sf,
    int M, int K,
    cudaStream_t stream)
{
    return nmoe::quant::launch_quantize_nvfp4(
        reinterpret_cast<const __nv_bfloat16*>(x), ldx,
        reinterpret_cast<uint16_t*>(out), ld_out,
        reinterpret_cast<uint8_t*>(sfa), ld_sf,
        M, K, stream);
}

cudaError_t quant_fp8_sf_strided_mma(
    const void* x, int ldx,
    void* out, int ld_out,
    void* sf_mma,
    const int32_t* offs,
    int E, int M_e_stride,
    int M_pad, int K,
    cudaStream_t stream)
{
    if ((K & 31) != 0) return cudaErrorInvalidValue;
    if ((M_e_stride & 127) != 0) return cudaErrorInvalidValue;
    const int sf_k = K / 32;
    if ((sf_k & 3) != 0) return cudaErrorInvalidValue;  // require sf_k % 4 == 0 (K % 128 == 0)

    return nmoe::quant::launch_quantize_fp8_sf_strided_mma(
        reinterpret_cast<const __nv_bfloat16*>(x), ldx,
        reinterpret_cast<uint16_t*>(out), ld_out,
        reinterpret_cast<uint8_t*>(sf_mma),
        offs, E, M_e_stride, sf_k,
        M_pad, K, stream);
}

cudaError_t quant_nvfp4_sf_strided_mma(
    const void* x, int ldx,
    void* out, int ld_out,
    void* sf_mma,
    const int32_t* offs,
    int E, int M_e_stride,
    int M_pad, int K,
    cudaStream_t stream)
{
    if ((K & 31) != 0) return cudaErrorInvalidValue;
    if ((M_e_stride & 127) != 0) return cudaErrorInvalidValue;
    const int sf_k = K / 32;
    if ((sf_k & 3) != 0) return cudaErrorInvalidValue;

    return nmoe::quant::launch_quantize_nvfp4_sf_strided_mma(
        reinterpret_cast<const __nv_bfloat16*>(x), ldx,
        reinterpret_cast<uint16_t*>(out), ld_out,
        reinterpret_cast<uint8_t*>(sf_mma),
        offs, E, M_e_stride, sf_k,
        M_pad, K, stream);
}

cudaError_t swiglu_bwd_bf16(
    const void* h1, int ld_h1,
    const void* h3, int ld_h3,
    const void* dA, int ld_dA,
    void* A_out, int ld_A,
    void* dH1_out, int ld_dH1,
    void* dH3_out, int ld_dH3,
    int M, int K,
    cudaStream_t stream)
{
    return nmoe::quant::launch_swiglu_bwd_bf16(
        reinterpret_cast<const __nv_bfloat16*>(h1), ld_h1,
        reinterpret_cast<const __nv_bfloat16*>(h3), ld_h3,
        reinterpret_cast<const __nv_bfloat16*>(dA), ld_dA,
        reinterpret_cast<__nv_bfloat16*>(A_out), ld_A,
        reinterpret_cast<__nv_bfloat16*>(dH1_out), ld_dH1,
        reinterpret_cast<__nv_bfloat16*>(dH3_out), ld_dH3,
        M, K, stream);
}

cudaError_t swiglu_quant_fp8(
    const void* h13, int ld_h13,
    void* out, int ld_out,
    void* sfa, int ld_sf,
    int M, int K,
    cudaStream_t stream)
{
    return nmoe::quant::launch_swiglu_quantize_fp8(
        reinterpret_cast<const __nv_bfloat16*>(h13), ld_h13,
        reinterpret_cast<uint16_t*>(out), ld_out,
        reinterpret_cast<uint8_t*>(sfa), ld_sf,
        M, K, stream);
}

cudaError_t swiglu_quant_nvfp4(
    const void* h13, int ld_h13,
    void* out, int ld_out,
    void* sfa, int ld_sf,
    int M, int K,
    cudaStream_t stream)
{
    return nmoe::quant::launch_swiglu_quantize_nvfp4(
        reinterpret_cast<const __nv_bfloat16*>(h13), ld_h13,
        reinterpret_cast<uint16_t*>(out), ld_out,
        reinterpret_cast<uint8_t*>(sfa), ld_sf,
        M, K, stream);
}

cudaError_t swiglu_quant_fp8_sf_strided_mma(
    const void* h13, int ld_h13,
    void* out, int ld_out,
    void* sf_mma,
    const int32_t* offs,
    int E, int M_e_stride,
    int M_pad, int K,
    cudaStream_t stream)
{
    if ((K & 31) != 0) return cudaErrorInvalidValue;
    if ((M_e_stride & 127) != 0) return cudaErrorInvalidValue;
    const int sf_k = K / 32;
    if ((sf_k & 3) != 0) return cudaErrorInvalidValue;

    return nmoe::quant::launch_swiglu_quantize_fp8_sf_strided_mma(
        reinterpret_cast<const __nv_bfloat16*>(h13), ld_h13,
        reinterpret_cast<uint16_t*>(out), ld_out,
        reinterpret_cast<uint8_t*>(sf_mma),
        offs, E, M_e_stride, sf_k,
        M_pad, K, stream);
}

cudaError_t swiglu_quant_nvfp4_sf_strided_mma(
    const void* h13, int ld_h13,
    void* out, int ld_out,
    void* sf_mma,
    const int32_t* offs,
    int E, int M_e_stride,
    int M_pad, int K,
    cudaStream_t stream)
{
    if ((K & 31) != 0) return cudaErrorInvalidValue;
    if ((M_e_stride & 127) != 0) return cudaErrorInvalidValue;
    const int sf_k = K / 32;
    if ((sf_k & 3) != 0) return cudaErrorInvalidValue;

    return nmoe::quant::launch_swiglu_quantize_nvfp4_sf_strided_mma(
        reinterpret_cast<const __nv_bfloat16*>(h13), ld_h13,
        reinterpret_cast<uint16_t*>(out), ld_out,
        reinterpret_cast<uint8_t*>(sf_mma),
        offs, E, M_e_stride, sf_k,
        M_pad, K, stream);
}

// Dense-specific NVFP4 SF swizzles
extern "C" cudaError_t swizzle_sf_dense_nvfp4(
    const void* sf_mkl, void* sf_mma,
    int M, int sf_k, cudaStream_t stream)
{
    return nmoe::quant::launch_swizzle_sf_dense_nvfp4(
        reinterpret_cast<const uint8_t*>(sf_mkl),
        reinterpret_cast<uint8_t*>(sf_mma),
        M, sf_k, stream);
}

// Quantization with PRECOMPUTED SFA (consumes E8M0 scales)
cudaError_t quant_fp8_with_sfa(
    const void* x, int ldx,
    void* out, int ld_out,
    const void* sfa, int ld_sf,
    int M, int K,
    cudaStream_t stream)
{
    return nmoe::quant::launch_quantize_fp8_with_sfa(
        reinterpret_cast<const __nv_bfloat16*>(x), ldx,
        reinterpret_cast<uint16_t*>(out), ld_out,
        reinterpret_cast<const uint8_t*>(sfa), ld_sf,
        M, K, stream);
}

cudaError_t quant_nvfp4_with_sfa(
    const void* x, int ldx,
    void* out, int ld_out,
    const void* sfa, int ld_sf,
    int M, int K,
    cudaStream_t stream)
{
    return nmoe::quant::launch_quantize_nvfp4_with_sfa(
        reinterpret_cast<const __nv_bfloat16*>(x), ldx,
        reinterpret_cast<uint16_t*>(out), ld_out,
        reinterpret_cast<const uint8_t*>(sfa), ld_sf,
        M, K, stream);
}

// ============================================================================
// NVFP4 -> BF16 Dequantization (dense helpers)
// ============================================================================
// Input:  q_u16 [M, K/4] packed NVFP4 (4 nibbles per uint16), row-major with stride ldq
//         sfa   [M, ceil(K/32)] E8M0 scale factors (per-32), row-major with stride ld_sf
// Output: out   [M, K] BF16, row-major with stride ldo

namespace nmoe { namespace quant {

__global__ void k_dequantize_nvfp4_to_bf16(
    const uint16_t* __restrict__ q_u16, int ldq,
    const uint8_t*  __restrict__ sfa,  int ld_sf,
    __nv_bfloat16*  __restrict__ out,  int ldo,
    int M, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int row0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int dr = blockDim.x * gridDim.x;
    for (int m = row0; m < M; m += dr) {
        const uint16_t* q_row = q_u16 + static_cast<size_t>(m) * ldq;
        const uint8_t*  sf_row = sfa   + static_cast<size_t>(m) * ld_sf;
        __nv_bfloat16*  o_row  = out   + static_cast<size_t>(m) * ldo;

        for (int k0 = 0; k0 < K; k0 += SF_VEC_FP4) {
            const int span = min(SF_VEC_FP4, K - k0);
            const int sf_idx = k0 / SF_VEC_FP4;
            const float scale = ptx::e8m0_decode_to_f32(sf_row[sf_idx]);

            // process in groups of 4 values (one uint16)
            #pragma unroll
            for (int i = 0; i < SF_VEC_FP4 / 4; ++i) {
                const int c = 4 * i;
                if (c >= span) break;
                const uint16_t packed = q_row[(k0 + c) / 4];
                float x0, x1, x2, x3;
                ptx::e2m1x4_packed_to_f32x4(packed, x0, x1, x2, x3);
                if (k0 + c + 0 < K) o_row[k0 + c + 0] = __float2bfloat16(x0 * scale);
                if (k0 + c + 1 < K) o_row[k0 + c + 1] = __float2bfloat16(x1 * scale);
                if (k0 + c + 2 < K) o_row[k0 + c + 2] = __float2bfloat16(x2 * scale);
                if (k0 + c + 3 < K) o_row[k0 + c + 3] = __float2bfloat16(x3 * scale);
            }
        }
    }
#endif
}

inline cudaError_t launch_dequantize_nvfp4_to_bf16(
    const uint16_t* q_u16, int ldq,
    const uint8_t*  sfa,   int ld_sf,
    __nv_bfloat16*  out,   int ldo,
    int M, int K,
    cudaStream_t stream = 0)
{
    const int threads = 128;
    const int blocks  = (M + threads - 1) / threads;
    k_dequantize_nvfp4_to_bf16<<<blocks, threads, 0, stream>>>(q_u16, ldq, sfa, ld_sf, out, ldo, M, K);
    return cudaGetLastError();
}

}} // namespace nmoe::quant

extern "C" cudaError_t dequant_nvfp4_to_bf16(
    const void* q, int ldq,
    const void* sfa, int ld_sf,
    int M, int K,
    void* out, int ldo,
    cudaStream_t stream)
{
    return nmoe::quant::launch_dequantize_nvfp4_to_bf16(
        reinterpret_cast<const uint16_t*>(q), ldq,
        reinterpret_cast<const uint8_t*>(sfa), ld_sf,
        reinterpret_cast<__nv_bfloat16*>(out), ldo,
        M, K, stream);
}

extern "C" cudaError_t repack_nvfp4_dense_perm(
    const void* in_u8, int ldi_bytes,
    void* out_u8, int ldo_bytes,
    const void* perm32,
    int M, int K,
    cudaStream_t stream)
{
    return nmoe::quant::launch_repack_nvfp4_dense_perm(
        reinterpret_cast<const uint8_t*>(in_u8), ldi_bytes,
        reinterpret_cast<uint8_t*>(out_u8), ldo_bytes,
        reinterpret_cast<const uint8_t*>(perm32),
        M, K, stream);
}

// ============================================================================
// Fused Dense GEMM: (NVFP4 A, per-32 SFA) x (BF16 W) -> BF16 Y
// ============================================================================
// A_q:  [M, K/4]  uint16  (packed fp4, 4 nibbles per u16), row-major, ldq
// SFA:  [M, K/32] uint8   (E8M0, per-32), row-major, ld_sf
// W:    [N, K]    bf16    (row-major), ldw
// Bias: [N]       bf16    (optional, may be nullptr)
// Y:    [M, N]    bf16    (row-major), ldy
//
// Contract: Y[m,n] = sum_k dequant(A_q[m,k], SFA[m,k/32]) * W[n,k] + bias[n]
// Dequant: fp4 -> f32 via e2m1x4_packed_to_f32x4; multiply by e8m0_decode(scale)

namespace nmoe { namespace quant {

__global__ void k_fused_dense_nvfp4_gemm_bf16(
    const uint16_t* __restrict__ A_q,  int ldq,
    const uint8_t*  __restrict__ SFA,  int ld_sf,
    const __nv_bfloat16* __restrict__ W, int ldw,
    const __nv_bfloat16* __restrict__ bias,            // optional
    __nv_bfloat16* __restrict__ Y, int ldy,
    int M, int N, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int n = blockIdx.x * blockDim.x + threadIdx.x; // column in Y
    const int m = blockIdx.y * blockDim.y + threadIdx.y; // row in Y
    if (m >= M || n >= N) return;

    const uint16_t* Aq_row = A_q + (size_t)m * ldq;
    const uint8_t*  sfa_row = SFA + (size_t)m * ld_sf;
    const __nv_bfloat16* W_row = W + (size_t)n * ldw; // W[n, :]

    float acc = 0.0f;
    // Iterate K in chunks of 32 (one scale)
    for (int k0 = 0; k0 < K; k0 += 32) {
        const int span = min(32, K - k0);
        const float scale = ptx::e8m0_decode_to_f32(sfa_row[k0/32]);
        // process 4 at a time from packed
        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            if (i >= span) break;
            const uint16_t packed = Aq_row[(k0 + i) / 4];
            float x0, x1, x2, x3;
            ptx::e2m1x4_packed_to_f32x4(packed, x0, x1, x2, x3);
            // Load weights and FMA
            const __nv_bfloat16 *wptr = W_row + (k0 + i);
            if (k0 + i + 0 < K) acc += (x0 * scale) * __bfloat162float(wptr[0]);
            if (k0 + i + 1 < K) acc += (x1 * scale) * __bfloat162float(wptr[1]);
            if (k0 + i + 2 < K) acc += (x2 * scale) * __bfloat162float(wptr[2]);
            if (k0 + i + 3 < K) acc += (x3 * scale) * __bfloat162float(wptr[3]);
        }
    }
    if (bias) acc += __bfloat162float(bias[n]);
    Y[(size_t)m * ldy + n] = __float2bfloat16(acc);
#endif
}

inline cudaError_t launch_fused_dense_nvfp4_gemm_bf16(
    const uint16_t* A_q, int ldq,
    const uint8_t*  SFA, int ld_sf,
    const __nv_bfloat16* W, int ldw,
    const __nv_bfloat16* bias,
    __nv_bfloat16* Y, int ldy,
    int M, int N, int K,
    cudaStream_t stream = 0)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
    k_fused_dense_nvfp4_gemm_bf16<<<grid, block, 0, stream>>>(A_q, ldq, SFA, ld_sf, W, ldw, bias, Y, ldy, M, N, K);
    return cudaGetLastError();
}

}} // namespace nmoe::quant

extern "C" cudaError_t fused_dense_nvfp4_gemm_bf16(
    const void* A_q, int ldq,
    const void* SFA, int ld_sf,
    const void* W, int ldw,
    const void* bias,
    void* Y, int ldy,
    int M, int N, int K,
    cudaStream_t stream)
{
    return nmoe::quant::launch_fused_dense_nvfp4_gemm_bf16(
        reinterpret_cast<const uint16_t*>(A_q), ldq,
        reinterpret_cast<const uint8_t*>(SFA), ld_sf,
        reinterpret_cast<const __nv_bfloat16*>(W), ldw,
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(Y), ldy,
        M, N, K, stream);
}

// ============================================================================
// Fused Dense GEMM: BF16 A --quantize NVFP4 per-32 on-the-fly--> x BF16 W -> BF16 Y
// ============================================================================
namespace nmoe { namespace quant {

__global__ void k_fused_dense_quant_nvfp4_gemm_bf16(
    const __nv_bfloat16* __restrict__ A, int lda,
    const __nv_bfloat16* __restrict__ W, int ldw,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ Y, int ldy,
    int M, int N, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    const __nv_bfloat16* A_row = A + (size_t)m * lda;
    const __nv_bfloat16* W_row = W + (size_t)n * ldw;
    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += 32) {
        const int span = min(32, K - k0);
        // amax over 32
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (i >= span) break;
            float v = __bfloat162float(A_row[k0 + i]);
            amax = fmaxf(amax, fabsf(v));
        }
        float scale_pos = (amax > 0.f) ? (amax / 6.0f) : 1.0f;
        uint8_t sb = nmoe::ptx::e8m0_encode_from_pos_f32(scale_pos);
        float scale = nmoe::ptx::e8m0_decode_to_f32(sb);
        float inv_scale = (scale > 0.f) ? (1.0f/scale) : 0.f;

        // process in groups of 4 using existing pack/depack helpers
        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            if (i >= span) break;
            float f0 = __bfloat162float(A_row[k0 + i + 0]) * inv_scale;
            float f1 = (i+1 < span) ? __bfloat162float(A_row[k0 + i + 1]) * inv_scale : 0.0f;
            float f2 = (i+2 < span) ? __bfloat162float(A_row[k0 + i + 2]) * inv_scale : 0.0f;
            float f3 = (i+3 < span) ? __bfloat162float(A_row[k0 + i + 3]) * inv_scale : 0.0f;
            uint16_t packed = nmoe::ptx::f32x4_to_e2m1x4_packed(f0, f1, f2, f3);
            float x0,x1,x2,x3;
            nmoe::ptx::e2m1x4_packed_to_f32x4(packed, x0, x1, x2, x3);
            x0 *= scale; x1 *= scale; x2 *= scale; x3 *= scale;
            const __nv_bfloat16* wptr = W_row + (k0 + i);
            acc += x0 * __bfloat162float(wptr[0]);
            if (i+1 < span) acc += x1 * __bfloat162float(wptr[1]);
            if (i+2 < span) acc += x2 * __bfloat162float(wptr[2]);
            if (i+3 < span) acc += x3 * __bfloat162float(wptr[3]);
        }
    }

    if (bias) acc += __bfloat162float(bias[n]);
    Y[(size_t)m * ldy + n] = __float2bfloat16(acc);
#endif
}

inline cudaError_t launch_fused_dense_quant_nvfp4_gemm_bf16(
    const __nv_bfloat16* A, int lda,
    const __nv_bfloat16* W, int ldw,
    const __nv_bfloat16* bias,
    __nv_bfloat16* Y, int ldy,
    int M, int N, int K,
    cudaStream_t stream = 0)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
    k_fused_dense_quant_nvfp4_gemm_bf16<<<grid, block, 0, stream>>>(A, lda, W, ldw, bias, Y, ldy, M, N, K);
    return cudaGetLastError();
}

}} // namespace nmoe::quant

extern "C" cudaError_t fused_dense_quant_nvfp4_gemm_bf16(
    const void* A, int lda,
    const void* W, int ldw,
    const void* bias,
    void* Y, int ldy,
    int M, int N, int K,
    cudaStream_t stream)
{
    return nmoe::quant::launch_fused_dense_quant_nvfp4_gemm_bf16(
        reinterpret_cast<const __nv_bfloat16*>(A), lda,
        reinterpret_cast<const __nv_bfloat16*>(W), ldw,
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(Y), ldy,
        M, N, K, stream);
}

// Fused dispatch + quantize for single-GPU MoE
// NOTE: Removed legacy dispatch+quant entrypoints (single-GPU convenience path).

// Swizzle scale factors from MKL (row-major) to MMA layout - rowwise (for SFA)
cudaError_t swizzle_sf_mkl_to_mma(
    const void* sf_mkl,  // [M, sf_k] uint8, row-major
    void* sf_mma,        // [M * sf_k] uint8, swizzled output
    int M, int sf_k,
    cudaStream_t stream)
{
    return nmoe::quant::launch_swizzle_sf(
        reinterpret_cast<const uint8_t*>(sf_mkl),
        reinterpret_cast<uint8_t*>(sf_mma),
        M, sf_k, stream);
}

// Per-expert strided swizzle for activation SFs
cudaError_t swizzle_sf_strided(
    const void* sf_mkl,   // [M_pad, sf_k] uint8, row-major
    void* sf_mma,         // [E, M_e_swizzle, sf_k_pad] uint8, per-expert swizzled
    const int32_t* offs,  // [E+1] cumulative offsets with leading 0
    int E, int sf_k, int sf_k_pad, int M_pad, int M_e_swizzle,
    cudaStream_t stream)
{
    return nmoe::quant::launch_swizzle_sf_strided(
        reinterpret_cast<const uint8_t*>(sf_mkl),
        reinterpret_cast<uint8_t*>(sf_mma),
        offs, E, sf_k, sf_k_pad, M_pad, M_e_swizzle, stream);
}

cudaError_t unswizzle_sf_strided(
    const void* sf_mma,   // [E, M_e_swizzle, sf_k_pad] uint8, per-expert swizzled
    void* sf_mkl,         // [M_pad, sf_k] uint8, row-major
    const int32_t* offs,  // [E+1] cumulative offsets with leading 0
    int E, int sf_k, int sf_k_pad, int M_pad, int M_e_swizzle,
    cudaStream_t stream)
{
    return nmoe::quant::launch_unswizzle_sf_strided(
        reinterpret_cast<const uint8_t*>(sf_mma),
        reinterpret_cast<uint8_t*>(sf_mkl),
        offs, E, sf_k, sf_k_pad, M_pad, M_e_swizzle, stream);
}

// Build grouped GEMM metadata on GPU (no CPU sync needed)
cudaError_t build_grouped_gemm_metadata(
    const int32_t* offs, int E,
    // Byte strides for pointer arithmetic
    int64_t A_base, int64_t A_row_bytes,
    int64_t B_base, int64_t B_expert_bytes,
    int64_t C_base, int64_t C_row_bytes,
    int64_t SFA_base, int64_t SFA_expert_bytes,
    int64_t SFB_base, int64_t SFB_expert_bytes,
    // Element strides for CUTLASS
    int32_t A_stride0_elem, int32_t A_stride1_elem,
    int32_t B_stride0_elem, int32_t B_stride1_elem,
    int32_t C_stride0_elem, int32_t C_stride1_elem,
    int32_t N, int32_t K,
    int32_t* sizes_mnkl, int32_t* strides_abc, int64_t* ptrs_abc, int64_t* ptrs_sfasfb,
    cudaStream_t stream)
{
    return nmoe::quant::launch_build_grouped_gemm_metadata(
        offs, E,
        A_base, A_row_bytes,
        B_base, B_expert_bytes,
        C_base, C_row_bytes,
        SFA_base, SFA_expert_bytes,
        SFB_base, SFB_expert_bytes,
        A_stride0_elem, A_stride1_elem,
        B_stride0_elem, B_stride1_elem,
        C_stride0_elem, C_stride1_elem,
        N, K,
        sizes_mnkl, strides_abc, ptrs_abc, ptrs_sfasfb,
        stream);
}

} // extern "C"

// ============================================================================
// Grouped Dense GEMM (skeleton): loop experts and launch fused NVFP4->BF16 GEMM
// ============================================================================
extern "C" cudaError_t grouped_dense_nvfp4_gemm_bf16_strided(
    const int32_t* d_sizes_mnkl,    // [E,4] (M_e, N, K, 1) on device
    const int32_t* d_strides_abc,   // [E,3,2] element strides on device
    const int64_t* d_ptrs_abc,      // [E,3] device array of device pointers
    const int64_t* d_ptrs_sfasfb,   // [E,2] device array of device pointers
    int E, int sf_k,
    cudaStream_t stream)
{
    // Copy metadata to host once (small arrays) to avoid host-side device deref
    std::vector<int32_t> sizes_h(E * 4);
    std::vector<int32_t> strides_h(E * 6);
    std::vector<int64_t> ptrs_abc_h(E * 3);
    std::vector<int64_t> ptrs_sfasfb_h(E * 2);
    cudaError_t err;
    err = cudaMemcpyAsync(sizes_h.data(), d_sizes_mnkl, sizes_h.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(strides_h.data(), d_strides_abc, strides_h.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(ptrs_abc_h.data(), d_ptrs_abc, ptrs_abc_h.size()*sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(ptrs_sfasfb_h.data(), d_ptrs_sfasfb, ptrs_sfasfb_h.size()*sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    for (int e = 0; e < E; ++e) {
        const int32_t M = sizes_h[e*4 + 0];
        const int32_t N = sizes_h[e*4 + 1];
        const int32_t K = sizes_h[e*4 + 2];
        if (M <= 0 || N <= 0 || K <= 0) continue;
        const int32_t A_s0 = strides_h[e*6 + 0];
        [[maybe_unused]] const int32_t A_s1 = strides_h[e*6 + 1];
        const int32_t B_s0 = strides_h[e*6 + 2];
        [[maybe_unused]] const int32_t B_s1 = strides_h[e*6 + 3];
        const int32_t C_s0 = strides_h[e*6 + 4];
        [[maybe_unused]] const int32_t C_s1 = strides_h[e*6 + 5];

        const uint16_t* A_q = reinterpret_cast<const uint16_t*>(ptrs_abc_h[e*3 + 0]);
        const __nv_bfloat16* B = reinterpret_cast<const __nv_bfloat16*>(ptrs_abc_h[e*3 + 1]);
        __nv_bfloat16* C = reinterpret_cast<__nv_bfloat16*>(ptrs_abc_h[e*3 + 2]);
        const uint8_t* SFA = reinterpret_cast<const uint8_t*>(ptrs_sfasfb_h[e*2 + 0]);

        err = fused_dense_nvfp4_gemm_bf16(
            A_q, A_s0,
            SFA, sf_k,
            B, B_s0,
            nullptr,
            C, C_s0,
            M, N, K,
            stream);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

// ============================================================================
// Fused Dense GEMM (FP8 on-the-fly): BF16 A -> FP8 quant (per-32) -> dequant -> BF16 GEMM
// ============================================================================
namespace nmoe { namespace quant {
__global__ void k_fused_dense_quant_fp8_gemm_bf16(
    const __nv_bfloat16* __restrict__ A, int lda,
    const __nv_bfloat16* __restrict__ W, int ldw,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ Y, int ldy,
    int M, int N, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int BM = 16, BN = 16;
    const int m = blockIdx.y * BM + threadIdx.y;
    const int n = blockIdx.x * BN + threadIdx.x;
    if (m >= M || n >= N) return;

    float acc = 0.0f;
    const __nv_bfloat16* a_row = A + static_cast<size_t>(m) * lda;
    const __nv_bfloat16* w_row = W + static_cast<size_t>(n) * ldw; // W[n, :]

    for (int k0 = 0; k0 < K; k0 += SF_VEC_FP8) {
        const int span = min(SF_VEC_FP8, K - k0);
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < SF_VEC_FP8; ++i) {
            if (i >= span) break;
            float v = __bfloat162float(a_row[k0 + i]);
            amax = fmaxf(amax, fabsf(v));
        }
        float scale = (amax > 0.0f) ? (amax / FP8_MAX) : 1.0f;
        uint8_t sbyte = ptx::e8m0_encode_from_pos_f32(scale);
        float s = ptx::e8m0_decode_to_f32(sbyte);
        float invs = 1.0f / s;

        #pragma unroll
        for (int i = 0; i < SF_VEC_FP8; ++i) {
            if (i >= span) break;
            float af = __bfloat162float(a_row[k0 + i]) * invs;
            uint8_t q = ptx::f32_to_e4m3_byte(af);
            float a_deq = ptx::e4m3_byte_to_f32(q) * s;
            float wf = __bfloat162float(w_row[k0 + i]);
            acc += a_deq * wf;
        }
    }

    if (bias) acc += __bfloat162float(bias[n]);
    Y[static_cast<size_t>(m) * ldy + n] = __float2bfloat16(acc);
#endif
}

inline cudaError_t launch_fused_dense_quant_fp8_gemm_bf16(
    const __nv_bfloat16* A, int lda,
    const __nv_bfloat16* W, int ldw,
    const __nv_bfloat16* bias,
    __nv_bfloat16* Y, int ldy,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 block(16,16);
    dim3 grid(ceil_div(N, 16), ceil_div(M, 16));
    k_fused_dense_quant_fp8_gemm_bf16<<<grid, block, 0, stream>>>(A, lda, W, ldw, bias, Y, ldy, M, N, K);
    return cudaGetLastError();
}
} } // namespace nmoe::quant

extern "C" cudaError_t dequant_fp8_to_bf16(
    const void* q, int ldq,
    const void* sfa, int ld_sf,
    int M, int K,
    void* out, int ldo,
    cudaStream_t stream)
{
    return nmoe::quant::launch_dequantize_fp8_to_bf16(
        reinterpret_cast<const uint16_t*>(q), ldq,
        reinterpret_cast<const uint8_t*>(sfa), ld_sf,
        reinterpret_cast<__nv_bfloat16*>(out), ldo,
        M, K, stream);
}

extern "C" cudaError_t fused_dense_quant_fp8_gemm_bf16(
    const void* A, int lda,
    const void* W, int ldw,
    const void* bias,
    void* Y, int ldy,
    int M, int N, int K,
    cudaStream_t stream)
{
    return nmoe::quant::launch_fused_dense_quant_fp8_gemm_bf16(
        reinterpret_cast<const __nv_bfloat16*>(A), lda,
        reinterpret_cast<const __nv_bfloat16*>(W), ldw,
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(Y), ldy,
        M, N, K, stream);
}
