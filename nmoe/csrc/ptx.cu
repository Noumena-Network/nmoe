// nmoe/ptx.cu
// PTX primitives for FP8/NVFP4 quantization on NVIDIA Blackwell (sm_100a).
//
// Device functions only - no kernels here.
//
// Contents:
//   - E8M0 encode/decode (power-of-two scale byte)
//   - FP8 E4M3 conversion (uses CUDA native type)
//   - NVFP4 E2M1 conversion (PTX, guarded by NMOE_ENABLE_PTX_E2M1)
//   - Pack helpers
//
// Target: sm_100a (Blackwell B200). Will fail loudly on older architectures.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#define NMOE_HAS_CUDA_FP8 1
#else
#define NMOE_HAS_CUDA_FP8 0
#endif

// Architecture guard - device code requires sm_100+
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
#error "nmoe/ptx.cu requires sm_100a or newer (Blackwell)."
#endif

namespace nmoe {
namespace ptx {

// ============================================================================
// E8M0 Scale Factor Encoding
// ============================================================================
// E8M0: pure exponent format, value = 2^(byte - 127)
// byte 127 = 2^0 = 1.0 (unit scale)
// byte 0 = 2^-127 ≈ 0 (minimum)
// byte 254 = 2^127 (maximum, byte 255 reserved for NaN/Inf)

__device__ __forceinline__ uint8_t e8m0_encode_from_pos_f32(float s) {
    if (!(s > 0.0f)) return 0;
    // Fast path: avoid log2f/exp2f. For normalized FP32:
    //   s = 2^(E-127) * (1.mantissa)
    //   ceil(log2(s)) = (E-127) + (mantissa != 0)
    // so encoded byte = E + (mantissa != 0).
    const uint32_t bits = __float_as_uint(s);
    uint32_t e = (bits >> 23) & 0xFFu;
    const uint32_t mant = bits & 0x7FFFFFu;
    e += (mant != 0);
    e = (e > 254u) ? 254u : e;
    return static_cast<uint8_t>(e);
}

__device__ __forceinline__ float e8m0_decode_to_f32(uint8_t byte) {
    // byte encodes pure exponent: scale = 2^(byte-127).
    // For byte==0, scale = 2^-127 which is a FP32 subnormal (mantissa 1<<22).
    const uint32_t bits = (byte == 0) ? 0x00400000u : (static_cast<uint32_t>(byte) << 23);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float e8m0_inv_decode_to_f32(uint8_t byte) {
    // inv_scale = 2^(127-byte). This avoids a float division in hot quantization paths.
    const uint8_t inv_byte = static_cast<uint8_t>(254u - static_cast<uint32_t>(byte));
    const uint32_t bits = (inv_byte == 0) ? 0x00400000u : (static_cast<uint32_t>(inv_byte) << 23);
    return __uint_as_float(bits);
}

// ============================================================================
// FP8 E4M3 Conversion
// ============================================================================
// Uses CUDA native __nv_fp8_e4m3 type for correct saturation and rounding.

#if NMOE_HAS_CUDA_FP8

__device__ __forceinline__ uint8_t f32_to_e4m3_byte(float x) {
    __nv_fp8_e4m3 v = __nv_fp8_e4m3(x);
    return *reinterpret_cast<uint8_t*>(&v);
}

__device__ __forceinline__ float e4m3_byte_to_f32(uint8_t b) {
    __nv_fp8_e4m3 v = *reinterpret_cast<__nv_fp8_e4m3*>(&b);
    return static_cast<float>(v);
}

#else

__device__ __forceinline__ uint8_t f32_to_e4m3_byte(float) {
    // No CUDA FP8 support - fail at runtime
    __trap();
    return 0;
}

__device__ __forceinline__ float e4m3_byte_to_f32(uint8_t) {
    __trap();
    return 0.0f;
}

#endif // NMOE_HAS_CUDA_FP8

// ============================================================================
// NVFP4 E2M1 Conversion (PTX)
// ============================================================================
// Requires NMOE_ENABLE_PTX_E2M1=1 and sm_100a.
// Uses cvt.rn.satfinite.e2m1x2.f32 PTX instruction.

#ifndef NMOE_ENABLE_PTX_E2M1
#define NMOE_ENABLE_PTX_E2M1 0
#endif

#if NMOE_ENABLE_PTX_E2M1

// Convert 4 floats to 4 E2M1 nibbles packed in low 16 bits of uint16.
// Uses round-to-nearest (rn) mode.
// Pattern from TransformerEngine: use .reg.b8 for byte outputs, then mov.b32 to pack.
// NOTE: TransformerEngine swaps element pairs (.y, .x order) to match MMA tensor core layout.
// Elements [e0, e1, e2, e3] are packed as: byte0=(e1_low, e0_high), byte1=(e3_low, e2_high)
__device__ __forceinline__ uint16_t f32x4_to_e2m1x4_packed(float x0, float x1, float x2, float x3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t out_4x;
    asm volatile(
        "{\n"
        ".reg.b8 f0; \n\t"
        ".reg.b8 f1; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, %1, %2;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, %3, %4;\n\t"
        "mov.b32 %0, {f0, f1, f0, f1};\n\t"
        "}"
        : "=r"(out_4x)
        : "f"(x1), "f"(x0), "f"(x3), "f"(x2));  // Swapped pairs to match TransformerEngine
    return static_cast<uint16_t>(out_4x & 0xFFFF);
#else
    return 0;
#endif
}

// With stochastic rounding (requires random bits)
// Convert 2 E2M1 nibbles (8 bits) to half2
// Uses cvt.rn.f16x2.e2m1x2 PTX instruction (sm_100+)
__device__ __forceinline__ void e2m1x2_to_f16x2(uint8_t packed, __half& h0, __half& h1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    unsigned int half2_storage;
    unsigned short tmp = static_cast<unsigned short>(packed);
    asm("{ .reg .b8 __$temp1, __$tempz;                 \n"
        " mov.b16 {__$temp1, __$tempz}, %1;             \n"
        " cvt.rn.f16x2.e2m1x2 %0, __$temp1;            }\n"
        : "=r"(half2_storage)
        : "h"(tmp));
    __half2 h2 = *reinterpret_cast<__half2*>(&half2_storage);
    h0 = __low2half(h2);
    h1 = __high2half(h2);
#else
    h0 = __float2half(0.0f);
    h1 = __float2half(0.0f);
#endif
}

// Convert 4 E2M1 nibbles (16 bits) to 4 floats
// Unpacks as: nibble0->x0, nibble1->x1, nibble2->x2, nibble3->x3
__device__ __forceinline__ void e2m1x4_packed_to_f32x4(uint16_t packed, float& x0, float& x1, float& x2, float& x3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // Low byte has nibbles 0,1; high byte has nibbles 2,3
    uint8_t lo = static_cast<uint8_t>(packed & 0xFF);
    uint8_t hi = static_cast<uint8_t>(packed >> 8);
    __half h0, h1, h2, h3;
    e2m1x2_to_f16x2(lo, h0, h1);
    e2m1x2_to_f16x2(hi, h2, h3);
    x0 = __half2float(h0);
    x1 = __half2float(h1);
    x2 = __half2float(h2);
    x3 = __half2float(h3);
#else
    x0 = x1 = x2 = x3 = 0.0f;
#endif
}

#else // !NMOE_ENABLE_PTX_E2M1

// Stubs that trap at runtime if called without enabling the feature
__device__ __forceinline__ uint16_t f32x4_to_e2m1x4_packed(float, float, float, float) {
#if defined(__CUDA_ARCH__)
    __trap(); // NVFP4 not enabled
#endif
    return 0;
}

__device__ __forceinline__ void e2m1x2_to_f16x2(uint8_t, __half& h0, __half& h1) {
#if defined(__CUDA_ARCH__)
    __trap();
#endif
    h0 = __float2half(0.0f);
    h1 = __float2half(0.0f);
}

__device__ __forceinline__ void e2m1x4_packed_to_f32x4(uint16_t, float& x0, float& x1, float& x2, float& x3) {
#if defined(__CUDA_ARCH__)
    __trap();
#endif
    x0 = x1 = x2 = x3 = 0.0f;
}

#endif // NMOE_ENABLE_PTX_E2M1

// ============================================================================
// Pack Helpers
// ============================================================================

__device__ __forceinline__ uint16_t pack2_u8_to_u16(uint8_t lo, uint8_t hi) {
    return static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
}

__device__ __forceinline__ void unpack_u16_to_2u8(uint16_t packed, uint8_t& lo, uint8_t& hi) {
    lo = static_cast<uint8_t>(packed & 0xFF);
    hi = static_cast<uint8_t>(packed >> 8);
}

// ============================================================================
// IPC / P2P Memory Ordering Primitives (from DeepEP utils.cuh)
// ============================================================================
// For CUDA IPC intranode NVLink communication.
// These ensure proper visibility of P2P writes across GPUs.

// --- Memory Fences (different scopes) ---
__device__ __forceinline__ void fence_acq_rel_sys() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ void fence_acq_rel_gpu() {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

__device__ __forceinline__ void fence_acq_rel_cta() {
    asm volatile("fence.acq_rel.cta;" ::: "memory");
}

// --- System-scope stores (for P2P writes visible to other GPUs) ---
__device__ __forceinline__ void st_release_sys_s32(int* ptr, int val) {
    asm volatile("st.release.sys.global.s32 [%0], %1;" :: "l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void st_relaxed_sys_s32(int* ptr, int val) {
    asm volatile("st.relaxed.sys.global.s32 [%0], %1;" :: "l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void st_relaxed_sys_b16(uint16_t* ptr, uint16_t val) {
    asm volatile("st.relaxed.sys.global.b16 [%0], %1;" :: "l"(ptr), "h"(val) : "memory");
}

__device__ __forceinline__ void st_relaxed_sys_v4_s32(int4* ptr, int4 val) {
    asm volatile("st.relaxed.sys.global.v4.s32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");
}

// --- System-scope loads (for reading P2P data from other GPUs) ---
__device__ __forceinline__ int ld_acquire_sys_s32(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint64_t ld_acquire_sys_u64(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

// --- Volatile loads (for polling) ---
__device__ __forceinline__ int ld_volatile_s32(const int* ptr) {
    int ret;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int64_t ld_volatile_s64(const int64_t* ptr) {
    int64_t ret;
    asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint64_t ld_volatile_u64(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ float ld_volatile_f32(const float* ptr) {
    float ret;
    asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

// --- Non-allocating stores (avoid polluting L1 on remote writes) ---
__device__ __forceinline__ void st_na_relaxed_gpu_b8(uint8_t* ptr, uint8_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" :: "l"(ptr), "h"((unsigned short)val));
}

__device__ __forceinline__ void st_na_relaxed_gpu_b16(uint16_t* ptr, uint16_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" :: "l"(ptr), "h"(val));
}

__device__ __forceinline__ void st_na_relaxed_gpu_b32(int* ptr, int val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" :: "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_relaxed_gpu_b64(int64_t* ptr, int64_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b64 [%0], %1;" :: "l"(ptr), "l"(val));
}

__device__ __forceinline__ void st_na_relaxed_gpu_v4(int4* ptr, int4 val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

__device__ __forceinline__ void st_na_release_gpu_b32(int* ptr, int val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" :: "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_release_gpu_b64(int64_t* ptr, int64_t val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" :: "l"(ptr), "l"(val));
}

// --- Non-allocating loads (avoid polluting L1 on remote reads) ---
__device__ __forceinline__ uint8_t ld_na_relaxed_gpu_b8(const uint8_t* ptr) {
    unsigned short ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);
}

__device__ __forceinline__ uint16_t ld_na_relaxed_gpu_b16(const uint16_t* ptr) {
    unsigned short ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int ld_na_relaxed_gpu_b32(const int* ptr) {
    int ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int64_t ld_na_relaxed_gpu_b64(const int64_t* ptr) {
    int64_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

// --- Non-coherent loads with L2 hint (streaming from remote) ---
// DeepEP pattern: ld.global.nc.L1::no_allocate.L2::256B for streaming P2P data
// This gives 256B L2 cache line allocation for better streaming performance
#ifndef NMOE_DISABLE_AGGRESSIVE_PTX
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

__device__ __forceinline__ uint8_t ld_nc_u8(const uint8_t* ptr) {
    unsigned short ret;
    asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);
}

__device__ __forceinline__ int ld_nc_s32(const int* ptr) {
    int ret;
    asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int64_t ld_nc_s64(const int64_t* ptr) {
    int64_t ret;
    asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ float ld_nc_f32(const float* ptr) {
    float ret;
    asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int2 ld_nc_v2_s32(const int2* ptr) {
    int2 ret;
    asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];"
                 : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int4 ld_nc_v4_s32(const int4* ptr) {
    int4 ret;
    asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}

// --- Non-allocating stores (standard, no cache policy) ---
__device__ __forceinline__ void st_na_s32(int* ptr, int val) {
    asm volatile("st.global.L1::no_allocate.s32 [%0], %1;" :: "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_s64(int64_t* ptr, int64_t val) {
    asm volatile("st.global.L1::no_allocate.s64 [%0], %1;" :: "l"(ptr), "l"(val));
}

__device__ __forceinline__ void st_na_f32(float* ptr, float val) {
    asm volatile("st.global.L1::no_allocate.f32 [%0], %1;" :: "l"(ptr), "f"(val));
}

__device__ __forceinline__ void st_na_v4_s32(int4* ptr, int4 val) {
    asm volatile("st.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

// --- System-scope atomics (for barrier signaling across GPUs) ---
// Note: atomicAdd_system/atomicSub_system are CUDA intrinsics, but we provide
// release-semantics versions for explicit memory ordering.

__device__ __forceinline__ int atom_add_release_sys_s32(int* ptr, int val) {
    int ret;
    asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(val));
    return ret;
}

__device__ __forceinline__ int atom_add_release_gpu_s32(int* ptr, int val) {
    int ret;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(val));
    return ret;
}

// System-scope atomics using CUDA intrinsics (for DeepEP-style barriers)
// These use the _system suffix which provides system-scope visibility
__device__ __forceinline__ int atomicAdd_sys(int* ptr, int val) {
    return atomicAdd_system(ptr, val);
}

__device__ __forceinline__ int atomicSub_sys(int* ptr, int val) {
    return atomicSub_system(ptr, val);
}

// --- Shared memory atomics (CTA scope) ---

// --- Lane ID ---
__device__ __forceinline__ int get_laneid() {
    int lane_id;
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

// --- Named barrier sync (for intra-kernel synchronization) ---
__device__ __forceinline__ void bar_sync_count(int bar_id, int thread_count) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(thread_count));
}

// --- Warp elect (leader selection) ---
__device__ __forceinline__ bool elect_sync(unsigned mask = 0xffffffff) {
    int pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred) : "r"(mask));
    return pred != 0;
}

// --- Trap (halt execution on error) ---
__device__ __forceinline__ void trap() {
    asm("trap;");
}

// ============================================================================
// Warp Reduction Templates (from DeepEP)
// ============================================================================

template <typename T> struct ReduceSum { __device__ T operator()(T a, T b) const { return a + b; } };
template <typename T> struct ReduceMax { __device__ T operator()(T a, T b) const { return a > b ? a : b; } };
template <typename T> struct ReduceMin { __device__ T operator()(T a, T b) const { return a < b ? a : b; } };
template <typename T> struct ReduceAnd { __device__ T operator()(T a, T b) const { return a & b; } };
template <typename T> struct ReduceOr  { __device__ T operator()(T a, T b) const { return a | b; } };

// Unified reduction function
template <int kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
__device__ __forceinline__ T warp_reduce(T value, Op op) {
    constexpr uint32_t mask = 0xffffffff;
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <=  1) value = op(value, __shfl_xor_sync(mask, value,  1));
        if constexpr (kNumLanesPerGroup <=  2) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup <=  4) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup <=  8) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup <= 16) value = op(value, __shfl_xor_sync(mask, value, 16));
    } else {
        if constexpr (kNumLanesPerGroup >= 32) value = op(value, __shfl_xor_sync(mask, value, 16));
        if constexpr (kNumLanesPerGroup >= 16) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup >=  8) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup >=  4) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup >=  2) value = op(value, __shfl_xor_sync(mask, value,  1));
    }
    return value;
}

// Convenience aliases
template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__device__ __forceinline__ T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__device__ __forceinline__ T warp_reduce_max(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMax<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__device__ __forceinline__ T warp_reduce_min(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMin<T>{});
}

// ============================================================================
// UNROLLED_WARP_COPY Macro (from DeepEP)
// ============================================================================
// Efficient warp-level copy with unrolling and custom load/store functions.
// Useful for P2P copy with non-allocating stores.

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC) \
{ \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR); \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC); \
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]); \
    } \
    { \
        int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]); \
            } \
        } \
    } \
}

// Wrapper functions for UNROLLED_WARP_COPY compatibility
// Load wrapper (returns value)
template <typename T>
__device__ __forceinline__ T ld_global(const T* ptr) { return __ldg(ptr); }

// Store wrapper for non-allocating stores (st_na_global pattern)
template <typename T>
__device__ __forceinline__ void st_na_global(T* ptr, const T& val);

template <>
__device__ __forceinline__ void st_na_global<int>(int* ptr, const int& val) {
    st_na_s32(ptr, val);
}

template <>
__device__ __forceinline__ void st_na_global<int4>(int4* ptr, const int4& val) {
    st_na_v4_s32(ptr, val);
}

} // namespace ptx
} // namespace nmoe
