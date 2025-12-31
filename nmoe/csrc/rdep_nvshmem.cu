// NVSHMEM implementation for RDEP hybrid mode
//
// Provides hybrid dispatch/return that uses:
//   - CUDA IPC for intra-node communication (faster, lower latency)
//   - NVSHMEM for inter-node communication (required for multi-node)
//
// Architecture:
//   rank = rdma_rank * local_world + nvl_rank
//   - rdma_rank: node index
//   - nvl_rank: GPU within node
//
// Only compiled when WITH_NVSHMEM is defined.

#ifdef WITH_NVSHMEM

#include "rdep_nvshmem.cuh"
#include "ptx.cu"

// Vendored DeepEP primitives - proper PTX semantics + IBGDA WQE support
#include "rdep/configs.cuh"
#include "rdep/utils.cuh"
#include "rdep/ibgda_device.cuh"

#include <cub/cub.cuh>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

using namespace nmoe::ptx;

// Forward declaration for swizzle_sf_strided (defined in quant.cu)
extern "C" cudaError_t swizzle_sf_strided(
    const void* sf_mkl,
    void* sf_mma,
    const int32_t* offs,
    int E, int sf_k, int sf_k_pad, int M_pad, int M_e_swizzle,
    cudaStream_t stream);

// Forward declaration for swizzle_sf_mkl_to_mma (defined in quant.cu)
extern "C" cudaError_t swizzle_sf_mkl_to_mma(
    const void* sf_mkl,
    void* sf_mma,
    int M, int sf_k,
    cudaStream_t stream);

// NVSHMEM error checking macro
#define NVSHMEM_CHECK(call)                                                   \
    do {                                                                      \
        int status = call;                                                    \
        if (status != 0) {                                                    \
            fprintf(stderr, "NVSHMEM error at %s:%d: %s returned %d\n",        \
                    __FILE__, __LINE__, #call, status);                       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Helper for vectorized non-allocating store (int2)
__device__ __forceinline__ void st_na_v2_s32(int2* ptr, int2 val) {
    asm volatile(
        "st.global.relaxed.gpu.v2.s32 [%0], {%1, %2};"
        :
        : "l"(ptr), "r"(val.x), "r"(val.y)
        : "memory"
    );
}

namespace rdep {
namespace nvshmem {

// Import vendored DeepEP primitives for use in hybrid kernels
using nmoe::rdep::memory_fence;
using nmoe::rdep::memory_fence_gpu;
using nmoe::rdep::memory_fence_cta;
using nmoe::rdep::st_release_sys_global;
using nmoe::rdep::ld_acquire_sys_global;
using nmoe::rdep::st_na_release;
using nmoe::rdep::ld_na_relaxed;
using nmoe::rdep::st_na_relaxed;
using nmoe::rdep::ld_nc_global;
using nmoe::rdep::st_na_global;
using nmoe::rdep::barrier_block;
using nmoe::rdep::ibgda_put_nbi_warp;
using nmoe::rdep::ibgda_quiet;
using nmoe::rdep::ibgda_get_state;
using nmoe::rdep::ibgda_get_rc;
using nmoe::rdep::get_lane_id;
using nmoe::rdep::warp_reduce_sum;
using nmoe::rdep::ceil_div;
using nmoe::rdep::align_up;

// ============================================================================
// Global State
// ============================================================================

NvshmemState g_nvshmem = {};

// ============================================================================
// Constants
// ============================================================================

constexpr int SF_VEC = 32;
constexpr float FP8_MAX = 448.0f;
constexpr float FP4_MAX = 6.0f;
constexpr uint64_t TIMEOUT_CYCLES = 200000000000ull;  // ~100s at 2GHz

// ============================================================================
// Metadata (same as IPC version in rdep.cu)
// ============================================================================

struct alignas(16) Meta {
    int64_t row_id;      // encodes (src_rank, tok, slot)
    int32_t local_eid;   // expert index on owner GPU
    float   gate;        // gating weight
};
static_assert(sizeof(Meta) == 16, "Meta must be 16 bytes");

// Hybrid internode routing uses per-node proxies (DeepEP pattern): inter-node sends
// target the peer with the same `nvl_rank` on the destination node, then proxy
// forwards intra-node over IPC. We pack the final destination `dest_nvl_rank` in
// the high 16 bits of Meta.local_eid when writing to the proxy.
static constexpr int META_DEST_NVL_SHIFT = 16;
__device__ __host__ __forceinline__ int meta_pack_local_eid_dest_nvl(int local_eid, int dest_nvl_rank) {
    return (dest_nvl_rank << META_DEST_NVL_SHIFT) | (local_eid & 0xFFFF);
}
__device__ __host__ __forceinline__ int meta_unpack_local_eid(int packed) {
    return packed & 0xFFFF;
}
__device__ __host__ __forceinline__ int meta_unpack_dest_nvl(int packed) {
    return (packed >> META_DEST_NVL_SHIFT) & 0xFFFF;
}

__device__ __forceinline__ void nvshmem_meta_p(
    Meta* meta, int pe, int64_t row_id, int32_t local_eid, float gate) {
    unsigned long long* dst = reinterpret_cast<unsigned long long*>(meta);
    unsigned int gate_bits = __float_as_uint(gate);
    unsigned long long w0 = static_cast<unsigned long long>(row_id);
    unsigned long long w1 =
        static_cast<unsigned long long>(static_cast<unsigned int>(local_eid)) |
        (static_cast<unsigned long long>(gate_bits) << 32);
    nvshmem_ulonglong_p(dst + 0, w0, pe);
    nvshmem_ulonglong_p(dst + 1, w1, pe);
}

// Row ID encoding: (rank * T + tok) * K + slot
__device__ __host__ __forceinline__
int64_t encode_rid(int rank, int tok, int slot, int T, int K) {
    return (static_cast<int64_t>(rank) * T + tok) * K + slot;
}

__device__ __host__ __forceinline__
void decode_rid(int64_t rid, int T, int K, int* rank, int* tok, int* slot) {
    *slot = rid % K;
    int64_t tmp = rid / K;
    *tok = tmp % T;
    *rank = tmp / T;
}

// ============================================================================
// Quantization Helpers (same as rdep.cu)
// ============================================================================

__device__ __forceinline__ uint8_t to_fp8(float v) {
    return f32_to_e4m3_byte(v);
}

__device__ __forceinline__ uint16_t to_fp4x4(float x0, float x1, float x2, float x3) {
    return f32x4_to_e2m1x4_packed(x0, x1, x2, x3);
}

__device__ __forceinline__ uint8_t e8m0_encode(float scale) {
    return e8m0_encode_from_pos_f32(scale);
}

__device__ __forceinline__ float e8m0_decode(uint8_t byte) {
    return e8m0_decode_to_f32(byte);
}

// ============================================================================
// Intra-node IPC barrier (CUDA IPC + system atomics)
//
// Inter-node synchronization uses NVSHMEM host collectives (e.g. nvshmemx_barrier_all_on_stream).
// ============================================================================

__device__ __forceinline__ void ipc_barrier_dynamic(int** barrier_ptrs, int nvl_rank, int local_world) {
    int thread_id = static_cast<int>(threadIdx.x);

    fence_acq_rel_sys();
    __syncthreads();

    if (thread_id < local_world) {
        atomicAdd_sys(barrier_ptrs[nvl_rank] + thread_id, RDMA_BARRIER_TAG);
        atomicSub_sys(barrier_ptrs[thread_id] + nvl_rank, RDMA_BARRIER_TAG);
    }

    uint64_t start_time = clock64();
    while (true) {
        int value = (thread_id < local_world) ? ld_volatile_s32(barrier_ptrs[nvl_rank] + thread_id) : 0;
        if (__all_sync(0xffffffff, value <= 0)) break;
        if (clock64() - start_time > TIMEOUT_CYCLES && thread_id < local_world) {
            printf("nmoe IPC barrier timeout\n");
            trap();
        }
    }
    __syncthreads();
}

// ============================================================================
// Initialization
// ============================================================================

void get_uid(void* uid_out) {
    nvshmemx_uniqueid_t uid;
    nvshmemx_get_uniqueid(&uid);
    memcpy(uid_out, &uid, sizeof(nvshmemx_uniqueid_t));
}

int get_uid_size() {
    return sizeof(nvshmemx_uniqueid_t);
}

void init(const void* uid, int rank, int world, int local_world) {
    if (g_nvshmem.initialized) return;

    // NOTE: Do NOT call cudaSetDevice before nvshmem init
    // DeepEP doesn't do this and it causes problems with PyTorch

    // Initialize with UID using the proper helper function
    nvshmemx_uniqueid_t nvshmem_uid;
    memcpy(&nvshmem_uid, uid, sizeof(nvshmemx_uniqueid_t));

    nvshmemx_init_attr_t attr = {};
    // Use the helper to set up UID args properly
    int status = nvshmemx_set_attr_uniqueid_args(rank, world, &nvshmem_uid, &attr);
    if (status != 0) {
        fprintf(stderr, "RDEP: nvshmemx_set_attr_uniqueid_args failed with status %d\n", status);
        return;
    }

    status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    if (status != 0) {
        fprintf(stderr, "RDEP: nvshmemx_init_attr failed with status %d\n", status);
        return;
    }

    // Call nvshmem_barrier_all like DeepEP does after init
    nvshmem_barrier_all();

    g_nvshmem.rank = rank;
    g_nvshmem.world = world;
    g_nvshmem.local_world = local_world;
    g_nvshmem.num_nodes = world / local_world;
    g_nvshmem.rdma_rank = rank / local_world;
    g_nvshmem.nvl_rank = rank % local_world;
    g_nvshmem.initialized = true;

    fprintf(stderr, "RDEP: NVSHMEM initialized (rank=%d, world=%d, local_world=%d, nodes=%d)\n",
            rank, world, local_world, g_nvshmem.num_nodes);
}

void finalize() {
    if (!g_nvshmem.initialized) return;

    // Free symmetric allocations (NVSHMEM buffers).
    //
    // DeepEP pattern: prefer a single aligned symmetric allocation and slice it.
    // If the aligned base pointers are set, free them; otherwise fall back to the
    // legacy per-buffer frees.
    if (g_nvshmem.sym_bf16_base) {
        nvshmem_free(g_nvshmem.sym_bf16_base);
    } else {
        if (g_nvshmem.x_buf_bf16) nvshmem_free(g_nvshmem.x_buf_bf16);
        if (g_nvshmem.tok_y) nvshmem_free(g_nvshmem.tok_y);
        if (g_nvshmem.tok_gate) nvshmem_free(g_nvshmem.tok_gate);
        if (g_nvshmem.tok_tag) nvshmem_free(g_nvshmem.tok_tag);
        if (g_nvshmem.meta) nvshmem_free(g_nvshmem.meta);
        if (g_nvshmem.counter) nvshmem_free(g_nvshmem.counter);
        if (g_nvshmem.dropped) nvshmem_free(g_nvshmem.dropped);
        if (g_nvshmem.barrier_signals) nvshmem_free(g_nvshmem.barrier_signals);
    }
    if (g_nvshmem.sym_block_base) {
        nvshmem_free(g_nvshmem.sym_block_base);
    } else {
        if (g_nvshmem.x_buf_block) nvshmem_free(g_nvshmem.x_buf_block);
        if (g_nvshmem.sfa_buf) nvshmem_free(g_nvshmem.sfa_buf);
        if (g_nvshmem.y_buf) nvshmem_free(g_nvshmem.y_buf);
    }

    // Close IPC handles for remote buffers (not own buffer)
    for (int r = 0; r < g_nvshmem.local_world; r++) {
        if (r != g_nvshmem.nvl_rank && g_nvshmem.ipc_buffer_ptrs[r]) {
            cudaIpcCloseMemHandle(g_nvshmem.ipc_buffer_ptrs[r]);
        }
    }

    // Free local IPC buffer (cudaMalloc'd)
    if (g_nvshmem.ipc_buffer) cudaFree(g_nvshmem.ipc_buffer);

    // Free local work buffers
    if (g_nvshmem.local_eid) cudaFree(g_nvshmem.local_eid);
    if (g_nvshmem.order) cudaFree(g_nvshmem.order);
    if (g_nvshmem.offsets) cudaFree(g_nvshmem.offsets);
    if (g_nvshmem.dest) cudaFree(g_nvshmem.dest);
    if (g_nvshmem.M_pad_dev) cudaFree(g_nvshmem.M_pad_dev);
    if (g_nvshmem.meta_copy) cudaFree(g_nvshmem.meta_copy);
    if (g_nvshmem.sfa_gather_tmp) cudaFree(g_nvshmem.sfa_gather_tmp);
    if (g_nvshmem.sort_temp) cudaFree(g_nvshmem.sort_temp);
    if (g_nvshmem.d_ipc_buffer_ptrs) cudaFree(g_nvshmem.d_ipc_buffer_ptrs);
    if (g_nvshmem.d_ipc_barrier_signal_ptrs) cudaFree(g_nvshmem.d_ipc_barrier_signal_ptrs);

    nvshmem_finalize();
    g_nvshmem = {};
}

// Helper function to compute IPC buffer layout (same as rdep.cu)
static inline size_t align_up(size_t x, size_t align) {
    return ((x + align - 1) / align) * align;
}

static void compute_ipc_buffer_layout_bf16(
    size_t capacity, int Ha, int world,
    size_t* x_off, size_t* meta_off, size_t* counter_off,
    size_t* dropped_off, size_t* barrier_off,
    size_t* tok_y_off, size_t* tok_gate_off,
    size_t* total_size)
{
    constexpr size_t BUFFER_ALIGNMENT = 128;
    *x_off = 0;
    *meta_off = capacity * Ha * sizeof(uint16_t);
    *counter_off = *meta_off + capacity * sizeof(Meta);
    *dropped_off = *counter_off + sizeof(int);
    *barrier_off = align_up(*dropped_off + sizeof(int), BUFFER_ALIGNMENT);
    const size_t ptrs_end = align_up(*barrier_off + MAX_LOCAL_GPUS * sizeof(int), BUFFER_ALIGNMENT);

    const size_t tok_slots = (world > 0) ? (capacity / static_cast<size_t>(world)) : 0;
    *tok_y_off = ptrs_end;
    *tok_gate_off = align_up(*tok_y_off + tok_slots * static_cast<size_t>(Ha) * sizeof(uint16_t), BUFFER_ALIGNMENT);
    *total_size = align_up(*tok_gate_off + tok_slots * sizeof(float), BUFFER_ALIGNMENT);
}

static void compute_ipc_buffer_layout_blockscaled(
    size_t capacity, int H, int Hp, int Hsf, int world,
    size_t* x_off, size_t* sfa_off, size_t* y_off,
    size_t* meta_off, size_t* counter_off,
    size_t* dropped_off, size_t* barrier_off,
    size_t* tok_y_off, size_t* tok_gate_off,
    size_t* total_size)
{
    constexpr size_t BUFFER_ALIGNMENT = 128;
    *x_off = 0;
    *sfa_off = capacity * static_cast<size_t>(Hp) * sizeof(uint16_t);
    *y_off = *sfa_off + capacity * static_cast<size_t>(Hsf) * sizeof(uint8_t);
    *meta_off = *y_off + capacity * static_cast<size_t>(H) * sizeof(uint16_t);
    *counter_off = *meta_off + capacity * sizeof(Meta);
    *dropped_off = *counter_off + sizeof(int);
    *barrier_off = align_up(*dropped_off + sizeof(int), BUFFER_ALIGNMENT);
    const size_t ptrs_end = align_up(*barrier_off + MAX_LOCAL_GPUS * sizeof(int), BUFFER_ALIGNMENT);

    const size_t tok_slots = (world > 0) ? (capacity / static_cast<size_t>(world)) : 0;
    const int tok_Ha = ((H + 7) / 8) * 8;
    *tok_y_off = ptrs_end;
    *tok_gate_off = align_up(*tok_y_off + tok_slots * static_cast<size_t>(tok_Ha) * sizeof(uint16_t), BUFFER_ALIGNMENT);
    *total_size = align_up(*tok_gate_off + tok_slots * sizeof(float), BUFFER_ALIGNMENT);
}

void alloc_bf16(size_t capacity, int H, int n_local) {
    int Ha = ((H + 7) / 8) * 8;
    const size_t tok_slots = (g_nvshmem.world > 0) ? (capacity / static_cast<size_t>(g_nvshmem.world)) : 0;

    // Free old NVSHMEM allocations
    if (g_nvshmem.sym_bf16_base) {
        nvshmem_free(g_nvshmem.sym_bf16_base);
    } else {
        if (g_nvshmem.x_buf_bf16) nvshmem_free(g_nvshmem.x_buf_bf16);
        if (g_nvshmem.tok_y) nvshmem_free(g_nvshmem.tok_y);
        if (g_nvshmem.tok_gate) nvshmem_free(g_nvshmem.tok_gate);
        if (g_nvshmem.tok_tag) nvshmem_free(g_nvshmem.tok_tag);
        if (g_nvshmem.meta) nvshmem_free(g_nvshmem.meta);
        if (g_nvshmem.counter) nvshmem_free(g_nvshmem.counter);
        if (g_nvshmem.dropped) nvshmem_free(g_nvshmem.dropped);
        if (g_nvshmem.barrier_signals) nvshmem_free(g_nvshmem.barrier_signals);
    }
    g_nvshmem.sym_bf16_base = nullptr;
    g_nvshmem.sym_bf16_bytes = 0;

    // Allocate symmetric heap for INTER-NODE communication.
    // DeepEP pattern: one aligned symmetric allocation + slicing into sub-buffers.
    // NOTE: NVSHMEM uses a fixed symmetric heap sized by NVSHMEM_SYMMETRIC_SIZE (default: 1GiB).
    // For moonlight-scale configs, this must be increased or nvshmem_malloc will fail and later
    // CUDA ops will surface as illegal memory access.
    const size_t x_bytes = capacity * static_cast<size_t>(Ha) * sizeof(uint16_t);
    const size_t tok_y_bytes = tok_slots * static_cast<size_t>(Ha) * sizeof(uint16_t);
    const size_t tok_gate_bytes = tok_slots * sizeof(float);
    const size_t tok_tag_bytes = tok_slots * sizeof(int);
    const size_t meta_bytes = capacity * sizeof(Meta);
    const size_t counter_bytes = sizeof(int);
    const size_t dropped_bytes = sizeof(int);
    const size_t barrier_bytes = MAX_NODES * sizeof(int);
    const size_t sym_total =
        x_bytes + tok_y_bytes + tok_gate_bytes + tok_tag_bytes +
        meta_bytes + counter_bytes + dropped_bytes + barrier_bytes;

    (void)sym_total;
    constexpr size_t kAlign = 128;
    const size_t x_off = 0;
    const size_t tok_y_off = align_up(x_off + x_bytes, kAlign);
    const size_t tok_gate_off = align_up(tok_y_off + tok_y_bytes, kAlign);
    const size_t tok_tag_off = align_up(tok_gate_off + tok_gate_bytes, kAlign);
    const size_t meta_off = align_up(tok_tag_off + tok_tag_bytes, kAlign);
    const size_t counter_off = align_up(meta_off + meta_bytes, kAlign);
    const size_t dropped_off = align_up(counter_off + counter_bytes, kAlign);
    const size_t barrier_off = align_up(dropped_off + dropped_bytes, kAlign);
    const size_t total_bytes = align_up(barrier_off + barrier_bytes, kAlign);

    g_nvshmem.sym_bf16_base = nvshmem_align(kAlign, total_bytes);
    g_nvshmem.sym_bf16_bytes = total_bytes;
    if (!g_nvshmem.sym_bf16_base) {
        fprintf(stderr,
            "RDEP ERROR: nvshmem_align failed (bf16). capacity=%zu H=%d Ha=%d tok_slots=%zu "
            "sym_bytes_total=%zu. Increase NVSHMEM_SYMMETRIC_SIZE and retry.\n",
            capacity, H, Ha, tok_slots, total_bytes);
        exit(EXIT_FAILURE);
    }
    char* base = static_cast<char*>(g_nvshmem.sym_bf16_base);
    g_nvshmem.x_buf_bf16 = reinterpret_cast<uint16_t*>(base + x_off);
    g_nvshmem.tok_y = reinterpret_cast<uint16_t*>(base + tok_y_off);
    g_nvshmem.tok_gate = reinterpret_cast<float*>(base + tok_gate_off);
    g_nvshmem.tok_tag = reinterpret_cast<int*>(base + tok_tag_off);
    g_nvshmem.meta = reinterpret_cast<void*>(base + meta_off);
    g_nvshmem.counter = reinterpret_cast<int*>(base + counter_off);
    g_nvshmem.dropped = reinterpret_cast<int*>(base + dropped_off);
    g_nvshmem.barrier_signals = reinterpret_cast<int*>(base + barrier_off);

    // Initialize counters and barriers
    CUDA_CHECK(cudaMemset(g_nvshmem.counter, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(g_nvshmem.dropped, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(g_nvshmem.barrier_signals, 0, MAX_NODES * sizeof(int)));
    CUDA_CHECK(cudaMemset(g_nvshmem.tok_tag, 0, tok_slots * sizeof(int)));

    // CRITICAL: Barrier + sync after NVSHMEM allocations, before any cudaMalloc
    // This matches DeepEP's pattern and prevents CUDA context corruption
    nvshmem_barrier_all();
    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================================================================
    // Allocate SEPARATE IPC buffer for INTRA-NODE communication (via cudaMalloc)
    // This buffer CAN be used with cudaIpcGetMemHandle/cudaIpcOpenMemHandle
    // because it's allocated with cudaMalloc, NOT nvshmem_malloc
    // =========================================================================
    if (g_nvshmem.ipc_buffer) cudaFree(g_nvshmem.ipc_buffer);

    // Compute IPC buffer layout
    compute_ipc_buffer_layout_bf16(
        capacity, Ha, g_nvshmem.world,
        &g_nvshmem.ipc_x_off,
        &g_nvshmem.ipc_meta_off,
        &g_nvshmem.ipc_counter_off,
        &g_nvshmem.ipc_dropped_off,
        &g_nvshmem.ipc_barrier_off,
        &g_nvshmem.ipc_tok_y_off,
        &g_nvshmem.ipc_tok_gate_off,
        &g_nvshmem.ipc_buffer_size);
    g_nvshmem.ipc_sfa_off = 0;
    g_nvshmem.ipc_y_off = 0;

    // Allocate IPC buffer with cudaMalloc
    CUDA_CHECK(cudaMalloc(&g_nvshmem.ipc_buffer, g_nvshmem.ipc_buffer_size));
    CUDA_CHECK(cudaMemset(g_nvshmem.ipc_buffer, 0, g_nvshmem.ipc_buffer_size));

    // Reset local pointer arrays; open_ipc_handles_* will populate peers.
    for (int r = 0; r < g_nvshmem.local_world; r++) {
        g_nvshmem.ipc_buffer_ptrs[r] = nullptr;
        g_nvshmem.ipc_barrier_signal_ptrs[r] = nullptr;
    }

    // Set local IPC buffer pointer
    g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank] = g_nvshmem.ipc_buffer;
    char* local_ipc = static_cast<char*>(g_nvshmem.ipc_buffer);
    g_nvshmem.ipc_barrier_signal_ptrs[g_nvshmem.nvl_rank] =
        reinterpret_cast<int*>(local_ipc + g_nvshmem.ipc_barrier_off);

    fprintf(stderr, "RDEP: Allocated IPC buffer (size=%zu bytes) for intra-node communication\n",
            g_nvshmem.ipc_buffer_size);

    // Allocate local work buffers
    if (g_nvshmem.local_eid) cudaFree(g_nvshmem.local_eid);
    if (g_nvshmem.order) cudaFree(g_nvshmem.order);
    if (g_nvshmem.offsets) cudaFree(g_nvshmem.offsets);
    if (g_nvshmem.dest) cudaFree(g_nvshmem.dest);
    if (g_nvshmem.M_pad_dev) cudaFree(g_nvshmem.M_pad_dev);
    if (g_nvshmem.sort_temp) cudaFree(g_nvshmem.sort_temp);

    CUDA_CHECK(cudaMalloc(&g_nvshmem.local_eid, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.order, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.offsets, (n_local + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.dest, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.M_pad_dev, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.meta_copy, capacity * sizeof(Meta)));

    // CUB sort temp storage
    g_nvshmem.sort_temp = nullptr;
    g_nvshmem.sort_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, g_nvshmem.sort_temp_bytes,
        g_nvshmem.local_eid, g_nvshmem.local_eid, g_nvshmem.order, g_nvshmem.order,
        static_cast<int>(capacity), 0, 32));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.sort_temp, g_nvshmem.sort_temp_bytes));

    g_nvshmem.capacity = capacity;
    g_nvshmem.H = H;
    g_nvshmem.Ha = Ha;
    g_nvshmem.tok_Ha = Ha;
    g_nvshmem.n_local = n_local;
    g_nvshmem.align = 128;  // Match blockscaled for consistent padding
    g_nvshmem.profile = -1;  // BF16 mode
}

// ============================================================================
// IPC Buffer Management Functions
// These use the SEPARATE cudaMalloc'd IPC buffer, NOT the NVSHMEM buffers
// ============================================================================

void get_ipc_handle_bf16(void* handle_out) {
    if (!g_nvshmem.initialized || !g_nvshmem.ipc_buffer) {
        fprintf(stderr, "RDEP: get_ipc_handle_bf16 called but IPC buffer not allocated!\n");
        return;
    }
    cudaIpcMemHandle_t handle;
    cudaError_t err = cudaIpcGetMemHandle(&handle, g_nvshmem.ipc_buffer);
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP: cudaIpcGetMemHandle failed: %s\n", cudaGetErrorString(err));
        return;
    }
    memcpy(handle_out, &handle, sizeof(cudaIpcMemHandle_t));
}

void get_ipc_handle_blockscaled(void* handle_out) {
    if (!g_nvshmem.initialized || !g_nvshmem.ipc_buffer) {
        fprintf(stderr, "RDEP: get_ipc_handle_blockscaled called but IPC buffer not allocated!\n");
        return;
    }
    cudaIpcMemHandle_t handle;
    cudaError_t err = cudaIpcGetMemHandle(&handle, g_nvshmem.ipc_buffer);
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP: cudaIpcGetMemHandle failed: %s\n", cudaGetErrorString(err));
        return;
    }
    memcpy(handle_out, &handle, sizeof(cudaIpcMemHandle_t));
}

void open_ipc_handles_bf16(const void* handles, int local_world) {
    const cudaIpcMemHandle_t* all_handles = static_cast<const cudaIpcMemHandle_t*>(handles);
    int my_nvl_rank = g_nvshmem.nvl_rank;

    for (int r = 0; r < local_world; r++) {
        if (r == my_nvl_rank) {
            // Local buffer already set in alloc_bf16
            continue;
        }
        // Open remote IPC buffer
        CUDA_CHECK(cudaIpcOpenMemHandle(
            &g_nvshmem.ipc_buffer_ptrs[r],
            all_handles[r],
            cudaIpcMemLazyEnablePeerAccess));
        // Set barrier signal pointer for remote buffer
        char* remote_buf = static_cast<char*>(g_nvshmem.ipc_buffer_ptrs[r]);
        g_nvshmem.ipc_barrier_signal_ptrs[r] =
            reinterpret_cast<int*>(remote_buf + g_nvshmem.ipc_barrier_off);
    }
    fprintf(stderr, "RDEP: Opened %d IPC handles for intra-node communication\n", local_world);
}

void open_ipc_handles_blockscaled(const void* handles, int local_world) {
    open_ipc_handles_bf16(handles, local_world);
}

void sync_ipc_buffer_ptrs_bf16() {
    // Copy IPC buffer pointers to device memory so kernels can access them
    // The host arrays ipc_buffer_ptrs and ipc_barrier_signal_ptrs cannot be
    // dereferenced from GPU code, so we need device copies

    int local_world = g_nvshmem.local_world;

    // Allocate device arrays if not already done
    if (g_nvshmem.d_ipc_buffer_ptrs == nullptr) {
        CUDA_CHECK(cudaMalloc(&g_nvshmem.d_ipc_buffer_ptrs, MAX_LOCAL_GPUS * sizeof(void*)));
    }
    if (g_nvshmem.d_ipc_barrier_signal_ptrs == nullptr) {
        CUDA_CHECK(cudaMalloc(&g_nvshmem.d_ipc_barrier_signal_ptrs, MAX_LOCAL_GPUS * sizeof(int*)));
    }

    // Copy from host to device
    CUDA_CHECK(cudaMemcpy(g_nvshmem.d_ipc_buffer_ptrs, g_nvshmem.ipc_buffer_ptrs,
                          local_world * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_nvshmem.d_ipc_barrier_signal_ptrs, g_nvshmem.ipc_barrier_signal_ptrs,
                          local_world * sizeof(int*), cudaMemcpyHostToDevice));

    fprintf(stderr, "RDEP: Synced %d IPC buffer pointers to device\n", local_world);
}

void sync_ipc_buffer_ptrs_blockscaled() {
    sync_ipc_buffer_ptrs_bf16();
}

void alloc_blockscaled(size_t capacity, int H, int n_local, int profile) {
    int pack_factor = (profile == 0) ? 2 : 4;  // FP8: 2, NVFP4: 4
    int Hp = H / pack_factor;
    int Hsf = (H + SF_VEC - 1) / SF_VEC;
    int Ha = H;  // Return buffers are BF16, unpadded
    int tok_Ha = ((H + 7) / 8) * 8;
    const size_t tok_slots = (g_nvshmem.world > 0) ? (capacity / static_cast<size_t>(g_nvshmem.world)) : 0;
    constexpr int align = 128;
    const size_t max_pad = capacity + static_cast<size_t>(n_local) * static_cast<size_t>(align - 1);

    // Free old allocations
    if (g_nvshmem.sym_block_base) {
        nvshmem_free(g_nvshmem.sym_block_base);
    } else {
        if (g_nvshmem.x_buf_block) nvshmem_free(g_nvshmem.x_buf_block);
        if (g_nvshmem.sfa_buf) nvshmem_free(g_nvshmem.sfa_buf);
        if (g_nvshmem.y_buf) nvshmem_free(g_nvshmem.y_buf);
        if (g_nvshmem.tok_y) nvshmem_free(g_nvshmem.tok_y);
        if (g_nvshmem.tok_gate) nvshmem_free(g_nvshmem.tok_gate);
        if (g_nvshmem.tok_tag) nvshmem_free(g_nvshmem.tok_tag);
        if (g_nvshmem.meta) nvshmem_free(g_nvshmem.meta);
        if (g_nvshmem.counter) nvshmem_free(g_nvshmem.counter);
        if (g_nvshmem.dropped) nvshmem_free(g_nvshmem.dropped);
        if (g_nvshmem.barrier_signals) nvshmem_free(g_nvshmem.barrier_signals);
    }
    g_nvshmem.sym_block_base = nullptr;
    g_nvshmem.sym_block_bytes = 0;

    // Allocate symmetric heap.
    // DeepEP pattern: one aligned symmetric allocation + slicing into sub-buffers.
    const size_t x_bytes = capacity * static_cast<size_t>(Hp) * sizeof(uint16_t);
    const size_t sfa_bytes = capacity * static_cast<size_t>(Hsf) * sizeof(uint8_t);
    const size_t y_bytes = capacity * static_cast<size_t>(H) * sizeof(uint16_t);
    const size_t tok_y_bytes = tok_slots * static_cast<size_t>(tok_Ha) * sizeof(uint16_t);
    const size_t tok_gate_bytes = tok_slots * sizeof(float);
    const size_t tok_tag_bytes = tok_slots * sizeof(int);
    const size_t meta_bytes = capacity * sizeof(Meta);
    const size_t counter_bytes = sizeof(int);
    const size_t dropped_bytes = sizeof(int);
    const size_t barrier_bytes = MAX_NODES * sizeof(int);
    const size_t sym_total =
        x_bytes + sfa_bytes + y_bytes + tok_y_bytes + tok_gate_bytes + tok_tag_bytes +
        meta_bytes + counter_bytes + dropped_bytes + barrier_bytes;

    (void)sym_total;
    constexpr size_t kAlignBuf = 128;
    const size_t x_off = 0;
    const size_t sfa_off = align_up(x_off + x_bytes, kAlignBuf);
    const size_t y_off = align_up(sfa_off + sfa_bytes, kAlignBuf);
    const size_t tok_y_off = align_up(y_off + y_bytes, kAlignBuf);
    const size_t tok_gate_off = align_up(tok_y_off + tok_y_bytes, kAlignBuf);
    const size_t tok_tag_off = align_up(tok_gate_off + tok_gate_bytes, kAlignBuf);
    const size_t meta_off = align_up(tok_tag_off + tok_tag_bytes, kAlignBuf);
    const size_t counter_off = align_up(meta_off + meta_bytes, kAlignBuf);
    const size_t dropped_off = align_up(counter_off + counter_bytes, kAlignBuf);
    const size_t barrier_off = align_up(dropped_off + dropped_bytes, kAlignBuf);
    const size_t total_bytes = align_up(barrier_off + barrier_bytes, kAlignBuf);

    g_nvshmem.sym_block_base = nvshmem_align(kAlignBuf, total_bytes);
    g_nvshmem.sym_block_bytes = total_bytes;
    if (!g_nvshmem.sym_block_base) {
        fprintf(stderr,
            "RDEP ERROR: nvshmem_align failed (blockscaled). profile=%d capacity=%zu H=%d Hp=%d Hsf=%d tok_slots=%zu "
            "sym_bytes_total=%zu. Increase NVSHMEM_SYMMETRIC_SIZE and retry.\n",
            profile, capacity, H, Hp, Hsf, tok_slots, total_bytes);
        exit(EXIT_FAILURE);
    }
    char* base = static_cast<char*>(g_nvshmem.sym_block_base);
    g_nvshmem.x_buf_block = reinterpret_cast<uint16_t*>(base + x_off);
    g_nvshmem.sfa_buf = reinterpret_cast<uint8_t*>(base + sfa_off);
    g_nvshmem.y_buf = reinterpret_cast<uint16_t*>(base + y_off);
    g_nvshmem.tok_y = reinterpret_cast<uint16_t*>(base + tok_y_off);
    g_nvshmem.tok_gate = reinterpret_cast<float*>(base + tok_gate_off);
    g_nvshmem.tok_tag = reinterpret_cast<int*>(base + tok_tag_off);
    g_nvshmem.meta = reinterpret_cast<void*>(base + meta_off);
    g_nvshmem.counter = reinterpret_cast<int*>(base + counter_off);
    g_nvshmem.dropped = reinterpret_cast<int*>(base + dropped_off);
    g_nvshmem.barrier_signals = reinterpret_cast<int*>(base + barrier_off);

    // Initialize
    CUDA_CHECK(cudaMemset(g_nvshmem.counter, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(g_nvshmem.dropped, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(g_nvshmem.barrier_signals, 0, MAX_NODES * sizeof(int)));
    CUDA_CHECK(cudaMemset(g_nvshmem.tok_tag, 0, tok_slots * sizeof(int)));

    // CRITICAL: Barrier + sync after NVSHMEM allocations, before any cudaMalloc
    // This matches DeepEP's pattern and prevents CUDA context corruption
    nvshmem_barrier_all();
    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================================================================
    // Allocate SEPARATE IPC buffer for INTRA-NODE communication (via cudaMalloc)
    // =========================================================================
    if (g_nvshmem.ipc_buffer) cudaFree(g_nvshmem.ipc_buffer);

    compute_ipc_buffer_layout_blockscaled(
        capacity, H, Hp, Hsf, g_nvshmem.world,
        &g_nvshmem.ipc_x_off,
        &g_nvshmem.ipc_sfa_off,
        &g_nvshmem.ipc_y_off,
        &g_nvshmem.ipc_meta_off,
        &g_nvshmem.ipc_counter_off,
        &g_nvshmem.ipc_dropped_off,
        &g_nvshmem.ipc_barrier_off,
        &g_nvshmem.ipc_tok_y_off,
        &g_nvshmem.ipc_tok_gate_off,
        &g_nvshmem.ipc_buffer_size);

    CUDA_CHECK(cudaMalloc(&g_nvshmem.ipc_buffer, g_nvshmem.ipc_buffer_size));
    CUDA_CHECK(cudaMemset(g_nvshmem.ipc_buffer, 0, g_nvshmem.ipc_buffer_size));

    // Reset local pointer arrays; open_ipc_handles_* will populate peers.
    for (int r = 0; r < g_nvshmem.local_world; r++) {
        g_nvshmem.ipc_buffer_ptrs[r] = nullptr;
        g_nvshmem.ipc_barrier_signal_ptrs[r] = nullptr;
    }

    g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank] = g_nvshmem.ipc_buffer;
    char* local_ipc = static_cast<char*>(g_nvshmem.ipc_buffer);
    g_nvshmem.ipc_barrier_signal_ptrs[g_nvshmem.nvl_rank] =
        reinterpret_cast<int*>(local_ipc + g_nvshmem.ipc_barrier_off);

    fprintf(stderr, "RDEP: Allocated IPC buffer (size=%zu bytes) for intra-node communication\n",
            g_nvshmem.ipc_buffer_size);

    // Allocate local work buffers
    if (g_nvshmem.local_eid) cudaFree(g_nvshmem.local_eid);
    if (g_nvshmem.order) cudaFree(g_nvshmem.order);
    if (g_nvshmem.offsets) cudaFree(g_nvshmem.offsets);
    if (g_nvshmem.dest) cudaFree(g_nvshmem.dest);
    if (g_nvshmem.M_pad_dev) cudaFree(g_nvshmem.M_pad_dev);
    if (g_nvshmem.meta_copy) cudaFree(g_nvshmem.meta_copy);
    if (g_nvshmem.sfa_gather_tmp) cudaFree(g_nvshmem.sfa_gather_tmp);
    if (g_nvshmem.sort_temp) cudaFree(g_nvshmem.sort_temp);

    CUDA_CHECK(cudaMalloc(&g_nvshmem.local_eid, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.order, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.offsets, (n_local + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.dest, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.M_pad_dev, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.meta_copy, capacity * sizeof(Meta)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.sfa_gather_tmp, max_pad * static_cast<size_t>(Hsf) * sizeof(uint8_t)));

    g_nvshmem.sort_temp = nullptr;
    g_nvshmem.sort_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, g_nvshmem.sort_temp_bytes,
        g_nvshmem.local_eid, g_nvshmem.local_eid, g_nvshmem.order, g_nvshmem.order,
        static_cast<int>(capacity), 0, 32));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.sort_temp, g_nvshmem.sort_temp_bytes));

    g_nvshmem.capacity = capacity;
    g_nvshmem.H = H;
    g_nvshmem.Ha = Ha;
    g_nvshmem.Hp = Hp;
    g_nvshmem.Hsf = Hsf;
    g_nvshmem.n_local = n_local;
    g_nvshmem.tok_Ha = tok_Ha;
    g_nvshmem.align = align;  // Blockscaled alignment
    g_nvshmem.profile = profile;
}

void reset_counters(cudaStream_t stream) {
    cudaMemsetAsync(g_nvshmem.counter, 0, sizeof(int), stream);
    cudaMemsetAsync(g_nvshmem.dropped, 0, sizeof(int), stream);
}

// ============================================================================
// Synchronization (Host API)
// ============================================================================

void barrier() {
    nvshmem_barrier_all();
}

void quiet() {
    nvshmem_quiet();
}

void quiet_on_stream(cudaStream_t stream) {
    nvshmemx_quiet_on_stream(stream);
}

__global__ void k_ipc_barrier(
    int** nvl_barrier_ptrs,
    int nvl_rank,
    int local_world)
{
    ipc_barrier_dynamic(nvl_barrier_ptrs, nvl_rank, local_world);
}

static inline void hybrid_barrier_on_stream(cudaStream_t stream) {
    // Intra-node (IPC) completion + inter-node (NVSHMEM) completion.
    k_ipc_barrier<<<1, 256, 0, stream>>>(
        g_nvshmem.d_ipc_barrier_signal_ptrs,
        g_nvshmem.nvl_rank,
        g_nvshmem.local_world);
    nvshmemx_barrier_all_on_stream(stream);
}

static inline void ipc_barrier_on_stream(cudaStream_t stream) {
    k_ipc_barrier<<<1, 256, 0, stream>>>(
        g_nvshmem.d_ipc_barrier_signal_ptrs,
        g_nvshmem.nvl_rank,
        g_nvshmem.local_world);
}

// ============================================================================
// Hybrid Dispatch Kernel (BF16)
// ============================================================================

__global__ void k_dispatch_hybrid_bf16(
    const __nv_bfloat16* __restrict__ x,   // [T, H]
    const int* __restrict__ eids,           // [T, K]
    const float* __restrict__ gates,        // [T, K]
    int my_rank, int T, int H, int Ha, int K,
    int n_local, int capacity,
    int local_world, int num_nodes, int rdma_rank, int nvl_rank,
    // NVSHMEM buffers (for inter-node)
    uint16_t* nvshmem_x_buf,
    Meta* nvshmem_meta,
    int* nvshmem_counter,
    int* nvshmem_dropped,
    // IPC buffers (for intra-node)
    void** ipc_buffer_ptrs,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    int M = T * K;

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = i / K;
        int slot = i % K;

        int eid = eids[tok * K + slot];
        float gate = gates[tok * K + slot];
        int dest = eid / n_local;
        int local_eid = eid % n_local;

        int dest_rdma_rank = dest / local_world;
        int dest_nvl_rank = dest % local_world;
        bool is_remote_node = (dest_rdma_rank != rdma_rank);
        bool is_remote_gpu = (dest != my_rank);

	        int slot_r;

		        if (is_remote_node) {
		            // Inter-node: send to proxy peer (same nvl_rank) on destination node.
		            const int proxy_pe = dest_rdma_rank * local_world + nvl_rank;
		            if (lane == 0) {
		                // Atomic increment counter on destination node
		                slot_r = nvshmem_int_atomic_fetch_add(nvshmem_counter, 1, proxy_pe);
		            }
		            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

		            if (slot_r >= capacity) {
		                if (lane == 0) {
		                    nvshmem_int_atomic_add(nvshmem_dropped, 1, proxy_pe);  // Fire and forget
		                }
		                continue;
		            }

		            // Write metadata via NVSHMEM
		            if (lane == 0) {
		                int64_t row_id = encode_rid(my_rank, tok, slot, T, K);
		                const int local_eid_packed = meta_pack_local_eid_dest_nvl(local_eid, dest_nvl_rank);
		                nvshmem_meta_p(nvshmem_meta + slot_r, proxy_pe, row_id, local_eid_packed, gate);
		            }

		            // Write BF16 payload via NVSHMEM (warp-cooperative)
		            const __nv_bfloat16* row = x + (int64_t)tok * H;
		            uint16_t* dst = nvshmem_x_buf + (int64_t)slot_r * Ha;

		            for (int h = lane * 4; h < H; h += 32 * 4) {
		                if (h + 4 <= H) {
		                    nvshmem_put64_nbi(reinterpret_cast<uint64_t*>(dst + h),
		                                      reinterpret_cast<const uint64_t*>(row + h),
		                                      1, proxy_pe);
		                } else {
		                    for (int hh = h; hh < H && hh < h + 4; hh++) {
		                        nvshmem_put16_nbi(dst + hh, reinterpret_cast<const uint16_t*>(row) + hh, 1, proxy_pe);
		                    }
		                }
		            }
		        } else {
		            // Intra-node: use IPC
		            char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[dest_nvl_rank]);
	            uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
            Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
            int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

            if (lane == 0) {
                slot_r = atomicAdd(counter, 1);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) continue;

            // Write metadata
            if (lane == 0) {
                Meta m{encode_rid(my_rank, tok, slot, T, K), local_eid, gate};
                if (is_remote_gpu) {
                    int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                    int4 meta_val = *reinterpret_cast<const int4*>(&m);
                    st_na_v4_s32(meta_dst, meta_val);
                } else {
                    meta_buf[slot_r] = m;
                }
            }

            // Write BF16 payload
            const __nv_bfloat16* row = x + (int64_t)tok * H;
            uint16_t* dst = x_buf + (int64_t)slot_r * Ha;

            if (is_remote_gpu) {
                // Vectorized non-allocating stores for P2P
                for (int h = lane * 4; h < H; h += 32 * 4) {
                    if (h + 4 <= H) {
                        int2* d = reinterpret_cast<int2*>(dst + h);
                        int2 v = *reinterpret_cast<const int2*>(row + h);
                        st_na_v2_s32(d, v);
                    } else {
                        for (int hh = h; hh < H && hh < h + 4; hh++) {
                            st_na_relaxed_gpu_b16(dst + hh, reinterpret_cast<const uint16_t*>(row)[hh]);
                        }
                    }
                }
            } else {
                // Local write
                for (int h = lane; h < H; h += 32) {
                    dst[h] = reinterpret_cast<const uint16_t*>(row)[h];
                }
            }
	    }
	}
}

__global__ void k_forward_nvshmem_dispatch_to_ipc_bf16(
    const uint16_t* __restrict__ nv_x_buf,   // [nv_count, Ha]
    const Meta* __restrict__ nv_meta,        // [nv_count]
    int nv_count,
    int H, int Ha,
    int capacity,
    int nvl_rank,
    int local_world,
    void** ipc_buffer_ptrs,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < nv_count; i += num_warps) {
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(nv_meta + i));
        const Meta in_m = mv.m;

        const int dest_nvl = meta_unpack_dest_nvl(in_m.local_eid);
        const int local_eid = meta_unpack_local_eid(in_m.local_eid);
        if (dest_nvl < 0 || dest_nvl >= local_world) continue;

        char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[dest_nvl]);
        uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
        Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
        int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

        int slot_r;
        if (lane == 0) {
            slot_r = atomicAdd(counter, 1);
        }
        slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);
        if (slot_r < 0 || slot_r >= capacity) continue;

        // Metadata (strip packed dest_nvl).
        if (lane == 0) {
            Meta out_m{in_m.row_id, local_eid, in_m.gate};
            if (dest_nvl != nvl_rank) {
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&out_m);
                st_na_v4_s32(meta_dst, meta_val);
            } else {
                meta_buf[slot_r] = out_m;
            }
        }

        const bool remote_gpu = (dest_nvl != nvl_rank);
        const uint16_t* src = nv_x_buf + (int64_t)i * Ha;
        uint16_t* dst = x_buf + (int64_t)slot_r * Ha;
        for (int h = lane * 4; h < H; h += 32 * 4) {
            if (h + 4 <= H) {
                int2 v = ld_nc_v2_s32(reinterpret_cast<const int2*>(src + h));
                int2* d = reinterpret_cast<int2*>(dst + h);
                if (remote_gpu) {
                    st_na_v2_s32(d, v);
                } else {
                    *d = v;
                }
            } else {
                for (int hh = h; hh < H && hh < h + 4; hh++) {
                    uint16_t u = src[hh];
                    if (remote_gpu) {
                        st_na_relaxed_gpu_b16(dst + hh, u);
                    } else {
                        dst[hh] = u;
                    }
                }
            }
        }
    }
}

__global__ void k_merge_nvshmem_into_ipc_bf16(
    const uint16_t* __restrict__ nv_x_buf,
    const Meta* __restrict__ nv_meta,
    uint16_t* __restrict__ ipc_x_buf,
    Meta* __restrict__ ipc_meta,
    int ipc_base,
    int nv_count,
    int H, int Ha)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < nv_count; i += num_warps) {
        int out_i = ipc_base + i;
        if (lane == 0) {
            ipc_meta[out_i] = nv_meta[i];
        }
        const uint16_t* src = nv_x_buf + (int64_t)i * Ha;
        uint16_t* dst = ipc_x_buf + (int64_t)out_i * Ha;
        for (int h = lane; h < H; h += 32) {
            dst[h] = src[h];
        }
    }
}

// ============================================================================
// Hybrid Dispatch Kernel (Blockscaled: FP8/NVFP4)
// ============================================================================

__global__ void k_dispatch_hybrid_blockscaled(
    const __nv_bfloat16* __restrict__ x,   // [T, H]
    const int* __restrict__ eids,          // [T, K]
    const float* __restrict__ gates,       // [T, K]
    int my_rank, int T, int H, int Hp, int Hsf, int K,
    int n_local, int capacity, int profile,
    int local_world, int num_nodes, int rdma_rank, int nvl_rank,
    // NVSHMEM buffers (inter-node)
    uint16_t* nvshmem_x_buf,
    uint8_t* nvshmem_sfa_buf,
    Meta* nvshmem_meta,
    int* nvshmem_counter,
    int* nvshmem_dropped,
    // IPC buffers (intra-node)
    void** ipc_buffer_ptrs,
    size_t ipc_sfa_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    int M = T * K;

    float dtype_max = (profile == 0) ? FP8_MAX : FP4_MAX;

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = i / K;
        int slot = i % K;

        int eid = eids[tok * K + slot];
        float gate = gates[tok * K + slot];
        int dest = eid / n_local;
        int local_eid = eid % n_local;

        int dest_rdma_rank = dest / local_world;
        int dest_nvl_rank = dest % local_world;
        bool is_remote_node = (dest_rdma_rank != rdma_rank);
        bool is_remote_gpu = (dest != my_rank);

        int slot_r;

        const __nv_bfloat16* row = x + (int64_t)tok * H;

        if (is_remote_node) {
            if (lane == 0) {
                slot_r = nvshmem_int_atomic_fetch_add(nvshmem_counter, 1, dest);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) {
                if (lane == 0) {
                    nvshmem_int_atomic_add(nvshmem_dropped, 1, dest);
                }
                continue;
            }

            if (lane == 0) {
                int64_t row_id = encode_rid(my_rank, tok, slot, T, K);
                nvshmem_meta_p(nvshmem_meta + slot_r, dest, row_id, local_eid, gate);
            }

            uint16_t* dst_pack = nvshmem_x_buf + (int64_t)slot_r * Hp;
            uint8_t* dst_sfa = nvshmem_sfa_buf + (int64_t)slot_r * Hsf;

            for (int blk = 0; blk < Hsf; blk++) {
                int h0 = blk * SF_VEC;
                int h_end = min(h0 + SF_VEC, H);
                int blk_size = h_end - h0;

                float val = 0.0f;
                if (lane < blk_size) val = __bfloat162float(row[h0 + lane]);

                float blk_amax = warp_reduce_max(fabsf(val));
                float scale = blk_amax / dtype_max;
                if (!(scale > 0.0f)) scale = 1.0f;
                uint8_t scale_byte = e8m0_encode(scale);
                float inv_scale = e8m0_inv_decode_to_f32(scale_byte);

                if (lane == 0) {
                    nvshmem_uint8_p(dst_sfa + blk, scale_byte, dest);
                }

                float qf = val * inv_scale;
                if (profile == 0) {
                    uint8_t q8 = to_fp8(qf);
                    uint8_t q8_neighbor = __shfl_xor_sync(0xFFFFFFFF, q8, 1);
                    if ((lane & 1) == 0 && lane < blk_size) {
                        uint16_t packed = (uint16_t)q8 | ((uint16_t)q8_neighbor << 8);
                        int pack_idx = blk * (SF_VEC / 2) + (lane / 2);
                        nvshmem_uint16_p(dst_pack + pack_idx, packed, dest);
                    }
                } else {
                    float qf0 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 0);
                    float qf1 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 1);
                    float qf2 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 2);
                    float qf3 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 3);
                    if ((lane & 3) == 0 && lane < blk_size) {
                        uint16_t packed = to_fp4x4(qf0, qf1, qf2, qf3);
                        int pack_idx = blk * (SF_VEC / 4) + (lane / 4);
                        nvshmem_uint16_p(dst_pack + pack_idx, packed, dest);
                    }
                }
            }
        } else {
            // Intra-node: use IPC
            char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[dest_nvl_rank]);
            uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
            uint8_t* sfa_buf = reinterpret_cast<uint8_t*>(dest_buf + ipc_sfa_off);
            Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
            int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

            if (lane == 0) {
                slot_r = atomicAdd(counter, 1);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) continue;

            if (lane == 0) {
                Meta m{encode_rid(my_rank, tok, slot, T, K), local_eid, gate};
                if (is_remote_gpu) {
                    int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                    int4 meta_val = *reinterpret_cast<const int4*>(&m);
                    st_na_v4_s32(meta_dst, meta_val);
                } else {
                    meta_buf[slot_r] = m;
                }
            }

            uint16_t* dst_pack = x_buf + (int64_t)slot_r * Hp;
            uint8_t* dst_sfa = sfa_buf + (int64_t)slot_r * Hsf;

            for (int blk = 0; blk < Hsf; blk++) {
                int h0 = blk * SF_VEC;
                int h_end = min(h0 + SF_VEC, H);
                int blk_size = h_end - h0;

                float val = 0.0f;
                if (lane < blk_size) val = __bfloat162float(row[h0 + lane]);

                float blk_amax = warp_reduce_max(fabsf(val));
                float scale = blk_amax / dtype_max;
                if (!(scale > 0.0f)) scale = 1.0f;
                uint8_t scale_byte = e8m0_encode(scale);
                float inv_scale = e8m0_inv_decode_to_f32(scale_byte);

                if (lane == 0) {
                    if (is_remote_gpu) st_na_relaxed_gpu_b8(dst_sfa + blk, scale_byte);
                    else dst_sfa[blk] = scale_byte;
                }

                float qf = val * inv_scale;
                if (profile == 0) {
                    uint8_t q8 = to_fp8(qf);
                    uint8_t q8_neighbor = __shfl_xor_sync(0xFFFFFFFF, q8, 1);
                    if ((lane & 1) == 0 && lane < blk_size) {
                        uint16_t packed = (uint16_t)q8 | ((uint16_t)q8_neighbor << 8);
                        int pack_idx = blk * (SF_VEC / 2) + (lane / 2);
                        if (is_remote_gpu) st_na_relaxed_gpu_b16(dst_pack + pack_idx, packed);
                        else dst_pack[pack_idx] = packed;
                    }
                } else {
                    float qf0 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 0);
                    float qf1 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 1);
                    float qf2 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 2);
                    float qf3 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 3);
                    if ((lane & 3) == 0 && lane < blk_size) {
                        uint16_t packed = to_fp4x4(qf0, qf1, qf2, qf3);
                        int pack_idx = blk * (SF_VEC / 4) + (lane / 4);
                        if (is_remote_gpu) st_na_relaxed_gpu_b16(dst_pack + pack_idx, packed);
                        else dst_pack[pack_idx] = packed;
                    }
                }
            }
        }
    }
}

__global__ void k_merge_nvshmem_into_ipc_blockscaled(
    const uint16_t* __restrict__ nv_x_buf,
    const uint8_t* __restrict__ nv_sfa,
    const Meta* __restrict__ nv_meta,
    uint16_t* __restrict__ ipc_x_buf,
    uint8_t* __restrict__ ipc_sfa,
    Meta* __restrict__ ipc_meta,
    int ipc_base,
    int nv_count,
    int Hp, int Hsf)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < nv_count; i += num_warps) {
        int out_i = ipc_base + i;
        if (lane == 0) {
            ipc_meta[out_i] = nv_meta[i];
        }
        const uint16_t* src = nv_x_buf + (int64_t)i * Hp;
        uint16_t* dst = ipc_x_buf + (int64_t)out_i * Hp;
        for (int hp = lane; hp < Hp; hp += 32) {
            dst[hp] = src[hp];
        }

        const uint8_t* sfa_src = nv_sfa + (int64_t)i * Hsf;
        uint8_t* sfa_dst = ipc_sfa + (int64_t)out_i * Hsf;
        for (int sf = lane; sf < Hsf; sf += 32) {
            sfa_dst[sf] = sfa_src[sf];
        }
    }
}

__global__ void k_gather_blockscaled_hybrid(
    const uint16_t* __restrict__ x_recv,
    const uint8_t* __restrict__ sfa_recv,
    const int* __restrict__ order,
    const int* __restrict__ dest,
    uint16_t* __restrict__ Xe_out,
    uint8_t* __restrict__ sfa_out,
    int M, int Hp, int Hsf)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int sorted_i = warp_id; sorted_i < M; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        int out_i = (dest != nullptr) ? dest[sorted_i] : sorted_i;

        const uint16_t* src = x_recv + (int64_t)orig_i * Hp;
        uint16_t* dst = Xe_out + (int64_t)out_i * Hp;
        for (int hp = lane; hp < Hp; hp += 32) {
            dst[hp] = src[hp];
        }

        const uint8_t* sfa_src = sfa_recv + (int64_t)orig_i * Hsf;
        uint8_t* sfa_dst = sfa_out + (int64_t)out_i * Hsf;
        for (int sf = lane; sf < Hsf; sf += 32) {
            sfa_dst[sf] = sfa_src[sf];
        }
    }
}

__global__ void k_gather_meta_sorted_hybrid(
    const Meta* __restrict__ meta,
    const int* __restrict__ order,
    int64_t* __restrict__ row_id_out,
    float* __restrict__ gate_out,
    int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        int orig_i = order[i];
        Meta m = meta[orig_i];
        row_id_out[i] = m.row_id;
        gate_out[i] = m.gate;
    }
}

// ============================================================================
// Helper Kernels for Sort/Gather (same as rdep.cu)
// ============================================================================

__global__ void k_extract_local_eid_hybrid(
    const Meta* meta, int* local_eid, int* order, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        local_eid[i] = meta[i].local_eid;
        order[i] = i;
    }
}

__global__ void k_compute_offsets_hybrid(
    const int* sorted_eid, int* offsets, int M, int n_local)
{
    extern __shared__ int s_offs[];
    int tid = threadIdx.x;

    // Initialize shared memory
    for (int e = tid; e < n_local; e += blockDim.x)
        s_offs[e] = 0;
    __syncthreads();

    // Count per-expert
    for (int i = tid; i < M; i += blockDim.x) {
        int eid = sorted_eid[i];
        if (eid >= 0 && eid < n_local)
            atomicAdd(&s_offs[eid], 1);
    }
    __syncthreads();

    // Prefix sum
    if (tid == 0) {
        int sum = 0;
        for (int e = 0; e < n_local; e++) {
            int cnt = s_offs[e];
            offsets[e] = sum;
            sum += cnt;
        }
        offsets[n_local] = sum;
    }
}

__global__ void k_compute_padded_mapping_hybrid(
    const int* offsets, int* offs_pad, int* dest, int* M_pad_out,
    int M, int n_local, int align)
{
    extern __shared__ int s_data[];
    int* s_pad_start = s_data;
    int* s_pad_end = s_data + n_local;
    int* s_orig_start = s_data + 2 * n_local;
    int tid = threadIdx.x;

    if (tid < n_local) {
        int start = offsets[tid];
        int end = offsets[tid + 1];
        int cnt = end - start;
        s_orig_start[tid] = start;
        s_pad_start[tid] = 0;
        s_pad_end[tid] = ((cnt + align - 1) / align) * align;
    }
    __syncthreads();

    // Prefix sum for padded offsets
    if (tid == 0) {
        int sum = 0;
        for (int e = 0; e < n_local; e++) {
            int pad_cnt = s_pad_end[e];
            s_pad_start[e] = sum;
            sum += pad_cnt;
            offs_pad[e] = sum;
        }
        *M_pad_out = sum;
    }
    __syncthreads();

    // Fill dest mapping
    for (int i = tid; i < M; i += blockDim.x) {
        // Find which expert this row belongs to
        for (int e = 0; e < n_local; e++) {
            int orig_start = s_orig_start[e];
            int orig_end = offsets[e + 1];
            if (i >= orig_start && i < orig_end) {
                int local_idx = i - orig_start;
                dest[i] = s_pad_start[e] + local_idx;
                break;
            }
        }
    }
}

__global__ void k_gather_bf16_hybrid(
    const uint16_t* src_buf, const int* order, const int* dest,
    __nv_bfloat16* out, int M, int H, int Ha)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_idx = order[i];
        int dst_idx = dest[i];

        const uint16_t* src_row = src_buf + (int64_t)src_idx * Ha;
        __nv_bfloat16* dst_row = out + (int64_t)dst_idx * H;

        for (int h = lane; h < H; h += 32) {
            dst_row[h] = *reinterpret_cast<const __nv_bfloat16*>(src_row + h);
        }
    }
}

// ============================================================================
// Host API: Hybrid Dispatch (BF16)
// ============================================================================

int dispatch_hybrid_bf16(
    const __nv_bfloat16* x,
    const int* eids,
    const float* gates,
    int T, int K,
    int align,
    void* Xe_out,
    int* offs_pad_out,
    int* dest_out,
    int64_t* row_id_out,
    float* gate_out,
    int* M_pad_out,
    void** ipc_buffer_ptrs,
    size_t ipc_meta_off,
    size_t ipc_counter_off,
    int** ipc_barrier_ptrs,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) return -1;
    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid bf16 requires synced IPC pointers; call nvshmem_sync_ipc_buffer_ptrs_bf16()\n");
        return -2;
    }

    int M = T * K;
    int capacity = static_cast<int>(g_nvshmem.capacity);

    // Reset NVSHMEM counters
    reset_counters(stream);

    // Reset IPC counter (local buffer)
    char* local_ipc_buf = static_cast<char*>(ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    int* local_counter = reinterpret_cast<int*>(local_ipc_buf + ipc_counter_off);
    CUDA_CHECK(cudaMemsetAsync(local_counter, 0, sizeof(int), stream));

    // Global barrier: ensure all ranks reset counters before any remote atomicAdd.
    hybrid_barrier_on_stream(stream);

    // Launch hybrid dispatch kernel
    int threads = 256;
    int warps_needed = M;
    int blocks = std::max(1, (warps_needed * 32 + threads - 1) / threads);

    // CRITICAL: Pass device pointer array (d_ipc_buffer_ptrs) to kernel, NOT host array.
    // The kernel cannot dereference host pointers - it needs device-accessible pointers.
	    k_dispatch_hybrid_bf16<<<blocks, threads, 0, stream>>>(
	        x, eids, gates,
	        g_nvshmem.rank, T, g_nvshmem.H, g_nvshmem.Ha, K,
	        g_nvshmem.n_local, capacity,
	        g_nvshmem.local_world, g_nvshmem.num_nodes, g_nvshmem.rdma_rank, g_nvshmem.nvl_rank,
	        g_nvshmem.x_buf_bf16,
	        static_cast<Meta*>(g_nvshmem.meta),
	        g_nvshmem.counter,
	        g_nvshmem.dropped,
	        g_nvshmem.d_ipc_buffer_ptrs,
	        ipc_meta_off,
	        ipc_counter_off);
	    CUDA_CHECK(cudaGetLastError());

		    // Ensure all NVSHMEM puts are complete and globally visible before counting/sorting.
		    nvshmemx_quiet_on_stream(stream);
		    hybrid_barrier_on_stream(stream);

		    // Forward inter-node receives (NVSHMEM proxy mailbox) to their true destination GPU via IPC.
		    int nvshmem_recv = 0;
		    CUDA_CHECK(cudaMemcpy(&nvshmem_recv, g_nvshmem.counter, sizeof(int), cudaMemcpyDeviceToHost));
		    nvshmem_recv = std::min(nvshmem_recv, capacity);
		    if (nvshmem_recv > 0) {
		        int f_threads = 256;
		        int f_blocks = std::max(1, (nvshmem_recv * 32 + f_threads - 1) / f_threads);
		        k_forward_nvshmem_dispatch_to_ipc_bf16<<<f_blocks, f_threads, 0, stream>>>(
		            g_nvshmem.x_buf_bf16,
		            static_cast<const Meta*>(g_nvshmem.meta),
		            nvshmem_recv,
		            g_nvshmem.H, g_nvshmem.Ha,
		            capacity,
		            g_nvshmem.nvl_rank,
		            g_nvshmem.local_world,
		            g_nvshmem.d_ipc_buffer_ptrs,
		            ipc_meta_off,
		            ipc_counter_off);
		        CUDA_CHECK(cudaGetLastError());
		    }
		    // Node-level barrier: ensure all ranks finish forwarding before counting/sorting.
		    ipc_barrier_on_stream(stream);

		    // Count total received rows (IPC only; inter-node rows were forwarded into IPC buffers).
		    int ipc_recv = 0;
		    CUDA_CHECK(cudaMemcpy(&ipc_recv, local_counter, sizeof(int), cudaMemcpyDeviceToHost));

		    int M_recv = ipc_recv;
		    if (M_recv <= 0) {
		        CUDA_CHECK(cudaMemsetAsync(offs_pad_out, 0, g_nvshmem.n_local * sizeof(int), stream));
		        *M_pad_out = 0;
		        return 0;
		    }
		    M_recv = std::min(M_recv, capacity);

		    Meta* meta_buf = reinterpret_cast<Meta*>(local_ipc_buf + ipc_meta_off);
		    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_ipc_buf);

	    // Extract local expert IDs
	    k_extract_local_eid_hybrid<<<(M_recv + 255) / 256, 256, 0, stream>>>(
	        meta_buf, g_nvshmem.local_eid, g_nvshmem.order, M_recv);
	    CUDA_CHECK(cudaGetLastError());

    // Sort by expert ID
    cub::DeviceRadixSort::SortPairs(g_nvshmem.sort_temp, g_nvshmem.sort_temp_bytes,
        g_nvshmem.local_eid, g_nvshmem.local_eid,
        g_nvshmem.order, g_nvshmem.order,
        M_recv, 0, 32, stream);

    // Compute expert offsets
    k_compute_offsets_hybrid<<<1, 256, g_nvshmem.n_local * sizeof(int), stream>>>(
        g_nvshmem.local_eid, g_nvshmem.offsets, M_recv, g_nvshmem.n_local);
    CUDA_CHECK(cudaGetLastError());

    // Compute padded mapping
    size_t pad_smem = 3 * g_nvshmem.n_local * sizeof(int);
    k_compute_padded_mapping_hybrid<<<1, 256, pad_smem, stream>>>(
        g_nvshmem.offsets, offs_pad_out, g_nvshmem.dest, g_nvshmem.M_pad_dev,
        M_recv, g_nvshmem.n_local, align);
    CUDA_CHECK(cudaGetLastError());

    // Copy M_pad to host
    CUDA_CHECK(cudaMemcpyAsync(M_pad_out, g_nvshmem.M_pad_dev, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int M_pad = *M_pad_out;

    if (Xe_out != nullptr) {
        // Zero output buffer
        CUDA_CHECK(cudaMemsetAsync(Xe_out, 0, (size_t)M_pad * g_nvshmem.H * sizeof(__nv_bfloat16), stream));

        // Gather sorted rows into output
        int gather_threads = 256;
        int gather_blocks = std::max(1, (M_recv * 32 + gather_threads - 1) / gather_threads);
        k_gather_bf16_hybrid<<<gather_blocks, gather_threads, 0, stream>>>(
            x_buf, g_nvshmem.order, g_nvshmem.dest,
            static_cast<__nv_bfloat16*>(Xe_out),
            M_recv, g_nvshmem.H, g_nvshmem.Ha);
        CUDA_CHECK(cudaGetLastError());
    }

    // Copy dest to output if requested
    if (dest_out) {
        CUDA_CHECK(cudaMemcpyAsync(dest_out, g_nvshmem.dest, M_recv * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    }

    if (row_id_out && gate_out) {
        int meta_blocks = std::max(1, (M_recv + 255) / 256);
        k_gather_meta_sorted_hybrid<<<meta_blocks, 256, 0, stream>>>(
            meta_buf, g_nvshmem.order, row_id_out, gate_out, M_recv);
        CUDA_CHECK(cudaGetLastError());
    }

    return M_recv;
}

// ============================================================================
// Blockscaled Hybrid Dispatch (stub - similar structure)
// ============================================================================

int dispatch_hybrid_blockscaled(
    const __nv_bfloat16* x,
    const int* eids,
    const float* gates,
    int T, int K,
    void* Xe_q_out,
    void* Xe_sf_out,
    int* offs_pad_out,
    int* dest_out,
    int64_t* row_id_out,
    float* gate_out,
    int* M_pad_out,
    void** ipc_buffer_ptrs,
    size_t ipc_x_off,
    size_t ipc_sfa_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off,
    int** ipc_barrier_ptrs,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) return -1;

    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid blockscaled requires synced IPC pointers; call nvshmem_sync_ipc_buffer_ptrs_blockscaled()\n");
        return -2;
    }

    int M = T * K;
    int capacity = static_cast<int>(g_nvshmem.capacity);

    // Reset counters
    reset_counters(stream);

    // Reset IPC counter (local buffer)
    char* local_ipc_buf = static_cast<char*>(ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    int* local_counter = reinterpret_cast<int*>(local_ipc_buf + ipc_counter_off);
    cudaMemsetAsync(local_counter, 0, sizeof(int), stream);

    // Global barrier: ensure all ranks reset counters before any remote atomicAdd.
    hybrid_barrier_on_stream(stream);

    // Launch hybrid dispatch kernel
    int threads = 256;
    int warps_needed = M;
    int blocks = std::max(1, (warps_needed * 32 + threads - 1) / threads);

    k_dispatch_hybrid_blockscaled<<<blocks, threads, 0, stream>>>(
        x, eids, gates,
        g_nvshmem.rank, T, g_nvshmem.H, g_nvshmem.Hp, g_nvshmem.Hsf, K,
        g_nvshmem.n_local, capacity, g_nvshmem.profile,
        g_nvshmem.local_world, g_nvshmem.num_nodes, g_nvshmem.rdma_rank, g_nvshmem.nvl_rank,
        g_nvshmem.x_buf_block,
        g_nvshmem.sfa_buf,
        static_cast<Meta*>(g_nvshmem.meta),
        g_nvshmem.counter,
        g_nvshmem.dropped,
        g_nvshmem.d_ipc_buffer_ptrs,
        ipc_sfa_off,
        ipc_meta_off,
        ipc_counter_off);

    // Ensure all NVSHMEM puts are complete and globally visible before counting/sorting.
    nvshmemx_quiet_on_stream(stream);
    hybrid_barrier_on_stream(stream);

    cudaStreamSynchronize(stream);

    // Count total received rows (IPC + NVSHMEM)
    int ipc_recv = 0;
    cudaMemcpy(&ipc_recv, local_counter, sizeof(int), cudaMemcpyDeviceToHost);

    int nvshmem_recv = 0;
    cudaMemcpy(&nvshmem_recv, g_nvshmem.counter, sizeof(int), cudaMemcpyDeviceToHost);

    int M_recv = ipc_recv + nvshmem_recv;
    if (M_recv <= 0) {
        cudaMemsetAsync(offs_pad_out, 0, g_nvshmem.n_local * sizeof(int), stream);
        *M_pad_out = 0;
        return 0;
    }
    M_recv = std::min(M_recv, capacity);

    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_ipc_buf + ipc_x_off);
    uint8_t* sfa_buf = reinterpret_cast<uint8_t*>(local_ipc_buf + ipc_sfa_off);
    Meta* meta_buf = reinterpret_cast<Meta*>(local_ipc_buf + ipc_meta_off);

    // Merge NVSHMEM receives into local IPC buffer [ipc_recv .. ipc_recv+nvshmem_recv).
    if (nvshmem_recv > 0) {
        int merge_threads = 256;
        int merge_warps = nvshmem_recv;
        int merge_blocks = std::max(1, (merge_warps * 32 + merge_threads - 1) / merge_threads);
        k_merge_nvshmem_into_ipc_blockscaled<<<merge_blocks, merge_threads, 0, stream>>>(
            g_nvshmem.x_buf_block,
            g_nvshmem.sfa_buf,
            static_cast<const Meta*>(g_nvshmem.meta),
            x_buf,
            sfa_buf,
            meta_buf,
            ipc_recv,
            nvshmem_recv,
            g_nvshmem.Hp,
            g_nvshmem.Hsf);
    }

    // Extract local expert IDs
    k_extract_local_eid_hybrid<<<(M_recv + 255) / 256, 256, 0, stream>>>(
        meta_buf, g_nvshmem.local_eid, g_nvshmem.order, M_recv);

    // Sort by expert ID
    cub::DeviceRadixSort::SortPairs(g_nvshmem.sort_temp, g_nvshmem.sort_temp_bytes,
        g_nvshmem.local_eid, g_nvshmem.local_eid,
        g_nvshmem.order, g_nvshmem.order,
        M_recv, 0, 32, stream);

    // Compute expert offsets
    k_compute_offsets_hybrid<<<1, 256, g_nvshmem.n_local * sizeof(int), stream>>>(
        g_nvshmem.local_eid, g_nvshmem.offsets, M_recv, g_nvshmem.n_local);

    // Compute padded mapping
    size_t pad_smem = 3 * g_nvshmem.n_local * sizeof(int);
    k_compute_padded_mapping_hybrid<<<1, 256, pad_smem, stream>>>(
        g_nvshmem.offsets, offs_pad_out, g_nvshmem.dest, g_nvshmem.M_pad_dev,
        M_recv, g_nvshmem.n_local, g_nvshmem.align);

    // Copy M_pad to host
    cudaMemcpyAsync(M_pad_out, g_nvshmem.M_pad_dev, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (dest_out) {
        cudaMemcpyAsync(dest_out, g_nvshmem.dest, M_recv * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);
    int M_pad = *M_pad_out;

    // Optional output materialization: meta-only mode passes Xe_q_out/Xe_sf_out as nullptr.
    if (Xe_q_out && Xe_sf_out) {
        // Zero output buffer
        cudaMemsetAsync(Xe_q_out, 0, (size_t)M_pad * g_nvshmem.Hp * sizeof(uint16_t), stream);
        // Ensure padding rows have deterministic, safe SF (scale=1.0 -> e8m0 byte 127).
        cudaMemsetAsync(g_nvshmem.sfa_gather_tmp, 127, (size_t)M_pad * g_nvshmem.Hsf * sizeof(uint8_t), stream);

        // Gather packed activations to Xe_q_out and rowwise SFA to a temporary buffer
        int gather_threads = 256;
        int gather_blocks = std::max(1, (M_recv * 32 + gather_threads - 1) / gather_threads);
        k_gather_blockscaled_hybrid<<<gather_blocks, gather_threads, 0, stream>>>(
            x_buf, sfa_buf, g_nvshmem.order, g_nvshmem.dest,
            static_cast<uint16_t*>(Xe_q_out), g_nvshmem.sfa_gather_tmp,
            M_recv, g_nvshmem.Hp, g_nvshmem.Hsf);

        const int sf_k = g_nvshmem.Hsf;
        if ((sf_k & 3) != 0) {
            fprintf(stderr, "RDEP ERROR: H=%d requires sf_k=H/32 multiple of 4 (H%%128==0). Got sf_k=%d\n",
                    g_nvshmem.H, sf_k);
            return -3;
        }
        cudaError_t err = swizzle_sf_mkl_to_mma(
            static_cast<const void*>(g_nvshmem.sfa_gather_tmp),
            Xe_sf_out,
            M_pad,
            sf_k,
            stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "RDEP ERROR: swizzle_sf_mkl_to_mma failed: %s\n", cudaGetErrorString(err));
            return -4;
        }
    }

    if (row_id_out && gate_out) {
        int meta_blocks = std::max(1, (M_recv + 255) / 256);
        k_gather_meta_sorted_hybrid<<<meta_blocks, 256, 0, stream>>>(
            meta_buf, g_nvshmem.order, row_id_out, gate_out, M_recv);
    }

    return M_recv;
}

void gather_xe_hybrid_blockscaled(
    void* Xe_q_out,
    void* Xe_sf_out,
    int M_recv,
    int M_pad,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) return;
    if (g_nvshmem.profile < 0) {
        fprintf(stderr, "RDEP ERROR: gather_xe_hybrid_blockscaled requires blockscaled NVSHMEM state\n");
        return;
    }
    if (!Xe_q_out || !Xe_sf_out) {
        fprintf(stderr, "RDEP ERROR: gather_xe_hybrid_blockscaled requires non-null outputs\n");
        return;
    }
    if (M_recv <= 0 || M_pad <= 0) return;

    char* local_ipc_buf = static_cast<char*>(g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_ipc_buf + g_nvshmem.ipc_x_off);
    uint8_t* sfa_buf = reinterpret_cast<uint8_t*>(local_ipc_buf + g_nvshmem.ipc_sfa_off);

    cudaMemsetAsync(Xe_q_out, 0, (size_t)M_pad * g_nvshmem.Hp * sizeof(uint16_t), stream);
    cudaMemsetAsync(g_nvshmem.sfa_gather_tmp, 127, (size_t)M_pad * g_nvshmem.Hsf * sizeof(uint8_t), stream);

    int threads = 256;
    int blocks = std::max(1, (M_recv * 32 + threads - 1) / threads);
    k_gather_blockscaled_hybrid<<<blocks, threads, 0, stream>>>(
        x_buf, sfa_buf, g_nvshmem.order, g_nvshmem.dest,
        static_cast<uint16_t*>(Xe_q_out), g_nvshmem.sfa_gather_tmp,
        M_recv, g_nvshmem.Hp, g_nvshmem.Hsf);

    const int sf_k = g_nvshmem.Hsf;
    if ((sf_k & 3) != 0) {
        fprintf(stderr, "RDEP ERROR: H=%d requires sf_k=H/32 multiple of 4 (H%%128==0). Got sf_k=%d\n",
                g_nvshmem.H, sf_k);
        return;
    }
    cudaError_t err = swizzle_sf_mkl_to_mma(
        static_cast<const void*>(g_nvshmem.sfa_gather_tmp),
        Xe_sf_out,
        M_pad,
        sf_k,
        stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP ERROR: swizzle_sf_mkl_to_mma failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

// ============================================================================
// Hybrid Return Scatter Kernel (BF16)
// ============================================================================

__global__ void k_return_scatter_hybrid_bf16(
    const __nv_bfloat16* __restrict__ Ye,    // [M_recv, H] expert outputs (sorted order)
    const int* __restrict__ order,            // [M_recv] original indices
    const Meta* __restrict__ meta,            // [capacity] metadata from dispatch
    float* __restrict__ out,                  // [T, H] local output accumulator
    int M_recv, int H, int Ha, int T, int K,
    const int my_rank, const int local_world, const int num_nodes, const int rdma_rank, const int nvl_rank,
    int capacity,
    // NVSHMEM buffers (for inter-node return)
    uint16_t* nvshmem_y_buf,
    Meta* nvshmem_meta,
    int* nvshmem_counter,
    // IPC buffers (for intra-node return)
    void** ipc_buffer_ptrs,
    size_t ipc_y_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta + orig_i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid(m.row_id, T, K, &src_rank, &tok, &slot);

        int src_rdma_rank = src_rank / local_world;
        int src_nvl_rank = src_rank % local_world;
        bool is_remote_node = (src_rdma_rank != rdma_rank);
        bool is_local = (src_rank == my_rank);

        if (is_local) {
            // Local: scatter directly with gate weighting
            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            float* out_row = out + (int64_t)tok * H;
            for (int h = lane; h < H; h += 32) {
                atomicAdd(out_row + h, __bfloat162float(y_row[h]) * m.gate);
            }
        } else if (is_remote_node) {
            // Inter-node: write to proxy peer (same nvl_rank) on source node, then proxy forwards via IPC.
            const int proxy_pe = src_rdma_rank * local_world + nvl_rank;
            int slot_r;
            if (lane == 0) {
                slot_r = nvshmem_int_atomic_fetch_add(nvshmem_counter, 1, proxy_pe);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) continue;

            // Write metadata via NVSHMEM
            if (lane == 0) {
                const int local_eid_packed = meta_pack_local_eid_dest_nvl(/*local_eid=*/0, src_nvl_rank);
                nvshmem_meta_p(nvshmem_meta + slot_r, proxy_pe, m.row_id, local_eid_packed, m.gate);
            }

            // Write BF16 payload via NVSHMEM
            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            uint16_t* dst = nvshmem_y_buf + (int64_t)slot_r * Ha;

            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    nvshmem_put64_nbi(reinterpret_cast<uint64_t*>(dst + h),
                                      reinterpret_cast<const uint64_t*>(y_row + h),
                                      1, proxy_pe);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        nvshmem_put16_nbi(dst + hh, reinterpret_cast<const uint16_t*>(y_row) + hh, 1, proxy_pe);
                    }
                }
            }
        } else {
            // Intra-node (different GPU, same node): use IPC
            char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[src_nvl_rank]);
            uint16_t* y_buf = reinterpret_cast<uint16_t*>(dest_buf + ipc_y_off);
            Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
            int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

            int slot_r;
            if (lane == 0) {
                slot_r = atomicAdd(counter, 1);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) continue;

            // Write metadata
            if (lane == 0) {
                Meta mr{m.row_id, 0, m.gate};
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&mr);
                st_na_v4_s32(meta_dst, meta_val);
            }

            // Write BF16 payload via IPC
            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            uint16_t* dst = y_buf + (int64_t)slot_r * Ha;

            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    int2* d = reinterpret_cast<int2*>(dst + h);
                    int2 v = *reinterpret_cast<const int2*>(y_row + h);
                    st_na_v2_s32(d, v);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        st_na_relaxed_gpu_b16(dst + hh, reinterpret_cast<const uint16_t*>(y_row)[hh]);
                    }
                }
            }
        }
    }
}

__global__ void k_return_scatter_hybrid_bf16_from_pad(
    const __nv_bfloat16* __restrict__ Ye_pad, // [M_pad, H] expert outputs (padded)
    const int* __restrict__ dest,             // [M_recv] sorted_i -> pad_i
    const int* __restrict__ order,            // [M_recv] original indices
    const Meta* __restrict__ meta,            // [capacity] metadata from dispatch
    float* __restrict__ out,                  // [T, H] local output accumulator
    int M_recv, int H, int Ha, int T, int K,
    const int my_rank, const int local_world, const int num_nodes, const int rdma_rank, const int nvl_rank,
    int capacity,
    // NVSHMEM buffers (for inter-node return)
    uint16_t* nvshmem_y_buf,
    Meta* nvshmem_meta,
    int* nvshmem_counter,
    // IPC buffers (for intra-node return)
    void** ipc_buffer_ptrs,
    size_t ipc_y_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    (void)num_nodes;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        const int pad_i = dest[sorted_i];
        if (pad_i < 0) continue;

        int orig_i = order[sorted_i];
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta + orig_i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid(m.row_id, T, K, &src_rank, &tok, &slot);

        int src_rdma_rank = src_rank / local_world;
        int src_nvl_rank = src_rank % local_world;
        bool is_remote_node = (src_rdma_rank != rdma_rank);
        bool is_local = (src_rank == my_rank);

        const __nv_bfloat16* y_row = Ye_pad + (int64_t)pad_i * H;

        if (is_local) {
            float* out_row = out + (int64_t)tok * H;
            for (int h = lane; h < H; h += 32) {
                atomicAdd(out_row + h, __bfloat162float(y_row[h]) * m.gate);
            }
        } else if (is_remote_node) {
            const int proxy_pe = src_rdma_rank * local_world + nvl_rank;
            int slot_r;
            if (lane == 0) {
                slot_r = nvshmem_int_atomic_fetch_add(nvshmem_counter, 1, proxy_pe);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) continue;

            if (lane == 0) {
                const int local_eid_packed = meta_pack_local_eid_dest_nvl(/*local_eid=*/0, src_nvl_rank);
                nvshmem_meta_p(nvshmem_meta + slot_r, proxy_pe, m.row_id, local_eid_packed, m.gate);
            }

            uint16_t* dst = nvshmem_y_buf + (int64_t)slot_r * Ha;
            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    nvshmem_put64_nbi(reinterpret_cast<uint64_t*>(dst + h),
                                      reinterpret_cast<const uint64_t*>(y_row + h),
                                      1, proxy_pe);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        nvshmem_put16_nbi(dst + hh, reinterpret_cast<const uint16_t*>(y_row) + hh, 1, proxy_pe);
                    }
                }
            }
        } else {
            char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[src_nvl_rank]);
            uint16_t* y_buf = reinterpret_cast<uint16_t*>(dest_buf + ipc_y_off);
            Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
            int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

            int slot_r;
            if (lane == 0) {
                slot_r = atomicAdd(counter, 1);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) continue;

            if (lane == 0) {
                Meta mr{m.row_id, 0, m.gate};
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&mr);
                st_na_v4_s32(meta_dst, meta_val);
            }

            uint16_t* dst = y_buf + (int64_t)slot_r * Ha;
            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    int2 v = *reinterpret_cast<const int2*>(y_row + h);
                    int2* d = reinterpret_cast<int2*>(dst + h);
                    st_na_v2_s32(d, v);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        st_na_relaxed_gpu_b16(dst + hh, reinterpret_cast<const uint16_t*>(y_row)[hh]);
                    }
                }
            }
        }
    }
}

__global__ void k_forward_nvshmem_return_to_ipc_bf16(
    const uint16_t* __restrict__ nv_y_buf,   // [nv_count, Ha]
    const Meta* __restrict__ nv_meta,        // [nv_count]
    int nv_count,
    int H, int Ha,
    int capacity,
    int nvl_rank,
    int local_world,
    void** ipc_buffer_ptrs,
    size_t ipc_y_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < nv_count; i += num_warps) {
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(nv_meta + i));
        const Meta in_m = mv.m;

        const int dest_nvl = meta_unpack_dest_nvl(in_m.local_eid);
        if (dest_nvl < 0 || dest_nvl >= local_world) continue;

        char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[dest_nvl]);
        uint16_t* y_buf = reinterpret_cast<uint16_t*>(dest_buf + ipc_y_off);
        Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
        int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

        int slot_r;
        if (lane == 0) {
            slot_r = atomicAdd(counter, 1);
        }
        slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);
        if (slot_r < 0 || slot_r >= capacity) continue;

        if (lane == 0) {
            Meta out_m{in_m.row_id, 0, in_m.gate};
            if (dest_nvl != nvl_rank) {
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&out_m);
                st_na_v4_s32(meta_dst, meta_val);
            } else {
                meta_buf[slot_r] = out_m;
            }
        }

        const bool remote_gpu = (dest_nvl != nvl_rank);
        const uint16_t* src = nv_y_buf + (int64_t)i * Ha;
        uint16_t* dst = y_buf + (int64_t)slot_r * Ha;
        for (int h = lane * 4; h < H; h += 32 * 4) {
            if (h + 4 <= H) {
                int2 v = ld_nc_v2_s32(reinterpret_cast<const int2*>(src + h));
                int2* d = reinterpret_cast<int2*>(dst + h);
                if (remote_gpu) {
                    st_na_v2_s32(d, v);
                } else {
                    *d = v;
                }
            } else {
                for (int hh = h; hh < H && hh < h + 4; hh++) {
                    uint16_t u = src[hh];
                    if (remote_gpu) {
                        st_na_relaxed_gpu_b16(dst + hh, u);
                    } else {
                        dst[hh] = u;
                    }
                }
            }
        }
    }
}

// Kernel to scatter received rows from return buffer to output
__global__ void k_scatter_received_hybrid_bf16(
    const uint16_t* __restrict__ y_buf,
    const Meta* __restrict__ meta,
    float* __restrict__ out,
    int M_ret, int H, int Ha, int T, int K)
{
    int lane = threadIdx.x % 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < M_ret; i += num_warps) {
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta + i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid(m.row_id, T, K, &src_rank, &tok, &slot);

        const uint16_t* y_row = y_buf + (int64_t)i * Ha;
        float* out_row = out + (int64_t)tok * H;

        // NOTE: y_row/meta may be written by peer GPUs (IPC) or by remote nodes
        // (NVSHMEM). Receiver-side L2 is not coherent with those writes, so use
        // non-caching loads to observe updates.
        for (int h = lane * 8; h < Ha; h += 32 * 8) {
            int4 v = ld_nc_v4_s32(reinterpret_cast<const int4*>(y_row + h));
            union BF16x8 {
                int4 v;
                uint16_t u[8];
            };
            BF16x8 x;
            x.v = v;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int hh = h + j;
                if (hh < H) {
                    const __nv_bfloat16 bf = *reinterpret_cast<const __nv_bfloat16*>(&x.u[j]);
                    atomicAdd(out_row + hh, __bfloat162float(bf) * m.gate);
                }
            }
        }
    }
}

// ============================================================================
// Host API: Hybrid Return Scatter (BF16)
// ============================================================================

static void return_scatter_hybrid_impl(
    const __nv_bfloat16* Ye,
    float* out,
    int M_recv, int T, int K,
    void** ipc_buffer_ptrs,
    size_t ipc_y_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off,
    uint16_t* nvshmem_y_buf,
    int H, int Ha,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) return;

    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid return_scatter requires synced IPC pointers\n");
        return;
    }

    int capacity = static_cast<int>(g_nvshmem.capacity);

    // Reset counters
    reset_counters(stream);

    // Reset local IPC counter
    char* local_ipc_buf = static_cast<char*>(ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    int* local_counter = reinterpret_cast<int*>(local_ipc_buf + ipc_counter_off);
    cudaMemsetAsync(local_counter, 0, sizeof(int), stream);

    // Snapshot dispatch metadata before return writes reuse the same IPC meta buffer.
    Meta* local_meta = reinterpret_cast<Meta*>(local_ipc_buf + ipc_meta_off);
    if (M_recv > 0) {
        cudaMemcpyAsync(g_nvshmem.meta_copy, local_meta,
                        static_cast<size_t>(M_recv) * sizeof(Meta),
                        cudaMemcpyDeviceToDevice, stream);
    }

	    // Launch return scatter kernel
	    int threads = 256;
	    int warps_needed = std::max(M_recv, 1);
	    int blocks = std::max(1, (warps_needed * 32 + threads - 1) / threads);

	    // Global barrier: ensure all ranks reset counters *and* snapshot their
	    // dispatch metadata before any remote atomicAdd()/writes begin.
		    hybrid_barrier_on_stream(stream);

	    if (M_recv > 0) {
	        k_return_scatter_hybrid_bf16<<<blocks, threads, 0, stream>>>(
	            Ye, g_nvshmem.order, g_nvshmem.meta_copy, out,
	            M_recv, H, Ha, T, K,
	            g_nvshmem.rank, g_nvshmem.local_world, g_nvshmem.num_nodes,
            g_nvshmem.rdma_rank, g_nvshmem.nvl_rank,
            capacity,
            nvshmem_y_buf,
            static_cast<Meta*>(g_nvshmem.meta),
            g_nvshmem.counter,
            g_nvshmem.d_ipc_buffer_ptrs,
            ipc_y_off,
            ipc_meta_off,
            ipc_counter_off);
    }

	    // Wait for all NVSHMEM puts to complete, then forward proxy mailbox to true destination GPUs via IPC.
		    nvshmemx_quiet_on_stream(stream);
			    hybrid_barrier_on_stream(stream);

	    int nvshmem_ret = 0;
	    cudaMemcpy(&nvshmem_ret, g_nvshmem.counter, sizeof(int), cudaMemcpyDeviceToHost);
	    nvshmem_ret = std::min(nvshmem_ret, capacity);
	    if (nvshmem_ret > 0) {
	        int f_threads = 256;
	        int f_blocks = std::max(1, (nvshmem_ret * 32 + f_threads - 1) / f_threads);
	        k_forward_nvshmem_return_to_ipc_bf16<<<f_blocks, f_threads, 0, stream>>>(
	            nvshmem_y_buf,
	            static_cast<const Meta*>(g_nvshmem.meta),
	            nvshmem_ret,
	            H, Ha,
	            capacity,
	            g_nvshmem.nvl_rank,
	            g_nvshmem.local_world,
	            g_nvshmem.d_ipc_buffer_ptrs,
	            ipc_y_off,
	            ipc_meta_off,
	            ipc_counter_off);
	        CUDA_CHECK(cudaGetLastError());
	    }
	    // Node-level barrier: ensure all ranks finish forwarding before scattering to output.
	    ipc_barrier_on_stream(stream);

	    // Count received rows from IPC
	    int ipc_ret = 0;
	    cudaMemcpy(&ipc_ret, local_counter, sizeof(int), cudaMemcpyDeviceToHost);

	    // Scatter received rows from IPC buffer
	    if (ipc_ret > 0) {
	        ipc_ret = std::min(ipc_ret, capacity);
	        uint16_t* ipc_y_buf = reinterpret_cast<uint16_t*>(local_ipc_buf + ipc_y_off);
	        Meta* ipc_meta_buf = reinterpret_cast<Meta*>(local_ipc_buf + ipc_meta_off);

	        int scatter_blocks = std::max(1, (ipc_ret * 32 + 255) / 256);
	        k_scatter_received_hybrid_bf16<<<scatter_blocks, 256, 0, stream>>>(
	            ipc_y_buf, ipc_meta_buf, out,
	            ipc_ret, H, Ha, T, K);
	    }
}

static void return_scatter_hybrid_from_pad_impl(
    const __nv_bfloat16* Ye_pad,
    float* out,
    int M_recv, int T, int K,
    void** ipc_buffer_ptrs,
    size_t ipc_y_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off,
    uint16_t* nvshmem_y_buf,
    int H, int Ha,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) return;

    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid return_scatter requires synced IPC pointers\n");
        return;
    }

    int capacity = static_cast<int>(g_nvshmem.capacity);

    reset_counters(stream);

    char* local_ipc_buf = static_cast<char*>(ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    int* local_counter = reinterpret_cast<int*>(local_ipc_buf + ipc_counter_off);
    cudaMemsetAsync(local_counter, 0, sizeof(int), stream);

    Meta* local_meta = reinterpret_cast<Meta*>(local_ipc_buf + ipc_meta_off);
    if (M_recv > 0) {
        cudaMemcpyAsync(g_nvshmem.meta_copy, local_meta,
                        static_cast<size_t>(M_recv) * sizeof(Meta),
                        cudaMemcpyDeviceToDevice, stream);
    }

    int threads = 256;
    int warps_needed = std::max(M_recv, 1);
    int blocks = std::max(1, (warps_needed * 32 + threads - 1) / threads);

    hybrid_barrier_on_stream(stream);

    if (M_recv > 0) {
        k_return_scatter_hybrid_bf16_from_pad<<<blocks, threads, 0, stream>>>(
            Ye_pad,
            g_nvshmem.dest,
            g_nvshmem.order,
            g_nvshmem.meta_copy,
            out,
            M_recv, H, Ha, T, K,
            g_nvshmem.rank, g_nvshmem.local_world, g_nvshmem.num_nodes,
            g_nvshmem.rdma_rank, g_nvshmem.nvl_rank,
            capacity,
            nvshmem_y_buf,
            static_cast<Meta*>(g_nvshmem.meta),
            g_nvshmem.counter,
            g_nvshmem.d_ipc_buffer_ptrs,
            ipc_y_off,
            ipc_meta_off,
            ipc_counter_off);
    }

    nvshmemx_quiet_on_stream(stream);
    hybrid_barrier_on_stream(stream);

    int nvshmem_ret = 0;
    cudaMemcpy(&nvshmem_ret, g_nvshmem.counter, sizeof(int), cudaMemcpyDeviceToHost);
    nvshmem_ret = std::min(nvshmem_ret, capacity);
    if (nvshmem_ret > 0) {
        int f_threads = 256;
        int f_blocks = std::max(1, (nvshmem_ret * 32 + f_threads - 1) / f_threads);
        k_forward_nvshmem_return_to_ipc_bf16<<<f_blocks, f_threads, 0, stream>>>(
            nvshmem_y_buf,
            static_cast<const Meta*>(g_nvshmem.meta),
            nvshmem_ret,
            H, Ha,
            capacity,
            g_nvshmem.nvl_rank,
            g_nvshmem.local_world,
            g_nvshmem.d_ipc_buffer_ptrs,
            ipc_y_off,
            ipc_meta_off,
            ipc_counter_off);
        CUDA_CHECK(cudaGetLastError());
    }

    ipc_barrier_on_stream(stream);

    int ipc_ret = 0;
    cudaMemcpy(&ipc_ret, local_counter, sizeof(int), cudaMemcpyDeviceToHost);

    if (ipc_ret > 0) {
        ipc_ret = std::min(ipc_ret, capacity);
        uint16_t* ipc_y_buf = reinterpret_cast<uint16_t*>(local_ipc_buf + ipc_y_off);
        Meta* ipc_meta_buf = reinterpret_cast<Meta*>(local_ipc_buf + ipc_meta_off);

        int scatter_blocks = std::max(1, (ipc_ret * 32 + 255) / 256);
        k_scatter_received_hybrid_bf16<<<scatter_blocks, 256, 0, stream>>>(
            ipc_y_buf, ipc_meta_buf, out,
            ipc_ret, H, Ha, T, K);
    }
}

void return_scatter_hybrid_bf16(
    const __nv_bfloat16* Ye,
    float* out,
    int M_recv, int T, int K,
    void** ipc_buffer_ptrs,
    int** ipc_barrier_ptrs,
    cudaStream_t stream)
{
    (void)ipc_barrier_ptrs;
    return_scatter_hybrid_impl(
        Ye,
        out,
        M_recv, T, K,
        ipc_buffer_ptrs,
        /*ipc_y_off=*/0,
        g_nvshmem.ipc_meta_off,
        g_nvshmem.ipc_counter_off,
        g_nvshmem.x_buf_bf16,  // BF16 reuses x_buf for return
        g_nvshmem.H,
        g_nvshmem.Ha,
        stream);
}

void return_scatter_hybrid_bf16_from_pad(
    const __nv_bfloat16* Ye_pad,
    float* out,
    int M_recv, int T, int K,
    void** ipc_buffer_ptrs,
    int** ipc_barrier_ptrs,
    cudaStream_t stream)
{
    (void)ipc_barrier_ptrs;
    return_scatter_hybrid_from_pad_impl(
        Ye_pad,
        out,
        M_recv, T, K,
        ipc_buffer_ptrs,
        /*ipc_y_off=*/0,
        g_nvshmem.ipc_meta_off,
        g_nvshmem.ipc_counter_off,
        g_nvshmem.x_buf_bf16,
        g_nvshmem.H,
        g_nvshmem.Ha,
        stream);
}

// ============================================================================
// Host API: Hybrid Return Scatter (Blockscaled)
// ============================================================================

void return_scatter_hybrid_blockscaled(
    const __nv_bfloat16* Ye,
    float* out,
    int M_recv, int T, int K,
    void** ipc_buffer_ptrs,
    int** ipc_barrier_ptrs,
    cudaStream_t stream)
{
    (void)ipc_barrier_ptrs;
    return_scatter_hybrid_impl(
        Ye,
        out,
        M_recv, T, K,
        ipc_buffer_ptrs,
        g_nvshmem.ipc_y_off,
        g_nvshmem.ipc_meta_off,
        g_nvshmem.ipc_counter_off,
        g_nvshmem.y_buf,
        g_nvshmem.H,
        g_nvshmem.Ha,
        stream);
}

void return_scatter_hybrid_blockscaled_from_pad(
    const __nv_bfloat16* Ye_pad,
    float* out,
    int M_recv, int T, int K,
    void** ipc_buffer_ptrs,
    int** ipc_barrier_ptrs,
    cudaStream_t stream)
{
    (void)ipc_barrier_ptrs;
    return_scatter_hybrid_from_pad_impl(
        Ye_pad,
        out,
        M_recv, T, K,
        ipc_buffer_ptrs,
        g_nvshmem.ipc_y_off,
        g_nvshmem.ipc_meta_off,
        g_nvshmem.ipc_counter_off,
        g_nvshmem.y_buf,
        g_nvshmem.H,
        g_nvshmem.Ha,
        stream);
}

// ============================================================================
// Hybrid Backward (BF16 payload)
// ============================================================================

static int g_bwd_phase = 0;

__global__ void k_stage_dy_push_hybrid(
    const __nv_bfloat16* __restrict__ dY,   // [T, H]
    const int* __restrict__ eids,           // [T, K]
    int my_rank, int T, int H, int stage_stride, int K,
    int n_local, int capacity,
    int local_world, int rdma_rank,
    uint16_t* nv_stage,
    void** ipc_buffer_ptrs,
    size_t ipc_stage_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    int M = T * K;

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = i / K;
        int slot = i - tok * K;

        int eid = eids[tok * K + slot];
        int dest = eid / n_local;
        int dest_rdma = dest / local_world;
        int dest_nvl = dest - dest_rdma * local_world;
        bool remote_node = (dest_rdma != rdma_rank);
        bool remote_gpu = (dest != my_rank);

        int64_t rid = encode_rid(my_rank, tok, slot, T, K);
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        const __nv_bfloat16* row = dY + (int64_t)tok * H;

        if (remote_node) {
            uint16_t* dst = nv_stage + rid * static_cast<int64_t>(stage_stride);
            // Vectorized NVSHMEM put: 64-bit (4 BF16 values).
            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    nvshmem_put64_nbi(reinterpret_cast<uint64_t*>(dst + h),
                                      reinterpret_cast<const uint64_t*>(row + h),
                                      1, dest);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        nvshmem_put16_nbi(dst + hh,
                                          reinterpret_cast<const uint16_t*>(row) + hh,
                                          1, dest);
                    }
                }
            }
        } else {
            char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[dest_nvl]);
            uint16_t* stage = reinterpret_cast<uint16_t*>(dest_buf + ipc_stage_off);
            uint16_t* dst = stage + rid * static_cast<int64_t>(stage_stride);

            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    int2 v = *reinterpret_cast<const int2*>(row + h);
                    int2* d = reinterpret_cast<int2*>(dst + h);
                    if (remote_gpu) {
                        st_na_v2_s32(d, v);
                    } else {
                        *d = v;
                    }
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                        if (remote_gpu) {
                            st_na_relaxed_gpu_b16(dst + hh, u);
                        } else {
                            dst[hh] = u;
                        }
                    }
                }
            }
        }
    }
}

__global__ void k_gather_dy_from_stage_and_send_gate_hybrid(
    const __nv_bfloat16* __restrict__ Ye_pad,      // [M_pad, H]
    const int* __restrict__ dest,                  // [M] sorted_i -> pad_i
    const int64_t* __restrict__ row_id,            // [M]
    const float* __restrict__ gate_sorted,         // [M]
    const uint16_t* __restrict__ stage_ipc,        // [capacity, stage_stride]
    const uint16_t* __restrict__ stage_nv,         // [capacity, stage_stride]
    int stage_stride,
    __nv_bfloat16* __restrict__ dYe_out,           // [M, H]
    float* __restrict__ dGate_sorted_out,          // [M]
    int M, int T, int H, int K,
    int capacity,
    int my_rank, int local_world, int rdma_rank,
    void** ipc_buffer_ptrs,
    size_t ipc_tok_gate_off,
    float* nv_tok_gate,
    int* nv_tok_tag,
    int phase)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < M; i += num_warps) {
        const int64_t rid = row_id[i];
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        // src_rank = rid / (T*K)
        const int64_t tk = static_cast<int64_t>(T) * static_cast<int64_t>(K);
        const int src_rank = static_cast<int>(rid / tk);
        const int src_rdma = src_rank / local_world;
        const bool src_same_node = (src_rdma == rdma_rank);

        const uint16_t* dy_u16 = (src_same_node ? stage_ipc : stage_nv) + rid * static_cast<int64_t>(stage_stride);
        const int pad_i = dest[i];
        if (pad_i < 0) continue;
        const uint16_t* ye_u16 = reinterpret_cast<const uint16_t*>(Ye_pad + (int64_t)pad_i * H);
        __nv_bfloat16* dye_row = dYe_out + (int64_t)i * H;

        const float g = gate_sorted[i];
        float dot = 0.0f;

        for (int h = lane * 8; h < stage_stride; h += 32 * 8) {
            int4 dy_v = ld_nc_v4_s32(reinterpret_cast<const int4*>(dy_u16 + h));
            union U16x8 { int4 v; uint16_t u[8]; };
            U16x8 dy8; dy8.v = dy_v;

            U16x8 ye8;
            if (h + 8 <= H) {
                ye8.v = *reinterpret_cast<const int4*>(ye_u16 + h);
            }

#pragma unroll
            for (int j = 0; j < 8; j++) {
                int hh = h + j;
                if (hh >= H) break;
                const __nv_bfloat16 dy_bf = *reinterpret_cast<const __nv_bfloat16*>(&dy8.u[j]);
                const __nv_bfloat16 ye_bf = (h + 8 <= H) ? *reinterpret_cast<const __nv_bfloat16*>(&ye8.u[j])
                                                         : *reinterpret_cast<const __nv_bfloat16*>(ye_u16 + hh);
                float dy = __bfloat162float(dy_bf);
                float ye = __bfloat162float(ye_bf);
                dot += ye * dy;
                dye_row[hh] = __float2bfloat16(dy * g);
            }
        }

        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            dGate_sorted_out[i] = dot;

            // idx = tok*K + slot within the source rank's tok buffers.
            const int slot = static_cast<int>(rid % K);
            const int64_t tmp = rid / K;
            const int tok = static_cast<int>(tmp % T);
            const int64_t idx = (int64_t)tok * K + slot;

            if (src_same_node) {
                const int src_nvl = src_rank - src_rdma * local_world;
                char* src_buf = static_cast<char*>(ipc_buffer_ptrs[src_nvl]);
                float* tok_gate = reinterpret_cast<float*>(src_buf + ipc_tok_gate_off);
                if (src_rank == my_rank) {
                    tok_gate[idx] = dot;
                } else {
                    st_na_f32(tok_gate + idx, dot);
                }
            } else {
                nvshmem_float_p(nv_tok_gate + idx, dot, src_rank);
                nvshmem_int_p(nv_tok_tag + idx, phase, src_rank);
            }
        }
    }
}

__global__ void k_collect_tok_gate_hybrid(
    const float* __restrict__ ipc_tok_gate,
    const float* __restrict__ nv_tok_gate,
    const int* __restrict__ nv_tok_tag,
    float* __restrict__ dGates_tk_out,
    int tok_slots,
    int phase)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tok_slots) {
        int tag = ld_nc_s32(nv_tok_tag + i);
        float v = (tag == phase) ? ld_nc_f32(nv_tok_gate + i) : ld_nc_f32(ipc_tok_gate + i);
        dGates_tk_out[i] = v;
    }
}

void gather_dy_hybrid_bf16(
    const __nv_bfloat16* dY_local,
    const int* eids,
    const __nv_bfloat16* Ye_pad,
    const int64_t* row_id,
    const float* gate_sorted,
    __nv_bfloat16* dYe_out,
    float* dGate_sorted_out,
    float* dGates_tk_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) return;
    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid gather_dy requires synced IPC pointers; call nvshmem_sync_ipc_buffer_ptrs_*\n");
        return;
    }
    if (K <= 0 || K > 32) {
        fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
        return;
    }
    if (H != g_nvshmem.H) {
        fprintf(stderr, "RDEP ERROR: gather_dy H mismatch: got H=%d, state H=%d\n", H, g_nvshmem.H);
        return;
    }

    const int phase = ++g_bwd_phase;
    const int my_rank = g_nvshmem.rank;
    const int local_world = g_nvshmem.local_world;
    const int rdma_rank = g_nvshmem.rdma_rank;

    const size_t tok_cap = (g_nvshmem.world > 0) ? (g_nvshmem.capacity / static_cast<size_t>(g_nvshmem.world)) : 0;
    const int tok_slots = T * K;
    if (static_cast<size_t>(tok_slots) > tok_cap) {
        fprintf(stderr,
                "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                tok_slots, tok_cap, g_nvshmem.capacity, g_nvshmem.world);
        return;
    }

    const bool bf16_profile = (g_nvshmem.profile == -1);
    const uint16_t* nv_stage = bf16_profile ? g_nvshmem.x_buf_bf16 : g_nvshmem.y_buf;
    const size_t ipc_stage_off = bf16_profile ? g_nvshmem.ipc_x_off : g_nvshmem.ipc_y_off;
    const int stage_stride = bf16_profile ? g_nvshmem.Ha : g_nvshmem.H;

    char* local_ipc_buf = static_cast<char*>(g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    const uint16_t* stage_ipc = reinterpret_cast<const uint16_t*>(local_ipc_buf + ipc_stage_off);
    const float* ipc_tok_gate = reinterpret_cast<const float*>(local_ipc_buf + g_nvshmem.ipc_tok_gate_off);

	    // Stage dY (push) to expert owners.
	    const int threads = 256;
	    const int warps_needed = std::max(1, tok_slots);
	    const int blocks = std::max(1, (warps_needed * 32 + threads - 1) / threads);
	    k_stage_dy_push_hybrid<<<blocks, threads, 0, stream>>>(
        dY_local,
        eids,
        my_rank, T, H, stage_stride, K,
        g_nvshmem.n_local, static_cast<int>(g_nvshmem.capacity),
        local_world, rdma_rank,
        const_cast<uint16_t*>(nv_stage),
        g_nvshmem.d_ipc_buffer_ptrs,
        ipc_stage_off);

    nvshmemx_quiet_on_stream(stream);
    hybrid_barrier_on_stream(stream);
	    cudaStreamSynchronize(stream);

	    // Compute dYe/dGate locally and return dGate to token owners.
	    const int g_threads = 256;
	    const int g_blocks = std::max(1, (M * 32 + g_threads - 1) / g_threads);
    k_gather_dy_from_stage_and_send_gate_hybrid<<<g_blocks, g_threads, 0, stream>>>(
        Ye_pad,
        g_nvshmem.dest,
        row_id,
        gate_sorted,
        stage_ipc,
        nv_stage,
        stage_stride,
        dYe_out,
        dGate_sorted_out,
        M, T, H, K,
        static_cast<int>(g_nvshmem.capacity),
        my_rank, local_world, rdma_rank,
        g_nvshmem.d_ipc_buffer_ptrs,
        g_nvshmem.ipc_tok_gate_off,
        g_nvshmem.tok_gate,
        g_nvshmem.tok_tag,
        phase);

    nvshmemx_quiet_on_stream(stream);
    hybrid_barrier_on_stream(stream);

    // Collect per-(tok,slot) dGate into output tensor.
    const int c_threads = 256;
    const int c_blocks = (tok_slots + c_threads - 1) / c_threads;
    k_collect_tok_gate_hybrid<<<c_blocks, c_threads, 0, stream>>>(
        ipc_tok_gate,
        g_nvshmem.tok_gate,
        g_nvshmem.tok_tag,
        dGates_tk_out,
        tok_slots,
        phase);
}

__global__ void k_send_dx_tokslot_hybrid(
    const __nv_bfloat16* __restrict__ dXe_sorted,  // [M, H]
    const int64_t* __restrict__ row_id,            // [M]
    int M, int T, int H, int K,
    int tok_Ha,
    int capacity,
    int my_rank, int local_world, int rdma_rank,
    void** ipc_buffer_ptrs,
    size_t ipc_tok_y_off,
    uint16_t* nv_tok_y,
    int* nv_tok_tag,
    int phase)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < M; i += num_warps) {
        const int64_t rid = row_id[i];
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        const int slot = static_cast<int>(rid % K);
        const int64_t tmp = rid / K;
        const int tok = static_cast<int>(tmp % T);
        const int src_rank = static_cast<int>(tmp / T);

        const int src_rdma = src_rank / local_world;
        const bool same_node = (src_rdma == rdma_rank);
        const int64_t idx = (int64_t)tok * K + slot;

        const __nv_bfloat16* row = dXe_sorted + (int64_t)i * H;

        if (same_node) {
            const int src_nvl = src_rank - src_rdma * local_world;
            char* src_buf = static_cast<char*>(ipc_buffer_ptrs[src_nvl]);
            uint16_t* tok_y = reinterpret_cast<uint16_t*>(src_buf + ipc_tok_y_off);
            uint16_t* dst = tok_y + idx * static_cast<int64_t>(tok_Ha);

            const bool remote_gpu = (src_rank != my_rank);
            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    int2 v = *reinterpret_cast<const int2*>(row + h);
                    int2* d = reinterpret_cast<int2*>(dst + h);
                    if (remote_gpu) {
                        st_na_v2_s32(d, v);
                    } else {
                        *d = v;
                    }
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                        if (remote_gpu) {
                            st_na_relaxed_gpu_b16(dst + hh, u);
                        } else {
                            dst[hh] = u;
                        }
                    }
                }
            }
        } else {
            uint16_t* dst = nv_tok_y + idx * static_cast<int64_t>(tok_Ha);
            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    nvshmem_put64_nbi(reinterpret_cast<uint64_t*>(dst + h),
                                      reinterpret_cast<const uint64_t*>(row + h),
                                      1, src_rank);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        nvshmem_put16_nbi(dst + hh,
                                          reinterpret_cast<const uint16_t*>(row) + hh,
                                          1, src_rank);
                    }
                }
            }
            if (lane == 0) {
                nvshmem_int_p(nv_tok_tag + idx, phase, src_rank);
            }
        }
    }
}

__global__ void k_reduce_dx_tokslot_hybrid(
    const uint16_t* __restrict__ ipc_tok_y,
    const uint16_t* __restrict__ nv_tok_y,
    const int* __restrict__ nv_tok_tag,
    float* __restrict__ dX_out,
    int T, int H, int tok_Ha, int K,
    int phase)
{
    int tok = static_cast<int>(blockIdx.x);
    if (tok >= T) return;
    if (K <= 0 || K > 32) return;

    int vec = static_cast<int>(threadIdx.x);
    for (int h0 = vec * 8; h0 < H; h0 += static_cast<int>(blockDim.x) * 8) {
        float acc[8] = {0};
        for (int slot = 0; slot < K; ++slot) {
            const int64_t idx = (int64_t)tok * K + slot;
            const bool remote = (ld_nc_s32(nv_tok_tag + idx) == phase);
            const uint16_t* base = remote ? nv_tok_y : ipc_tok_y;
            const uint16_t* y_row = base + idx * static_cast<int64_t>(tok_Ha) + h0;

            int4 v = ld_nc_v4_s32(reinterpret_cast<const int4*>(y_row));
            union BF16x8 {
                int4 v;
                uint16_t u[8];
            };
            BF16x8 x;
            x.v = v;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                int hh = h0 + j;
                if (hh < H) {
                    const __nv_bfloat16 bf = *reinterpret_cast<const __nv_bfloat16*>(&x.u[j]);
                    acc[j] += __bfloat162float(bf);
                }
            }
        }

        float* out_row = dX_out + (int64_t)tok * H + h0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int hh = h0 + j;
            if (hh < H) out_row[j] = acc[j];
        }
    }
}

void scatter_dx_hybrid_bf16(
    const __nv_bfloat16* dXe_sorted,
    const int64_t* row_id,
    float* dX_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) return;
    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid scatter_dx requires synced IPC pointers; call nvshmem_sync_ipc_buffer_ptrs_*\n");
        return;
    }
    if (K <= 0 || K > 32) {
        fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
        return;
    }
    if (H != g_nvshmem.H) {
        fprintf(stderr, "RDEP ERROR: scatter_dx H mismatch: got H=%d, state H=%d\n", H, g_nvshmem.H);
        return;
    }
    if ((H & 7) != 0) {
        fprintf(stderr, "RDEP ERROR: scatter_dx requires H multiple of 8 (H=%d)\n", H);
        return;
    }

    const int phase = ++g_bwd_phase;
    const int my_rank = g_nvshmem.rank;
    const int local_world = g_nvshmem.local_world;
    const int rdma_rank = g_nvshmem.rdma_rank;

    const size_t tok_cap = (g_nvshmem.world > 0) ? (g_nvshmem.capacity / static_cast<size_t>(g_nvshmem.world)) : 0;
    const int tok_slots = T * K;
    if (static_cast<size_t>(tok_slots) > tok_cap) {
        fprintf(stderr,
                "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                tok_slots, tok_cap, g_nvshmem.capacity, g_nvshmem.world);
        return;
    }

    char* local_ipc_buf = static_cast<char*>(g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    const uint16_t* ipc_tok_y = reinterpret_cast<const uint16_t*>(local_ipc_buf + g_nvshmem.ipc_tok_y_off);

	    const int threads = 256;
	    const int warps_needed = std::max(1, M);
	    const int blocks = std::max(1, (warps_needed * 32 + threads - 1) / threads);
	    k_send_dx_tokslot_hybrid<<<blocks, threads, 0, stream>>>(
        dXe_sorted,
        row_id,
        M, T, H, K,
        g_nvshmem.tok_Ha,
        static_cast<int>(g_nvshmem.capacity),
        my_rank, local_world, rdma_rank,
        g_nvshmem.d_ipc_buffer_ptrs,
        g_nvshmem.ipc_tok_y_off,
        g_nvshmem.tok_y,
        g_nvshmem.tok_tag,
        phase);

    nvshmemx_quiet_on_stream(stream);
    hybrid_barrier_on_stream(stream);
	    cudaStreamSynchronize(stream);

	    k_reduce_dx_tokslot_hybrid<<<T, 256, 0, stream>>>(
	        ipc_tok_y,
	        g_nvshmem.tok_y,
        g_nvshmem.tok_tag,
        dX_out,
        T, H, g_nvshmem.tok_Ha, K,
        phase);
}

}  // namespace nvshmem
}  // namespace rdep

// ============================================================================
// C API wrappers for Python bindings
// Use rdep_ prefix to avoid conflicts with NVSHMEM's own functions
// ============================================================================

extern "C" {

void rdep_nvshmem_get_uid(void* uid_out) {
    rdep::nvshmem::get_uid(uid_out);
}

int rdep_nvshmem_get_uid_size() {
    return rdep::nvshmem::get_uid_size();
}

void rdep_nvshmem_init_with_uid(const void* uid, int rank, int world, int local_world) {
    rdep::nvshmem::init(uid, rank, world, local_world);
}

void rdep_nvshmem_finalize() {
    rdep::nvshmem::finalize();
}

void rdep_nvshmem_alloc_bf16(size_t capacity, int H, int n_local) {
    rdep::nvshmem::alloc_bf16(capacity, H, n_local);
}

void rdep_nvshmem_alloc_blockscaled(size_t capacity, int H, int n_local, int profile) {
    rdep::nvshmem::alloc_blockscaled(capacity, H, n_local, profile);
}

void rdep_nvshmem_barrier() {
    rdep::nvshmem::barrier();
}

void rdep_nvshmem_quiet() {
    rdep::nvshmem::quiet();
}

// IPC buffer management functions
void rdep_nvshmem_get_ipc_handle_bf16(void* handle_out) {
    rdep::nvshmem::get_ipc_handle_bf16(handle_out);
}

void rdep_nvshmem_open_ipc_handles_bf16(const void* handles, int local_world) {
    rdep::nvshmem::open_ipc_handles_bf16(handles, local_world);
}

void rdep_nvshmem_sync_ipc_buffer_ptrs_bf16() {
    rdep::nvshmem::sync_ipc_buffer_ptrs_bf16();
}

void rdep_nvshmem_get_ipc_handle_blockscaled(void* handle_out) {
    rdep::nvshmem::get_ipc_handle_blockscaled(handle_out);
}

void rdep_nvshmem_open_ipc_handles_blockscaled(const void* handles, int local_world) {
    rdep::nvshmem::open_ipc_handles_blockscaled(handles, local_world);
}

void rdep_nvshmem_sync_ipc_buffer_ptrs_blockscaled() {
    rdep::nvshmem::sync_ipc_buffer_ptrs_blockscaled();
}

}  // extern "C"

#endif  // WITH_NVSHMEM
