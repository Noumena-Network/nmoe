// RDEP: Expert-parallel dispatch/return for MoE
//
// Three modes:
//   - MODE_SINGLE: world=1, local sort/pad/quant only
//   - MODE_IPC: world=local_world, CUDA IPC for intra-node NVLink
//   - MODE_HYBRID: world>local_world, IPC intra-node + NVSHMEM inter-node
//
// Bootstrap (one-time at init):
//   - IPC: NCCL all_gather to exchange cudaIpcMemHandle_t
//   - NVSHMEM: NCCL broadcast to share NVSHMEM UID
//
// Hot path (zero NCCL):
//   - GPU-side atomics for sync (atomicAdd_system/atomicSub_system)
//   - Direct P2P writes via IPC handles or NVSHMEM puts

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <type_traits>
#include <vector>

// NVSHMEM support (optional, for multi-node)
#ifdef WITH_NVSHMEM
#include "rdep_nvshmem.cuh"
#endif

namespace rdep {

// ============================================================================
// Constants (following DeepEP configs.cuh)
// ============================================================================

constexpr int SF_VEC = 32;        // Scale factor granularity
constexpr float FP8_MAX = 448.0f;
constexpr float FP4_MAX = 6.0f;
constexpr int MAX_RANKS = 8;      // Max NVLink peers (like DeepEP)
constexpr int BUFFER_ALIGNMENT = 128;  // DeepEP's NUM_BUFFER_ALIGNMENT_BYTES
constexpr int BARRIER_TAG = 1024;      // DeepEP's FINISHED_SUM_TAG
constexpr uint64_t TIMEOUT_CYCLES = 200000000000ull;  // ~100s at 2GHz
constexpr int IPC_PHASE_IDX = 0;

// ============================================================================
// PTX Primitives - imported from ptx.cu
// ============================================================================
// Use nmoe::ptx:: namespace for all PTX primitives.
// See ptx.cu for the full list of IPC/P2P memory ordering primitives.

#include "ptx.cu"

using namespace nmoe::ptx;

// Forward declaration for swizzle_sf_strided (defined in quant.cu)
extern "C" cudaError_t swizzle_sf_strided(
    const void* sf_mkl,
    void* sf_mma,
    const int32_t* offs,
    int E, int sf_k, int sf_k_pad, int M_pad, int M_e_swizzle,
    cudaStream_t stream);

// ============================================================================
// Helpers
// ============================================================================

__device__ __host__ __forceinline__ int H_aligned(int H) {
    return ((H + 7) / 8) * 8;
}

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
// FP8 E4M3 / FP4 E2M1 Conversion - use ptx.cu versions
// ============================================================================
// Aliases to match existing code using local names

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
// GPU-Side Barrier (DeepEP pattern)
// ============================================================================
// Cross-GPU barrier using atomicAdd_system/atomicSub_system.
// Each rank adds to its own signal array and subtracts from other ranks' arrays.
// When all signals reach zero, barrier is complete.
//
// Pattern: barrier_signal_ptrs[rank][i] tracks arrivals from rank i.
// - Add BARRIER_TAG to own array: barrier_signal_ptrs[my_rank][thread_id]
// - Sub BARRIER_TAG from other array: barrier_signal_ptrs[thread_id][my_rank]
// - Poll until all values <= 0

template <int kNumRanks, bool kSyncOnly = false>
__device__ __forceinline__ void
barrier_block(int** barrier_signal_ptrs, int rank) {
    int thread_id = static_cast<int>(threadIdx.x);

    // For non-sync-only cases, flush all P2P writes to system scope
    if constexpr (!kSyncOnly) {
        fence_acq_rel_sys();
        __syncthreads();
    }

    // Add to self signals, subtract from others
    // This ensures all ranks must have arrived before any can proceed
    if (thread_id < kNumRanks) {
        atomicAdd_sys(barrier_signal_ptrs[rank] + thread_id, BARRIER_TAG);
        atomicSub_sys(barrier_signal_ptrs[thread_id] + rank, BARRIER_TAG);
    }

    // Wait for all signals to reach zero (all ranks have arrived)
    uint64_t start_time = clock64();
    while (true) {
        int value = (thread_id < kNumRanks) ? ld_volatile_s32(barrier_signal_ptrs[rank] + thread_id) : 0;
        if (__all_sync(0xffffffff, value <= 0))
            break;

        // Timeout check
        if (clock64() - start_time > TIMEOUT_CYCLES && thread_id < kNumRanks) {
            printf("nmoe barrier timeout: rank=%d, thread=%d, value=%d\n", rank, thread_id, value);
            trap();
        }
    }
    // Acquire remote writes that happened-before peers signaled arrival.
    fence_acq_rel_sys();
    __syncthreads();
}

// Non-templated version for dynamic num_ranks
__device__ __forceinline__ void
barrier_block_dynamic(int** barrier_signal_ptrs, int rank, int num_ranks) {
    int thread_id = static_cast<int>(threadIdx.x);

    fence_acq_rel_sys();
    __syncthreads();

    if (thread_id < num_ranks) {
        atomicAdd_sys(barrier_signal_ptrs[rank] + thread_id, BARRIER_TAG);
        atomicSub_sys(barrier_signal_ptrs[thread_id] + rank, BARRIER_TAG);
    }

    uint64_t start_time = clock64();
    while (true) {
        int value = (thread_id < num_ranks) ? ld_volatile_s32(barrier_signal_ptrs[rank] + thread_id) : 0;
        if (__all_sync(0xffffffff, value <= 0))
            break;

        if (clock64() - start_time > TIMEOUT_CYCLES && thread_id < num_ranks) {
            printf("nmoe barrier timeout: rank=%d, thread=%d, value=%d\n", rank, thread_id, value);
            trap();
        }
    }
    // Acquire remote writes that happened-before peers signaled arrival.
    fence_acq_rel_sys();
    __syncthreads();
}

// ============================================================================
// Metadata (16-byte aligned)
// ============================================================================

struct alignas(16) Meta {
    int64_t row_id;
    int32_t local_eid;
    float   gate;
};
static_assert(sizeof(Meta) == 16, "Meta must be 16 bytes");

// ============================================================================
// Mode Selection
// ============================================================================

enum RdepMode {
    MODE_SINGLE = 0,   // world=1, local only
    MODE_IPC = 1,      // world=local_world, CUDA IPC
    MODE_HYBRID = 2    // world>local_world, IPC intra-node + NVSHMEM inter-node
};

static RdepMode g_mode = MODE_SINGLE;

// ============================================================================
// Global State with IPC Buffer Pointers
// ============================================================================

	struct StateBF16 {
	    // IPC buffer pointers - [rank] -> remote buffer on that rank
	    // buffer_ptrs[my_rank] is local cudaMalloc, others are IPC-opened
	    void* buffer_ptrs[MAX_RANKS];

    // Barrier signal pointers - [rank] -> signal array on that rank
    // Each rank's buffer has MAX_RANKS ints for barrier signals
    int* barrier_signal_ptrs[MAX_RANKS];

	    // Buffer layout within each rank's allocation (IPC, BF16):
	    // [capacity * Ha * sizeof(uint16_t)]          - x_buf (BF16 activations, dispatch receive)
	    // [capacity * sizeof(Meta)]                   - meta (dispatch metadata)
	    // [sizeof(int)]                               - counter (legacy append counter; avoided on hot paths where possible)
	    // [sizeof(int)]                               - dropped (dispatch overflow counter)
	    // [MAX_RANKS * sizeof(int)]                   - barrier_signals (GPU-side sync)
	    // [MAX_RANKS * sizeof(void*)]                 - buffer_ptrs_gpu (pointers on GPU)
	    // [MAX_RANKS * sizeof(int*)]                  - barrier_signal_ptrs_gpu
	    // [tok_slots * Ha * sizeof(uint16_t)]         - tok_y (BF16 per-(tok,slot) buffer, used for return/dX)
	    // [tok_slots * sizeof(float)]                 - tok_gate (float per-(tok,slot) buffer, used for return gating / scratch)
	    //
	    // Where tok_slots = capacity / world (must be >= T*K).

    // Local work buffers
    int*      local_eid;
    int*      order;
    int*      offsets;
    int*      offs_pad;
    int*      dest;
    int*      M_pad_dev;
    Meta*     meta_copy;
    void*     sort_temp;
    size_t    sort_temp_bytes;

    // 2-phase dispatch: local atomic counters for ordering within rank's sends
    int* local_counters;  // [MAX_RANKS] - local atomics for 2-phase dispatch

    // Dimensions
    size_t capacity;
    size_t buffer_size;  // Total bytes per rank
    int M_pad;
    int H, Ha;
    int world, rank;
    int n_local;
    int align;
    bool initialized;

};

struct StateBlockscaled {
    void* buffer_ptrs[MAX_RANKS];
    int* barrier_signal_ptrs[MAX_RANKS];

    // Buffer layout:
    // [capacity * Hp * sizeof(uint16_t)]  - x_buf (packed)
    // [capacity * Hsf * sizeof(uint8_t)]  - sfa_buf
    // [capacity * H * sizeof(uint16_t)]   - y_buf (return BF16)
    // [capacity * sizeof(Meta)]           - meta
    // [sizeof(int)]                        - counter
    // [sizeof(int)]                        - dropped
    // [MAX_RANKS * sizeof(int)]           - barrier_signals
    // [MAX_RANKS * sizeof(void*)]         - buffer_ptrs_gpu
    // [MAX_RANKS * sizeof(int*)]          - barrier_signal_ptrs_gpu
    // [tok_slots * Ha * sizeof(uint16_t)] - tok_y (BF16 per-(tok,slot) scratch for return/dX)
    // [tok_slots * sizeof(float)]         - tok_gate (float per-(tok,slot) scratch for return/dGate)

    int*      local_eid;
    int*      order;
    int*      offsets;
    int*      offs_pad;
    int*      dest;
    int*      M_pad_dev;
    void*     sort_temp;
    size_t    sort_temp_bytes;

    size_t capacity;
    size_t buffer_size;
    int M_pad;
    int H, Ha, Hp, Hsf;
    int world, rank;
    int n_local;
    int align;
    int profile;
    bool initialized;
    // Additional workspace
    uint8_t* sfa_gather_tmp;   // [max_pad * Hsf] row-major gathered SFA (max_pad = capacity + n_local*(align-1))
    int*     offs_with0;       // [n_local + 1] offsets with leading 0 for swizzle
    int      M_e_swizzle_cap;  // aligned capacity per-expert (128)
};

static StateBF16 g_bf16 = {};
static StateBlockscaled g_block = {};

// IPC handles stored for cleanup
static cudaIpcMemHandle_t g_ipc_handles_bf16[MAX_RANKS];
static cudaIpcMemHandle_t g_ipc_handles_block[MAX_RANKS];

static int g_ipc_phase_bf16 = 0;
static int g_ipc_phase_block = 0;

// 2-phase dispatch: enabled by default for IPC mode (eliminates remote atomics)
// Set NMOE_DISPATCH_2PHASE=0 to disable and use legacy atomic-counter dispatch
static bool g_use_2phase_dispatch = true;

// ============================================================================
// Helper: Get buffer offsets (following DeepEP layout pattern)
// ============================================================================

__host__ __device__ __forceinline__
size_t align_up(size_t x, size_t align) {
    return ((x + align - 1) / align) * align;
}

	__host__ __device__ __forceinline__
	void bf16_buffer_offsets(size_t capacity, int Ha, int world,
	                         size_t* x_off, size_t* meta_off,
	                         size_t* counter_off, size_t* dropped_off,
	                         size_t* barrier_off, size_t* buf_ptrs_off, size_t* sig_ptrs_off,
	                         size_t* tok_y_off, size_t* tok_gate_off,
	                         size_t* total_size,
	                         // 2-phase dispatch areas (optional, can be nullptr)
	                         size_t* recv_counts_off = nullptr,
	                         size_t* recv_offsets_off = nullptr) {
	    *x_off = 0;
	    *meta_off = capacity * Ha * sizeof(uint16_t);
	    *counter_off = *meta_off + capacity * sizeof(Meta);
	    *dropped_off = *counter_off + sizeof(int);
	    // Align barrier signals for atomic operations
	    *barrier_off = align_up(*dropped_off + sizeof(int), BUFFER_ALIGNMENT);
	    *buf_ptrs_off = *barrier_off + MAX_RANKS * sizeof(int);
	    *sig_ptrs_off = *buf_ptrs_off + MAX_RANKS * sizeof(void*);
	    const size_t ptrs_end = *sig_ptrs_off + MAX_RANKS * sizeof(int*);

	    // Token-slot buffers (fixed size per rank; used by IPC return/dX to avoid append counters).
	    const size_t tok_slots = (world > 0) ? (capacity / static_cast<size_t>(world)) : 0;
	    *tok_y_off = align_up(ptrs_end, BUFFER_ALIGNMENT);
	    *tok_gate_off = align_up(*tok_y_off + tok_slots * static_cast<size_t>(Ha) * sizeof(uint16_t), BUFFER_ALIGNMENT);

	    // 2-phase dispatch: count exchange area (MAX_RANKS ints per rank for recv_from[src] counts)
	    // Layout: recv_counts[src] = how many tokens I receive from rank src
	    //         recv_offsets[src] = where rank src's data starts in my buffer (prefix sum)
	    const size_t tok_gate_end = *tok_gate_off + tok_slots * sizeof(float);
	    const size_t _recv_counts_off = align_up(tok_gate_end, BUFFER_ALIGNMENT);
	    const size_t _recv_offsets_off = _recv_counts_off + MAX_RANKS * sizeof(int);

	    if (recv_counts_off) *recv_counts_off = _recv_counts_off;
	    if (recv_offsets_off) *recv_offsets_off = _recv_offsets_off;

	    *total_size = align_up(_recv_offsets_off + MAX_RANKS * sizeof(int), BUFFER_ALIGNMENT);
	}

__host__ __device__ __forceinline__
void blockscaled_buffer_offsets(size_t capacity, int H, int Hp, int Hsf,
                                int world,
                                size_t* x_off, size_t* sfa_off, size_t* y_off,
                                size_t* meta_off, size_t* counter_off, size_t* dropped_off,
                                size_t* barrier_off, size_t* buf_ptrs_off, size_t* sig_ptrs_off,
                                size_t* tok_y_off, size_t* tok_gate_off,
                                size_t* total_size) {
    *x_off = 0;
    *sfa_off = capacity * Hp * sizeof(uint16_t);
    *y_off = *sfa_off + capacity * Hsf * sizeof(uint8_t);
    *meta_off = *y_off + capacity * H * sizeof(uint16_t);
    *counter_off = *meta_off + capacity * sizeof(Meta);
    *dropped_off = *counter_off + sizeof(int);
    *barrier_off = align_up(*dropped_off + sizeof(int), BUFFER_ALIGNMENT);
    *buf_ptrs_off = *barrier_off + MAX_RANKS * sizeof(int);
    *sig_ptrs_off = *buf_ptrs_off + MAX_RANKS * sizeof(void*);
    const size_t ptrs_end = *sig_ptrs_off + MAX_RANKS * sizeof(int*);

    // Token-slot buffers (fixed size per rank; used by IPC return/dX).
    const size_t tok_slots = (world > 0) ? (capacity / static_cast<size_t>(world)) : 0;
    const int Ha = H_aligned(H);
    *tok_y_off = align_up(ptrs_end, BUFFER_ALIGNMENT);
    *tok_gate_off = align_up(*tok_y_off + tok_slots * static_cast<size_t>(Ha) * sizeof(uint16_t), BUFFER_ALIGNMENT);

    *total_size = align_up(*tok_gate_off + tok_slots * sizeof(float), BUFFER_ALIGNMENT);
}

// ============================================================================
// Init / Alloc - IPC Setup
// ============================================================================

// Get local IPC handle (call on each rank after alloc)
extern "C" void rdep_get_ipc_handle_bf16(void* handle_out) {
    if (!g_bf16.initialized || !g_bf16.buffer_ptrs[g_bf16.rank]) return;
    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, g_bf16.buffer_ptrs[g_bf16.rank]);
    memcpy(handle_out, &handle, sizeof(cudaIpcMemHandle_t));
}

extern "C" void rdep_get_ipc_handle_blockscaled(void* handle_out) {
    if (!g_block.initialized || !g_block.buffer_ptrs[g_block.rank]) return;
    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, g_block.buffer_ptrs[g_block.rank]);
    memcpy(handle_out, &handle, sizeof(cudaIpcMemHandle_t));
}

// Initialize with rank info and determine mode
// Mode is auto-selected based on world vs local_world:
//   - world=1: MODE_SINGLE (local only)
//   - world=local_world: MODE_IPC (CUDA IPC for intra-node)
//   - world>local_world: MODE_HYBRID (IPC intra-node + NVSHMEM inter-node)
extern "C" void rdep_init(int rank, int world, int local_world) {
    g_bf16.rank = rank;
    g_bf16.world = world;
    g_block.rank = rank;
    g_block.world = world;

    // Mode selection
    if (world == 1) {
        g_mode = MODE_SINGLE;
        return;
    } else if (world == local_world) {
        g_mode = MODE_IPC;
    } else {
        g_mode = MODE_HYBRID;
#ifndef WITH_NVSHMEM
        fprintf(stderr, "RDEP ERROR: Multi-node (world=%d > local_world=%d) requires NVSHMEM.\n", world, local_world);
        fprintf(stderr, "           Rebuild with NVSHMEM support or use single-node configuration.\n");
        exit(1);
#endif
    }

    // For IPC mode: check P2P access for local peers only
    // For HYBRID mode: check P2P for local peers, NVSHMEM handles inter-node
    int my_device;
    cudaGetDevice(&my_device);
    int local_rank = rank % local_world;

    for (int peer = 0; peer < local_world; peer++) {
        if (peer == local_rank) continue;
        int can_access = 0;
        cudaDeviceCanAccessPeer(&can_access, my_device, peer);
        if (!can_access) {
            fprintf(stderr, "RDEP ERROR: GPU %d cannot access local peer GPU %d\n", my_device, peer);
        }
        int native_atomic = 0;
        cudaDeviceGetP2PAttribute(&native_atomic, cudaDevP2PAttrNativeAtomicSupported, my_device, peer);
        if (!native_atomic) {
            fprintf(stderr, "RDEP WARNING: Native atomics not supported between GPU %d and %d\n", my_device, peer);
        }
    }

#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        // NVSHMEM initialization will be done separately after UID broadcast
    }
#endif
}

// Query current mode
extern "C" int rdep_get_mode() {
    return static_cast<int>(g_mode);
}

// Check if NVSHMEM support is compiled in
extern "C" bool rdep_has_nvshmem() {
#ifdef WITH_NVSHMEM
    return true;
#else
    return false;
#endif
}

// Open remote IPC handles after all_gather
extern "C" void rdep_open_ipc_handles_bf16(const void* handles, int world) {
    const cudaIpcMemHandle_t* all_handles = static_cast<const cudaIpcMemHandle_t*>(handles);
    int my_rank = g_bf16.rank;

    // Calculate barrier offset
    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);

    for (int r = 0; r < world; r++) {
        if (r == my_rank) {
            // Local buffer already allocated
            memcpy(&g_ipc_handles_bf16[r], &all_handles[r], sizeof(cudaIpcMemHandle_t));
        } else {
            // Open remote buffer
            memcpy(&g_ipc_handles_bf16[r], &all_handles[r], sizeof(cudaIpcMemHandle_t));
            cudaIpcOpenMemHandle(&g_bf16.buffer_ptrs[r], g_ipc_handles_bf16[r],
                                 cudaIpcMemLazyEnablePeerAccess);
            // Set barrier signal pointer for remote buffer
            char* remote_buf = static_cast<char*>(g_bf16.buffer_ptrs[r]);
            g_bf16.barrier_signal_ptrs[r] = reinterpret_cast<int*>(remote_buf + barrier_off);
        }
    }
}

extern "C" void rdep_open_ipc_handles_blockscaled(const void* handles, int world) {
    const cudaIpcMemHandle_t* all_handles = static_cast<const cudaIpcMemHandle_t*>(handles);
    int my_rank = g_block.rank;

    // Calculate barrier offset
    size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, world,
                               &x_off, &sfa_off, &y_off, &meta_off, &counter_off, &dropped_off,
                               &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                               &tok_y_off, &tok_gate_off,
                               &total_size);

    for (int r = 0; r < world; r++) {
        if (r == my_rank) {
            memcpy(&g_ipc_handles_block[r], &all_handles[r], sizeof(cudaIpcMemHandle_t));
        } else {
            memcpy(&g_ipc_handles_block[r], &all_handles[r], sizeof(cudaIpcMemHandle_t));
            cudaIpcOpenMemHandle(&g_block.buffer_ptrs[r], g_ipc_handles_block[r],
                                 cudaIpcMemLazyEnablePeerAccess);
            // Set barrier signal pointer for remote buffer
            char* remote_buf = static_cast<char*>(g_block.buffer_ptrs[r]);
            g_block.barrier_signal_ptrs[r] = reinterpret_cast<int*>(remote_buf + barrier_off);
        }
    }
}

// Allocate local buffer (BF16 path)
extern "C" void rdep_alloc_bf16(size_t capacity, int H, int n_local) {
    int Ha = H_aligned(H);

    // Calculate buffer layout
    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(capacity, Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);

    // Free old local allocation
    if (g_bf16.buffer_ptrs[g_bf16.rank]) {
        cudaFree(g_bf16.buffer_ptrs[g_bf16.rank]);
        g_bf16.buffer_ptrs[g_bf16.rank] = nullptr;
    }
    if (g_bf16.local_eid) cudaFree(g_bf16.local_eid);
    if (g_bf16.order) cudaFree(g_bf16.order);
    if (g_bf16.offsets) cudaFree(g_bf16.offsets);
    if (g_bf16.offs_pad) cudaFree(g_bf16.offs_pad);
    if (g_bf16.dest) cudaFree(g_bf16.dest);
    if (g_bf16.M_pad_dev) cudaFree(g_bf16.M_pad_dev);
    if (g_bf16.meta_copy) cudaFree(g_bf16.meta_copy);
    if (g_bf16.sort_temp) cudaFree(g_bf16.sort_temp);
    if (g_bf16.local_counters) cudaFree(g_bf16.local_counters);

    // Allocate local buffer (aligned for DeepEP pattern)
    cudaMalloc(&g_bf16.buffer_ptrs[g_bf16.rank], total_size);
    cudaMemset(g_bf16.buffer_ptrs[g_bf16.rank], 0, total_size);

    // Set local barrier signal pointer
    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    g_bf16.barrier_signal_ptrs[g_bf16.rank] = reinterpret_cast<int*>(local_buf + barrier_off);

    // Allocate work buffers
    cudaMalloc(&g_bf16.local_eid, capacity * sizeof(int));
    cudaMalloc(&g_bf16.order, capacity * sizeof(int));
    cudaMalloc(&g_bf16.offsets, (n_local + 1) * sizeof(int));
    cudaMalloc(&g_bf16.offs_pad, n_local * sizeof(int));
    cudaMalloc(&g_bf16.dest, capacity * sizeof(int));
    cudaMalloc(&g_bf16.M_pad_dev, sizeof(int));
    cudaMalloc(&g_bf16.meta_copy, capacity * sizeof(Meta));

    g_bf16.sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, g_bf16.sort_temp_bytes,
        g_bf16.local_eid, g_bf16.local_eid, g_bf16.order, g_bf16.order, (int)capacity);
    cudaMalloc(&g_bf16.sort_temp, g_bf16.sort_temp_bytes);

    // 2-phase dispatch: local atomic counters (one per destination rank)
    cudaMalloc(&g_bf16.local_counters, MAX_RANKS * sizeof(int));

    g_bf16.capacity = capacity;
    g_bf16.buffer_size = total_size;
    g_bf16.H = H;
    g_bf16.Ha = Ha;
    g_bf16.n_local = n_local;
    g_bf16.align = 128;  // Match blockscaled for consistent padding
    g_bf16.initialized = true;
    g_ipc_phase_bf16 = 0;
}

// Allocate local buffer (Blockscaled path)
extern "C" void rdep_alloc_blockscaled(size_t capacity, int H, int n_local, int profile) {
    int pack_factor = (profile == 0) ? 2 : 4;
    int Hp = H / pack_factor;
    int Hsf = (H + SF_VEC - 1) / SF_VEC;
    int Ha = H_aligned(H);
    const int align = 128;
    const size_t max_pad = capacity + static_cast<size_t>(n_local) * static_cast<size_t>(align - 1);

    size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    blockscaled_buffer_offsets(capacity, H, Hp, Hsf, g_block.world,
                               &x_off, &sfa_off, &y_off,
                               &meta_off, &counter_off, &dropped_off,
                               &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                               &tok_y_off, &tok_gate_off,
                               &total_size);

    if (g_block.buffer_ptrs[g_block.rank]) {
        cudaFree(g_block.buffer_ptrs[g_block.rank]);
        g_block.buffer_ptrs[g_block.rank] = nullptr;
    }
    if (g_block.local_eid) cudaFree(g_block.local_eid);
    if (g_block.order) cudaFree(g_block.order);
    if (g_block.offsets) cudaFree(g_block.offsets);
    if (g_block.offs_pad) cudaFree(g_block.offs_pad);
    if (g_block.dest) cudaFree(g_block.dest);
    if (g_block.M_pad_dev) cudaFree(g_block.M_pad_dev);
    if (g_block.sort_temp) cudaFree(g_block.sort_temp);
    if (g_block.sfa_gather_tmp) cudaFree(g_block.sfa_gather_tmp);
    if (g_block.offs_with0) cudaFree(g_block.offs_with0);

    cudaMalloc(&g_block.buffer_ptrs[g_block.rank], total_size);
    cudaMemset(g_block.buffer_ptrs[g_block.rank], 0, total_size);

    // Set local barrier signal pointer
    char* local_buf = static_cast<char*>(g_block.buffer_ptrs[g_block.rank]);
    g_block.barrier_signal_ptrs[g_block.rank] = reinterpret_cast<int*>(local_buf + barrier_off);

    cudaMalloc(&g_block.local_eid, capacity * sizeof(int));
    cudaMalloc(&g_block.order, capacity * sizeof(int));
    cudaMalloc(&g_block.offsets, (n_local + 1) * sizeof(int));
    cudaMalloc(&g_block.offs_pad, n_local * sizeof(int));
    cudaMalloc(&g_block.dest, capacity * sizeof(int));
    cudaMalloc(&g_block.M_pad_dev, sizeof(int));

    g_block.sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, g_block.sort_temp_bytes,
        g_block.local_eid, g_block.local_eid, g_block.order, g_block.order, (int)capacity);
    cudaMalloc(&g_block.sort_temp, g_block.sort_temp_bytes);

    g_block.capacity = capacity;
    g_block.buffer_size = total_size;
    g_block.H = H;
    g_block.Ha = Ha;
    g_block.Hp = Hp;
    g_block.Hsf = Hsf;
    g_block.n_local = n_local;
    g_block.align = align;
    g_block.profile = profile;
    g_block.initialized = true;

    // Allocate additional workspace for SFA swizzle and offsets-with-zero
    cudaMalloc(&g_block.sfa_gather_tmp, max_pad * g_block.Hsf * sizeof(uint8_t));
    cudaMalloc(&g_block.offs_with0, (n_local + 1) * sizeof(int));
    g_block.M_e_swizzle_cap = ((static_cast<int>(capacity) + 127) / 128) * 128;
    g_ipc_phase_block = 0;
}

// ============================================================================
// IPC Dispatch Kernel - Direct P2P writes via IPC pointers
// ============================================================================

// Device pointers to all ranks' buffers (DeepEP pattern: kernel-accessible arrays)
__device__ void* d_buffer_ptrs_bf16[MAX_RANKS];
__device__ int*  d_barrier_signal_ptrs_bf16[MAX_RANKS];
__device__ void* d_buffer_ptrs_block[MAX_RANKS];
__device__ int*  d_barrier_signal_ptrs_block[MAX_RANKS];
__device__ int   d_my_rank_bf16;
__device__ int   d_my_rank_block;
__device__ int   d_world_bf16;
__device__ int   d_world_block;

// One-CTA, system-scope cross-GPU barriers (IPC mode).
// Declared here for use in forward dispatch/return; defined below with other IPC helpers.
__global__ void k_ipc_barrier_phase_bf16(int phase);
__global__ void k_ipc_barrier_phase_block(int phase);

__host__ __forceinline__ void ipc_barrier_bf16(cudaStream_t stream) {
    if (g_mode != MODE_IPC) return;
    if (g_bf16.world <= 1) return;
    k_ipc_barrier_phase_bf16<<<1, 256, 0, stream>>>(++g_ipc_phase_bf16);
}

__host__ __forceinline__ void ipc_barrier_block(cudaStream_t stream) {
    if (g_mode != MODE_IPC) return;
    if (g_block.world <= 1) return;
    k_ipc_barrier_phase_block<<<1, 256, 0, stream>>>(++g_ipc_phase_block);
}

// Copy buffer and barrier signal pointers to device
extern "C" void rdep_sync_buffer_ptrs_bf16() {
    cudaMemcpyToSymbol(d_buffer_ptrs_bf16, g_bf16.buffer_ptrs,
                       g_bf16.world * sizeof(void*), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_barrier_signal_ptrs_bf16, g_bf16.barrier_signal_ptrs,
                       g_bf16.world * sizeof(int*), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_my_rank_bf16, &g_bf16.rank, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_world_bf16, &g_bf16.world, sizeof(int), 0, cudaMemcpyHostToDevice);
}

extern "C" void rdep_sync_buffer_ptrs_blockscaled() {
    cudaMemcpyToSymbol(d_buffer_ptrs_block, g_block.buffer_ptrs,
                       g_block.world * sizeof(void*), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_barrier_signal_ptrs_block, g_block.barrier_signal_ptrs,
                       g_block.world * sizeof(int*), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_my_rank_block, &g_block.rank, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_world_block, &g_block.world, sizeof(int), 0, cudaMemcpyHostToDevice);
}

// ============================================================================
// BF16 Dispatch Kernel
// Each warp handles one (token, slot) pair
// ============================================================================
__global__ void k_dispatch_bf16(
    const __nv_bfloat16* __restrict__ x,  // [T, H] - NOT expanded
    const int* __restrict__ eids,          // [T, K] - expert IDs
    const float* __restrict__ gates,       // [T, K] - gate values
    int my_rank, int T, int H, int Ha, int K,
    int n_local, int capacity,
    size_t meta_off, size_t counter_off, size_t dropped_off)
{
    // Each warp processes one (tok, slot) pair
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
        bool is_remote = (dest != my_rank);

        // Get destination buffer pointer
        char* dest_buf = static_cast<char*>(d_buffer_ptrs_bf16[dest]);
        uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
        Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + meta_off);
        int* counter = reinterpret_cast<int*>(dest_buf + counter_off);
        int* dropped = reinterpret_cast<int*>(dest_buf + dropped_off);

        // One warp, one slot - leader does atomic
        int slot_r;
        if (lane == 0)
            slot_r = atomicAdd(counter, 1);
        slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

        if (slot_r >= capacity) {
            if (lane == 0)
                atomicAdd(dropped, 1);
            continue;
        }

        // Write metadata
        if (lane == 0) {
            Meta m{encode_rid(my_rank, tok, slot, T, K), local_eid, gate};
            if (is_remote) {
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&m);
                st_relaxed_sys_v4_s32(meta_dst, meta_val);  // sys-scope for cross-GPU
            } else {
                meta_buf[slot_r] = m;
            }
        }

        // Write BF16 payload - each lane handles H/32 elements
        const __nv_bfloat16* row = x + (int64_t)tok * H;  // Read from original [T,H]
        uint16_t* dst = x_buf + (int64_t)slot_r * Ha;

        if (is_remote) {
            // Vectorized P2P writes - sys-scope for cross-GPU visibility
            int h = lane * 8;  // Each lane starts at different offset
            for (; h < H; h += 32 * 8) {
                if (h + 8 <= H) {
                    int4* d = reinterpret_cast<int4*>(dst + h);
                    int4 v = *reinterpret_cast<const int4*>(row + h);
                    st_relaxed_sys_v4_s32(d, v);
                } else {
                    // Handle tail
                    for (int hh = h; hh < H && hh < h + 8; hh++) {
                        st_relaxed_sys_b16(reinterpret_cast<uint16_t*>(dst + hh),
                                           *reinterpret_cast<const uint16_t*>(row + hh));
                    }
                }
            }
        } else {
            // Local copy - vectorized int4 (same pattern as remote, but regular stores)
            int h = lane * 8;
            for (; h < H; h += 32 * 8) {
                if (h + 8 <= H) {
                    int4* d = reinterpret_cast<int4*>(dst + h);
                    int4 v = *reinterpret_cast<const int4*>(row + h);
                    *d = v;  // Regular store (no NA hint needed for local)
                } else {
                    // Handle tail
                    for (int hh = h; hh < H && hh < h + 8; hh++)
                        dst[hh] = reinterpret_cast<const uint16_t*>(row)[hh];
                }
            }
        }
    }

    // Fence to ensure all sys-scope writes are visible before kernel completes
    fence_acq_rel_sys();
}

// ============================================================================
// 2-Phase Dispatch: Count tokens per destination (Phase 1)
// No remote atomics - just counts locally, then writes counts to each dest
// ============================================================================
__global__ void k_count_dispatch_bf16(
    const int* __restrict__ eids,          // [T, K] - expert IDs
    int T, int K, int n_local,
    size_t recv_counts_off)
{
    // Each warp counts tokens for different destination ranges
    // Use block-level reduction to aggregate counts
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int M = T * K;
    int my_rank = d_my_rank_bf16;
    int world = d_world_bf16;

    // Shared memory for per-destination counts
    extern __shared__ int shared_counts[];

    // Initialize shared counts
    if (threadIdx.x < world) {
        shared_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    // Each thread processes multiple (tok, slot) pairs and accumulates to shared
    for (int i = tid; i < M; i += gridDim.x * blockDim.x) {
        int tok = i / K;
        int slot = i % K;
        int eid = eids[tok * K + slot];
        int dest = eid / n_local;
        if (dest >= 0 && dest < world) {
            atomicAdd(&shared_counts[dest], 1);  // Local atomic only
        }
    }
    __syncthreads();

    // Block leader writes to global (one block aggregates, then atomicAdd to global)
    // Actually, we need a global counter per dest. Use the recv_counts area of MY buffer,
    // then write to dests after all blocks finish.
    // Simpler: use atomicAdd to a local send_counts array, then one kernel writes to dests.
    // Let's use the recv_counts area of our own buffer temporarily as send_counts.
    if (threadIdx.x < world) {
        char* my_buf = static_cast<char*>(d_buffer_ptrs_bf16[my_rank]);
        int* my_send_counts = reinterpret_cast<int*>(my_buf + recv_counts_off);
        atomicAdd(&my_send_counts[threadIdx.x], shared_counts[threadIdx.x]);
    }
}

// Write send counts to each destination's recv_counts area (after k_count_dispatch completes)
__global__ void k_write_counts_to_dests_bf16(size_t recv_counts_off) {
    int my_rank = d_my_rank_bf16;
    int world = d_world_bf16;
    int dest = threadIdx.x;

    if (dest >= world) return;

    // Read my send count to dest
    char* my_buf = static_cast<char*>(d_buffer_ptrs_bf16[my_rank]);
    int* my_send_counts = reinterpret_cast<int*>(my_buf + recv_counts_off);
    int count = my_send_counts[dest];

    // Write to dest's recv_counts[my_rank]
    char* dest_buf = static_cast<char*>(d_buffer_ptrs_bf16[dest]);
    int* dest_recv_counts = reinterpret_cast<int*>(dest_buf + recv_counts_off);

    if (dest != my_rank) {
        // P2P write - sys-scope for cross-GPU visibility
        st_relaxed_sys_s32(&dest_recv_counts[my_rank], count);
    } else {
        dest_recv_counts[my_rank] = count;
    }

    // Fence to ensure writes are sys-visible before kernel completes
    fence_acq_rel_sys();
}

// Compute prefix sums from recv_counts and write offsets back to sources
__global__ void k_compute_and_write_offsets_bf16(size_t recv_counts_off, size_t recv_offsets_off) {
    int my_rank = d_my_rank_bf16;
    int world = d_world_bf16;

    // Only one thread does this (simple serial prefix sum for small world)
    if (threadIdx.x != 0) return;

    char* my_buf = static_cast<char*>(d_buffer_ptrs_bf16[my_rank]);
    int* recv_counts = reinterpret_cast<int*>(my_buf + recv_counts_off);
    int* recv_offsets = reinterpret_cast<int*>(my_buf + recv_offsets_off);

    // Compute prefix sums
    int offset = 0;
    for (int src = 0; src < world; ++src) {
        recv_offsets[src] = offset;
        offset += recv_counts[src];
    }

    // Write offsets back to each source's buffer so they know where to write
    // Source src writes at our recv_offsets[src]
    for (int src = 0; src < world; ++src) {
        char* src_buf = static_cast<char*>(d_buffer_ptrs_bf16[src]);
        int* src_recv_offsets = reinterpret_cast<int*>(src_buf + recv_offsets_off);

        if (src != my_rank) {
            // P2P write - sys-scope for cross-GPU visibility
            st_relaxed_sys_s32(&src_recv_offsets[my_rank], recv_offsets[src]);
        } else {
            src_recv_offsets[my_rank] = recv_offsets[src];
        }
    }

    // Fence to ensure writes are sys-visible before kernel completes
    fence_acq_rel_sys();
}

// 2-Phase Dispatch: Deterministic write (Phase 2)
// Uses pre-computed offsets, LOCAL atomics only for ordering within a rank's batch
__global__ void k_dispatch_2phase_bf16(
    const __nv_bfloat16* __restrict__ x,  // [T, H]
    const int* __restrict__ eids,          // [T, K]
    const float* __restrict__ gates,       // [T, K]
    int* __restrict__ local_counters,      // [world] - local atomic counters (device memory)
    int T, int H, int Ha, int K,
    int n_local, int capacity,
    size_t meta_off, size_t recv_offsets_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    int M = T * K;
    int my_rank = d_my_rank_bf16;
    int world = d_world_bf16;

    // Read my starting offsets from each destination (written by k_compute_and_write_offsets)
    char* my_buf = static_cast<char*>(d_buffer_ptrs_bf16[my_rank]);
    int* my_recv_offsets = reinterpret_cast<int*>(my_buf + recv_offsets_off);

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = i / K;
        int slot = i % K;

        int eid = eids[tok * K + slot];
        float gate = gates[tok * K + slot];
        int dest = eid / n_local;
        int local_eid = eid % n_local;
        bool is_remote = (dest != my_rank);

        // Get local offset within this rank's batch to dest (LOCAL atomic only)
        int local_idx;
        if (lane == 0) {
            local_idx = atomicAdd(&local_counters[dest], 1);
        }
        local_idx = __shfl_sync(0xFFFFFFFF, local_idx, 0);

        // Read where this rank's data starts at dest
        int base_offset = my_recv_offsets[dest];
        int slot_r = base_offset + local_idx;

        if (slot_r >= capacity) continue;  // Overflow protection

        // Get destination buffer
        char* dest_buf = static_cast<char*>(d_buffer_ptrs_bf16[dest]);
        uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
        Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + meta_off);

        // Write metadata
        if (lane == 0) {
            Meta m{encode_rid(my_rank, tok, slot, T, K), local_eid, gate};
            if (is_remote) {
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&m);
                st_relaxed_sys_v4_s32(meta_dst, meta_val);  // sys-scope for cross-GPU visibility
            } else {
                meta_buf[slot_r] = m;
            }
        }

        // Write BF16 payload
        const __nv_bfloat16* row = x + (int64_t)tok * H;
        uint16_t* dst = x_buf + (int64_t)slot_r * Ha;

        if (is_remote) {
            for (int h = lane * 8; h < H; h += 32 * 8) {
                if (h + 8 <= H) {
                    int4* d = reinterpret_cast<int4*>(dst + h);
                    int4 v = *reinterpret_cast<const int4*>(row + h);
                    st_relaxed_sys_v4_s32(d, v);  // sys-scope for cross-GPU visibility
                } else {
                    for (int hh = h; hh < H && hh < h + 8; hh++) {
                        st_relaxed_sys_b16(reinterpret_cast<uint16_t*>(dst + hh),
                                           *reinterpret_cast<const uint16_t*>(row + hh));
                    }
                }
            }
        } else {
            for (int h = lane * 8; h < H; h += 32 * 8) {
                if (h + 8 <= H) {
                    int4* d = reinterpret_cast<int4*>(dst + h);
                    int4 v = *reinterpret_cast<const int4*>(row + h);
                    *d = v;
                } else {
                    for (int hh = h; hh < H && hh < h + 8; hh++)
                        dst[hh] = reinterpret_cast<const uint16_t*>(row)[hh];
                }
            }
        }
    }

    // Ensure all sys-scope writes are visible before kernel completes.
    // The barrier kernel's fence cannot order writes from this kernel since
    // they are in different threads' program order. This fence must happen
    // HERE, in the kernel that did the writes.
    fence_acq_rel_sys();
}

// ============================================================================
// Blockscaled Dispatch Kernel
// Each warp handles one (token, slot) pair
// Quantization happens in registers, writes directly to remote buffer
// ============================================================================
__global__ void k_dispatch_blockscaled(
    const __nv_bfloat16* __restrict__ x,  // [T, H] - NOT expanded
    const int* __restrict__ eids,          // [T, K] - expert IDs
    const float* __restrict__ gates,       // [T, K] - gate values
    int my_rank, int T, int H, int Hp, int Hsf, int K,
    int n_local, int capacity, int profile,
    size_t sfa_off, size_t meta_off, size_t counter_off, size_t dropped_off)
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
        bool is_remote = (dest != my_rank);

        // Get destination buffer pointers
        char* dest_buf = static_cast<char*>(d_buffer_ptrs_block[dest]);
        uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
        uint8_t* sfa_buf = reinterpret_cast<uint8_t*>(dest_buf + sfa_off);
        Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + meta_off);
        int* counter = reinterpret_cast<int*>(dest_buf + counter_off);
        int* dropped = reinterpret_cast<int*>(dest_buf + dropped_off);

        // One warp, one slot - leader does atomic
        int slot_r;
        if (lane == 0)
            slot_r = atomicAdd(counter, 1);
        slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

        if (slot_r >= capacity) {
            if (lane == 0)
                atomicAdd(dropped, 1);
            continue;
        }

        // Write metadata
        if (lane == 0) {
            Meta m{encode_rid(my_rank, tok, slot, T, K), local_eid, gate};
            if (is_remote) {
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&m);
                st_na_v4_s32(meta_dst, meta_val);
            } else {
                meta_buf[slot_r] = m;
            }
        }

        // Pointers to destination row
        const __nv_bfloat16* row = x + (int64_t)tok * H;
        uint16_t* dst_pack = x_buf + (int64_t)slot_r * Hp;
        uint8_t* dst_sfa = sfa_buf + (int64_t)slot_r * Hsf;

        // Process each 32-element block
        // Lanes 0-31 each handle one element within the block
        for (int blk = 0; blk < Hsf; blk++) {
            int h0 = blk * SF_VEC;  // SF_VEC = 32
            int h_end = min(h0 + SF_VEC, H);
            int blk_size = h_end - h0;

            // Each lane loads its element (if within bounds)
            float val = 0.0f;
            if (lane < blk_size) {
                val = __bfloat162float(row[h0 + lane]);
            }

            // Warp reduction for max absolute value
            float local_amax = fabsf(val);
            float blk_amax = warp_reduce_max(local_amax);

            // Compute and broadcast scale
            float scale = blk_amax / dtype_max;
            if (!(scale > 0.0f)) scale = 1.0f;
            uint8_t scale_byte = e8m0_encode(scale);
            float s = e8m0_decode(scale_byte);
            float inv_scale = (s > 0.0f) ? (1.0f / s) : 1.0f;

            // Write scale factor (lane 0 only)
            if (lane == 0) {
                if (is_remote) {
                    st_na_relaxed_gpu_b8(dst_sfa + blk, scale_byte);
                } else {
                    dst_sfa[blk] = scale_byte;
                }
            }

            // Quantize value in register
            float qf = val * inv_scale;

            if (profile == 0) {
                // FP8 E4M3: pack 2 values into uint16
                // Lanes 0,1 -> pack[0], lanes 2,3 -> pack[1], etc.
                uint8_t q8 = to_fp8(qf);

                // Get neighbor's quantized value via shuffle
                uint8_t q8_neighbor = __shfl_xor_sync(0xFFFFFFFF, q8, 1);

                // Even lanes pack [self, neighbor], odd lanes idle
                if ((lane & 1) == 0 && lane < blk_size) {
                    uint16_t packed = (uint16_t)q8 | ((uint16_t)q8_neighbor << 8);
                    int pack_idx = blk * (SF_VEC / 2) + (lane / 2);
                    if (is_remote) {
                        st_na_relaxed_gpu_b16(dst_pack + pack_idx, packed);
                    } else {
                        dst_pack[pack_idx] = packed;
                    }
                }
            } else {
                // NVFP4 E2M1: pack 4 values into uint16
                // Lanes 0,1,2,3 -> pack[0], lanes 4,5,6,7 -> pack[1], etc.
                // First get all 4 quantized values via shuffles
                float qf0 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 0);
                float qf1 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 1);
                float qf2 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 2);
                float qf3 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 3);

                // Only lane 0 of each group of 4 writes
                if ((lane & 3) == 0 && lane < blk_size) {
                    uint16_t packed = to_fp4x4(qf0, qf1, qf2, qf3);
                    int pack_idx = blk * (SF_VEC / 4) + (lane / 4);
                    if (is_remote) {
                        st_na_relaxed_gpu_b16(dst_pack + pack_idx, packed);
                    } else {
                        dst_pack[pack_idx] = packed;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Sort + Gather Kernels
// ============================================================================

__global__ void k_extract_local_eid(
    const Meta* __restrict__ meta,
    int* __restrict__ local_eid,
    int* __restrict__ order,
    int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        local_eid[i] = meta[i].local_eid;
        order[i] = i;
    }
}

__global__ void k_compute_offsets(
    const int* __restrict__ sorted_eid,
    int* __restrict__ offsets,
    int M, int n_local)
{
    extern __shared__ int hist[];
    int tid = threadIdx.x;

    for (int e = tid; e < n_local; e += blockDim.x)
        hist[e] = 0;
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + tid; i < M; i += gridDim.x * blockDim.x)
        atomicAdd(&hist[sorted_eid[i]], 1);
    __syncthreads();

    if (blockIdx.x == 0 && tid == 0) {
        offsets[0] = 0;
        for (int e = 0; e < n_local; e++)
            offsets[e + 1] = offsets[e] + hist[e];
    }
}

__global__ void k_compute_padded_mapping(
    const int* __restrict__ offsets,
    int* __restrict__ offs_pad,
    int* __restrict__ dest,
    int* __restrict__ M_pad_out,
    int M, int n_local, int align)
{
    extern __shared__ int sh[];
    int* cnt = sh;
    int* cnt_pad = sh + n_local;
    int* starts_pad = sh + 2*n_local;
    int tid = threadIdx.x;

    for (int e = tid; e < n_local; e += blockDim.x) {
        int c = offsets[e + 1] - offsets[e];
        cnt[e] = c;
        cnt_pad[e] = ((c + align - 1) / align) * align;
    }
    __syncthreads();

    if (tid == 0) {
        int sum = 0;
        for (int e = 0; e < n_local; e++) {
            starts_pad[e] = sum;
            sum += cnt_pad[e];
        }
        *M_pad_out = sum;

        sum = 0;
        for (int e = 0; e < n_local; e++) {
            sum += cnt_pad[e];
            offs_pad[e] = sum;
        }
    }
    __syncthreads();

    for (int i = tid; i < M; i += blockDim.x) {
        int lo = 0, hi = n_local;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (offsets[mid + 1] <= i) lo = mid + 1;
            else hi = mid;
        }
        dest[i] = starts_pad[lo] + (i - offsets[lo]);
    }
}

__global__ void k_gather_bf16(
    const uint16_t* __restrict__ x_recv,
    const int* __restrict__ order,
    const int* __restrict__ dest,
    __nv_bfloat16* __restrict__ Xe_out,
    int M, int H, int Ha)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    // Use int4 vectorized copy for better bandwidth (H/8 int4s = H/2 BF16s per int4)
    int hidden_int4 = H / 8;

    for (int sorted_i = warp_id; sorted_i < M; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        const int4* src = reinterpret_cast<const int4*>(x_recv + (int64_t)orig_i * Ha);
        int out_i = (dest != nullptr) ? dest[sorted_i] : sorted_i;
        int4* dst = reinterpret_cast<int4*>(Xe_out + (int64_t)out_i * H);

        // Use UNROLLED_WARP_COPY for efficient vectorized copy
        UNROLLED_WARP_COPY(4, lane, hidden_int4, dst, src, ld_nc_v4_s32, st_na_global);

        // Handle remaining elements if H not divisible by 8
        int remaining_start = hidden_int4 * 8;
        const __nv_bfloat16* src_bf16 = reinterpret_cast<const __nv_bfloat16*>(x_recv + (int64_t)orig_i * Ha);
        __nv_bfloat16* dst_bf16 = Xe_out + (int64_t)out_i * H;
        for (int h = remaining_start + lane; h < H; h += 32)
            dst_bf16[h] = src_bf16[h];
    }
}

__global__ void k_gather_blockscaled(
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
        for (int hp = lane; hp < Hp; hp += 32)
            dst[hp] = src[hp];

        const uint8_t* sfa_src = sfa_recv + (int64_t)orig_i * Hsf;
        uint8_t* sfa_dst = sfa_out + (int64_t)out_i * Hsf;
        for (int sf = lane; sf < Hsf; sf += 32)
            sfa_dst[sf] = sfa_src[sf];
    }
}

__global__ void k_gather_meta_sorted(
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

__global__ void k_gather_from_pad_bf16(
    const __nv_bfloat16* __restrict__ in_pad,
    const int* __restrict__ dest,
    __nv_bfloat16* __restrict__ out_sorted,
    int M, int H)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    // Use int4 vectorized copy for better bandwidth (H/8 int4s = H/2 BF16s per int4)
    int hidden_int4 = H / 8;

    for (int sorted_i = warp_id; sorted_i < M; sorted_i += num_warps) {
        int pad_i = dest[sorted_i];
        const int4* src = reinterpret_cast<const int4*>(in_pad + (int64_t)pad_i * H);
        int4* dst = reinterpret_cast<int4*>(out_sorted + (int64_t)sorted_i * H);

        UNROLLED_WARP_COPY(4, lane, hidden_int4, dst, src, ld_nc_v4_s32, st_na_global);

        // Handle remaining elements if H not divisible by 8
        int remaining_start = hidden_int4 * 8;
        const __nv_bfloat16* src_bf16 = in_pad + (int64_t)pad_i * H;
        __nv_bfloat16* dst_bf16 = out_sorted + (int64_t)sorted_i * H;
        for (int h = remaining_start + lane; h < H; h += 32)
            dst_bf16[h] = src_bf16[h];
    }
}

__global__ void k_scatter_sorted_to_pad_bf16(
    const __nv_bfloat16* __restrict__ in_sorted,
    const int* __restrict__ dest,
    __nv_bfloat16* __restrict__ out_pad,
    int M, int H)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    int hidden_int4 = H / 8;

    for (int sorted_i = warp_id; sorted_i < M; sorted_i += num_warps) {
        int pad_i = dest[sorted_i];
        const int4* src = reinterpret_cast<const int4*>(in_sorted + (int64_t)sorted_i * H);
        int4* dst = reinterpret_cast<int4*>(out_pad + (int64_t)pad_i * H);

        UNROLLED_WARP_COPY(4, lane, hidden_int4, dst, src, ld_nc_v4_s32, st_na_global);

        int remaining_start = hidden_int4 * 8;
        const __nv_bfloat16* src_bf16 = in_sorted + (int64_t)sorted_i * H;
        __nv_bfloat16* dst_bf16 = out_pad + (int64_t)pad_i * H;
        for (int h = remaining_start + lane; h < H; h += 32)
            dst_bf16[h] = src_bf16[h];
    }
}

// ============================================================================
// Host API: BF16 Dispatch
// ============================================================================

extern "C" int rdep_dispatch_meta_bf16(
    const void* x,           // [T, H] - NOT expanded
    const int* eids,         // [T, K] - expert IDs (NOT flattened)
    const float* gates,      // [T, K] - gate values (NOT flattened)
    int T, int K,
    int align,               // Per-expert row padding (8 for BF16, 128 for blockscaled)
    int* offs_pad_out,       // [n_local] device int32
    int* M_pad_out,          // host int32 (pinned recommended). Used as a host scratch for M_recv.
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return -1;
        }
        if (nvshmem::g_nvshmem.profile != -1) {
            fprintf(stderr, "RDEP ERROR: rdep_dispatch_meta_bf16 requires BF16 NVSHMEM state (profile=-1)\n");
            return -2;
        }

        // Reuse the hybrid dispatch pipeline but skip Xe_out materialization.
        return nvshmem::dispatch_hybrid_bf16(
            static_cast<const __nv_bfloat16*>(x),
            eids,
            gates,
            T, K,
            align,
            /*Xe_out=*/nullptr,
            offs_pad_out,
            /*dest_out=*/nullptr,
            /*row_id_out=*/nullptr,
            /*gate_out=*/nullptr,
            M_pad_out,
            nvshmem::g_nvshmem.ipc_buffer_ptrs,
            nvshmem::g_nvshmem.ipc_meta_off,
            nvshmem::g_nvshmem.ipc_counter_off,
            nvshmem::g_nvshmem.ipc_barrier_signal_ptrs,
            stream);
    }
#endif

    if (!g_bf16.initialized) {
        fprintf(stderr, "RDEP ERROR: BF16 buffers not initialized\n");
        return -1;
    }

    if (g_bf16.H % 8 != 0) {
        fprintf(stderr, "RDEP ERROR: H=%d must be multiple of 8 for vectorized copies\n", g_bf16.H);
        return -2;
    }

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);

    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    int capacity = static_cast<int>(g_bf16.capacity);
    int M = T * K;

    cudaMemsetAsync(local_buf + counter_off, 0, sizeof(int), stream);
    cudaMemsetAsync(local_buf + dropped_off, 0, sizeof(int), stream);

    ipc_barrier_bf16(stream);

    int warps_needed = M;
    int threads = 256;
    int blocks = std::max(1, (warps_needed * 32 + threads - 1) / threads);

    k_dispatch_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), eids, gates,
        g_bf16.rank, T, g_bf16.H, g_bf16.Ha, K,
        g_bf16.n_local, capacity,
        meta_off, counter_off, dropped_off);

    ipc_barrier_bf16(stream);

    // Read back M_recv (host sync point). Use pinned host memory when available.
    if (M_pad_out == nullptr) {
        fprintf(stderr, "RDEP ERROR: M_pad_out (host scratch) is null\n");
        return -3;
    }
    cudaMemcpyAsync(M_pad_out, local_buf + counter_off, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int M_recv = *M_pad_out;

    if (M_recv <= 0) {
        cudaMemsetAsync(offs_pad_out, 0, g_bf16.n_local * sizeof(int), stream);
        *M_pad_out = 0;
        return 0;
    }
    M_recv = std::min(M_recv, capacity);

    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);

    k_extract_local_eid<<<(M_recv + 255) / 256, 256, 0, stream>>>(
        meta_buf, g_bf16.local_eid, g_bf16.order, M_recv);

    cub::DeviceRadixSort::SortPairs(g_bf16.sort_temp, g_bf16.sort_temp_bytes,
        g_bf16.local_eid, g_bf16.local_eid, g_bf16.order, g_bf16.order, M_recv, 0, 32, stream);

    k_compute_offsets<<<1, 256, g_bf16.n_local * sizeof(int), stream>>>(
        g_bf16.local_eid, g_bf16.offsets, M_recv, g_bf16.n_local);

    size_t pad_smem = 3 * g_bf16.n_local * sizeof(int);
    k_compute_padded_mapping<<<1, 256, pad_smem, stream>>>(
        g_bf16.offsets, offs_pad_out, g_bf16.dest, g_bf16.M_pad_dev,
        M_recv, g_bf16.n_local, align);
    return M_recv;
}

extern "C" void rdep_gather_xe_bf16(
    void* Xe_out,           // [M_pad, H] BF16
    int M_recv,
    int M_pad,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return;
        }
        if (nvshmem::g_nvshmem.profile != -1) {
            fprintf(stderr, "RDEP ERROR: rdep_gather_xe_bf16 requires BF16 NVSHMEM state (profile=-1)\n");
            return;
        }
        if (M_recv <= 0 || M_pad <= 0) return;

        const int H = nvshmem::g_nvshmem.H;
        const int Ha = nvshmem::g_nvshmem.Ha;
        char* local_ipc_buf = static_cast<char*>(nvshmem::g_nvshmem.ipc_buffer_ptrs[nvshmem::g_nvshmem.nvl_rank]);
        uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_ipc_buf + nvshmem::g_nvshmem.ipc_x_off);

        cudaMemsetAsync(Xe_out, 0, (size_t)M_pad * (size_t)H * sizeof(__nv_bfloat16), stream);

        int gather_threads = 256;
        int gather_blocks = std::max(1, (M_recv * 32 + gather_threads - 1) / gather_threads);
        k_gather_bf16<<<gather_blocks, gather_threads, 0, stream>>>(
            x_buf, nvshmem::g_nvshmem.order, nvshmem::g_nvshmem.dest,
            static_cast<__nv_bfloat16*>(Xe_out),
            M_recv, H, Ha);
        return;
    }
#endif
    if (!g_bf16.initialized) return;
    if (M_recv <= 0 || M_pad <= 0) return;

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);
    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_buf + x_off);

    cudaMemsetAsync(Xe_out, 0, (size_t)M_pad * g_bf16.H * sizeof(__nv_bfloat16), stream);

    int gather_threads = 256;
    int gather_blocks = std::max(1, (M_recv * 32 + gather_threads - 1) / gather_threads);
    k_gather_bf16<<<gather_blocks, gather_threads, 0, stream>>>(
        x_buf, g_bf16.order, g_bf16.dest,
        static_cast<__nv_bfloat16*>(Xe_out),
        M_recv, g_bf16.H, g_bf16.Ha);
}

extern "C" void rdep_gather_meta_sorted_bf16(
    int64_t* row_id_out,     // [M_recv] int64 (device)
    float* gate_out,         // [M_recv] float32 (device)
    int M_recv,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return;
        }
        if (nvshmem::g_nvshmem.profile != -1) {
            fprintf(stderr, "RDEP ERROR: rdep_gather_meta_sorted_bf16 requires BF16 NVSHMEM state (profile=-1)\n");
            return;
        }
        if (M_recv <= 0) return;

        char* local_ipc_buf = static_cast<char*>(nvshmem::g_nvshmem.ipc_buffer_ptrs[nvshmem::g_nvshmem.nvl_rank]);
        Meta* meta_buf = reinterpret_cast<Meta*>(local_ipc_buf + nvshmem::g_nvshmem.ipc_meta_off);

        int t = 256;
        int b = (M_recv + t - 1) / t;
        k_gather_meta_sorted<<<b, t, 0, stream>>>(
            meta_buf, nvshmem::g_nvshmem.order, row_id_out, gate_out, M_recv);
        return;
    }
#endif
    if (!g_bf16.initialized) return;
    if (M_recv <= 0) return;

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);
    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);

    int t = 256;
    int b = (M_recv + t - 1) / t;
    k_gather_meta_sorted<<<b, t, 0, stream>>>(
        meta_buf, g_bf16.order, row_id_out, gate_out, M_recv);
}

extern "C" void rdep_gather_from_pad_bf16(
    const void* in_pad,      // [M_pad, H] BF16
    void* out_sorted,        // [M_recv, H] BF16
    int M_recv,
    int H,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return;
        }
        if (nvshmem::g_nvshmem.profile != -1) {
            fprintf(stderr, "RDEP ERROR: rdep_gather_from_pad_bf16 requires BF16 NVSHMEM state (profile=-1)\n");
            return;
        }
        if (H != nvshmem::g_nvshmem.H) {
            fprintf(stderr, "RDEP ERROR: rdep_gather_from_pad_bf16 H mismatch: got H=%d state H=%d\n",
                    H, nvshmem::g_nvshmem.H);
            return;
        }
        if (M_recv <= 0 || H <= 0) return;

        int threads = 256;
        int blocks = std::max(1, (M_recv * 32 + threads - 1) / threads);
        k_gather_from_pad_bf16<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(in_pad),
            nvshmem::g_nvshmem.dest,
            static_cast<__nv_bfloat16*>(out_sorted),
            M_recv, H);
        return;
    }
#endif
    if (!g_bf16.initialized) return;
    if (M_recv <= 0 || H <= 0) return;

    int threads = 256;
    int blocks = std::max(1, (M_recv * 32 + threads - 1) / threads);
    k_gather_from_pad_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in_pad),
        g_bf16.dest,
        static_cast<__nv_bfloat16*>(out_sorted),
        M_recv, H);
}

extern "C" void rdep_scatter_sorted_to_pad_bf16(
    const void* in_sorted,   // [M_recv, H] BF16
    void* out_pad,           // [M_pad, H] BF16
    int M_recv,
    int H,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return;
        }
        if (nvshmem::g_nvshmem.profile != -1) {
            fprintf(stderr, "RDEP ERROR: rdep_scatter_sorted_to_pad_bf16 requires BF16 NVSHMEM state (profile=-1)\n");
            return;
        }
        if (H != nvshmem::g_nvshmem.H) {
            fprintf(stderr, "RDEP ERROR: rdep_scatter_sorted_to_pad_bf16 H mismatch: got H=%d state H=%d\n",
                    H, nvshmem::g_nvshmem.H);
            return;
        }
        if (M_recv <= 0 || H <= 0) return;

        int threads = 256;
        int blocks = std::max(1, (M_recv * 32 + threads - 1) / threads);
        k_scatter_sorted_to_pad_bf16<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(in_sorted),
            nvshmem::g_nvshmem.dest,
            static_cast<__nv_bfloat16*>(out_pad),
            M_recv, H);
        return;
    }
#endif
    if (!g_bf16.initialized) return;
    if (M_recv <= 0 || H <= 0) return;

    int threads = 256;
    int blocks = std::max(1, (M_recv * 32 + threads - 1) / threads);
    k_scatter_sorted_to_pad_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in_sorted),
        g_bf16.dest,
        static_cast<__nv_bfloat16*>(out_pad),
        M_recv, H);
}

// ============================================================================
// 2-Phase Dispatch Implementation (IPC mode only)
// Eliminates remote atomics by exchanging counts before writing data
// ============================================================================
static int dispatch_2phase_bf16(
    const __nv_bfloat16* x,
    const int* eids,
    const float* gates,
    int T, int K, int M,
    size_t meta_off, size_t recv_counts_off, size_t recv_offsets_off,
    int capacity,
    cudaStream_t stream)
{
    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);

    // Phase 1: Count tokens per destination
    // Reset recv_counts (we use it as send_counts first, then it gets overwritten)
    cudaMemsetAsync(local_buf + recv_counts_off, 0, MAX_RANKS * sizeof(int), stream);

    // Reset local_counters for the final write phase
    cudaMemsetAsync(g_bf16.local_counters, 0, MAX_RANKS * sizeof(int), stream);

    // Barrier: ensure all resets complete before counting
    ipc_barrier_bf16(stream);

    // Count tokens per destination
    int count_threads = 256;
    int count_blocks = std::max(1, (M + count_threads - 1) / count_threads);
    k_count_dispatch_bf16<<<count_blocks, count_threads, MAX_RANKS * sizeof(int), stream>>>(
        eids, T, K, g_bf16.n_local, recv_counts_off);

    // Write counts to each destination's buffer
    k_write_counts_to_dests_bf16<<<1, MAX_RANKS, 0, stream>>>(recv_counts_off);

    // Barrier: ensure all counts are visible
    ipc_barrier_bf16(stream);

    // Phase 2: Compute prefix sums and exchange offsets
    k_compute_and_write_offsets_bf16<<<1, 32, 0, stream>>>(recv_counts_off, recv_offsets_off);

    // Barrier: ensure all offsets are visible
    ipc_barrier_bf16(stream);

    // Reset local_counters again (they may have been corrupted by multi-block counting)
    cudaMemsetAsync(g_bf16.local_counters, 0, MAX_RANKS * sizeof(int), stream);

    // Phase 3: Write data at deterministic offsets
    int dispatch_threads = 256;
    int dispatch_warps = M;
    int dispatch_blocks = std::max(1, (dispatch_warps * 32 + dispatch_threads - 1) / dispatch_threads);
    k_dispatch_2phase_bf16<<<dispatch_blocks, dispatch_threads, 0, stream>>>(
        x, eids, gates,
        g_bf16.local_counters,
        T, g_bf16.H, g_bf16.Ha, K,
        g_bf16.n_local, capacity,
        meta_off, recv_offsets_off);

    // Barrier: ensure all data writes are visible
    ipc_barrier_bf16(stream);

    // Compute M_recv from recv_counts (sum of all recv_counts)
    cudaStreamSynchronize(stream);

    int* recv_counts = reinterpret_cast<int*>(local_buf + recv_counts_off);
    int M_recv = 0;
    int h_recv_counts[MAX_RANKS];
    cudaMemcpy(h_recv_counts, recv_counts, g_bf16.world * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < g_bf16.world; ++i) {
        M_recv += h_recv_counts[i];
    }

    return M_recv;
}

extern "C" int rdep_dispatch(
    const void* x,           // [T, H] - NOT expanded
    const int* eids,         // [T, K] - expert IDs (NOT flattened)
    const float* gates,      // [T, K] - gate values (NOT flattened)
    int T, int K,
    void* Xe_out,
    int* offs_pad_out,
    int* dest_out,
    int64_t* row_id_out,
    float* gate_out,
    int* M_pad_out,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    // Hybrid mode: use NVSHMEM for inter-node + IPC for intra-node
    // CRITICAL: Check for hybrid mode FIRST before g_bf16.initialized,
    // because hybrid mode uses g_nvshmem state, not g_bf16 state
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return -1;
        }
        // Alignment check for hybrid mode
        if (nvshmem::g_nvshmem.H % 8 != 0) {
            fprintf(stderr, "RDEP ERROR: H=%d must be multiple of 8 for vectorized copies\n", nvshmem::g_nvshmem.H);
            return -2;
        }
        return nvshmem::dispatch_hybrid_bf16(
            static_cast<const __nv_bfloat16*>(x), eids, gates,
            T, K,
            nvshmem::g_nvshmem.align,  // Use NVSHMEM state's alignment
            Xe_out, offs_pad_out,
            dest_out, row_id_out, gate_out,
            M_pad_out,
            nvshmem::g_nvshmem.ipc_buffer_ptrs,
            nvshmem::g_nvshmem.ipc_meta_off,
            nvshmem::g_nvshmem.ipc_counter_off,
            nvshmem::g_nvshmem.ipc_barrier_signal_ptrs,
            stream);
    }
#endif

    // Single-GPU and IPC modes: use local IPC path
    // Check g_bf16.initialized for non-hybrid mode
    if (!g_bf16.initialized) {
        fprintf(stderr, "RDEP ERROR: BF16 buffers not initialized\n");
        return -1;
    }

    // Alignment assertion: H must be multiple of 8 for vectorized int4 copies
    // (8 BF16 = 16 bytes = sizeof(int4))
    if (g_bf16.H % 8 != 0) {
        fprintf(stderr, "RDEP ERROR: H=%d must be multiple of 8 for vectorized copies\n", g_bf16.H);
        return -2;
    }

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    size_t recv_counts_off, recv_offsets_off;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size,
                        &recv_counts_off, &recv_offsets_off);

    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    int capacity = static_cast<int>(g_bf16.capacity);
    int M = T * K;

    int M_recv = 0;

    // Use 2-phase dispatch for IPC mode with world > 1 (eliminates remote atomics)
    if (g_use_2phase_dispatch && g_bf16.world > 1) {
        M_recv = dispatch_2phase_bf16(
            static_cast<const __nv_bfloat16*>(x), eids, gates,
            T, K, M,
            meta_off, recv_counts_off, recv_offsets_off,
            capacity, stream);
    } else {
        // Legacy atomic-counter dispatch (single GPU or disabled 2-phase)
        // Reset counter (async, no sync needed before dispatch starts)
        cudaMemsetAsync(local_buf + counter_off, 0, sizeof(int), stream);
        cudaMemsetAsync(local_buf + dropped_off, 0, sizeof(int), stream);

        // IPC mode requires a global barrier to avoid counter races:
        // all ranks must reset their receive counters before any rank begins
        // remote atomicAdd() into those counters.
        if (g_bf16.world > 1) {
            ipc_barrier_bf16(stream);
        }

        // Launch fused dispatch kernel (reads [T,H] directly)
        // One warp per (tok, slot) pair
        int warps_needed = M;
        int threads = 256;
        int blocks = std::max(1, (warps_needed * 32 + threads - 1) / threads);

        k_dispatch_bf16<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x), eids, gates,
            g_bf16.rank, T, g_bf16.H, g_bf16.Ha, K,
            g_bf16.n_local, capacity,
            meta_off, counter_off, dropped_off);

        // IPC mode requires a global barrier to ensure all remote writes are
        // complete before reading the local counter and sorting.
        if (g_bf16.world > 1) {
            ipc_barrier_bf16(stream);
        }

        // Single sync after dispatch - required for M_recv readback
        // This is the ONLY required sync in the dispatch path
        cudaStreamSynchronize(stream);

        cudaMemcpy(&M_recv, local_buf + counter_off, sizeof(int), cudaMemcpyDeviceToHost);
    }

    if (M_recv <= 0) {
        cudaMemsetAsync(offs_pad_out, 0, g_bf16.n_local * sizeof(int), stream);
        *M_pad_out = 0;
        return 0;
    }
    M_recv = std::min(M_recv, capacity);

    // Sort and gather pipeline - all async
    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);
    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_buf + x_off);

    k_extract_local_eid<<<(M_recv + 255) / 256, 256, 0, stream>>>(
        meta_buf, g_bf16.local_eid, g_bf16.order, M_recv);

    cub::DeviceRadixSort::SortPairs(g_bf16.sort_temp, g_bf16.sort_temp_bytes,
        g_bf16.local_eid, g_bf16.local_eid, g_bf16.order, g_bf16.order, M_recv, 0, 32, stream);

    k_compute_offsets<<<1, 256, g_bf16.n_local * sizeof(int), stream>>>(
        g_bf16.local_eid, g_bf16.offsets, M_recv, g_bf16.n_local);

    size_t pad_smem = 3 * g_bf16.n_local * sizeof(int);
    k_compute_padded_mapping<<<1, 256, pad_smem, stream>>>(
        g_bf16.offsets, offs_pad_out, g_bf16.dest, g_bf16.M_pad_dev,
        M_recv, g_bf16.n_local, g_bf16.align);

    // Copy M_pad back (need sync for this)
    cudaMemcpyAsync(M_pad_out, g_bf16.M_pad_dev, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (dest_out)
        cudaMemcpyAsync(dest_out, g_bf16.dest, M_recv * sizeof(int), cudaMemcpyDeviceToDevice, stream);

    cudaStreamSynchronize(stream);
    int M_pad = *M_pad_out;

    cudaMemsetAsync(Xe_out, 0, (size_t)M_pad * g_bf16.H * sizeof(__nv_bfloat16), stream);

    int gather_threads = 256;
    int gather_blocks = std::max(1, (M_recv * 32 + gather_threads - 1) / gather_threads);
    k_gather_bf16<<<gather_blocks, gather_threads, 0, stream>>>(
        x_buf, g_bf16.order, g_bf16.dest,
        static_cast<__nv_bfloat16*>(Xe_out),
        M_recv, g_bf16.H, g_bf16.Ha);

    if (row_id_out != nullptr && gate_out != nullptr) {
        int t = 256;
        int b = (M_recv + t - 1) / t;
        k_gather_meta_sorted<<<b, t, 0, stream>>>(
            meta_buf, g_bf16.order, row_id_out, gate_out, M_recv);
    }

    g_bf16.M_pad = M_pad;
    return M_recv;
}

// ============================================================================
// Host API: Blockscaled Dispatch
// ============================================================================

extern "C" int rdep_dispatch_blockscaled(
    const void* x,          // [T, H] BF16 - NOT expanded
    const int* eids,        // [T, K] expert IDs
    const float* gates,     // [T, K] gate values
    int T, int K,
    void* Xe_q_out,
    void* Xe_sf_out,
    int* offs_pad_out,
    int* dest_out,
    int64_t* row_id_out,
    float* gate_out,
    int* M_pad_out,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    // Hybrid mode: use NVSHMEM for inter-node + IPC for intra-node
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return -1;
        }
        return nvshmem::dispatch_hybrid_blockscaled(
            static_cast<const __nv_bfloat16*>(x), eids, gates,
            T, K, Xe_q_out, Xe_sf_out,
            offs_pad_out, dest_out,
            row_id_out, gate_out,
            M_pad_out,
            nvshmem::g_nvshmem.ipc_buffer_ptrs,
            nvshmem::g_nvshmem.ipc_x_off,
            nvshmem::g_nvshmem.ipc_sfa_off,
            nvshmem::g_nvshmem.ipc_meta_off,
            nvshmem::g_nvshmem.ipc_counter_off,
            nvshmem::g_nvshmem.ipc_barrier_signal_ptrs,
            stream);
    }
#endif

    if (!g_block.initialized) return -1;

    // Single-GPU and IPC modes: use local IPC path
    int M = T * K;
    size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, g_block.world,
                               &x_off, &sfa_off, &y_off, &meta_off, &counter_off, &dropped_off,
                               &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                               &tok_y_off, &tok_gate_off,
                               &total_size);

    char* local_buf = static_cast<char*>(g_block.buffer_ptrs[g_block.rank]);
    cudaMemsetAsync(local_buf + counter_off, 0, sizeof(int), stream);
    cudaMemsetAsync(local_buf + dropped_off, 0, sizeof(int), stream);

    // No cudaStreamSynchronize here - fused kernel handles ordering

    // IPC mode requires a global barrier to avoid counter races (same as BF16).
    ipc_barrier_block(stream);

    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = std::max(1, (M + warps_per_block - 1) / warps_per_block);
    int capacity = static_cast<int>(g_block.capacity);

    // NO shared memory needed - this is the key benefit of the fused version
    k_dispatch_blockscaled<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), eids, gates,
        g_block.rank, T, g_block.H, g_block.Hp, g_block.Hsf, K,
        g_block.n_local, capacity, g_block.profile,
        sfa_off, meta_off, counter_off, dropped_off);

    ipc_barrier_block(stream);

    if (M_pad_out == nullptr) {
        fprintf(stderr, "RDEP ERROR: M_pad_out (host scratch) is null\n");
        return -3;
    }
    cudaMemcpyAsync(M_pad_out, local_buf + counter_off, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int M_recv = *M_pad_out;

    if (M_recv <= 0) {
        cudaMemsetAsync(offs_pad_out, 0, g_block.n_local * sizeof(int), stream);
        *M_pad_out = 0;
        return 0;
    }
    M_recv = std::min(M_recv, capacity);

    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);
    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_buf + x_off);
    uint8_t* sfa_buf = reinterpret_cast<uint8_t*>(local_buf + sfa_off);

    k_extract_local_eid<<<(M_recv + 255) / 256, 256, 0, stream>>>(
        meta_buf, g_block.local_eid, g_block.order, M_recv);

    cub::DeviceRadixSort::SortPairs(g_block.sort_temp, g_block.sort_temp_bytes,
        g_block.local_eid, g_block.local_eid, g_block.order, g_block.order, M_recv, 0, 32, stream);

    k_compute_offsets<<<1, 256, g_block.n_local * sizeof(int), stream>>>(
        g_block.local_eid, g_block.offsets, M_recv, g_block.n_local);

    size_t pad_smem = 3 * g_block.n_local * sizeof(int);
    k_compute_padded_mapping<<<1, 256, pad_smem, stream>>>(
        g_block.offsets, offs_pad_out, g_block.dest, g_block.M_pad_dev,
        M_recv, g_block.n_local, g_block.align);

    if (dest_out)
        cudaMemcpyAsync(dest_out, g_block.dest, M_recv * sizeof(int), cudaMemcpyDeviceToDevice, stream);

    // Avoid a second host sync for exact M_pad:
    // - Exact padded total is sum_e align_up(cnt_e, align) and depends on routing.
    // - For blockscaled grouped GEMM we only need per-expert offsets to be aligned.
    // - Over-approximate to a deterministic upper bound and extend the last expert.
    //
    // Upper bound (aligned, >= exact): floor((M_recv + n_local*(align-1)) / align) * align.
    int M_pad_bound = M_recv + g_block.n_local * (g_block.align - 1);
    int M_pad = (M_pad_bound / g_block.align) * g_block.align;
    if (g_block.n_local > 0) {
        cudaMemcpyAsync(offs_pad_out + (g_block.n_local - 1), &M_pad, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }
    *M_pad_out = M_pad;

    cudaMemsetAsync(Xe_q_out, 0, (size_t)M_pad * g_block.Hp * sizeof(uint16_t), stream);
    // Ensure padding rows have deterministic, safe SF (scale=1.0 -> e8m0 byte 127).
    // k_gather_blockscaled only writes SF for real rows (M_recv), so we prefill.
    cudaMemsetAsync(g_block.sfa_gather_tmp, 127, (size_t)M_pad * g_block.Hsf * sizeof(uint8_t), stream);

    int warps_needed = M_recv;
    int gather_threads = 256;
    int gather_blocks = std::max(1, (warps_needed * 32 + gather_threads - 1) / gather_threads);
    // Gather packed activations to Xe_q_out and rowwise SFA to a temporary buffer
    k_gather_blockscaled<<<gather_blocks, gather_threads, 0, stream>>>(
        x_buf, sfa_buf, g_block.order, g_block.dest,
        static_cast<uint16_t*>(Xe_q_out), static_cast<uint8_t*>(g_block.sfa_gather_tmp),
        M_recv, g_block.Hp, g_block.Hsf);

    g_block.M_pad = M_pad;

    // Build offsets with leading 0 into device buffer
    // offs_pad_out is length E; offs_with0[0]=0, offs_with0[1:]=offs_pad_out[:]
    cudaMemsetAsync(g_block.offs_with0, 0, sizeof(int), stream);
    cudaMemcpyAsync(g_block.offs_with0 + 1, offs_pad_out,
                    g_block.n_local * sizeof(int), cudaMemcpyDeviceToDevice, stream);

    // Fixed per-expert stride for swizzled SFA output.
    // Must match the Python-side view/stride (capacity-aligned).
    int M_e_swizzle = ((static_cast<int>(g_block.capacity) + 127) / 128) * 128;
    int sf_k = g_block.Hsf;
    int sf_k_pad = ((sf_k + 3) / 4) * 4;

    // Swizzle rowwise SFA into per-expert MMA layout directly into Xe_sf_out
    swizzle_sf_strided(
        static_cast<const void*>(g_block.sfa_gather_tmp),
        static_cast<void*>(Xe_sf_out),
        g_block.offs_with0,
        g_block.n_local, sf_k, sf_k_pad, M_pad, M_e_swizzle,
        stream);

    if (row_id_out != nullptr && gate_out != nullptr) {
        int t = 256;
        int b = (M_recv + t - 1) / t;
        k_gather_meta_sorted<<<b, t, 0, stream>>>(
            meta_buf, g_block.order, row_id_out, gate_out, M_recv);
    }

    return M_recv;
}

// ============================================================================
// Return Scatter (IPC version)
// ============================================================================

// BF16 return scatter kernel (uses d_buffer_ptrs_bf16)
__global__ void k_return_scatter_bf16(
    const __nv_bfloat16* __restrict__ Ye,
    const int* __restrict__ order,
    const Meta* __restrict__ meta,
    float* __restrict__ out,
    int M_recv, int H, int Ha, int T, int K,
    int my_rank, int world, int capacity,
    size_t meta_off, size_t counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        // Bounds check on orig_i
        if (orig_i < 0 || orig_i >= capacity) {
            if (lane == 0) printf("RDEP BUG k_return_scatter: orig_i=%d out of bounds [0,%d)\n", orig_i, capacity);
            continue;
        }
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

        // Bounds check on decoded values
        if (src_rank < 0 || src_rank >= world) {
            if (lane == 0) printf("RDEP BUG k_return_scatter: src_rank=%d out of bounds [0,%d), row_id=%lld\n", src_rank, world, (long long)m.row_id);
            continue;
        }
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) {
            if (lane == 0) printf("RDEP BUG k_return_scatter: tok=%d slot=%d out of bounds T=%d K=%d\n", tok, slot, T, K);
            continue;
        }

        if (src_rank == my_rank) {
            // Local: scatter directly
            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            float* out_row = out + (int64_t)tok * H;
            for (int h = lane; h < H; h += 32)
                atomicAdd(out_row + h, __bfloat162float(y_row[h]) * m.gate);
        } else {
            // Remote: write to source rank's buffer via IPC (BF16 buffers)
            char* dest_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
            uint16_t* y_buf = reinterpret_cast<uint16_t*>(dest_buf);
            Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + meta_off);
            int* counter = reinterpret_cast<int*>(dest_buf + counter_off);

            int slot_r;
            if (lane == 0)
                slot_r = atomicAdd(counter, 1);
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) continue;

            // Write metadata (16B) - sys-scope for cross-GPU visibility.
            if (lane == 0) {
                Meta mr{m.row_id, 0, m.gate};
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&mr);
                st_relaxed_sys_v4_s32(meta_dst, meta_val);
            }

            // Write BF16 payload - sys-scope for cross-GPU visibility.
            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            uint16_t* dst = y_buf + (int64_t)slot_r * Ha;

            int h = lane * 8;
            for (; h < H; h += 32 * 8) {
                if (h + 8 <= H) {
                    int4* d = reinterpret_cast<int4*>(dst + h);
                    int4 v = *reinterpret_cast<const int4*>(y_row + h);
                    st_relaxed_sys_v4_s32(d, v);
                } else {
                    for (int hh = h; hh < H && hh < h + 8; hh++) {
                        st_relaxed_sys_b16(dst + hh, reinterpret_cast<const uint16_t*>(y_row)[hh]);
                    }
                }
            }
        }
    }

    // Fence to ensure all sys-scope writes are visible before kernel completes
    fence_acq_rel_sys();
}

// Blockscaled return scatter kernel (uses d_buffer_ptrs_block)
__global__ void k_return_scatter_blockscaled_bf16(
    const __nv_bfloat16* __restrict__ Ye,
    const int* __restrict__ order,
    const Meta* __restrict__ meta,
    float* __restrict__ out,
    int M_recv, int H, int Ha, int T, int K,
    int my_rank, int world, int capacity,
    size_t y_off, size_t meta_off, size_t counter_off)
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

        if (src_rank == my_rank) {
            // Local: scatter directly
            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            float* out_row = out + (int64_t)tok * H;
            for (int h = lane; h < H; h += 32)
                atomicAdd(out_row + h, __bfloat162float(y_row[h]) * m.gate);
        } else {
            // Remote: write to source rank's buffer via IPC (blockscaled buffers)
            char* dest_buf = static_cast<char*>(d_buffer_ptrs_block[src_rank]);
            uint16_t* y_buf = reinterpret_cast<uint16_t*>(dest_buf + y_off);
            Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + meta_off);
            int* counter = reinterpret_cast<int*>(dest_buf + counter_off);

            int slot_r;
            if (lane == 0)
                slot_r = atomicAdd(counter, 1);
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) continue;

            if (lane == 0) {
                Meta mr{m.row_id, 0, m.gate};
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&mr);
                st_na_v4_s32(meta_dst, meta_val);
            }

            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            uint16_t* dst = y_buf + (int64_t)slot_r * Ha;

            int h = lane * 8;
            for (; h < H; h += 32 * 8) {
                if (h + 8 <= H) {
                    int4* d = reinterpret_cast<int4*>(dst + h);
                    int4 v = *reinterpret_cast<const int4*>(y_row + h);
                    st_na_v4_s32(d, v);
                } else {
                    for (int hh = h; hh < H && hh < h + 8; hh++) {
                        st_na_relaxed_gpu_b16(dst + hh, reinterpret_cast<const uint16_t*>(y_row)[hh]);
                    }
                }
            }
        }
    }
}

__global__ void k_scatter_received_bf16(
    const uint16_t* __restrict__ y_recv,
    const Meta* __restrict__ meta_recv,
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
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta_recv + i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid(m.row_id, T, K, &src_rank, &tok, &slot);

        const uint16_t* y_row = y_recv + (int64_t)i * Ha;
        float* out_row = out + (int64_t)tok * H;

        // NOTE: y_row is written by peer GPUs via IPC. Receiver-side L2 is not
        // coherent with peer writes; use non-caching loads to observe updates.
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
// IPC Token-Slot Return / dX (BF16)
//
// For operations where every (tok,slot) exists exactly once on the source rank
// (forward return, backward dX), avoid dynamic append+metadata:
// write directly into a per-(tok,slot) buffer on the source rank:
//   idx = tok*K + slot
// This eliminates receive counters, meta overwrites, and atomicAdd scatter.
// ============================================================================

__global__ void k_return_write_tokslot_bf16(
    const __nv_bfloat16* __restrict__ Ye_sorted,
    const int* __restrict__ order,
    const Meta* __restrict__ meta_buf,
    int M_recv, int H, int Ha, int T, int K,
    int world, int capacity,
    size_t tok_y_off, size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        // Bounds check on orig_i
        if (orig_i < 0 || orig_i >= capacity) {
            if (lane == 0) printf("RDEP BUG: orig_i=%d out of bounds [0,%d)\n", orig_i, capacity);
            continue;
        }
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta_buf + orig_i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid(m.row_id, T, K, &src_rank, &tok, &slot);

        // Bounds check on decoded values
        if (src_rank < 0 || src_rank >= world) {
            if (lane == 0) printf("RDEP BUG: src_rank=%d out of bounds [0,%d), row_id=%lld\n", src_rank, world, (long long)m.row_id);
            continue;
        }
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) {
            if (lane == 0) printf("RDEP BUG: tok=%d slot=%d out of bounds T=%d K=%d\n", tok, slot, T, K);
            continue;
        }

        const int64_t idx = (int64_t)tok * K + slot;

        char* dst_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
        uint16_t* tok_y = reinterpret_cast<uint16_t*>(dst_buf + tok_y_off);
        float* tok_gate = reinterpret_cast<float*>(dst_buf + tok_gate_off);

        // Gate is a scalar; one lane writes. Use sys-scope for cross-GPU visibility.
        if (lane == 0) st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(m.gate));

        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(Ye_sorted + (int64_t)sorted_i * H);
        uint16_t* dst_u16 = tok_y + idx * Ha;

        // H is required to be multiple of 8 (int4 = 8 BF16). Sys-scope for cross-GPU.
        for (int h = lane * 8; h < H; h += 32 * 8) {
            int4 v = *reinterpret_cast<const int4*>(src_u16 + h);
            int4* d = reinterpret_cast<int4*>(dst_u16 + h);
            st_relaxed_sys_v4_s32(d, v);
        }
    }

    // Fence to ensure all sys-scope writes are visible before kernel completes
    fence_acq_rel_sys();
}

__global__ void k_return_write_tokslot_blockscaled(
    const __nv_bfloat16* __restrict__ Ye_sorted,
    const int* __restrict__ order,
    const Meta* __restrict__ meta_buf,
    int M_recv, int H, int Ha, int T, int K,
    int world, int capacity,
    size_t tok_y_off, size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        if (orig_i < 0 || orig_i >= capacity) {
            if (lane == 0) printf("RDEP BUG: orig_i=%d out of bounds [0,%d)\n", orig_i, capacity);
            continue;
        }
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta_buf + orig_i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid(m.row_id, T, K, &src_rank, &tok, &slot);

        if (src_rank < 0 || src_rank >= world) {
            if (lane == 0) printf("RDEP BUG: src_rank=%d out of bounds [0,%d), row_id=%lld\n", src_rank, world, (long long)m.row_id);
            continue;
        }
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) {
            if (lane == 0) printf("RDEP BUG: tok=%d slot=%d out of bounds T=%d K=%d\n", tok, slot, T, K);
            continue;
        }

        const int64_t idx = (int64_t)tok * K + slot;

        char* dst_buf = static_cast<char*>(d_buffer_ptrs_block[src_rank]);
        uint16_t* tok_y = reinterpret_cast<uint16_t*>(dst_buf + tok_y_off);
        float* tok_gate = reinterpret_cast<float*>(dst_buf + tok_gate_off);

        // Gate is a scalar; one lane writes. Use sys-scope for cross-GPU visibility.
        if (lane == 0) st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(m.gate));

        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(Ye_sorted + (int64_t)sorted_i * H);
        uint16_t* dst_u16 = tok_y + idx * Ha;

        // Sys-scope for cross-GPU visibility.
        for (int h = lane * 8; h < H; h += 32 * 8) {
            int4 v = *reinterpret_cast<const int4*>(src_u16 + h);
            int4* d = reinterpret_cast<int4*>(dst_u16 + h);
            st_relaxed_sys_v4_s32(d, v);
        }
    }

    // Fence to ensure all sys-scope writes are visible before kernel completes
    fence_acq_rel_sys();
}

__global__ void k_reduce_tokslot_gate_bf16(
    const uint16_t* __restrict__ tok_y,
    const float* __restrict__ tok_gate,
    float* __restrict__ out,
    int T, int H, int Ha, int K)
{
    int tok = static_cast<int>(blockIdx.x);
    if (tok >= T) return;
    if (K <= 0 || K > 32) return;

    __shared__ float g_shared[32];
    if (threadIdx.x < K) g_shared[threadIdx.x] = ld_nc_f32(tok_gate + (int64_t)tok * K + threadIdx.x);
    __syncthreads();

    int vec = static_cast<int>(threadIdx.x);
    for (int h0 = vec * 8; h0 < H; h0 += static_cast<int>(blockDim.x) * 8) {
        float acc[8] = {0};
        for (int slot = 0; slot < K; ++slot) {
            const float g = g_shared[slot];
            const uint16_t* y_row = tok_y + (int64_t)(tok * K + slot) * Ha + h0;
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
                    acc[j] += __bfloat162float(bf) * g;
                }
            }
        }

        float* out_row = out + (int64_t)tok * H + h0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int hh = h0 + j;
            if (hh < H) out_row[j] = acc[j];
        }
    }
}

__global__ void k_send_dx_tokslot_bf16(
    const __nv_bfloat16* __restrict__ dXe_sorted,
    const int64_t* __restrict__ row_id,
    int M, int T, int H, int Ha, int K,
    int world,
    size_t tok_y_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid(row_id[i], T, K, &src_rank, &tok, &slot);

        // Bounds check on decoded values
        if (src_rank < 0 || src_rank >= world) {
            if (lane == 0) printf("RDEP BUG k_send_dx: src_rank=%d out of bounds [0,%d), row_id=%lld\n", src_rank, world, (long long)row_id[i]);
            continue;
        }
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) {
            if (lane == 0) printf("RDEP BUG k_send_dx: tok=%d slot=%d out of bounds T=%d K=%d\n", tok, slot, T, K);
            continue;
        }

        const int64_t idx = (int64_t)tok * K + slot;
        char* dst_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
        uint16_t* tok_y = reinterpret_cast<uint16_t*>(dst_buf + tok_y_off);

        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(dXe_sorted + (int64_t)i * H);
        uint16_t* dst_u16 = tok_y + idx * Ha;

        // Sys-scope for cross-GPU visibility
        for (int h = lane * 8; h < H; h += 32 * 8) {
            int4 v = *reinterpret_cast<const int4*>(src_u16 + h);
            int4* d = reinterpret_cast<int4*>(dst_u16 + h);
            st_relaxed_sys_v4_s32(d, v);
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_send_dx_tokslot_blockscaled(
    const __nv_bfloat16* __restrict__ dXe_sorted,
    const int64_t* __restrict__ row_id,
    int M, int T, int H, int Ha, int K,
    int world,
    size_t tok_y_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid(row_id[i], T, K, &src_rank, &tok, &slot);

        // Bounds check on decoded values
        if (src_rank < 0 || src_rank >= world) {
            if (lane == 0) printf("RDEP BUG k_send_dx_block: src_rank=%d out of bounds [0,%d), row_id=%lld\n", src_rank, world, (long long)row_id[i]);
            continue;
        }
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) {
            if (lane == 0) printf("RDEP BUG k_send_dx_block: tok=%d slot=%d out of bounds T=%d K=%d\n", tok, slot, T, K);
            continue;
        }

        const int64_t idx = (int64_t)tok * K + slot;
        char* dst_buf = static_cast<char*>(d_buffer_ptrs_block[src_rank]);
        uint16_t* tok_y = reinterpret_cast<uint16_t*>(dst_buf + tok_y_off);

        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(dXe_sorted + (int64_t)i * H);
        uint16_t* dst_u16 = tok_y + idx * Ha;

        // Sys-scope for cross-GPU visibility
        for (int h = lane * 8; h < H; h += 32 * 8) {
            int4 v = *reinterpret_cast<const int4*>(src_u16 + h);
            int4* d = reinterpret_cast<int4*>(dst_u16 + h);
            st_relaxed_sys_v4_s32(d, v);
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_reduce_tokslot_sum_bf16(
    const uint16_t* __restrict__ tok_y,
    float* __restrict__ out,
    int T, int H, int Ha, int K)
{
    int tok = static_cast<int>(blockIdx.x);
    if (tok >= T) return;
    if (K <= 0 || K > 32) return;

    int vec = static_cast<int>(threadIdx.x);
    for (int h0 = vec * 8; h0 < H; h0 += static_cast<int>(blockDim.x) * 8) {
        float acc[8] = {0};
        for (int slot = 0; slot < K; ++slot) {
            const uint16_t* y_row = tok_y + (int64_t)(tok * K + slot) * Ha + h0;
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

        float* out_row = out + (int64_t)tok * H + h0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int hh = h0 + j;
            if (hh < H) out_row[j] = acc[j];
        }
    }
}

extern "C" void rdep_return_scatter(
    const void* Ye,
    void* out,
    int M_recv, int T, int K,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    // Hybrid mode: use NVSHMEM for inter-node + IPC for intra-node
    // CRITICAL: Use g_nvshmem.ipc_buffer_ptrs (populated by nvshmem_open_ipc_handles_bf16),
    // NOT g_bf16.buffer_ptrs (which is NOT populated in hybrid mode)
    // Check hybrid mode FIRST before g_bf16.initialized (hybrid doesn't use g_bf16)
    if (g_mode == MODE_HYBRID) {
        nvshmem::return_scatter_hybrid_bf16(
            static_cast<const __nv_bfloat16*>(Ye),
            static_cast<float*>(out),
            M_recv, T, K,
            nvshmem::g_nvshmem.ipc_buffer_ptrs,
            nvshmem::g_nvshmem.ipc_barrier_signal_ptrs,
            stream);
        return;
    }
#endif

    // Single-GPU and IPC modes require g_bf16 to be initialized
    if (!g_bf16.initialized) {
        return;
    }

    // Single-GPU and IPC modes: use local IPC path
    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);

    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);

    // IPC fast path: direct per-(tok,slot) writes + deterministic reduction.
    if (g_mode == MODE_IPC && g_bf16.world > 1) {
        const int H = g_bf16.H;
        const int Ha = g_bf16.Ha;
        const int tok_slots = T * K;
        const size_t tok_cap = g_bf16.capacity / static_cast<size_t>(g_bf16.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_bf16.capacity, g_bf16.world);
            return;
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            return;
        }

        const int warps_needed = M_recv;
        const int threads = 256;
        const int blocks = std::max(1, (warps_needed * 32 + threads - 1) / threads);
        k_return_write_tokslot_bf16<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(Ye),
            g_bf16.order,
            meta_buf,
            M_recv, H, Ha, T, K,
            g_bf16.world, static_cast<int>(g_bf16.capacity),
            tok_y_off, tok_gate_off);

        // Ensure all ranks finished writing tok-slot buffers before reduction.
        ipc_barrier_bf16(stream);

        const uint16_t* tok_y = reinterpret_cast<const uint16_t*>(local_buf + tok_y_off);
        const float* tok_gate = reinterpret_cast<const float*>(local_buf + tok_gate_off);
        k_reduce_tokslot_gate_bf16<<<T, 256, 0, stream>>>(
            tok_y,
            tok_gate,
            static_cast<float*>(out),
            T, H, Ha, K);
        return;
    }

    // Legacy path (single-GPU / non-IPC): append+meta + atomic scatter.
    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_buf + x_off);
    cudaMemsetAsync(local_buf + counter_off, 0, sizeof(int), stream);

    // Snapshot dispatch metadata before return writes reuse the same IPC buffer.
    if (M_recv > 0) {
        cudaMemcpyAsync(g_bf16.meta_copy, meta_buf,
                        static_cast<size_t>(M_recv) * sizeof(Meta),
                        cudaMemcpyDeviceToDevice, stream);
    }

    ipc_barrier_bf16(stream);

    int threads = 256;
    int blocks = std::max(1, (M_recv + threads - 1) / threads);
    int capacity = static_cast<int>(g_bf16.capacity);

    k_return_scatter_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Ye), g_bf16.order, g_bf16.meta_copy,
        static_cast<float*>(out),
        M_recv, g_bf16.H, g_bf16.Ha, T, K,
        g_bf16.rank, g_bf16.world, capacity,
        meta_off, counter_off);

    ipc_barrier_bf16(stream);

    int M_ret = 0;
    cudaMemcpyAsync(&M_ret, local_buf + counter_off, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (M_ret > 0) {
        M_ret = std::min(M_ret, capacity);
        k_scatter_received_bf16<<<(M_ret + 255) / 256, 256, 0, stream>>>(
            x_buf, meta_buf,
            static_cast<float*>(out),
            M_ret, g_bf16.H, g_bf16.Ha, T, K);
    }
}

extern "C" void rdep_return_scatter_blockscaled(
    const void* Ye,
    void* out,
    int M_recv, int T, int K,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    // Hybrid mode: use NVSHMEM for inter-node + IPC for intra-node
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return;
        }
        nvshmem::return_scatter_hybrid_blockscaled(
            static_cast<const __nv_bfloat16*>(Ye),
            static_cast<float*>(out),
            M_recv, T, K,
            nvshmem::g_nvshmem.ipc_buffer_ptrs,
            nvshmem::g_nvshmem.ipc_barrier_signal_ptrs,
            stream);
        return;
    }
#endif

    if (!g_block.initialized) return;

    // Single-GPU and IPC modes: use local IPC path
    size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, g_block.world,
                               &x_off, &sfa_off, &y_off, &meta_off, &counter_off, &dropped_off,
                               &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                               &tok_y_off, &tok_gate_off,
                               &total_size);

    char* local_buf = static_cast<char*>(g_block.buffer_ptrs[g_block.rank]);
    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);
    uint16_t* y_buf = reinterpret_cast<uint16_t*>(local_buf + y_off);

    // IPC fast path: deterministic per-(tok,slot) writes + deterministic reduction.
    // Avoids dynamic append counters and meta overwrites (matches BF16 IPC path).
    if (g_mode == MODE_IPC && g_block.world > 1) {
        const int H = g_block.H;
        const int Ha = g_block.Ha;
        const int tok_slots = T * K;
        const size_t tok_cap = g_block.capacity / static_cast<size_t>(g_block.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_block.capacity, g_block.world);
            return;
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            return;
        }

        const int warps_needed = std::max(M_recv, 1);
        const int threads = 256;
        const int blocks = std::max(1, (warps_needed * 32 + threads - 1) / threads);
        k_return_write_tokslot_blockscaled<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(Ye),
            g_block.order,
            meta_buf,
            M_recv, H, Ha, T, K,
            g_block.world, static_cast<int>(g_block.capacity),
            tok_y_off, tok_gate_off);

        ipc_barrier_block(stream);

        const uint16_t* tok_y = reinterpret_cast<const uint16_t*>(local_buf + tok_y_off);
        const float* tok_gate = reinterpret_cast<const float*>(local_buf + tok_gate_off);
        k_reduce_tokslot_gate_bf16<<<T, 256, 0, stream>>>(
            tok_y,
            tok_gate,
            static_cast<float*>(out),
            T, H, Ha, K);
        return;
    }

    cudaMemsetAsync(local_buf + counter_off, 0, sizeof(int), stream);
    cudaStreamSynchronize(stream);

    int threads = 256;
    int blocks = std::max(1, (M_recv + threads - 1) / threads);
    int capacity = static_cast<int>(g_block.capacity);
    int H = g_block.H;

    // Use blockscaled return scatter kernel (uses d_buffer_ptrs_block)
    k_return_scatter_blockscaled_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Ye), g_block.order, meta_buf,
        static_cast<float*>(out),
        M_recv, H, H, T, K,
        g_block.rank, g_block.world, capacity,
        y_off, meta_off, counter_off);

    cudaStreamSynchronize(stream);

    int M_ret = 0;
    cudaMemcpyAsync(&M_ret, local_buf + counter_off, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (M_ret > 0) {
        M_ret = std::min(M_ret, capacity);
        k_scatter_received_bf16<<<(M_ret + 255) / 256, 256, 0, stream>>>(
            y_buf, meta_buf,
            static_cast<float*>(out),
            M_ret, H, H, T, K);
    }
}

__global__ void k_gather_dy_bf16(
    const __nv_bfloat16* __restrict__ dY,          // [T, H]
    const __nv_bfloat16* __restrict__ Ye,          // [M, H] (sorted)
    const int64_t* __restrict__ row_id,            // [M]
    const float* __restrict__ gate,                // [M]
    __nv_bfloat16* __restrict__ dYe_out,           // [M, H]
    float* __restrict__ dGate_out,                 // [M]
    int M, int T, int H, int K)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid(row_id[i], T, K, &src_rank, &tok, &slot);
        const __nv_bfloat16* dy_row = dY + (int64_t)tok * H;
        const __nv_bfloat16* ye_row = Ye + (int64_t)i * H;
        __nv_bfloat16* dye_row = dYe_out + (int64_t)i * H;

        float g = gate[i];
        float dot = 0.0f;
        for (int h = lane; h < H; h += 32) {
            float dy = __bfloat162float(dy_row[h]);
            float ye = __bfloat162float(ye_row[h]);
            dot += ye * dy;
            dye_row[h] = __float2bfloat16(dy * g);
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            dGate_out[i] = dot;
        }
    }
}

__global__ void k_scatter_gate_bf16(
    const float* __restrict__ dGate_sorted,
    const int64_t* __restrict__ row_id,
    float* __restrict__ dGates_tk,
    int M, int T, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        int src_rank, tok, slot;
        decode_rid(row_id[i], T, K, &src_rank, &tok, &slot);
        dGates_tk[tok * K + slot] = dGate_sorted[i];
    }
}

__global__ void k_scatter_dx_bf16(
    const __nv_bfloat16* __restrict__ dXe_pad,   // [M_pad, H]
    const int* __restrict__ dest,                // [M]
    const int64_t* __restrict__ row_id,           // [M]
    float* __restrict__ dX,                       // [T, H] float32 accum
    int M, int T, int H, int K)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid(row_id[i], T, K, &src_rank, &tok, &slot);
        int pad_i = dest[i];
        const __nv_bfloat16* dxe_row = dXe_pad + (int64_t)pad_i * H;
        float* dx_row = dX + (int64_t)tok * H;
        for (int h = lane; h < H; h += 32) {
            atomicAdd(dx_row + h, __bfloat162float(dxe_row[h]));
        }
    }
}

extern "C" void rdep_gather_dy_bf16(
    const void* dY,
    const void* Ye,
    const int64_t* row_id,
    const float* gate,
    void* dYe_out,
    float* dGate_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = std::max(1, (M * 32 + threads - 1) / threads);
    k_gather_dy_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dY),
        static_cast<const __nv_bfloat16*>(Ye),
        row_id,
        gate,
        static_cast<__nv_bfloat16*>(dYe_out),
        dGate_out,
        M, T, H, K);
}

extern "C" void rdep_scatter_gate_bf16(
    const float* dGate_sorted,
    const int64_t* row_id,
    float* dGates_tk,
    int M, int T, int K,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    k_scatter_gate_bf16<<<blocks, threads, 0, stream>>>(
        dGate_sorted, row_id, dGates_tk, M, T, K);
}

extern "C" void rdep_scatter_dx_bf16(
    const void* dXe_pad,
    const int* dest,
    const int64_t* row_id,
    void* dX_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = std::max(1, (M * 32 + threads - 1) / threads);
    k_scatter_dx_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dXe_pad),
        dest,
        row_id,
        static_cast<float*>(dX_out),
        M, T, H, K);
}

extern "C" void rdep_scatter_dx_bf16_internal(
    const void* dXe_pad,
    const int64_t* row_id,
    void* dX_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    if (!g_bf16.initialized) return;
    int threads = 256;
    int blocks = std::max(1, (M * 32 + threads - 1) / threads);
    k_scatter_dx_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dXe_pad),
        g_bf16.dest,
        row_id,
        static_cast<float*>(dX_out),
        M, T, H, K);
}

// ============================================================================
// IPC Backward Helpers (single-node, BF16)
//
// These helpers implement the "flipped" RDEP backward using IPC memory only.
// They intentionally avoid relying on global mutable forward metadata across
// layers by using the materialized per-dispatch row_id tensors.
// ============================================================================

__global__ void k_ipc_barrier_phase_bf16(int phase) {
    // One-CTA IPC barrier using per-peer phase slots (DeepEP-style direction):
    // - Each rank writes `phase` into each peer's signal[my_rank] (sys-scope release store).
    // - Each rank waits on its local signal[peer] (sys-scope acquire load).
    //
    // This avoids remote polling traffic: waiting spins only on local memory.
    int tid = threadIdx.x;
    int world = d_world_bf16;
    int my_rank = d_my_rank_bf16;

    // Fence to ensure all prior writes (from previous kernels) are sys-visible
    // before we signal completion. Without this, the release store only orders
    // writes within THIS kernel, not writes from previous dispatch kernels.
    fence_acq_rel_sys();
    __syncthreads();

    if (tid < world) {
        int* peer_sig = d_barrier_signal_ptrs_bf16[tid] + my_rank;
        st_release_sys_s32(peer_sig, phase);
    }
    __syncthreads();

    uint64_t start_time = clock64();
    if (tid < world) {
        const int* local_sig = d_barrier_signal_ptrs_bf16[my_rank] + tid;
        while (ld_acquire_sys_s32(local_sig) < phase) {
            if (clock64() - start_time > TIMEOUT_CYCLES) {
                printf("nmoe phase barrier timeout: rank=%d wait_rank=%d phase=%d\n",
                       my_rank, tid, phase);
                trap();
            }
        }
    }
    __syncthreads();
}

__global__ void k_ipc_barrier_phase_block(int phase) {
    int tid = threadIdx.x;
    int world = d_world_block;
    int my_rank = d_my_rank_block;

    // Fence to ensure all prior writes (from previous kernels) are sys-visible
    fence_acq_rel_sys();
    __syncthreads();

    if (tid < world) {
        int* peer_sig = d_barrier_signal_ptrs_block[tid] + my_rank;
        st_release_sys_s32(peer_sig, phase);
    }
    __syncthreads();

    uint64_t start_time = clock64();
    if (tid < world) {
        const int* local_sig = d_barrier_signal_ptrs_block[my_rank] + tid;
        while (ld_acquire_sys_s32(local_sig) < phase) {
            if (clock64() - start_time > TIMEOUT_CYCLES) {
                printf("nmoe phase barrier timeout (blockscaled): rank=%d wait_rank=%d phase=%d\n",
                       my_rank, tid, phase);
                trap();
            }
        }
    }
    __syncthreads();
}

__global__ void k_stage_dy_to_xbuf_bf16(
    const __nv_bfloat16* __restrict__ dY,
    uint16_t* __restrict__ x_buf,
    int T, int H, int Ha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * H) {
        int t = i / H;
        int h = i - t * H;
        const uint16_t* src = reinterpret_cast<const uint16_t*>(dY + (int64_t)t * H);
        x_buf[(int64_t)t * Ha + h] = src[h];
    }
}

// ============================================================================
// IPC Backward (push staging) - BF16 payload
//
// Contract:
//  - Stage dY by row_id into destination buffers (push, no remote reads).
//  - Compute local dYe/dGate from staged dY and Ye_sorted.
//  - Return dGate via fixed (tok,slot) writes into src-rank tok_gate buffer.
//
// This path is used for BF16 *and* blockscaled profiles (STE backward), with
// the active runtime state selecting the underlying buffer layout.
// ============================================================================

__global__ void k_push_stage_dy_ipc_bf16(
    const __nv_bfloat16* __restrict__ dY,   // [T, H]
    const int* __restrict__ eids,           // [T, K]
    int T, int H, int Ha, int K,
    int n_local, int capacity,
    size_t x_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    int M = T * K;
    const int my_rank = d_my_rank_bf16;

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = i / K;
        int slot = i - tok * K;
        int eid = eids[tok * K + slot];
        int dest = eid / n_local;

        const int64_t rid = encode_rid(my_rank, tok, slot, T, K);
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        char* dest_buf = static_cast<char*>(d_buffer_ptrs_bf16[dest]);
        uint16_t* stage = reinterpret_cast<uint16_t*>(dest_buf + x_off);
        uint16_t* dst = stage + rid * Ha;
        const __nv_bfloat16* row = dY + (int64_t)tok * H;

        const bool is_remote = (dest != my_rank);
        int h = lane * 8;
        for (; h < H; h += 32 * 8) {
            if (h + 8 <= H) {
                int4 v = *reinterpret_cast<const int4*>(row + h);
                int4* d = reinterpret_cast<int4*>(dst + h);
                if (is_remote) {
                    st_relaxed_sys_v4_s32(d, v);  // sys-scope for cross-GPU
                } else {
                    *d = v;
                }
            } else {
                for (int hh = h; hh < H && hh < h + 8; hh++) {
                    const uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                    if (is_remote) {
                        st_relaxed_sys_b16(dst + hh, u);  // sys-scope for cross-GPU
                    } else {
                        dst[hh] = u;
                    }
                }
            }
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_push_stage_dy_ipc_blockscaled(
    const __nv_bfloat16* __restrict__ dY,   // [T, H]
    const int* __restrict__ eids,           // [T, K]
    int T, int H, int K,
    int n_local, int capacity,
    size_t y_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    int M = T * K;
    const int my_rank = d_my_rank_block;

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = i / K;
        int slot = i - tok * K;
        int eid = eids[tok * K + slot];
        int dest = eid / n_local;

        const int64_t rid = encode_rid(my_rank, tok, slot, T, K);
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        char* dest_buf = static_cast<char*>(d_buffer_ptrs_block[dest]);
        uint16_t* stage = reinterpret_cast<uint16_t*>(dest_buf + y_off);
        uint16_t* dst = stage + rid * static_cast<int64_t>(H);
        const __nv_bfloat16* row = dY + (int64_t)tok * H;

        const bool is_remote = (dest != my_rank);
        int h = lane * 8;
        for (; h < H; h += 32 * 8) {
            if (h + 8 <= H) {
                int4 v = *reinterpret_cast<const int4*>(row + h);
                int4* d = reinterpret_cast<int4*>(dst + h);
                if (is_remote) {
                    st_relaxed_sys_v4_s32(d, v);  // sys-scope for cross-GPU
                } else {
                    *d = v;
                }
            } else {
                for (int hh = h; hh < H && hh < h + 8; hh++) {
                    const uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                    if (is_remote) {
                        st_relaxed_sys_b16(dst + hh, u);  // sys-scope for cross-GPU
                    } else {
                        dst[hh] = u;
                    }
                }
            }
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_gather_dy_from_stage_and_send_gate_ipc_bf16(
    const __nv_bfloat16* __restrict__ Ye_sorted,   // [M, H]
    const int64_t* __restrict__ row_id,            // [M]
    const float* __restrict__ gate_sorted,         // [M]
    const uint16_t* __restrict__ stage,            // [capacity, Ha]
    __nv_bfloat16* __restrict__ dYe_out,           // [M, H]
    float* __restrict__ dGate_sorted_out,          // [M]
    int M, int T, int H, int Ha, int K,
    int capacity,
    size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const int my_rank = d_my_rank_bf16;

    for (int i = warp_id; i < M; i += num_warps) {
        const int64_t rid = row_id[i];
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        const uint16_t* dy_u16 = stage + rid * Ha;
        const uint16_t* ye_u16 = reinterpret_cast<const uint16_t*>(Ye_sorted + (int64_t)i * H);
        __nv_bfloat16* dye_row = dYe_out + (int64_t)i * H;

        const float g = gate_sorted[i];
        float dot = 0.0f;

        for (int h = lane * 8; h < Ha; h += 32 * 8) {
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

            // Return dGate to source rank via fixed tok-slot write.
            int slot = static_cast<int>(rid % K);
            const int64_t tmp = rid / K;
            int tok = static_cast<int>(tmp % T);
            int src_rank = static_cast<int>(tmp / T);
            const int64_t idx = (int64_t)tok * K + slot;
            if (idx < 0 || idx >= static_cast<int64_t>(capacity)) continue;

            char* src_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
            float* tok_gate = reinterpret_cast<float*>(src_buf + tok_gate_off);
            if (src_rank == my_rank) {
                tok_gate[idx] = dot;
            } else {
                // Sys-scope for cross-GPU visibility
                st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(dot));
            }
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_gather_dy_from_stage_and_send_gate_ipc_blockscaled(
    const __nv_bfloat16* __restrict__ Ye_sorted,   // [M, H]
    const int64_t* __restrict__ row_id,            // [M]
    const float* __restrict__ gate_sorted,         // [M]
    const uint16_t* __restrict__ stage,            // [capacity, H]
    __nv_bfloat16* __restrict__ dYe_out,           // [M, H]
    float* __restrict__ dGate_sorted_out,          // [M]
    int M, int T, int H, int K,
    int capacity,
    size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const int my_rank = d_my_rank_block;

    for (int i = warp_id; i < M; i += num_warps) {
        const int64_t rid = row_id[i];
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        const uint16_t* dy_u16 = stage + rid * static_cast<int64_t>(H);
        const uint16_t* ye_u16 = reinterpret_cast<const uint16_t*>(Ye_sorted + (int64_t)i * H);
        __nv_bfloat16* dye_row = dYe_out + (int64_t)i * H;

        const float g = gate_sorted[i];
        float dot = 0.0f;

        for (int h = lane * 8; h < H; h += 32 * 8) {
            union U16x8 { int4 v; uint16_t u[8]; };
            U16x8 dy8;
            U16x8 ye8;
            const bool full = (h + 8 <= H);
            if (full) {
                dy8.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(dy_u16 + h));
                ye8.v = *reinterpret_cast<const int4*>(ye_u16 + h);
            } else {
                // Tail: load scalar BF16 values.
                for (int j = 0; j < 8; j++) {
                    int hh = h + j;
                    if (hh < H) {
                        dy8.u[j] = reinterpret_cast<const uint16_t*>(dy_u16)[hh];
                        ye8.u[j] = reinterpret_cast<const uint16_t*>(ye_u16)[hh];
                    } else {
                        dy8.u[j] = 0;
                        ye8.u[j] = 0;
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < 8; j++) {
                int hh = h + j;
                if (hh >= H) break;
                const __nv_bfloat16 dy_bf = *reinterpret_cast<const __nv_bfloat16*>(&dy8.u[j]);
                const __nv_bfloat16 ye_bf = *reinterpret_cast<const __nv_bfloat16*>(&ye8.u[j]);
                float dy = __bfloat162float(dy_bf);
                float ye = __bfloat162float(ye_bf);
                dot += ye * dy;
                dye_row[hh] = __float2bfloat16(dy * g);
            }
        }

        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            dGate_sorted_out[i] = dot;

            int slot = static_cast<int>(rid % K);
            const int64_t tmp = rid / K;
            int tok = static_cast<int>(tmp % T);
            int src_rank = static_cast<int>(tmp / T);
            const int64_t idx = (int64_t)tok * K + slot;
            if (idx < 0 || idx >= static_cast<int64_t>(capacity)) continue;

            char* src_buf = static_cast<char*>(d_buffer_ptrs_block[src_rank]);
            float* tok_gate = reinterpret_cast<float*>(src_buf + tok_gate_off);
            if (src_rank == my_rank) {
                tok_gate[idx] = dot;
            } else {
                // Sys-scope for cross-GPU visibility
                st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(dot));
            }
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_collect_tok_gate_ipc(
    const float* __restrict__ tok_gate,
    float* __restrict__ dGates_tk_out,
    int tok_slots)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tok_slots) {
        // tok_gate may be written by peer GPUs; use non-caching loads.
        dGates_tk_out[i] = ld_nc_f32(tok_gate + i);
    }
}

__global__ void k_gather_dy_remote_ipc_bf16(
    const __nv_bfloat16* __restrict__ Ye,
    const int64_t* __restrict__ row_id,
    const float* __restrict__ gate,
    __nv_bfloat16* __restrict__ dYe_out,
    float* __restrict__ dGate_out,
    int M, int T, int H, int K, int Ha)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid(row_id[i], T, K, &src_rank, &tok, &slot);
        const char* src_buf = static_cast<const char*>(d_buffer_ptrs_bf16[src_rank]);
        const uint16_t* src_x = reinterpret_cast<const uint16_t*>(src_buf);
        const __nv_bfloat16* dy_row = reinterpret_cast<const __nv_bfloat16*>(src_x + (int64_t)tok * Ha);

        const __nv_bfloat16* ye_row = Ye + (int64_t)i * H;
        __nv_bfloat16* dye_row = dYe_out + (int64_t)i * H;

        float g = gate[i];
        float dot = 0.0f;
        for (int h = lane; h < H; h += 32) {
            float dy = __bfloat162float(dy_row[h]);
            float ye = __bfloat162float(ye_row[h]);
            dot += ye * dy;
            dye_row[h] = __float2bfloat16(dy * g);
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) dGate_out[i] = dot;
    }
}

__global__ void k_send_dgate_ipc_bf16(
    const int64_t* __restrict__ row_id,
    const float* __restrict__ dGate_sorted,
    float* __restrict__ dGates_tk_local,
    int M, int T, int K,
    int capacity,
    size_t meta_off, size_t counter_off)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    int src_rank, tok, slot;
    decode_rid(row_id[i], T, K, &src_rank, &tok, &slot);

    if (src_rank == d_my_rank_bf16) {
        atomicAdd(dGates_tk_local + tok * K + slot, dGate_sorted[i]);
        return;
    }

    char* dst_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
    Meta* meta_buf = reinterpret_cast<Meta*>(dst_buf + meta_off);
    int* counter = reinterpret_cast<int*>(dst_buf + counter_off);

    int slot_r = atomicAdd(counter, 1);
    if (slot_r < capacity) {
        meta_buf[slot_r] = Meta{row_id[i], 0, dGate_sorted[i]};
    }
}

__global__ void k_scatter_received_gate_ipc_bf16(
    const Meta* __restrict__ meta_recv,
    float* __restrict__ dGates_tk,
    int M_ret, int T, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M_ret) return;
    const Meta& m = meta_recv[i];
    int src_rank, tok, slot;
    decode_rid(m.row_id, T, K, &src_rank, &tok, &slot);
    atomicAdd(dGates_tk + tok * K + slot, m.gate);
}

__global__ void k_send_dx_ipc_bf16(
    const __nv_bfloat16* __restrict__ dXe_sorted,
    const int64_t* __restrict__ row_id,
    float* __restrict__ dX_local,
    int M, int T, int H, int K, int Ha,
    int capacity,
    size_t meta_off, size_t counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid(row_id[i], T, K, &src_rank, &tok, &slot);

        const __nv_bfloat16* src_row = dXe_sorted + (int64_t)i * H;

        if (src_rank == d_my_rank_bf16) {
            float* out_row = dX_local + (int64_t)tok * H;
            for (int h = lane; h < H; h += 32) atomicAdd(out_row + h, __bfloat162float(src_row[h]));
            continue;
        }

        char* dst_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
        uint16_t* y_buf = reinterpret_cast<uint16_t*>(dst_buf);  // x_off = 0
        Meta* meta_buf = reinterpret_cast<Meta*>(dst_buf + meta_off);
        int* counter = reinterpret_cast<int*>(dst_buf + counter_off);

        int slot_r;
        if (lane == 0) slot_r = atomicAdd(counter, 1);
        slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);
        if (slot_r >= capacity) continue;

        if (lane == 0) meta_buf[slot_r] = Meta{row_id[i], 0, 1.0f};

        uint16_t* dst = y_buf + (int64_t)slot_r * Ha;
        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(src_row);
        for (int h = lane; h < H; h += 32) dst[h] = src_u16[h];
    }
}

extern "C" void rdep_gather_dy_ipc_bf16(
    const void* dY_local,
    const void* Ye_sorted,
    const int64_t* row_id,
    const float* gate_sorted,
    void* dYe_out,
    float* dGate_sorted_out,
    float* dGates_tk_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    if (!g_bf16.initialized || g_mode != MODE_IPC) return;
    if (g_bf16.world <= 1) return;
    if (g_bf16.world > MAX_RANKS) {
        fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_bf16.world, MAX_RANKS);
        return;
    }

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);

    // 1) Stage local dY into local x_buf for IPC peers to read.
    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_buf + x_off);

    int threads = 256;
    int blocks = (T * H + threads - 1) / threads;
    k_stage_dy_to_xbuf_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dY_local),
        x_buf,
        T, H, g_bf16.Ha);

    // 2) Barrier so all ranks have staged dY.
    ipc_barrier_bf16(stream);

    // 3) Compute dYe and dGate using remote reads.
    int g_threads = 256;
    int g_blocks = std::max(1, (M * 32 + g_threads - 1) / g_threads);
    k_gather_dy_remote_ipc_bf16<<<g_blocks, g_threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Ye_sorted),
        row_id,
        gate_sorted,
        static_cast<__nv_bfloat16*>(dYe_out),
        dGate_sorted_out,
        M, T, H, K, g_bf16.Ha);

    // 4) Reset receive counter and local dGates buffer, then send dGate to source ranks via IPC.
    cudaMemsetAsync(local_buf + counter_off, 0, sizeof(int), stream);
    cudaMemsetAsync(dGates_tk_out, 0, (size_t)T * (size_t)K * sizeof(float), stream);

    // Global barrier: ensure all ranks reset counters before any remote atomicAdd() begins.
    ipc_barrier_bf16(stream);

    int s_threads = 256;
    int s_blocks = std::max(1, (M + s_threads - 1) / s_threads);
    k_send_dgate_ipc_bf16<<<s_blocks, s_threads, 0, stream>>>(
        row_id,
        dGate_sorted_out,
        dGates_tk_out,
        M, T, K,
        static_cast<int>(g_bf16.capacity),
        meta_off, counter_off);

    // 5) Barrier so all ranks finished sending dGate metadata.
    ipc_barrier_bf16(stream);

    // 6) Receive and scatter dGate into local [T,K].
    cudaStreamSynchronize(stream);
    int M_ret = 0;
    cudaMemcpy(&M_ret, local_buf + counter_off, sizeof(int), cudaMemcpyDeviceToHost);
    if (M_ret > 0) {
        int capacity = static_cast<int>(g_bf16.capacity);
        M_ret = std::min(M_ret, capacity);
        Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);
        int t = 256;
        int b = (M_ret + t - 1) / t;
        k_scatter_received_gate_ipc_bf16<<<b, t, 0, stream>>>(
            meta_buf,
            dGates_tk_out,
            M_ret, T, K);
    }
}

extern "C" void rdep_gather_dy_dist_bf16(
    const void* dY_local,
    const int* eids,
    const void* Ye_sorted,
    const int64_t* row_id,
    const float* gate_sorted,
    void* dYe_out,
    float* dGate_sorted_out,
    float* dGates_tk_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        nvshmem::gather_dy_hybrid_bf16(
            static_cast<const __nv_bfloat16*>(dY_local),
            eids,
            static_cast<const __nv_bfloat16*>(Ye_sorted),
            row_id,
            gate_sorted,
            static_cast<__nv_bfloat16*>(dYe_out),
            dGate_sorted_out,
            dGates_tk_out,
            M, T, H, K,
            stream);
        return;
    }
#endif
    if (g_mode != MODE_IPC) return;

    if (g_bf16.initialized) {
        if (g_bf16.world <= 1) return;
        if (g_bf16.world > MAX_RANKS) {
            fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_bf16.world, MAX_RANKS);
            return;
        }
        size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
        bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                            &x_off, &meta_off, &counter_off, &dropped_off,
                            &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                            &tok_y_off, &tok_gate_off,
                            &total_size);
        const int64_t cap = static_cast<int64_t>(g_bf16.capacity);
        const int tok_slots = T * K;
        const size_t tok_cap = g_bf16.capacity / static_cast<size_t>(g_bf16.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_bf16.capacity, g_bf16.world);
            return;
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            return;
        }

        const int threads = 256;
        const int warps_needed = std::max(1, tok_slots);
        const int blocks = std::max(1, (warps_needed * 32 + threads - 1) / threads);
        k_push_stage_dy_ipc_bf16<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dY_local),
            eids,
            T, H, g_bf16.Ha, K,
            g_bf16.n_local, static_cast<int>(g_bf16.capacity),
            x_off);

        ipc_barrier_bf16(stream);

        const char* local_buf = static_cast<const char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
        const uint16_t* stage = reinterpret_cast<const uint16_t*>(local_buf + x_off);
        const int g_threads = 256;
        const int g_blocks = std::max(1, (M * 32 + g_threads - 1) / g_threads);
        k_gather_dy_from_stage_and_send_gate_ipc_bf16<<<g_blocks, g_threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(Ye_sorted),
            row_id,
            gate_sorted,
            stage,
            static_cast<__nv_bfloat16*>(dYe_out),
            dGate_sorted_out,
            M, T, H, g_bf16.Ha, K,
            static_cast<int>(g_bf16.capacity),
            tok_gate_off);

        ipc_barrier_bf16(stream);

        const float* tok_gate = reinterpret_cast<const float*>(local_buf + tok_gate_off);
        const int t_threads = 256;
        const int t_blocks = (tok_slots + t_threads - 1) / t_threads;
        k_collect_tok_gate_ipc<<<t_blocks, t_threads, 0, stream>>>(
            tok_gate,
            dGates_tk_out,
            tok_slots);
        return;
    }

    if (g_block.initialized) {
        if (g_block.world <= 1) return;
        if (g_block.world > MAX_RANKS) {
            fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_block.world, MAX_RANKS);
            return;
        }
        size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
        blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, g_block.world,
                                   &x_off, &sfa_off, &y_off,
                                   &meta_off, &counter_off, &dropped_off,
                                   &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                                   &tok_y_off, &tok_gate_off,
                                   &total_size);
        const int tok_slots = T * K;
        const size_t tok_cap = g_block.capacity / static_cast<size_t>(g_block.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_block.capacity, g_block.world);
            return;
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            return;
        }

        const int threads = 256;
        const int warps_needed = std::max(1, tok_slots);
        const int blocks = std::max(1, (warps_needed * 32 + threads - 1) / threads);
        k_push_stage_dy_ipc_blockscaled<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dY_local),
            eids,
            T, H, K,
            g_block.n_local, static_cast<int>(g_block.capacity),
            y_off);

        ipc_barrier_block(stream);

        const char* local_buf = static_cast<const char*>(g_block.buffer_ptrs[g_block.rank]);
        const uint16_t* stage = reinterpret_cast<const uint16_t*>(local_buf + y_off);
        const int g_threads = 256;
        const int g_blocks = std::max(1, (M * 32 + g_threads - 1) / g_threads);
        k_gather_dy_from_stage_and_send_gate_ipc_blockscaled<<<g_blocks, g_threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(Ye_sorted),
            row_id,
            gate_sorted,
            stage,
            static_cast<__nv_bfloat16*>(dYe_out),
            dGate_sorted_out,
            M, T, H, K,
            static_cast<int>(g_block.capacity),
            tok_gate_off);

        ipc_barrier_block(stream);

        const float* tok_gate = reinterpret_cast<const float*>(local_buf + tok_gate_off);
        const int t_threads = 256;
        const int t_blocks = (tok_slots + t_threads - 1) / t_threads;
        k_collect_tok_gate_ipc<<<t_blocks, t_threads, 0, stream>>>(
            tok_gate,
            dGates_tk_out,
            tok_slots);
        return;
    }
}

extern "C" void rdep_scatter_dx_dist_bf16(
    const void* dXe_sorted,
    const int64_t* row_id,
    void* dX_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        nvshmem::scatter_dx_hybrid_bf16(
            static_cast<const __nv_bfloat16*>(dXe_sorted),
            row_id,
            static_cast<float*>(dX_out),
            M, T, H, K,
            stream);
        return;
    }
#endif
    if (g_mode != MODE_IPC) return;

    if (g_bf16.initialized) {
        if (g_bf16.world <= 1) return;
        if (g_bf16.world > MAX_RANKS) {
            fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_bf16.world, MAX_RANKS);
            return;
        }
        size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
        bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                            &x_off, &meta_off, &counter_off, &dropped_off,
                            &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                            &tok_y_off, &tok_gate_off,
                            &total_size);

        const int Ha = g_bf16.Ha;
        const int tok_slots = T * K;
        const size_t tok_cap = g_bf16.capacity / static_cast<size_t>(g_bf16.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_bf16.capacity, g_bf16.world);
            return;
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            return;
        }

        int threads = 256;
        int blocks = std::max(1, (M * 32 + threads - 1) / threads);
        k_send_dx_tokslot_bf16<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dXe_sorted),
            row_id,
            M, T, H, Ha, K,
            g_bf16.world,
            tok_y_off);

        ipc_barrier_bf16(stream);

        const char* local_buf = static_cast<const char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
        const uint16_t* tok_y = reinterpret_cast<const uint16_t*>(local_buf + tok_y_off);
        k_reduce_tokslot_sum_bf16<<<T, 256, 0, stream>>>(
            tok_y,
            static_cast<float*>(dX_out),
            T, H, Ha, K);
        return;
    }

    if (g_block.initialized) {
        if (g_block.world <= 1) return;
        if (g_block.world > MAX_RANKS) {
            fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_block.world, MAX_RANKS);
            return;
        }
        size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
        blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, g_block.world,
                                   &x_off, &sfa_off, &y_off,
                                   &meta_off, &counter_off, &dropped_off,
                                   &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                                   &tok_y_off, &tok_gate_off,
                                   &total_size);

        const int Ha = g_block.Ha;
        const int tok_slots = T * K;
        const size_t tok_cap = g_block.capacity / static_cast<size_t>(g_block.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_block.capacity, g_block.world);
            return;
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            return;
        }

        int threads = 256;
        int blocks = std::max(1, (M * 32 + threads - 1) / threads);
        k_send_dx_tokslot_blockscaled<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dXe_sorted),
            row_id,
            M, T, H, Ha, K,
            g_block.world,
            tok_y_off);

        ipc_barrier_block(stream);

        const char* local_buf = static_cast<const char*>(g_block.buffer_ptrs[g_block.rank]);
        const uint16_t* tok_y = reinterpret_cast<const uint16_t*>(local_buf + tok_y_off);
        k_reduce_tokslot_sum_bf16<<<T, 256, 0, stream>>>(
            tok_y,
            static_cast<float*>(dX_out),
            T, H, Ha, K);
        return;
    }
}

extern "C" void rdep_scatter_dx_ipc_bf16(
    const void* dXe_sorted,
    const int64_t* row_id,
    void* dX_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    if (!g_bf16.initialized || g_mode != MODE_IPC) return;
    if (g_bf16.world <= 1) return;
    if (g_bf16.world > MAX_RANKS) {
        fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_bf16.world, MAX_RANKS);
        return;
    }

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);

    const int Ha = g_bf16.Ha;
    const int tok_slots = T * K;
    const size_t tok_cap = g_bf16.capacity / static_cast<size_t>(g_bf16.world);
    if (static_cast<size_t>(tok_slots) > tok_cap) {
        fprintf(stderr,
                "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                tok_slots, tok_cap, g_bf16.capacity, g_bf16.world);
        return;
    }
    if (K <= 0 || K > 32) {
        fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
        return;
    }

    int threads = 256;
    int blocks = std::max(1, (M * 32 + threads - 1) / threads);
    k_send_dx_tokslot_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dXe_sorted),
        row_id,
        M, T, H, Ha, K,
        g_bf16.world,
        tok_y_off);

    // Ensure all ranks finished writing tok-slot buffers before reduction.
    ipc_barrier_bf16(stream);

    const char* local_buf = static_cast<const char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    const uint16_t* tok_y = reinterpret_cast<const uint16_t*>(local_buf + tok_y_off);
    k_reduce_tokslot_sum_bf16<<<T, 256, 0, stream>>>(
        tok_y,
        static_cast<float*>(dX_out),
        T, H, Ha, K);
}

} // namespace rdep
