// NVSHMEM support for RDEP hybrid mode (IPC intra-node + NVSHMEM inter-node)
//
// Only compiled when WITH_NVSHMEM is defined.
// Provides NVSHMEM initialization and hybrid dispatch/return kernels.
//
// Architecture:
//   - rank = rdma_rank * local_world + nvl_rank
//   - rdma_rank: node index (0..num_nodes-1)
//   - nvl_rank: GPU within node (0..local_world-1)
//   - Intra-node (same rdma_rank): use IPC via d_buffer_ptrs
//   - Inter-node (different rdma_rank): use NVSHMEM symmetric heap

#pragma once

#ifdef WITH_NVSHMEM

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nvshmem.h>
#include <nvshmemx.h>

namespace rdep {
namespace nvshmem {

// ============================================================================
// Constants
// ============================================================================

constexpr int MAX_NODES = 64;          // Max RDMA/NVSHMEM peers
constexpr int MAX_LOCAL_GPUS = 72;     // Max NVLink peers (GB300 NVL72)
constexpr int RDMA_BARRIER_TAG = 2048; // Different from IPC barrier tag

struct Meta;  // Defined in rdep_nvshmem.cu (layout must match IPC Meta).

// ============================================================================
// NVSHMEM State
// ============================================================================

struct NvshmemState {
    // =========================================================================
    // NVSHMEM Symmetric heap allocations (for INTER-NODE communication only)
    // These are allocated via nvshmem_malloc and are visible to all PEs
    // =========================================================================
    // Allocation discipline (DeepEP pattern): prefer a single aligned symmetric allocation
    // and slice it into sub-buffers. This guarantees identical heap layout across PEs
    // and avoids computed-offset translation failures under IBGDA.
    void*     sym_bf16_base;   // Base pointer for BF16 symmetric region (nvshmem_align)
    size_t    sym_bf16_bytes;  // Total bytes in sym_bf16_base
    void*     sym_block_base;  // Base pointer for blockscaled symmetric region (nvshmem_align)
    size_t    sym_block_bytes; // Total bytes in sym_block_base

    uint16_t* x_buf_bf16;      // BF16 activations [capacity * Ha]
    uint16_t* x_buf_block;     // Blockscaled packed [capacity * Hp]
    uint8_t*  sfa_buf;         // Scale factors [capacity * Hsf]
    uint16_t* y_buf;           // Return buffer [capacity * H]

    // Token-slot buffers for backward + deterministic return/dX (BF16 payload).
    // These are symmetric so inter-node sends can target them via NVSHMEM puts.
    uint16_t* tok_y;           // [tok_slots * tok_Ha] BF16 scratch (idx = tok*K + slot)
    float*    tok_gate;        // [tok_slots] float scratch (idx = tok*K + slot)
    int*      tok_tag;         // [tok_slots] int32 phase tag for remote writes

    // Metadata (symmetric)
    void*     meta;            // Meta structs [capacity]

    // Global counter (symmetric) for inter-node receives.
    // NOTE: must be a single counter to avoid per-source slot collisions.
    int*      counter;         // [1]
    int*      dropped;         // Dropped counter

    // RDMA barrier signals (symmetric)
    // barrier_signals[rdma_rank] = signal from that node
    int*      barrier_signals; // [MAX_NODES]

    // =========================================================================
    // IPC buffers (for INTRA-NODE communication only)
    // These are allocated via cudaMalloc and can be used with cudaIpcGetMemHandle
    // CRITICAL: NVSHMEM memory cannot be used with CUDA IPC, so we need separate
    // buffers for intra-node communication
    // =========================================================================
    void*     ipc_buffer;             // Local IPC buffer (cudaMalloc'd)
    void*     ipc_buffer_ptrs[MAX_LOCAL_GPUS];  // [local_world] pointers to all local buffers (HOST)
    void**    d_ipc_buffer_ptrs;      // Device copy of ipc_buffer_ptrs for kernel access
    int*      ipc_barrier_signal_ptrs[MAX_LOCAL_GPUS];  // [local_world] barrier signal pointers (HOST)
    int**     d_ipc_barrier_signal_ptrs;  // Device copy of barrier signal pointers
    size_t    ipc_buffer_size;        // Total size of IPC buffer

    // IPC buffer layout offsets (for BF16 path)
    size_t    ipc_x_off;
    size_t    ipc_sfa_off;            // Blockscaled only (packed SFA bytes)
    size_t    ipc_y_off;              // Blockscaled only (BF16 return buffer)
    size_t    ipc_meta_off;
    size_t    ipc_counter_off;
    size_t    ipc_dropped_off;
    size_t    ipc_barrier_off;
    size_t    ipc_tok_y_off;
    size_t    ipc_tok_gate_off;

    // =========================================================================
    // Local work buffers (not symmetric, regular cudaMalloc)
    // =========================================================================
    int*      local_eid;       // [capacity] extracted local expert IDs
    int*      order;           // [capacity] sort permutation
    int*      offsets;         // [n_local+1] expert offsets
    int*      dest;            // [capacity] dest mapping for gather
    int*      M_pad_dev;       // [1] padded row count
    Meta*     meta_copy;       // [capacity] snapshot of dispatch meta for return
    // Blockscaled-only workspace
    uint8_t*  sfa_gather_tmp;  // [max_pad * Hsf] gathered rowwise SFA (row-major)
    void*     sort_temp;       // CUB sort temp storage
    size_t    sort_temp_bytes;

    // Buffer layout info (for computing offsets)
    size_t    x_buf_offset;
    size_t    meta_offset;
    size_t    counter_offset;

    // Dimensions
    size_t capacity;
    int H, Ha, Hp, Hsf;
    int world, rank, local_world;
    int num_nodes;             // world / local_world
    int rdma_rank;             // rank / local_world
    int nvl_rank;              // rank % local_world
    int n_local;
    int tok_Ha;                // BF16 row stride for tok_y (>= H, multiple of 8)
    int align;                 // Row alignment (8 for BF16, 128 for blockscaled)
    int profile;               // 0=fp8, 1=nvfp4, -1=bf16
    bool initialized;
};

extern NvshmemState g_nvshmem;

// ============================================================================
// Initialization
// ============================================================================

// Get NVSHMEM UID for broadcast (call on rank 0 only)
void get_uid(void* uid_out);

// Get UID size in bytes
int get_uid_size();

// Initialize NVSHMEM with UID (call after NCCL broadcast of UID)
void init(const void* uid, int rank, int world, int local_world);

// Finalize NVSHMEM
void finalize();

// Allocate symmetric buffers for BF16 path (NVSHMEM + IPC)
void alloc_bf16(size_t capacity, int H, int n_local);

// Allocate symmetric buffers for blockscaled path (NVSHMEM + IPC)
void alloc_blockscaled(size_t capacity, int H, int n_local, int profile);

// Reset counters before dispatch
void reset_counters(cudaStream_t stream);

// ============================================================================
// IPC Buffer Management (for intra-node communication in hybrid mode)
// These functions manage the SEPARATE cudaMalloc'd IPC buffers that can be
// used with cudaIpcGetMemHandle/cudaIpcOpenMemHandle
// ============================================================================

// Get IPC handle for the local IPC buffer (BF16 path)
// Returns: cudaIpcMemHandle_t as raw bytes
void get_ipc_handle_bf16(void* handle_out);

// Get IPC handle for the local IPC buffer (blockscaled path)
void get_ipc_handle_blockscaled(void* handle_out);

// Open remote IPC handles and populate ipc_buffer_ptrs (BF16 path)
// handles: array of local_world cudaIpcMemHandle_t handles
void open_ipc_handles_bf16(const void* handles, int local_world);

// Open remote IPC handles (blockscaled path)
void open_ipc_handles_blockscaled(const void* handles, int local_world);

// Copy IPC buffer pointers to device (call after open_ipc_handles)
void sync_ipc_buffer_ptrs_bf16();
void sync_ipc_buffer_ptrs_blockscaled();

// ============================================================================
// Hybrid Dispatch (IPC intra-node, NVSHMEM inter-node)
// ============================================================================

// Dispatch BF16 tokens - hybrid path
// Writes to IPC buffer for local node, NVSHMEM for remote nodes
// Returns M_recv (number of rows received on this GPU)
int dispatch_hybrid_bf16(
    const __nv_bfloat16* x,    // [T, H]
    const int* eids,            // [T, K]
    const float* gates,         // [T, K]
    int T, int K,
    int align,                  // Per-expert row padding (8 for BF16, 128 for blockscaled)
    void* Xe_out,               // Output buffer [M_pad, H]
    int* offs_pad_out,          // [n_local] padded offsets
    int* dest_out,              // [M_recv] dest indices for unpad
    int64_t* row_id_out,         // [M_recv] (src_rank,tok,slot) sorted order
    float* gate_out,             // [M_recv] gate weights sorted order
    int* M_pad_out,             // Padded row count
    // IPC state from rdep.cu
    void** ipc_buffer_ptrs,     // [local_world] IPC buffer pointers
    size_t ipc_meta_off,        // Offset to meta in IPC buffer
    size_t ipc_counter_off,     // Offset to counter in IPC buffer
    int** ipc_barrier_ptrs,     // [local_world] IPC barrier signal pointers
    cudaStream_t stream);

// Dispatch blockscaled tokens - hybrid path
int dispatch_hybrid_blockscaled(
    const __nv_bfloat16* x,    // [T, H]
    const int* eids,            // [T, K]
    const float* gates,         // [T, K]
    int T, int K,
    void* Xe_q_out,             // Packed output [M_pad, Hp]
    void* Xe_sf_out,            // Scale factors [M_pad, Hsf]
    int* offs_pad_out,          // [n_local] padded offsets
    int* dest_out,              // [M_recv] dest indices
    int64_t* row_id_out,         // [M_recv] (src_rank,tok,slot) sorted order
    float* gate_out,             // [M_recv] gate weights sorted order
    int* M_pad_out,             // Padded row count
    // IPC state from rdep.cu
    void** ipc_buffer_ptrs,
    size_t ipc_x_off,
    size_t ipc_sfa_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off,
    int** ipc_barrier_ptrs,
    cudaStream_t stream);

// Gather blockscaled activations from hybrid IPC buffer into padded outputs.
// Requires a prior dispatch_hybrid_blockscaled call (meta-only or full) to populate:
// - g_nvshmem.order, g_nvshmem.dest, and local IPC receive buffers.
void gather_xe_hybrid_blockscaled(
    void* Xe_q_out,   // [M_pad, Hp] uint16
    void* Xe_sf_out,  // [M_pad, Hsf] uint8 (packed MMA layout)
    int M_recv,
    int M_pad,
    cudaStream_t stream);

// ============================================================================
// Hybrid Return Scatter
// ============================================================================

// Return scatter BF16 - hybrid path
// Gathers expert outputs from local GPU, writes to source GPUs
void return_scatter_hybrid_bf16(
    const __nv_bfloat16* Ye,    // [M_recv, H] expert outputs
    float* out,                  // [T, H] output (accumulated)
    int M_recv, int T, int K,
    // IPC state
    void** ipc_buffer_ptrs,
    int** ipc_barrier_ptrs,
    cudaStream_t stream);

// Return scatter BF16 - hybrid path (from padded output layout)
// Ye_pad: [M_pad, H], indexed via g_nvshmem.dest[sorted_i] -> pad_i.
void return_scatter_hybrid_bf16_from_pad(
    const __nv_bfloat16* Ye_pad, // [M_pad, H]
    float* out,                  // [T, H] output (accumulated)
    int M_recv, int T, int K,
    // IPC state
    void** ipc_buffer_ptrs,
    int** ipc_barrier_ptrs,
    cudaStream_t stream);

// Return scatter blockscaled - hybrid path
void return_scatter_hybrid_blockscaled(
    const __nv_bfloat16* Ye,    // [M_recv, H] expert outputs
    float* out,                  // [T, H] output (accumulated)
    int M_recv, int T, int K,
    // IPC state
    void** ipc_buffer_ptrs,
    int** ipc_barrier_ptrs,
    cudaStream_t stream);

// Return scatter blockscaled - hybrid path (from padded output layout)
void return_scatter_hybrid_blockscaled_from_pad(
    const __nv_bfloat16* Ye_pad, // [M_pad, H]
    float* out,                  // [T, H] output (accumulated)
    int M_recv, int T, int K,
    // IPC state
    void** ipc_buffer_ptrs,
    int** ipc_barrier_ptrs,
    cudaStream_t stream);

// ============================================================================
// Hybrid Backward (BF16 payload)
// ============================================================================

// Backward gather: stages dY via IPC+NVSHMEM (push), computes local dYe and returns dGate.
void gather_dy_hybrid_bf16(
    const __nv_bfloat16* dY_local,   // [T, H]
    const int* eids,                 // [T, K] global expert ids
    const __nv_bfloat16* Ye_pad,     // [M_pad, H] (indexed via dest)
    const int64_t* row_id,           // [M]
    const float* gate_sorted,        // [M]
    __nv_bfloat16* dYe_out,          // [M, H]
    float* dGate_sorted_out,         // [M]
    float* dGates_tk_out,            // [T, K]
    int M, int T, int H, int K,
    cudaStream_t stream);

// Backward scatter: sends dXe rows back to token owners via fixed tok-slot writes + local reduction.
void scatter_dx_hybrid_bf16(
    const __nv_bfloat16* dXe_sorted,  // [M, H]
    const int64_t* row_id,            // [M]
    float* dX_out,                    // [T, H] (float32 accum)
    int M, int T, int H, int K,
    cudaStream_t stream);

// ============================================================================
// Synchronization
// ============================================================================

// Two-phase barrier: NVL (intra-node) then RDMA (inter-node)
// nvl_barrier_ptrs: IPC barrier pointers for local node
// Must be called from a kernel
__device__ void hybrid_barrier(
    int** nvl_barrier_ptrs,
    int* rdma_barrier_signals,
    int nvl_rank, int rdma_rank,
    int local_world, int num_nodes);

// Host-side NVSHMEM barrier (all PEs)
void barrier();

// Host-side NVSHMEM quiet (ensure all puts complete)
void quiet();

// Quiet on stream
void quiet_on_stream(cudaStream_t stream);

}  // namespace nvshmem
}  // namespace rdep

#endif  // WITH_NVSHMEM
