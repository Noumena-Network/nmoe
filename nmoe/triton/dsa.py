"""DeepSeek Sparse Attention (DSA) Lightning Indexer.

Paper: DeepSeek-V3.2 Section 2.1, Equation 1:
    I_{t,s} = Σ_{j=1}^{H^I} w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)

Uses score computation kernel + torch.topk for top-k selection.
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Score Computation Kernel (Eq 1)
# =============================================================================

@triton.jit
def _compute_indexer_scores(
    Q, K, W,      # Input tensors
    Scores,       # Output: [B, M, N]
    M, N,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_wb, stride_wm, stride_wh,
    stride_sb, stride_sm, stride_sn,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute indexer scores: I_{t,s} = Σ_h w_{t,h} · ReLU(q_{t,h} · k_s)

    Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N), B)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Base pointers for this batch
    Q_base = Q + pid_b * stride_qb
    K_base = K + pid_b * stride_kb
    W_base = W + pid_b * stride_wb

    # Load keys for this block: [BLOCK_N, D]
    k_ptrs = K_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    k_block = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    # Compute scores: Σ_h w[m,h] * ReLU(q[m,h,:] · k[n,:])
    scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for h in tl.static_range(H):
        # Load q[m, h, :] for all m in block
        q_ptrs = Q_base + offs_m[:, None] * stride_qm + h * stride_qh + offs_d[None, :] * stride_qd
        q_h = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        # Load w[m, h] for all m in block
        w_ptrs = W_base + offs_m * stride_wm + h * stride_wh
        w_h = tl.load(w_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        # Compute q · k^T and apply ReLU
        dots = tl.dot(q_h, tl.trans(k_block), allow_tf32=False)  # [BLOCK_M, BLOCK_N]
        dots = tl.maximum(dots, 0.0)  # ReLU

        # Weighted sum
        scores += w_h[:, None] * dots

    # Apply causal mask
    if CAUSAL:
        causal_mask = offs_n[None, :] > offs_m[:, None]
        scores = tl.where(causal_mask, float('-inf'), scores)

    # Store scores
    score_ptrs = Scores + pid_b * stride_sb + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    tl.store(score_ptrs, scores, mask=mask_m[:, None] & mask_n[None, :])


def compute_indexer_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Compute indexer scores using Triton kernel.

    Args:
        q: [B, M, H, D] - indexer queries
        k: [B, N, D] - indexer keys (MQA-style)
        w: [B, M, H] - per-head weights
        causal: apply causal masking

    Returns:
        scores: [B, M, N] - indexer scores
    """
    B, M, H, D = q.shape
    N = k.shape[1]

    scores = torch.empty(B, M, N, device=q.device, dtype=torch.float32)

    # Block sizes tuned for B200
    BLOCK_M = 32
    BLOCK_N = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), B)

    _compute_indexer_scores[grid](
        q, k, w,
        scores,
        M, N, H, D,
        *q.stride(), *k.stride(), *w.stride(), *scores.stride(),
        CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    return scores


def lightning_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    top_k: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute lightning indexer scores and select top-k indices.

    This implementation computes scores using a Triton kernel then uses
    torch.topk for selection. This is efficient for training where we
    need reliable correctness for any k value.

    For inference with very long sequences, a fused streaming implementation
    would be more memory efficient.

    Args:
        q: [B, M, H, D] - indexer queries
        k: [B, N, D] - indexer keys (MQA-style)
        w: [B, M, H] - per-head weights
        top_k: number of keys to select per query
        causal: apply causal masking (query m attends only to keys n <= m)

    Returns:
        values: [B, M, k] - top-k scores
        indices: [B, M, k] - top-k key indices
    """
    B, M, H, D = q.shape
    N = k.shape[1]
    K_SEL = min(top_k, N)

    # Compute all scores using Triton kernel
    scores = compute_indexer_scores(q, k, w, causal=causal)

    # Select top-k using torch.topk (well-optimized for any k)
    vals, idxs = scores.topk(K_SEL, dim=-1)

    return vals, idxs.to(torch.int32)


def lightning_indexer_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    top_k: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation for correctness testing."""
    B, M, H, D = q.shape
    N = k.shape[1]

    # Equation 1: I_{t,s} = Σ_h w_{t,h} · ReLU(q_{t,h} · k_s)
    scores = torch.einsum('bmhd,bnd->bmhn', q.float(), k.float())
    scores = torch.relu(scores)
    scores = (w.float().unsqueeze(-1) * scores).sum(dim=2)  # [B, M, N]

    if causal:
        mask = torch.triu(torch.ones(M, N, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0), float('-inf'))

    K_SEL = min(top_k, N)
    vals, idxs = scores.topk(K_SEL, dim=-1)
    return vals, idxs.to(torch.int32)


# =============================================================================
# Fused Streaming Top-K (for small k <= BLOCK_N)
# =============================================================================
# This section provides a fused implementation for cases where k is small enough
# to fit in a single block. For larger k, use lightning_indexer above.

@triton.jit
def _fpval_to_key(x):
    """Convert float32 bits to sortable key."""
    # For uint32 input representing float32 bits
    tm: tl.constexpr = 0x80000000  # sign bit
    fm: tl.constexpr = 0xFFFFFFFF  # all bits
    tm_arr = tl.full(x.shape, tm, dtype=tl.uint32)
    fm_arr = tl.full(x.shape, fm, dtype=tl.uint32)
    return x ^ tl.where((x & tm_arr) != 0, fm_arr, tm_arr)


@triton.jit
def _key_to_fpval(x):
    """Convert sortable key back to float32 bits."""
    tm: tl.constexpr = 0x80000000
    fm: tl.constexpr = 0xFFFFFFFF
    tm_arr = tl.full(x.shape, tm, dtype=tl.uint32)
    fm_arr = tl.full(x.shape, fm, dtype=tl.uint32)
    return x ^ tl.where((x & tm_arr) == 0, fm_arr, tm_arr)


@triton.jit
def _lightning_indexer_fused_small_k(
    Q, K, W,
    OutVals, OutIdxs,
    M, N,
    H: tl.constexpr,
    D: tl.constexpr,
    K_SEL: tl.constexpr,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_wb, stride_wm, stride_wh,
    stride_vb, stride_vm, stride_vk,
    stride_ib, stride_im, stride_ik,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_PAD: tl.constexpr,
):
    """
    Fused lightning indexer with streaming top-k for small k.

    Requires: K_SEL <= BLOCK_N

    Grid: (cdiv(M, BLOCK_M), B)
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < M

    # Base pointers
    Q_base = Q + pid_b * stride_qb
    K_base = K + pid_b * stride_kb
    W_base = W + pid_b * stride_wb

    # Number of iterations (computed at compile time)
    loop_iterations: tl.constexpr = N_PAD // BLOCK_N - 1

    # Start from last block (peeled first iteration with masking)
    offs_n = loop_iterations * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n[None, :] < N

    # First iteration - load keys and compute scores
    k_ptrs = K_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    k_block = tl.load(k_ptrs, mask=(offs_n < N)[:, None], other=0.0).to(tl.float32)

    scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for h in tl.static_range(H):
        q_ptrs = Q_base + offs_m[:, None] * stride_qm + h * stride_qh + offs_d[None, :] * stride_qd
        q_h = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        w_ptrs = W_base + offs_m * stride_wm + h * stride_wh
        w_h = tl.load(w_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        dots = tl.dot(q_h, tl.trans(k_block), allow_tf32=False)
        dots = tl.maximum(dots, 0.0)
        scores += w_h[:, None] * dots

    # Apply masks
    if CAUSAL:
        causal_ok = offs_n[None, :] <= offs_m[:, None]
        valid = mask_m[:, None] & mask_n & causal_ok
    else:
        valid = mask_m[:, None] & mask_n
    scores = tl.where(valid, scores, float('-inf'))

    # Pack scores and indices for sorting
    score_bits = scores.to(tl.uint32, bitcast=True)
    score_key = _fpval_to_key(score_bits)
    idx_key = (N_PAD - offs_n).to(tl.uint32)
    packed = (score_key.to(tl.uint64) << 16) | idx_key[None, :].to(tl.uint64)

    # First topk
    acc = tl.topk(packed, K_SEL, dim=1)

    # Subsequent iterations
    for _i in (tl.static_range if loop_iterations <= 4 else range)(loop_iterations):
        acc = tl.bitonic_merge(acc)
        offs_n = offs_n - BLOCK_N

        k_ptrs = K_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_block = tl.load(k_ptrs).to(tl.float32)

        scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for h in tl.static_range(H):
            q_ptrs = Q_base + offs_m[:, None] * stride_qm + h * stride_qh + offs_d[None, :] * stride_qd
            q_h = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
            w_ptrs = W_base + offs_m * stride_wm + h * stride_wh
            w_h = tl.load(w_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            dots = tl.dot(q_h, tl.trans(k_block), allow_tf32=False)
            dots = tl.maximum(dots, 0.0)
            scores += w_h[:, None] * dots

        if CAUSAL:
            causal_ok = offs_n[None, :] <= offs_m[:, None]
            scores = tl.where(causal_ok, scores, float('-inf'))

        score_bits = scores.to(tl.uint32, bitcast=True)
        score_key = _fpval_to_key(score_bits)
        idx_key = (N_PAD - offs_n).to(tl.uint32)
        packed = (score_key.to(tl.uint64) << 16) | idx_key[None, :].to(tl.uint64)

        block_topk = tl.topk(packed, K_SEL, dim=1)
        acc = tl.maximum(acc, block_topk)

    # Rotate and sort by index
    acc = (acc << 48) | (acc >> 16)
    acc = tl.sort(acc, dim=1, descending=True)

    # Extract results
    y_indices_raw = (acc >> 48).to(tl.uint32)
    y_indices = (N_PAD - y_indices_raw).to(tl.int32)
    y_values_raw = acc.to(tl.uint32)
    y_values = _key_to_fpval(y_values_raw).to(tl.float32, bitcast=True)

    # Store
    offs_k = tl.arange(0, K_SEL)
    v_ptrs = OutVals + pid_b * stride_vb + offs_m[:, None] * stride_vm + offs_k[None, :] * stride_vk
    i_ptrs = OutIdxs + pid_b * stride_ib + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik

    tl.store(v_ptrs, y_values, mask=mask_m[:, None])
    tl.store(i_ptrs, y_indices, mask=mask_m[:, None])


def lightning_indexer_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    top_k: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused lightning indexer for small k values (k <= 64).

    For larger k, use lightning_indexer() which uses a two-pass approach.
    """
    B, M, H, D = q.shape
    N = k.shape[1]
    K_SEL = min(top_k, N)

    # This fused version only works for small k
    K_SEL_PAD = triton.next_power_of_2(K_SEL)
    BLOCK_N = max(K_SEL_PAD, 64)

    if BLOCK_N > 64:
        # Fall back to two-pass for large k
        return lightning_indexer(q, k, w, top_k, causal)

    out_vals = torch.empty(B, M, K_SEL_PAD, device=q.device, dtype=torch.float32)
    out_idxs = torch.empty(B, M, K_SEL_PAD, device=q.device, dtype=torch.int32)

    BLOCK_M = 32

    # N_PAD must be multiple of BLOCK_N
    N_PAD = ((N + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

    grid = (triton.cdiv(M, BLOCK_M), B)

    _lightning_indexer_fused_small_k[grid](
        q, k, w,
        out_vals, out_idxs,
        M, N, H, D, K_SEL_PAD,
        *q.stride(), *k.stride(), *w.stride(),
        *out_vals.stride(), *out_idxs.stride(),
        CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        N_PAD=N_PAD,
        num_warps=4,
        num_stages=2,
    )

    return out_vals[:, :, :K_SEL], out_idxs[:, :, :K_SEL]
