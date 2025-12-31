// Shared CUTLASS SF swizzle offset computation.
// Single source of truth - do NOT duplicate this math elsewhere.
//
// Canonical packed SFA contract:
//   - Shape: [capacity, sf_k] uint8, where sf_k = K/32 and sf_k % 4 == 0
//   - Expert e's SFA block starts at offs[e] * sf_k bytes
//   - Indexing within block uses cutlass_sf_swizzle_offset(m_local, k_sf, M_e, sf_k)
//   - Padding rows must be zeroed before consumption
//
// This layout is used by:
//   - Xe_sf: input activations for GEMM1 (K=H)
//   - A_sf:  post-SwiGLU activations for GEMM3 (K=Dff)

#pragma once

#include <cstddef>
#include <cstdint>

namespace nmoe {

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

__device__ __forceinline__ size_t cutlass_sf_swizzle_offset(
    size_t m, size_t k, uint32_t M, uint32_t sf_k)
{
    (void)M;  // M is for documentation/future use; swizzle doesn't depend on it

    constexpr uint32_t atom_m = 128;
    constexpr uint32_t atom_k = 4;
    constexpr uint32_t atom_size = atom_m * atom_k;  // 512
    const uint32_t rest_k = sf_k / atom_k;

    const size_t m_32 = m % 32;
    const size_t m_4  = (m / 32) % 4;
    const size_t m_rest = m / atom_m;

    const size_t k_4 = k % atom_k;
    const size_t k_rest = k / atom_k;

    const size_t atom_offset = m_32 * 16 + m_4 * 4 + k_4;
    const size_t atom_idx = m_rest * rest_k + k_rest;

    return atom_idx * atom_size + atom_offset;
}

}  // namespace nmoe

// Zero padding gaps kernel is defined in quant.cu to avoid multiple definition.
// Use the extern "C" wrapper: zero_sfa_padding_gaps(sfa, offs_pad, offs_unpad, E, sf_k, stream)
