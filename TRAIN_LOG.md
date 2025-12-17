# Training Log

## 2025-12-17: IPC Mode Validation - 8-GPU Moonlight (200 steps)

### Summary

Successfully validated 8-GPU IPC mode training across all three precision profiles (BF16, FP8, NVFP4) after fixing critical collectiveness violations in the RDEP dispatch/combine paths.

### Configuration

- **Model**: Moonlight (2B params, 27 layers, 64 experts, top-6 routing)
- **Hardware**: 8x NVIDIA B200 GPUs (single node, IPC mode)
- **Batch**: 64 global (8 per GPU), 4096 seq_len
- **Steps**: 200
- **Data**: FineWeb-Edu (streaming, ~8M tokens available, wrapping enabled)

### Results

| Profile | Final Loss | Start Loss | Peak Memory | Avg TPS (node) | ms/step |
|---------|-----------|------------|-------------|----------------|---------|
| BF16    | 7.27      | 12.34      | 137.87 GiB (77%) | ~122k | ~2100 |
| FP8     | 7.44      | 12.34      | 144.90 GiB (81%) | ~104k | ~2500 |
| NVFP4   | 7.63      | 12.34      | 143.75 GiB (81%) | ~110k | ~2350 |

### BF16 Training Curve

```
step:   1  loss: 12.3430  grad_norm: 192.8154  memory: 91.60GiB   tps: 2,105
step:  10  loss: 11.6710  grad_norm: 585.2727  memory: 136.27GiB  tps: 14,635
step:  20  loss: 10.5757  grad_norm: 327.2692  memory: 137.44GiB  tps: 15,168
step:  30  loss: 10.0417  grad_norm: 115.0629  memory: 137.66GiB  tps: 14,480
step:  40  loss:  9.3846  grad_norm: 473.7863  memory: 137.66GiB  tps: 14,866
step:  50  loss:  8.8403  grad_norm:  54.3825  memory: 137.66GiB  tps: 15,296
step:  60  loss:  8.2782  grad_norm:  16.9639  memory: 137.66GiB  tps: 15,521
step:  70  loss:  7.9111  grad_norm:  28.2059  memory: 137.83GiB  tps: 15,448
step:  80  loss:  7.7636  grad_norm:  12.3717  memory: 137.83GiB  tps: 15,150
step:  90  loss:  7.8360  grad_norm:  21.4122  memory: 137.87GiB  tps: 14,241
step: 100  loss:  7.8120  grad_norm:   4.5645  memory: 137.87GiB  tps: 14,834
step: 110  loss:  7.6934  grad_norm:   8.6194  memory: 137.87GiB  tps: 14,944
step: 120  loss:  7.6188  grad_norm:   6.4448  memory: 137.87GiB  tps: 15,113
step: 130  loss:  7.6708  grad_norm:   6.8947  memory: 137.87GiB  tps: 15,280
step: 140  loss:  7.5647  grad_norm:  14.0279  memory: 137.87GiB  tps: 14,974
step: 150  loss:  7.5816  grad_norm:  16.5601  memory: 137.87GiB  tps: 14,873
step: 160  loss:  7.5194  grad_norm:   3.1803  memory: 137.87GiB  tps: 14,460
step: 170  loss:  7.4050  grad_norm:   1.8129  memory: 137.87GiB  tps: 15,083
step: 180  loss:  7.5820  grad_norm:   5.1845  memory: 137.87GiB  tps: 15,333
step: 190  loss:  7.3799  grad_norm:   4.1170  memory: 137.87GiB  tps: 16,084
step: 200  loss:  7.2701  grad_norm:   6.8444  memory: 137.87GiB  tps: 15,657
```

### FP8 Training Curve

```
step:   1  loss: 12.3431  grad_norm:  63.6732  memory: 100.19GiB  tps: 1,608
step:  10  loss: 11.6883  grad_norm: 717.5409  memory: 144.61GiB  tps: 12,440
step:  20  loss: 10.5681  grad_norm: 1220.641  memory: 144.65GiB  tps: 12,920
step:  30  loss: 10.0521  grad_norm:  56.1716  memory: 144.65GiB  tps: 12,343
step:  40  loss:  9.3772  grad_norm:  28.2424  memory: 144.65GiB  tps: 12,335
step:  50  loss:  8.8243  grad_norm:  68.6010  memory: 144.65GiB  tps: 12,967
step:  60  loss:  8.2951  grad_norm: 100.7163  memory: 144.65GiB  tps: 13,473
step:  70  loss:  7.9088  grad_norm:  14.9101  memory: 144.65GiB  tps: 13,169
step:  80  loss:  7.7758  grad_norm:  16.4042  memory: 144.65GiB  tps: 12,636
step:  90  loss:  7.8626  grad_norm:   4.3627  memory: 144.65GiB  tps: 12,622
step: 100  loss:  7.8425  grad_norm:  14.4055  memory: 144.65GiB  tps: 12,223
step: 110  loss:  7.7157  grad_norm:   5.3438  memory: 144.65GiB  tps: 12,614
step: 120  loss:  7.6947  grad_norm:   9.7582  memory: 144.65GiB  tps: 13,069
step: 130  loss:  7.7169  grad_norm:   8.7002  memory: 144.65GiB  tps: 13,409
step: 140  loss:  7.6229  grad_norm:   1.3065  memory: 144.65GiB  tps: 12,999
step: 150  loss:  7.6564  grad_norm:   1.4123  memory: 144.65GiB  tps: 12,149
step: 160  loss:  7.5971  grad_norm:   1.8475  memory: 144.65GiB  tps: 12,645
step: 170  loss:  7.4977  grad_norm:   1.7556  memory: 144.65GiB  tps: 13,433
step: 180  loss:  7.6985  grad_norm:   2.1730  memory: 144.90GiB  tps: 13,655
step: 190  loss:  7.5219  grad_norm:   1.8343  memory: 144.90GiB  tps: 13,618
step: 200  loss:  7.4363  grad_norm:   1.6337  memory: 144.90GiB  tps: 13,761
```

### NVFP4 Training Curve

```
step:   1  loss: 12.3435  grad_norm:  343.772  memory: 98.31GiB   tps: 1,823
step:  10  loss: 11.7380  grad_norm:  206.888  memory: 142.42GiB  tps: 12,954
step:  20  loss: 10.5855  grad_norm: 35333.01  memory: 143.57GiB  tps: 13,303
step:  30  loss: 10.0418  grad_norm:  62.7358  memory: 143.57GiB  tps: 12,861
step:  40  loss:  9.4030  grad_norm:  49.0843  memory: 143.57GiB  tps: 13,204
step:  50  loss:  8.8292  grad_norm: 1513.939  memory: 143.75GiB  tps: 13,355
step:  60  loss:  8.2849  grad_norm:  60.2146  memory: 143.75GiB  tps: 13,683
step:  70  loss:  7.9172  grad_norm:  14.3719  memory: 143.75GiB  tps: 13,776
step:  80  loss:  7.7788  grad_norm:  46.5989  memory: 143.75GiB  tps: 13,248
step:  90  loss:  7.8584  grad_norm:  42.0544  memory: 143.75GiB  tps: 13,359
step: 100  loss:  7.8350  grad_norm:  19.0194  memory: 143.75GiB  tps: 13,170
step: 110  loss:  7.7391  grad_norm:   4.0108  memory: 143.75GiB  tps: 14,805
step: 120  loss:  7.6933  grad_norm:   1.5168  memory: 143.75GiB  tps: 15,930
step: 130  loss:  7.7567  grad_norm:   3.2053  memory: 143.75GiB  tps: 15,096
step: 140  loss:  7.6988  grad_norm:   1.4630  memory: 143.75GiB  tps: 14,305
step: 150  loss:  7.7576  grad_norm:   1.4027  memory: 143.75GiB  tps: 13,562
step: 160  loss:  7.7350  grad_norm:   1.3798  memory: 143.75GiB  tps: 14,063
step: 170  loss:  7.6694  grad_norm:   1.1640  memory: 143.75GiB  tps: 14,203
step: 180  loss:  7.8404  grad_norm:   1.9457  memory: 143.75GiB  tps: 14,554
step: 190  loss:  7.6871  grad_norm:   1.9864  memory: 143.75GiB  tps: 14,896
step: 200  loss:  7.6262  grad_norm:  11.5088  memory: 143.75GiB  tps: 14,069
```

### Bugs Fixed (This Session)

Prior to these runs, 8-GPU IPC mode was failing at ~step 30 with phase barrier timeouts. Two critical bugs were identified and fixed:

#### 1. Collectiveness Violation in Early Returns

**Problem**: When a rank received M_recv=0 tokens from dispatch, the forward/backward paths returned early and skipped IPC collectives (return_scatter, gather_dy_dist_bf16, scatter_dx_dist_bf16). This caused the global phase counter `g_ipc_phase_bf16` to desync across ranks, leading to deterministic deadlocks.

**Fix**: Modified early-return paths in `_MoEBf16Fused` and `_MoEBlockscaledFused` to still call IPC collectives with M=0, ensuring all ranks participate in barriers even when they have no local work.

```python
# Before (broken):
if M_recv <= 0:
    return zeros  # Skips barriers!

# After (fixed):
if M_recv <= 0:
    if is_dist:
        _C.return_scatter(dummy.data_ptr(), out.data_ptr(), 0, T, K, stream)  # Participate in barrier
    return zeros
```

#### 2. Invalid Output Pointer in Blockscaled Backward

**Problem**: `_MoEBlockscaledFused.backward` passed `torch.empty(0).data_ptr()` for the `dGates_tk_out` parameter, but the CUDA kernel `k_collect_tok_gate_ipc` unconditionally writes T*K floats to that pointer. This caused "unspecified launch failure" errors.

**Fix**: Allocate a properly-sized buffer:

```python
# Before (broken):
torch.empty(0, device=device).data_ptr()  # Invalid!

# After (fixed):
dGates_tk_f32 = torch.zeros(int(T), int(K), device=device, dtype=torch.float32)
```

### Observations

1. **Loss Convergence**: All profiles show healthy convergence from ~12.3 to ~7.3-7.6 over 200 steps
2. **Router Stability**: CV (coefficient of variation) stabilizes around 170-220%, max load ~11-14%
3. **Memory**: BF16 uses least memory (77%), FP8/NVFP4 use ~81% due to quantization buffers
4. **Throughput**: BF16 achieves highest throughput (~15k TPS per GPU) due to simpler compute path
5. **Gradient Norms**: Occasional spikes (especially NVFP4 step 20: 35k) but training remains stable

### Next Steps

- [ ] Longer training runs (1000+ steps) to verify sustained stability
- [ ] Multi-node validation with NVSHMEM hybrid mode
- [ ] Checkpoint save/resume testing
- [ ] Gradient accumulation validation
