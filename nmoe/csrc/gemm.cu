#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
 
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <mutex>
 
namespace {
 
inline cudaError_t cuda_from_cublas(cublasStatus_t s) {
  return (s == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}
 
struct LtState {
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 32ull << 20;
  std::mutex mu;
};
 
LtState& lt_state() {
  static LtState s;
  return s;
}
 
cudaError_t ensure_lt(cudaStream_t stream) {
  auto& s = lt_state();
  std::lock_guard<std::mutex> lock(s.mu);
  if (s.handle == nullptr) {
    auto st = cublasLtCreate(&s.handle);
    if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  }
  if (s.workspace == nullptr) {
    auto err = cudaMalloc(&s.workspace, s.workspace_bytes);
    if (err != cudaSuccess) return err;
  }
  return cudaSuccess;
}
 
struct MatmulKey {
  int opA;
  int opB;
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldc;
};
 
struct MatmulKeyHash {
  size_t operator()(const MatmulKey& x) const noexcept {
    size_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
      h ^= v;
      h *= 1099511628211ull;
    };
    mix(static_cast<uint64_t>(x.opA));
    mix(static_cast<uint64_t>(x.opB));
    mix(static_cast<uint64_t>(x.m));
    mix(static_cast<uint64_t>(x.n));
    mix(static_cast<uint64_t>(x.k));
    mix(static_cast<uint64_t>(x.lda));
    mix(static_cast<uint64_t>(x.ldb));
    mix(static_cast<uint64_t>(x.ldc));
    return h;
  }
};
 
inline bool operator==(const MatmulKey& a, const MatmulKey& b) {
  return a.opA == b.opA && a.opB == b.opB && a.m == b.m && a.n == b.n && a.k == b.k &&
         a.lda == b.lda && a.ldb == b.ldb && a.ldc == b.ldc;
}
 
struct MatmulPlan {
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatmulAlgo_t algo{};
 
  MatmulPlan() = default;
  MatmulPlan(const MatmulPlan&) = delete;
  MatmulPlan& operator=(const MatmulPlan&) = delete;
 
  ~MatmulPlan() {
    if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc);
  }
};
 
std::mutex plan_mu;
std::unordered_map<MatmulKey, std::unique_ptr<MatmulPlan>, MatmulKeyHash> plan_cache;
 
cudaError_t get_plan(cublasLtHandle_t handle,
                     cublasOperation_t opA,
                     cublasOperation_t opB,
                     int m,
                     int n,
                     int k,
                     int lda,
                     int ldb,
                     int ldc,
                     size_t workspace_bytes,
                     MatmulPlan** out) {
  const MatmulKey key{
      static_cast<int>(opA),
      static_cast<int>(opB),
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
  };
 
  {
    std::lock_guard<std::mutex> lock(plan_mu);
    auto it = plan_cache.find(key);
    if (it != plan_cache.end()) {
      *out = it->second.get();
      return cudaSuccess;
    }
  }
 
  auto plan = std::make_unique<MatmulPlan>();
 
  const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  const cudaDataType_t scale_type = CUDA_R_32F;
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
 
  cublasStatus_t st = cublasLtMatmulDescCreate(&plan->matmul_desc, compute_type, scale_type);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
 
  st = cublasLtMatmulDescSetAttribute(
      plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatmulDescSetAttribute(
      plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
 
  const int a_rows = (opA == CUBLAS_OP_N) ? m : k;
  const int a_cols = (opA == CUBLAS_OP_N) ? k : m;
  const int b_rows = (opB == CUBLAS_OP_N) ? k : n;
  const int b_cols = (opB == CUBLAS_OP_N) ? n : k;
 
  st = cublasLtMatrixLayoutCreate(&plan->a_layout, CUDA_R_16BF, a_rows, a_cols, lda);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatrixLayoutSetAttribute(
      plan->a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
 
  st = cublasLtMatrixLayoutCreate(&plan->b_layout, CUDA_R_16BF, b_rows, b_cols, ldb);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatrixLayoutSetAttribute(
      plan->b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
 
  st = cublasLtMatrixLayoutCreate(&plan->c_layout, CUDA_R_16BF, m, n, ldc);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatrixLayoutSetAttribute(
      plan->c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
 
  cublasLtMatmulPreference_t pref = nullptr;
  st = cublasLtMatmulPreferenceCreate(&pref);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes));
  if (st != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(pref);
    return cuda_from_cublas(st);
  }
 
  cublasLtMatmulHeuristicResult_t heur{};
  int returned = 0;
  st = cublasLtMatmulAlgoGetHeuristic(
      handle, plan->matmul_desc,
      plan->a_layout, plan->b_layout,
      plan->c_layout, plan->c_layout,
      pref,
      1,
      &heur,
      &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  if (st != CUBLAS_STATUS_SUCCESS || returned == 0) return cuda_from_cublas(st);
 
  plan->algo = heur.algo;
 
  {
    std::lock_guard<std::mutex> lock(plan_mu);
    auto [it, inserted] = plan_cache.emplace(key, std::move(plan));
    if (!inserted) {
      *out = it->second.get();
      return cudaSuccess;
    }
    *out = it->second.get();
  }
  return cudaSuccess;
}
 
cudaError_t lt_gemm_bf16_f32accum(
    const __nv_bfloat16* A,
    cublasOperation_t opA,
    const __nv_bfloat16* B,
    cublasOperation_t opB,
    __nv_bfloat16* C,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    float alpha,
    float beta,
    cudaStream_t stream) {
  auto err = ensure_lt(stream);
  if (err != cudaSuccess) return err;
 
  auto& s = lt_state();
 
  MatmulPlan* plan = nullptr;
  err = get_plan(s.handle, opA, opB, m, n, k, lda, ldb, ldc, s.workspace_bytes, &plan);
  if (err != cudaSuccess) return err;
 
  cublasStatus_t st = cublasLtMatmul(
      s.handle,
      plan->matmul_desc,
      &alpha,
      A,
      plan->a_layout,
      B,
      plan->b_layout,
      &beta,
      C,
      plan->c_layout,
      C,
      plan->c_layout,
      &plan->algo,
      s.workspace,
      s.workspace_bytes,
      stream);
  return cuda_from_cublas(st);
}
 
}  // namespace

extern "C" cudaError_t bf16_wgrad_w2_cublaslt(const void* A,
                                             const void* dY,
                                             void* dW2_out,
                                             const int32_t* offs_pad,
                                             int E,
                                             int H,
                                             int Dff,
                                             cudaStream_t stream) {
  if (E <= 0 || H <= 0 || Dff <= 0) return cudaSuccess;

  const auto* A_bf16 = reinterpret_cast<const __nv_bfloat16*>(A);
  const auto* dY_bf16 = reinterpret_cast<const __nv_bfloat16*>(dY);
  auto* dW2_bf16 = reinterpret_cast<__nv_bfloat16*>(dW2_out);

  const int lda = Dff;  // A [m, Dff]
  const int ldb = H;    // dY [m, H]
  const int ldc = H;    // dW2 [Dff, H]

  int32_t start = 0;
  for (int e = 0; e < E; ++e) {
    const int32_t end = offs_pad[e];
    const int m = static_cast<int>(end - start);
    if (m < 0) return cudaErrorInvalidValue;
    __nv_bfloat16* dW2e = dW2_bf16 + static_cast<size_t>(e) * static_cast<size_t>(Dff) * static_cast<size_t>(H);
    if (m > 0) {
      const __nv_bfloat16* Ae = A_bf16 + static_cast<size_t>(start) * static_cast<size_t>(lda);
      const __nv_bfloat16* dYe = dY_bf16 + static_cast<size_t>(start) * static_cast<size_t>(ldb);
      auto err = lt_gemm_bf16_f32accum(Ae, CUBLAS_OP_T, dYe, CUBLAS_OP_N, dW2e,
                                  /*m=*/Dff, /*n=*/H, /*k=*/m,
                                  /*lda=*/lda, /*ldb=*/ldb, /*ldc=*/ldc,
                                  /*alpha=*/1.0f, /*beta=*/0.0f,
                                  stream);
      if (err != cudaSuccess) return err;
    } else {
      // Expert has zero tokens: ensure its gradient slice is exactly zero.
      auto err = cudaMemsetAsync(dW2e, 0, static_cast<size_t>(Dff) * static_cast<size_t>(H) * sizeof(__nv_bfloat16), stream);
      if (err != cudaSuccess) return err;
    }
    start = end;
  }
  return cudaSuccess;
}
 
extern "C" cudaError_t bf16_wgrad_w13_cublaslt(const void* X,
                                              const void* dH,
                                              void* dW_out,
                                              const int32_t* offs_pad,
                                              int E,
                                              int H,
                                              int Dff,
                                              cudaStream_t stream) {
  if (E <= 0 || H <= 0 || Dff <= 0) return cudaSuccess;

  const auto* X_bf16 = reinterpret_cast<const __nv_bfloat16*>(X);
  const auto* dH_bf16 = reinterpret_cast<const __nv_bfloat16*>(dH);
  auto* dW_bf16 = reinterpret_cast<__nv_bfloat16*>(dW_out);

  const int lda = H;    // X [m, H]
  const int ldb = Dff;  // dH [m, Dff]
  const int ldc = Dff;  // dW [H, Dff]

  int32_t start = 0;
  for (int e = 0; e < E; ++e) {
    const int32_t end = offs_pad[e];
    const int m = static_cast<int>(end - start);
    if (m < 0) return cudaErrorInvalidValue;
    __nv_bfloat16* dWe = dW_bf16 + static_cast<size_t>(e) * static_cast<size_t>(H) * static_cast<size_t>(Dff);
    if (m > 0) {
      const __nv_bfloat16* Xe = X_bf16 + static_cast<size_t>(start) * static_cast<size_t>(lda);
      const __nv_bfloat16* dHe = dH_bf16 + static_cast<size_t>(start) * static_cast<size_t>(ldb);
      auto err = lt_gemm_bf16_f32accum(Xe, CUBLAS_OP_T, dHe, CUBLAS_OP_N, dWe,
                                  /*m=*/H, /*n=*/Dff, /*k=*/m,
                                  /*lda=*/lda, /*ldb=*/ldb, /*ldc=*/ldc,
                                  /*alpha=*/1.0f, /*beta=*/0.0f,
                                  stream);
      if (err != cudaSuccess) return err;
    } else {
      // Expert has zero tokens: ensure its gradient slice is exactly zero.
      auto err = cudaMemsetAsync(dWe, 0, static_cast<size_t>(H) * static_cast<size_t>(Dff) * sizeof(__nv_bfloat16), stream);
      if (err != cudaSuccess) return err;
    }
    start = end;
  }
  return cudaSuccess;
}
