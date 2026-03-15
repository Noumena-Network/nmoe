// GPU telemetry via NVML with a background poller.
// Provides non-blocking snapshot() for Python via pybind11.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <atomic>
#include <dlfcn.h>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <cstring>

namespace py = pybind11;

namespace {

struct GPUStat {
  int index = 0;
  double utilization_gpu = 0.0;      // percent
  double memory_used_gib = 0.0;
  double memory_total_gib = 0.0;
  double power_draw_w = 0.0;
  double power_limit_w = 0.0;
  double temperature_c = 0.0;
  double clocks_sm_mhz = 0.0;
  // Decoded throttle flags (B200 SXM minimal)
  bool throttle_thermal = false;
  bool throttle_power = false;
  bool throttle_hw_slowdown = false;
  bool throttle_apps = false;
  // ECC counters (aggregate)
  unsigned long long ecc_corrected = 0ULL;
  unsigned long long ecc_uncorrected = 0ULL;
};

std::atomic<bool> g_running{false};
std::thread g_thread;
std::mutex g_mu;
std::vector<GPUStat> g_snapshot;
std::atomic<int> g_interval_ms{1000};

#if __has_include(<nvml.h>)
#include <nvml.h>

struct NVMLFns {
  using InitV2Fn = nvmlReturn_t (*)(void);
  using ShutdownFn = nvmlReturn_t (*)(void);
  using DeviceGetCountV2Fn = nvmlReturn_t (*)(unsigned int*);
  using DeviceGetHandleByIndexV2Fn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
  using DeviceGetUtilizationRatesFn = nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t*);
  using DeviceGetMemoryInfoFn = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*);
  using DeviceGetPowerUsageFn = nvmlReturn_t (*)(nvmlDevice_t, unsigned int*);
  using DeviceGetEnforcedPowerLimitFn = nvmlReturn_t (*)(nvmlDevice_t, unsigned int*);
  using DeviceGetPowerManagementLimitFn = nvmlReturn_t (*)(nvmlDevice_t, unsigned int*);
  using DeviceGetTemperatureFn = nvmlReturn_t (*)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int*);
  using DeviceGetClockInfoFn = nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int*);
  using DeviceGetCurrentClocksThrottleReasonsFn = nvmlReturn_t (*)(nvmlDevice_t, unsigned long long*);
  using DeviceGetTotalEccErrorsFn = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, unsigned long long*);

  InitV2Fn init_v2 = nullptr;
  ShutdownFn shutdown = nullptr;
  DeviceGetCountV2Fn device_get_count_v2 = nullptr;
  DeviceGetHandleByIndexV2Fn device_get_handle_by_index_v2 = nullptr;
  DeviceGetUtilizationRatesFn device_get_utilization_rates = nullptr;
  DeviceGetMemoryInfoFn device_get_memory_info = nullptr;
  DeviceGetPowerUsageFn device_get_power_usage = nullptr;
  DeviceGetEnforcedPowerLimitFn device_get_enforced_power_limit = nullptr;
  DeviceGetPowerManagementLimitFn device_get_power_management_limit = nullptr;
  DeviceGetTemperatureFn device_get_temperature = nullptr;
  DeviceGetClockInfoFn device_get_clock_info = nullptr;
  DeviceGetCurrentClocksThrottleReasonsFn device_get_current_clocks_throttle_reasons = nullptr;
  DeviceGetTotalEccErrorsFn device_get_total_ecc_errors = nullptr;

  template <typename Fn>
  static bool load_required(void* lib, const char* name, Fn& out) {
    out = reinterpret_cast<Fn>(dlsym(lib, name));
    return out != nullptr;
  }

  template <typename Fn>
  static void load_optional(void* lib, const char* name, Fn& out) {
    out = reinterpret_cast<Fn>(dlsym(lib, name));
  }

  bool load(void* lib) {
    return load_required(lib, "nvmlInit_v2", init_v2) &&
           load_required(lib, "nvmlShutdown", shutdown) &&
           load_required(lib, "nvmlDeviceGetCount_v2", device_get_count_v2) &&
           load_required(lib, "nvmlDeviceGetHandleByIndex_v2", device_get_handle_by_index_v2) &&
           load_required(lib, "nvmlDeviceGetUtilizationRates", device_get_utilization_rates) &&
           load_required(lib, "nvmlDeviceGetMemoryInfo", device_get_memory_info) &&
           load_required(lib, "nvmlDeviceGetPowerUsage", device_get_power_usage) &&
           load_required(lib, "nvmlDeviceGetTemperature", device_get_temperature) &&
           load_required(lib, "nvmlDeviceGetClockInfo", device_get_clock_info) &&
           load_required(lib, "nvmlDeviceGetCurrentClocksThrottleReasons", device_get_current_clocks_throttle_reasons);
  }
};

struct NVMLContext {
  bool ok = false;
  bool init_called = false;
  void* lib = nullptr;
  NVMLFns fns;
  std::vector<nvmlDevice_t> handles;
  NVMLContext() {
    lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) return;
    if (!fns.load(lib)) return;
    NVMLFns::load_optional(lib, "nvmlDeviceGetEnforcedPowerLimit", fns.device_get_enforced_power_limit);
    NVMLFns::load_optional(lib, "nvmlDeviceGetPowerManagementLimit", fns.device_get_power_management_limit);
    NVMLFns::load_optional(lib, "nvmlDeviceGetTotalEccErrors", fns.device_get_total_ecc_errors);

    nvmlReturn_t r = fns.init_v2();
    if (r != NVML_SUCCESS) return;
    init_called = true;
    unsigned int count = 0;
    if (fns.device_get_count_v2(&count) != NVML_SUCCESS) return;
    handles.resize(count);
    for (unsigned i = 0; i < count; ++i) {
      nvmlDevice_t h{};
      if (fns.device_get_handle_by_index_v2(i, &h) == NVML_SUCCESS) {
        handles[i] = h;
      }
    }
    ok = true;
  }
  ~NVMLContext() {
    if (init_called && fns.shutdown != nullptr) fns.shutdown();
    if (lib != nullptr) dlclose(lib);
  }
};

static void sample_once_nvml(NVMLContext& ctx, std::vector<GPUStat>& out) {
  out.clear();
  if (!ctx.ok) return;
  out.reserve(ctx.handles.size());
  for (size_t i = 0; i < ctx.handles.size(); ++i) {
    nvmlDevice_t h = ctx.handles[i];
    GPUStat s;
    s.index = static_cast<int>(i);

    // Utilization
    nvmlUtilization_t util{};
    if (ctx.fns.device_get_utilization_rates(h, &util) == NVML_SUCCESS) {
      s.utilization_gpu = util.gpu;
    }
    // Memory
    nvmlMemory_t mem{};
    if (ctx.fns.device_get_memory_info(h, &mem) == NVML_SUCCESS) {
      s.memory_used_gib = static_cast<double>(mem.used) / (1024.0 * 1024.0 * 1024.0);
      s.memory_total_gib = static_cast<double>(mem.total) / (1024.0 * 1024.0 * 1024.0);
    }
    // Power
    unsigned int mw = 0;
    if (ctx.fns.device_get_power_usage(h, &mw) == NVML_SUCCESS) {
      s.power_draw_w = mw / 1000.0;
    }
    if ((ctx.fns.device_get_enforced_power_limit != nullptr &&
         ctx.fns.device_get_enforced_power_limit(h, &mw) == NVML_SUCCESS) ||
        (ctx.fns.device_get_power_management_limit != nullptr &&
         ctx.fns.device_get_power_management_limit(h, &mw) == NVML_SUCCESS)) {
      s.power_limit_w = mw / 1000.0;
    }
    // Temperature
    unsigned int tc = 0;
    if (ctx.fns.device_get_temperature(h, NVML_TEMPERATURE_GPU, &tc) == NVML_SUCCESS) {
      s.temperature_c = tc;
    }
    // Clocks
    unsigned int clk = 0;
    if (ctx.fns.device_get_clock_info(h, NVML_CLOCK_SM, &clk) == NVML_SUCCESS) s.clocks_sm_mhz = clk;

    // Decode throttle reasons into booleans (thermal, power, hw slowdown, apps clocks)
    unsigned long long thr = 0ULL;
    if (ctx.fns.device_get_current_clocks_throttle_reasons(h, &thr) == NVML_SUCCESS) {
#ifdef NVML_CLOCKS_THROTTLE_REASON_SW_THERMAL_SLOWDOWN
      unsigned long long THERMAL = NVML_CLOCKS_THROTTLE_REASON_SW_THERMAL_SLOWDOWN;
#ifdef NVML_CLOCKS_THROTTLE_REASON_HW_THERMAL_SLOWDOWN
      THERMAL |= NVML_CLOCKS_THROTTLE_REASON_HW_THERMAL_SLOWDOWN;
#endif
      s.throttle_thermal = (thr & THERMAL) != 0ULL;
#endif
#ifdef NVML_CLOCKS_THROTTLE_REASON_SW_POWER_CAP
      unsigned long long POWER = NVML_CLOCKS_THROTTLE_REASON_SW_POWER_CAP;
#ifdef NVML_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE_SLOWDOWN
      POWER |= NVML_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE_SLOWDOWN;
#endif
      s.throttle_power = (thr & POWER) != 0ULL;
#endif
#ifdef NVML_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN
      s.throttle_hw_slowdown = (thr & NVML_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN) != 0ULL;
#endif
#ifdef NVML_CLOCKS_THROTTLE_REASON_APPS_CLOCKS_SETTING
      s.throttle_apps = (thr & NVML_CLOCKS_THROTTLE_REASON_APPS_CLOCKS_SETTING) != 0ULL;
#endif
    }

    // ECC counters (aggregate totals); may be unsupported/disabled on some SKUs
    unsigned long long eccv = 0ULL;
    if (ctx.fns.device_get_total_ecc_errors != nullptr &&
        ctx.fns.device_get_total_ecc_errors(h, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_AGGREGATE_ECC, &eccv) == NVML_SUCCESS) s.ecc_corrected = eccv;
    if (ctx.fns.device_get_total_ecc_errors != nullptr &&
        ctx.fns.device_get_total_ecc_errors(h, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_AGGREGATE_ECC, &eccv) == NVML_SUCCESS) s.ecc_uncorrected = eccv;

    out.push_back(s);
  }
}

static void poll_loop() {
  NVMLContext ctx;
  while (g_running.load(std::memory_order_relaxed)) {
    std::vector<GPUStat> tmp;
    sample_once_nvml(ctx, tmp);
    {
      std::lock_guard<std::mutex> lk(g_mu);
      g_snapshot.swap(tmp);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(g_interval_ms.load()));
  }
}

#else

static void poll_loop() {
  // NVML not available at compile time; keep empty snapshot
  while (g_running.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(g_interval_ms.load()));
  }
}

#endif

} // anonymous

// Standalone module for GPU telemetry
PYBIND11_MODULE(gpu, m) {
  m.doc() = "GPU telemetry (NVML) with background poller";

  m.def("start", [](int interval_ms){
    if (interval_ms > 0) g_interval_ms.store(interval_ms);
    bool expected = false;
    if (g_running.compare_exchange_strong(expected, true)) {
      g_thread = std::thread(poll_loop);
    }
  }, py::arg("interval_ms") = 1000, "Start NVML poller (idempotent)");

  m.def("stop", [](){
    bool expected = true;
    if (g_running.compare_exchange_strong(expected, false)) {
      if (g_thread.joinable()) g_thread.join();
    }
  }, "Stop NVML poller");

  m.def("is_running", [](){ return g_running.load(); }, "Return whether the poller is running");

  m.def("snapshot", [](){
    std::vector<GPUStat> snap;
    {
      std::lock_guard<std::mutex> lk(g_mu);
      snap = g_snapshot; // copy
    }
    py::list out;
    for (const auto& s : snap) {
      py::dict d;
      d["index"] = s.index;
      d["utilization_gpu"] = s.utilization_gpu;
      d["memory_used_gib"] = s.memory_used_gib;
      d["memory_total_gib"] = s.memory_total_gib;
      d["power_draw_w"] = s.power_draw_w;
      d["power_limit_w"] = s.power_limit_w;
      d["temperature_c"] = s.temperature_c;
      d["clocks_sm_mhz"] = s.clocks_sm_mhz;
      d["throttle_thermal"] = s.throttle_thermal;
      d["throttle_power"] = s.throttle_power;
      d["throttle_hw_slowdown"] = s.throttle_hw_slowdown;
      d["throttle_apps"] = s.throttle_apps;
      d["ecc_corrected"] = py::int_(s.ecc_corrected);
      d["ecc_uncorrected"] = py::int_(s.ecc_uncorrected);
      out.append(std::move(d));
    }
    return out;
  }, "Return latest per‑GPU snapshot (non‑blocking)");
}
