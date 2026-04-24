#pragma once
#include "io/csv_main.h"
#include <chrono>
#include <format>
#include <spdlog/spdlog.h>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on

namespace bench_detail {
#define BENCH

using bench_clock =
    std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                       std::chrono::high_resolution_clock,
                       std::chrono::steady_clock>;

struct bench_duration {
  double value;
  std::string_view unit;
};

struct bench_status {
  bool valid = true;
  bool result_rejected = false;
  std::string reason;
};

struct bench_metadata_columns {
  std::string_view valid;
  std::string_view reason;
};

struct bench_context {
  std::string_view name;
  bench_status status;
  bench_context *prev = nullptr;
};

struct bench_outcome {
  std::string name;
  bool valid = true;
  bool result_rejected = false;
};

inline thread_local bench_context *g_current_bench = nullptr;
inline thread_local std::optional<bench_outcome> g_last_bench_outcome =
    std::nullopt;

inline std::optional<bench_metadata_columns>
metadata_columns_for(std::string_view name) {
  if (name == "discrepancy") {
    return bench_metadata_columns{"valid_discrepancy",
                                  "fallback_reason_discrepancy"};
  }
  if (name == "ac_discrepancy") {
    return bench_metadata_columns{"valid_ac_discrepancy",
                                  "fallback_reason_ac_discrepancy"};
  }
  return std::nullopt;
}

inline void invalidate_current_timing(std::string_view reason) {
  if (g_current_bench == nullptr)
    return;
  g_current_bench->status.valid = false;
  g_current_bench->status.reason = std::string(reason);
}

inline void reject_current_result(std::string_view reason) {
  if (g_current_bench == nullptr)
    return;
  g_current_bench->status.valid = false;
  g_current_bench->status.result_rejected = true;
  g_current_bench->status.reason = std::string(reason);
}

inline bool last_result_rejected_for(std::string_view name) {
  return g_last_bench_outcome.has_value() &&
         g_last_bench_outcome->name == name &&
         g_last_bench_outcome->result_rejected;
}

inline bool current_result_rejected() {
  return g_current_bench != nullptr && g_current_bench->status.result_rejected;
}

inline bench_duration format_duration(std::chrono::nanoseconds ns) {
  const auto n = static_cast<double>(ns.count());
  if (n < 1'000.0)
    return {n, "ns"};
  if (n < 1'000'000.0)
    return {n / 1'000.0, "\xE6s"};
  if (n < 1'000'000'000.0)
    return {n / 1'000'000.0, "ms"};
  return {n / 1'000'000'000.0, "s"};
}

inline int duration_precision(double value) {
  return value < 10    ? 3
         : value < 100 ? 2
                       : 1;
}

class ScopedBenchTimer {
public:
  explicit ScopedBenchTimer(std::string_view name)
      : name_(name), ctx_{name}, t0_(bench_clock::now()) {
    ctx_.prev = g_current_bench;
    g_current_bench = &ctx_;
  }

  ~ScopedBenchTimer() {
    g_current_bench = ctx_.prev;
    const auto t1 = bench_clock::now();
    const auto dur =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0_);
    g_last_bench_outcome = bench_outcome{std::string(name_), ctx_.status.valid,
                                         ctx_.status.result_rejected};
    const auto meta = metadata_columns_for(name_);
    if (ctx_.status.valid) {
      io::csv::main_set(std::format("ns_{}", name_), (long long)dur.count());
      if (meta.has_value()) {
        io::csv::main_set(meta->valid, true);
        io::csv::main_set(meta->reason, "");
      }
    } else if (meta.has_value()) {
      io::csv::main_set(meta->valid, false);
      io::csv::main_set(meta->reason, ctx_.status.reason);
    }
    const auto out = format_duration(dur);
    const int prec = duration_precision(out.value);
    spdlog::info("{}",
                 std::format("{:<28}  {:>10.{}f} {}", name_, out.value, prec,
                             out.unit));
  }

private:
  std::string name_;
  bench_context ctx_;
  bench_clock::time_point t0_;
};

template <class F>
inline decltype(auto) bench_run(std::string_view name, F &&f) {
#ifdef BENCH
  ScopedBenchTimer timer(name);
  return std::forward<F>(f)();
#else
  return std::forward<F>(f)();
#endif
}

[[noreturn]] inline void mismatch(std::string_view trusted_name, bool trusted,
                                  std::string_view other_name, bool other) {
  spdlog::error("Mismatch between {} ({}) and {} ({}).", trusted_name,
                trusted, other_name, other);
  std::exit(1);
}

template <class F>
inline void bench_run_check(std::string_view trusted_name, bool trusted,
                            std::string_view name, F &&f) {
  using R = decltype(std::forward<F>(f)());

  if constexpr (std::is_void_v<R>) {
    bench_run(name, std::forward<F>(f));
    return;
  } else {
    // Enforce that non-void is actually bool-like
    static_assert(std::is_convertible_v<R, bool>,
                  "bench_run_check expects f() to return bool (or void).");

    const bool v = static_cast<bool>(bench_run(name, std::forward<F>(f)));
    if (last_result_rejected_for(name))
      return;
    if (v != trusted)
      mismatch(trusted_name, trusted, name, v);
  }
}

} // namespace bench_detail

#define BENCH_RUN(name, expr)                                                  \
  bench_detail::bench_run((name), [&]() -> decltype(auto) { return (expr); })

#define BENCH_RUN_CHECK(trusted_name, trusted, name, expr)                     \
  bench_detail::bench_run_check((trusted_name), (trusted), (name),             \
                                [&]() -> decltype(auto) { return (expr); })

static inline void log_mem(std::string_view tag) {
  PROCESS_MEMORY_COUNTERS_EX2 pmc{};
  pmc.cb = sizeof(pmc);
  if (GetProcessMemoryInfo(GetCurrentProcess(),
                           reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmc),
                           sizeof(pmc))) {
    spdlog::info("[mem] {}WorkingSetMB={} PrivateMB={}", tag,
                 pmc.WorkingSetSize / (1024 * 1024),
                 pmc.PrivateUsage / (1024 * 1024));
  }
}
