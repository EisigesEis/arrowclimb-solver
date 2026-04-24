#pragma once

#include "profile_config.h"

#if defined(DISCREP_PROFILE)

#include <algorithm>
#include <chrono>
#include <spdlog/spdlog.h>

namespace oracle::discrep::profile {

using clock_t =
    std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                       std::chrono::high_resolution_clock,
                       std::chrono::steady_clock>;

inline long long ns_since(const clock_t::time_point &t0) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t::now() - t0)
      .count();
}

struct SolveProf {
  long long t_total = 0;
  long long t_analyze = 0;
  long long t_base = 0;
  long long t_layers = 0;
  long long t_merge = 0;
  int layers = 0;
  std::size_t max_prev = 0;
  std::size_t max_cur = 0;
  bool hit_target_base = false;
};

inline void emit_solve_profile(const SolveProf &prof) {
  spdlog::info(
      "[discrep::solve/profile] total_ms={:.3f} analyze_ms={:.3f} "
      "base_ms={:.3f} layers_ms={:.3f} merge_ms={:.3f} layers={} "
      "max_prev={} max_cur={} base_hit={}",
      prof.t_total / 1e6, prof.t_analyze / 1e6, prof.t_base / 1e6,
      prof.t_layers / 1e6, prof.t_merge / 1e6, prof.layers, prof.max_prev,
      prof.max_cur, prof.hit_target_base ? 1 : 0);
}

struct SolveFFTProf {
  long long t_total = 0;
  long long t_analyze = 0;
  long long t_base = 0;
  long long t_conv = 0;
  long long t_scatter = 0;
  long long t_fft_conv = 0;
  long long t_sparse_conv = 0;
  int layers = 0;
  int conv_layers = 0;
  int sparse_conv_layers = 0;
  int fft_conv_layers = 0;
  std::size_t sum_ones_prev = 0;
  std::size_t max_prevN = 0;
  std::size_t max_curN = 0;
  double sum_fill_ratio = 0.0;
  double max_fill_ratio = 0.0;
  double fft_useful_grid_weighted_sum = 0.0;
  long long fft_useful_grid_weight_total_ns = 0;
  bool used_sliced_n1 = false;
  bool hit_target_base = false;
};

inline const char *conv_mode_name(const SolveFFTProf &prof) {
  if (prof.used_sliced_n1)
    return "sliced_n1";
  if (prof.sparse_conv_layers > 0 && prof.fft_conv_layers > 0)
    return "mixed";
  if (prof.sparse_conv_layers > 0)
    return "sparse";
  return "fft";
}

inline void emit_solve_fft_profile(const char *solver_name,
                                   const SolveFFTProf &prof) {
  const std::size_t peakN = std::max(prof.max_prevN, prof.max_curN);
  const double ones_avg =
      (prof.conv_layers > 0)
          ? static_cast<double>(prof.sum_ones_prev) /
                static_cast<double>(prof.conv_layers)
          : 0.0;
  const double fill_avg_pct =
      (prof.conv_layers > 0)
          ? (100.0 * prof.sum_fill_ratio) / static_cast<double>(prof.conv_layers)
          : 0.0;
  const double fill_max_pct = 100.0 * prof.max_fill_ratio;

  spdlog::info(
      "[{} profile] total_ms={:.3f} analyze_ms={:.3f} "
      "base_ms={:.3f} conv_ms={:.3f} fft_conv_ms={:.3f} sparse_conv_ms={:.3f} scatter_ms={:.3f} "
      "layers={} peakN={} conv_mode={} ones_avg={:.1f} "
      "fill_avg_pct={:.3f} fill_max_pct={:.3f} base_hit={}",
      solver_name, prof.t_total / 1e6, prof.t_analyze / 1e6, prof.t_base / 1e6,
      prof.t_conv / 1e6, prof.t_fft_conv / 1e6, prof.t_sparse_conv / 1e6,
      prof.t_scatter / 1e6, prof.layers, peakN,
      conv_mode_name(prof), ones_avg, fill_avg_pct, fill_max_pct,
      prof.hit_target_base ? 1 : 0);
}

struct SlicedProf {
  long long t_total = 0;
  long long t_base = 0;
  long long t_layers = 0;
  long long t_fwd = 0;
  long long t_combine = 0;
  long long t_inv = 0;
  long long t_scatter = 0;
  int layers = 0;
  int active_slices_max = 0;
  int touched_slices_max = 0;
};

inline void emit_sliced_profile(const SlicedProf &prof) {
  spdlog::info(
      "[discrep_fft::sliced_n1/profile] total_ms={:.3f} base_ms={:.3f} "
      "layers_ms={:.3f} fwd_ms={:.3f} combine_ms={:.3f} inv_ms={:.3f} "
      "scatter_ms={:.3f} layers={} active_max={} touched_max={}",
      prof.t_total / 1e6, prof.t_base / 1e6, prof.t_layers / 1e6,
      prof.t_fwd / 1e6, prof.t_combine / 1e6, prof.t_inv / 1e6,
      prof.t_scatter / 1e6, prof.layers, prof.active_slices_max,
      prof.touched_slices_max);
}

} // namespace oracle::discrep::profile

#endif // defined(DISCREP_PROFILE)
