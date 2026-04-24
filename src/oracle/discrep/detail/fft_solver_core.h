#pragma once

#include "bench.h"
#include "io/csv_main.h"
#include "model/matrix/model.h"
#include "model/matrix/packed.h"

#include "convolve/augmented_base.h"
#include "convolve/fft_bool_r2c.h"
#include "convolve/fft_bool_sparse.h"

#include "profile_helpers.h"
#include "sliced_n1_solver.h"
#include "tube_embed.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace oracle::discrep::detail {

using ::Bounds;
using ::Vec;

constexpr std::size_t kDenseEmbedSizeCap = 50'000'000;

struct FFTWorkspace {
  std::unique_ptr<FFTBoolConvolver> conv;
  std::vector<std::uint8_t> fprev;
  std::vector<std::uint8_t> fcur;
  std::vector<std::uint8_t> g;
  std::vector<std::size_t> idx_buf;
};

enum class Status { Ready, SolvedTrue, SolvedFalse };

template <class MatrixLike> struct PreparedInstance {
  const MatrixLike *A = nullptr;
  int n = 0;
  int r = 0;
  int m_dim = 0;
  long long q = 0;
  int Delta_global = 0;
  long long K = 0;
  int ell = 0;
  Eigen::VectorXi b_up;
  Eigen::VectorXi b_down;
  Eigen::VectorXi b_aug;
  Vec target{};
  Bounds box0;
  std::vector<long long> H_up;
  std::vector<long long> H_block;

  const MatrixLike &matrix() const {
    assert(A != nullptr);
    return *A;
  }
};

template <class MatrixLike>
inline Status prepare_instance(const MatrixLike &A,
                               const Eigen::Ref<const Eigen::VectorXi> &b,
                               PreparedInstance<MatrixLike> &ctx) {
  ctx.A = &A;
  ctx.n = static_cast<int>(A.num_blocks());
  if (b.size() < ctx.n)
    return Status::SolvedFalse;

  ctx.r = static_cast<int>(b.size()) - ctx.n;
  ctx.m_dim = ctx.r + ctx.n;
  if (ctx.m_dim > MAX_M)
    return Status::SolvedFalse;

  ctx.b_up = b.head(ctx.r);
  ctx.b_down = b.tail(ctx.n);
  ctx.q = static_cast<long long>(ctx.b_down.sum());
  if (ctx.q == 0)
    return ctx.b_up.isZero() ? Status::SolvedTrue : Status::SolvedFalse;

  const UpMatrixStats up_stats = analyze_up_matrix(A, ctx.r);
  ctx.H_up = choose_H_up_per_row(up_stats, ctx.r);
  ctx.H_block = choose_H_block_per_block(A, ctx.r);
  ctx.Delta_global = A.maxCoeff();
  ctx.K = choose_K_default(ctx.r, ctx.Delta_global);
  ctx.ell = ceil_log_base_6_5(ctx.K);

  ctx.b_aug.resize(ctx.m_dim);
  for (int k = 0; k < ctx.r; ++k)
    ctx.b_aug[k] = ctx.b_up[k];
  for (int e = 0; e < ctx.n; ++e)
    ctx.b_aug[ctx.r + e] = ctx.b_down[e];

  ctx.target = Vec{};
  for (int k = 0; k < ctx.r; ++k)
    ctx.target.x[static_cast<std::size_t>(k)] = ctx.b_up[k];
  for (int e = 0; e < ctx.n; ++e)
    ctx.target.x[static_cast<std::size_t>(ctx.r + e)] = ctx.b_down[e];

  ctx.box0 = make_tube_box(ctx.b_aug, ctx.b_down, ctx.r, ctx.m_dim, 0,
                           ctx.ell, ctx.H_up, ctx.H_block);
  if (box_empty(ctx.box0, ctx.m_dim))
    return Status::SolvedFalse;

  return Status::Ready;
}

struct ReachableLayer {
  Embed embed;
};

enum class SolverStep { Continue, Infeasible, Rejected };

inline void warmup(const LPModel<PackedA> &m) {
  sliced_n1::warmup(m);
}

inline bool should_emit_csv_utilization(std::string_view profile_solver_name) {
  return profile_solver_name == "discrepancy::solve";
}

inline void
emit_csv_utilization(std::string_view profile_solver_name,
                     const oracle::discrep::profile::SolveFFTProf &prof) {
  if (!should_emit_csv_utilization(profile_solver_name))
    return;
  if (prof.conv_layers <= 0)
    return;

  const long long total_conv_runtime_ns = prof.t_fft_conv + prof.t_sparse_conv;
  if (total_conv_runtime_ns > 0) {
    io::csv::main_set(
        "discrepancy_fft_conv_runtime_share",
        static_cast<double>(prof.t_fft_conv) /
            static_cast<double>(total_conv_runtime_ns));
  }
  if (prof.fft_useful_grid_weight_total_ns > 0) {
    io::csv::main_set(
        "discrepancy_fft_useful_grid_share_conv_weighted",
        prof.fft_useful_grid_weighted_sum /
            static_cast<double>(prof.fft_useful_grid_weight_total_ns));
  }

  io::csv::main_set("discrepancy_fft_layer_share",
                    static_cast<double>(prof.fft_conv_layers) /
                        static_cast<double>(prof.conv_layers));
  io::csv::main_set("discrepancy_fill_ratio_avg",
                    prof.sum_fill_ratio / static_cast<double>(prof.conv_layers));
}

// Oversized dense embeddings are a representation limit, not infeasibility.
inline bool reject_solver_run(std::string_view solver_name,
                              std::string_view reason) {
  bench_detail::reject_current_result(reason);
  spdlog::info("{} timing invalidated: rejected solver result ({})",
               solver_name, reason);
  return false;
}

struct SolverRun {
  using clock_t = oracle::discrep::profile::clock_t;
  using time_point = clock_t::time_point;

  const char *solver_name = "";
  time_point t_total0 = clock_t::now();
  oracle::discrep::profile::SolveFFTProf data;

  explicit SolverRun(const char *name) : solver_name(name) {}

  static time_point now() { return clock_t::now(); }
  static long long ns_since(const time_point &t0) {
    return oracle::discrep::profile::ns_since(t0);
  }

  void add_analyze_since(const time_point &t0) { data.t_analyze += ns_since(t0); }

  void add_base_since(const time_point &t0, std::size_t base_N) {
    data.t_base += ns_since(t0);
    data.max_prevN = std::max(data.max_prevN, base_N);
  }

  void note_base_target_hit() { data.hit_target_base = true; }

  void note_sliced_n1() { data.used_sliced_n1 = true; }

  void note_layer_embed(std::size_t prev_N, std::size_t cur_N) {
    data.max_prevN = std::max(data.max_prevN, prev_N);
    data.max_curN = std::max(data.max_curN, cur_N);
  }

  void add_fft_convolution(std::size_t ones_prev, std::size_t prev_N,
                           long long dt_conv) {
    data.conv_layers++;
    data.fft_conv_layers++;
    data.sum_ones_prev += ones_prev;

    const double fill_ratio =
        (prev_N > 0) ? static_cast<double>(ones_prev) / static_cast<double>(prev_N)
                     : 0.0;
    data.sum_fill_ratio += fill_ratio;
    data.max_fill_ratio = std::max(data.max_fill_ratio, fill_ratio);

    data.t_fft_conv += dt_conv;
    data.t_conv += dt_conv;
    data.fft_useful_grid_weighted_sum += fill_ratio * static_cast<double>(dt_conv);
    data.fft_useful_grid_weight_total_ns += dt_conv;
  }

  void add_scatter_since(const time_point &t0) {
    data.t_scatter += ns_since(t0);
    data.layers++;
  }

  void emit() {
    data.t_total = ns_since(t_total0);
    oracle::discrep::profile::emit_solve_fft_profile(solver_name, data);
  }

  bool finish(bool result) {
    emit();
    emit_csv_utilization(solver_name, data);
    return result;
  }

  bool reject(std::string_view reason) {
    emit();
    return reject_solver_run(solver_name, reason);
  }
};

inline void run_self_convolution(FFTWorkspace &ws, std::size_t prev_N,
                                 const std::vector<std::uint8_t> &fprev) {
  if (!ws.conv || ws.conv->N != prev_N) {
    ws.conv.reset(new FFTBoolConvolver(prev_N, FFTW_ESTIMATE));
    ws.g.resize(prev_N);
  }
  ws.conv->mul_bool(ws.g, fprev, fprev);
}

inline void scatter_layer(const std::vector<std::uint8_t> &g, const Embed &prev,
                          const Bounds &box_cur, const Embed &cur, int m_dim,
                          std::vector<std::uint8_t> &fcur) {
  scatter_conv_to_next(g, prev, box_cur, cur, m_dim, fcur);
}

template <class MatrixLike>
class DenseFftSolver {
public:
  DenseFftSolver(FFTWorkspace &workspace, SolverRun &run)
      : ws_(workspace), run_(run) {}

  bool solve(const PreparedInstance<MatrixLike> &ctx) {
    ReachableLayer layer;
    const SolverStep base_status = build_base_layer(ctx, layer);
    if (base_status == SolverStep::Rejected)
      return false;

    if (target_reachable(ctx, layer)) {
      run_.note_base_target_hit();
      return run_.finish(true);
    }

    if (ctx.ell == 0)
      return run_.finish(false);

    for (int i = 1; i <= ctx.ell; ++i) {
      const SolverStep step_status = advance_layer(ctx, i, layer);
      if (step_status == SolverStep::Rejected)
        return false;
      if (step_status == SolverStep::Infeasible)
        return run_.finish(false);

      if (i == ctx.ell && target_reachable(ctx, layer))
        return run_.finish(true);
    }

    return run_.finish(false);
  }

private:
  FFTWorkspace &ws_;
  SolverRun &run_;

  SolverStep build_base_layer(const PreparedInstance<MatrixLike> &ctx,
                              ReachableLayer &layer) {
    Embed base_embed;
    if (!build_embed(ctx.box0, ctx.m_dim, base_embed))
      return run_.reject("embed_build_failed") ? SolverStep::Continue
                                               : SolverStep::Rejected;

    if (base_embed.N > kDenseEmbedSizeCap)
      return run_.reject("embed_size_cap_base") ? SolverStep::Continue
                                                : SolverStep::Rejected;

    const auto t_base0 = SolverRun::now();

    ws_.fprev.assign(base_embed.N, std::uint8_t{0});
    auto &base_bits = ws_.fprev;

    std::size_t zero_idx = 0;
    if (encode_in_box(base_embed, Vec{}, ctx.m_dim, zero_idx))
      base_bits[zero_idx] = 1;

    ws_.idx_buf.clear();
    ws_.idx_buf.reserve(static_cast<std::size_t>(ctx.matrix().cols()));
    auto &idx_buf = ws_.idx_buf;

    auto commit_unique = [&]() {
      std::sort(idx_buf.begin(), idx_buf.end());
      idx_buf.erase(std::unique(idx_buf.begin(), idx_buf.end()), idx_buf.end());
      for (std::size_t idx : idx_buf)
        base_bits[idx] = 1;
      idx_buf.clear();
    };

    convolve::for_each_aug_vec(
        ctx.matrix(), ctx.r, ctx.n,
        [&](const Vec &y) {
          std::size_t idx = 0;
          if (encode_in_box(base_embed, y, ctx.m_dim, idx))
            idx_buf.push_back(idx);
        },
        [&](int) { commit_unique(); });
    commit_unique();

    run_.add_base_since(t_base0, base_embed.N);
    layer.embed = std::move(base_embed);
    return SolverStep::Continue;
  }

  SolverStep advance_layer(const PreparedInstance<MatrixLike> &ctx,
                           int layer_index, ReachableLayer &layer) {
    Bounds next_box = make_tube_box(ctx.b_aug, ctx.b_down, ctx.r, ctx.m_dim,
                                    layer_index, ctx.ell, ctx.H_up,
                                    ctx.H_block);
    if (box_empty(next_box, ctx.m_dim))
      return SolverStep::Infeasible;

    Embed next_embed;
    if (!build_embed(next_box, ctx.m_dim, next_embed))
      return run_.reject("embed_build_failed_layer") ? SolverStep::Continue
                                                     : SolverStep::Rejected;

    if (layer.embed.N > kDenseEmbedSizeCap || next_embed.N > kDenseEmbedSizeCap)
      return run_.reject("embed_size_cap_layer") ? SolverStep::Continue
                                                 : SolverStep::Rejected;

    run_.note_layer_embed(layer.embed.N, next_embed.N);

    const std::size_t ones_prev = convolve::count_ones(ws_.fprev);
    const auto t_conv0 = SolverRun::now();
    run_self_convolution(ws_, layer.embed.N, ws_.fprev);
    run_.add_fft_convolution(ones_prev, layer.embed.N,
                             SolverRun::ns_since(t_conv0));

    if (ws_.fcur.size() != next_embed.N)
      ws_.fcur.resize(next_embed.N);
    std::fill(ws_.fcur.begin(), ws_.fcur.end(), std::uint8_t{0});

    const auto t_scatter0 = SolverRun::now();
    scatter_layer(ws_.g, layer.embed, next_box, next_embed, ctx.m_dim,
                  ws_.fcur);
    run_.add_scatter_since(t_scatter0);

    ws_.fprev.swap(ws_.fcur);
    layer.embed = std::move(next_embed);
    return SolverStep::Continue;
  }

  bool target_reachable(const PreparedInstance<MatrixLike> &ctx,
                        const ReachableLayer &layer) const {
    std::size_t target_idx = 0;
    return encode_in_box(layer.embed, ctx.target, ctx.m_dim, target_idx) &&
           ws_.fprev[target_idx];
  }
};

template <class MatrixLike>
inline bool try_sliced_n1_fast_path(const PreparedInstance<MatrixLike> &ctx,
                                    const LPModel<PackedA> *packed_model,
                                    SolverRun &profile, bool &result) {
  if (!sliced_n1::can_use(ctx, packed_model))
    return false;

  result = sliced_n1::solve(ctx, *packed_model);
  profile.note_sliced_n1();
  return true;
}

template <class MatrixLike>
inline bool solve_ready_instance(const PreparedInstance<MatrixLike> &ctx,
                                 const LPModel<PackedA> *packed_model,
                                 FFTWorkspace &ws, SolverRun &profile) {
  bool sliced_result = false;
  if (try_sliced_n1_fast_path(ctx, packed_model, profile, sliced_result))
    return profile.finish(sliced_result);

  return DenseFftSolver<MatrixLike>(ws, profile).solve(ctx);
}

template <class MatrixLike>
inline bool run_solver(const MatrixLike &A,
                       const Eigen::Ref<const Eigen::VectorXi> &b,
                       const LPModel<PackedA> *packed_model, FFTWorkspace &ws,
                       const char *profile_solver_name) {
  SolverRun profile(profile_solver_name);

  const auto t_analyze0 = SolverRun::now();
  PreparedInstance<MatrixLike> ctx;
  const auto status = prepare_instance(A, b, ctx);
  profile.add_analyze_since(t_analyze0);

  switch (status) {
  case Status::SolvedTrue:
    return profile.finish(true);
  case Status::SolvedFalse:
    return profile.finish(false);
  case Status::Ready:
    break;
  }

  return solve_ready_instance(ctx, packed_model, ws, profile);
}

} // namespace oracle::discrep::detail
