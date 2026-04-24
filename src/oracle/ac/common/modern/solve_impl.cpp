#include "oracle/ac/common/modern/solve_impl.h"

#include "convolve/sumset.h"
#include "oracle/ac/common/profile.h"
#include "oracle/ac/naive/util.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

using Eigen::MatrixXi;
using Eigen::VectorXi;

namespace {

struct ModernWorkspace {
  SumSet dt;
  SumSet next;
  SumSet bk;
  SumSet merged;
  SumSet tmp;
  std::vector<Bounds> digit_bounds;
  std::vector<PowTable> pt;
};

inline Bounds make_window_bounds(const VectorXi &lo, const VectorXi &hi, int r) {
  Bounds b;
  b.vacuous = false;
  b.lo_active = true;
  b.hi_active = true;
  for (int i = 0; i < r; ++i) {
    b.lo[i] = lo[i];
    b.hi[i] = hi[i];
  }
  return b;
}

/*
  Runs per-block DP for one digit:
  For each block k, get all feasible block sums
  for digit and merge into global sumset.
  Returns if any feasible sums remain.
 */
inline bool run_digit_dp_sumset(const PackedA &A, int r,
                                const Eigen::Ref<const VectorXi> &job_target,
                                const Eigen::Ref<const VectorXi> &picks_by_block,
                                const oracle::ac_modern::BaseTableFnSumSet get_block_sumset,
                                ModernWorkspace &ws) {
  reset_unit_sumset(ws.dt);

  Bounds vac;
  vac.vacuous = true;
  vac.lo_active = false;
  vac.hi_active = false;

  const int B = A.num_blocks();
  assert(picks_by_block.size() == B);

  for (int k = 0; k < B; ++k) {
    oracle::ac_profile::begin_base_call();
    const auto t0 = std::chrono::steady_clock::now();
    ws.bk = get_block_sumset(A, k, job_target, picks_by_block(k), ws.pt[k]);
    const auto t1 = std::chrono::steady_clock::now();
    const auto dt_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    oracle::ac_profile::finish_base_call(static_cast<std::size_t>(k), dt_ns);
    if (ws.bk.size() == 0)
      return false;

    merge_sets_into(ws.dt, ws.bk, r, job_target, vac, ws.next);
    if (ws.next.size() == 0)
      return false;
    std::swap(ws.dt, ws.next);
  }
  return ws.dt.size() > 0;
}

inline Vec make_target_vec(const Eigen::Ref<const VectorXi> &b_up, int r) {
  Vec tgt{};
  for (int i = 0; i < r; ++i)
    tgt.x[i] = b_up[i];
  return tgt;
}

inline void init_digit_bounds(const MatrixXi &lo, const MatrixXi &hi, int r,
                              int num_digits, ModernWorkspace &ws) {
  ws.digit_bounds.resize(num_digits);
  for (int i = 0; i < num_digits; ++i) {
    ws.digit_bounds[i] =
        make_window_bounds(lo.col(i), hi.col(i), r);
  }
}

/*
  Initialize digit merging.
  Check bounds on "Zero + N_i"
  merge with doubling applied
 */
inline bool run_base_digit(const SumSet &Ni, int r,
                           const Eigen::Ref<const Eigen::VectorXi> &b_up_i,
                           const Bounds &bounds, SumSet &merged) {
  SumSet zero;
  reset_unit_sumset(zero);
  merge_sets_scale2_into(zero, Ni, r, b_up_i, bounds, merged);
  return merged.size() > 0;
}

/*
  Intermediate digit.
  Check bounds on "merged + N_i"
  merge with doubling applied
*/
inline bool run_mid_digit(const SumSet &merged, const SumSet &Ni, int r,
                          const Eigen::Ref<const Eigen::VectorXi> &b_up_i,
                          const Bounds &bounds, SumSet &tmp, SumSet &next_merged) {
  merge_sets_scale2_into(merged, Ni, r, b_up_i, bounds, tmp);
  if (tmp.size() == 0)
    return false;
  std::swap(next_merged, tmp);
  return true;
}

/*
  final digit:
  Check exact match "target" for any "merged + N_i"
*/
inline bool run_final_digit(const SumSet &merged, const SumSet &Ni, int r,
                            const Eigen::Ref<const Eigen::VectorXi> &b_up,
                            const Bounds &bounds, const Vec &target) {
  return merge_sets_hits_target(merged, Ni, r, b_up, bounds, target);
}

bool solve_impl_core(const LPModel<PackedA> &m,
                     const oracle::ac_modern::BaseTableFnSumSet get_block_sumset) {
  const int Delta_global = m.A.maxCoeff();
  const auto &A = m.A;
  const auto &b = m.b;

  if (A.num_blocks() == 0) {
    return (b.array() == 0).all();
  }

  const int r = A.rows();
  const int n = A.num_blocks();

  const auto b_up = b.head(r);
  const auto b_down = b.tail(n);

  const int K = std::max(
      1,
      int(floor(2.0 * (r + 1) * log2(4.0 * (r + 1) * double(Delta_global)))));
  const int D = n * K * Delta_global;
  const int b_down_max = b_down.maxCoeff();
  const int I =
      std::max(1, int(ceil(log2(double(b_down_max + K) / (2 * K + 1)))) + 1);
  assert(K >= 1 && I >= 1 && D >= 1);

  MatrixXi b_down_tilde = MatrixXi::Zero(n, I);
  get_num_machines(I, n, K, b_down, b_down_tilde);
  const MatrixXi b_up_tilde = get_num_jobs(I, b_up);
  const MatrixXi LO = (b_up_tilde.array() - D).matrix().cwiseMax(0);

  static ModernWorkspace ws;
  ws.merged.clear();
  ws.tmp.clear();
  init_digit_bounds(LO, b_up_tilde, r, I, ws);
  ws.pt.resize(static_cast<std::size_t>(n));
  for (int k = 0; k < n; ++k)
    ws.pt[static_cast<std::size_t>(k)].clear_cache();

  const Vec tgt = make_target_vec(b_up, r);

  if (I == 1) {
    if (!run_digit_dp_sumset(A, r, b_up_tilde.col(0), b_down_tilde.col(0),
                             get_block_sumset, ws)) {
      return false;
    }
    if (!ws.dt.hashed)
      ws.dt.sort_unique();
    return ws.dt.contains(tgt);
  }

  for (int i = 0; i < I; ++i) {
    if (!run_digit_dp_sumset(A, r, b_up_tilde.col(i), b_down_tilde.col(i),
                             get_block_sumset, ws)) {
      return false;
    }
    const SumSet &Ni = ws.dt;

    if (i == 0) {
      if (!run_base_digit(Ni, r, b_up_tilde.col(i),
                          ws.digit_bounds[static_cast<std::size_t>(i)],
                          ws.merged)) {
        return false;
      }
      continue;
    }

    if (i < I - 1) {
      if (!run_mid_digit(ws.merged, Ni, r, b_up_tilde.col(i),
                         ws.digit_bounds[static_cast<std::size_t>(i)], ws.tmp,
                         ws.merged)) {
        return false;
      }
      continue;
    }

    return run_final_digit(ws.merged, Ni, r, b_up,
                           ws.digit_bounds[static_cast<std::size_t>(i)], tgt);
  }

  return false;
}

} // namespace

namespace oracle::ac_modern {

bool solve_impl(const LPModel<PackedA> &m, const BaseTableFnSumSet &bt) {
  return solve_impl_core(m, bt);
}

} // namespace oracle::ac_modern
