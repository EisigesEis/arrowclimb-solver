#include "oracle/ac/discrepancy/bt_enumerator.h"

#include "bench.h"
#include "convolve/sumset.h"
#include "oracle/ac/common/profile.h"
#include "oracle/discrep/solve.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace oracle::ac_discrepancy {

namespace {

int compute_K_default(int r, int Delta) {
  return std::max(
      1,
      int(std::floor(2.0 * (r + 1) *
                     std::log2(4.0 * (r + 1) * double(Delta)))));
}

} // namespace

Eigen::VectorXi compute_block_caps(const PackedA &A, int k,
                                   Eigen::Ref<const Eigen::VectorXi> target) {
  const int r = static_cast<int>(A.rows());
  Eigen::VectorXi caps = target;
  const int Delta_local =
      std::max(1, A.block_maxCoeff(static_cast<std::size_t>(k)));
  const int K = compute_K_default(r, Delta_local);
  for (int i = 0; i < r; ++i) {
    const long long local_cap = 1LL * K * Delta_local;
    if (local_cap < caps[i])
      caps[i] = static_cast<int>(local_cap);
  }
  return caps;
}

SumSet compute_base_sumset_for_block(const PackedA &A, int k,
                                     Eigen::Ref<const Eigen::VectorXi> target,
                                     int s, PowTable &pow_table) {
  (void)pow_table;

  SumSet empty;
  const int r = static_cast<int>(A.rows());
  if (target.size() != r) {
    assert(false &&
           "compute_base_sumset_for_block: target.size() must equal A.rows()");
  }

  const auto [c0, c1] = A.block_col_range(static_cast<std::size_t>(k));
  if (r == 0 || c0 == c1) {
    if (s == 0) {
      reset_unit_sumset(empty);
      return empty;
    }
    return empty;
  }
  if (s == 0) {
    reset_unit_sumset(empty);
    return empty;
  }

  const Eigen::VectorXi caps = compute_block_caps(A, k, target);
  Eigen::VectorXi cur = Eigen::VectorXi::Zero(r);
  SumSet out;
  out.len = 1;
  out.hashed = false;
  long long grid_points_total = 0;
  long long feasible_vectors_total = 0;

  while (true) {
    ++grid_points_total;
    if (oracle::discrep::solve_single_block(A, k, cur, s)) {
      ++feasible_vectors_total;
      Vec v{};
      for (int i = 0; i < r; ++i)
        v.x[static_cast<std::size_t>(i)] = cur[i];
      out.vec.push_back(v);
    }
    if (bench_detail::current_result_rejected())
      break;

    int idx = 0;
    for (; idx < r; ++idx) {
      if (cur[idx] < caps[idx]) {
        ++cur[idx];
        break;
      }
      cur[idx] = 0;
    }
    if (idx == r)
      break;
  }

  oracle::ac_profile::note_grid_enumeration_base_call(grid_points_total,
                                                      feasible_vectors_total);

  if (bench_detail::current_result_rejected())
    return empty;

  if (out.vec.empty())
    return empty;

  out.sort_unique();
  out.ensure_hashed();
  return out;
}

} // namespace oracle::ac_discrepancy
