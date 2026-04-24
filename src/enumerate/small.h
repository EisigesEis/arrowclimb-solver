#pragma once
#include "instance/types.h"
#include <Eigen/Core>
#include <algorithm>

namespace {
using Eigen::VectorXi;
using i64 = long long;
} // namespace

namespace enumerate_small {
/*
  Iterative DFS enumeration of all feasible small machine configurations.

  Enumerates all configs x \in [0..u_j]^N s.t.
    x p^T \le C

  where C is capacity bound for biggest small machine.

  - conf is our config vectors of job types 0 to N-1.
  - remain[j] stores unused capacity:
      remain[j] = C - sum_{i < j} conf[i] * p[i]
  - max_x[j] is largest feasible value for conf[j]:
      max_x[j] = min(u[j], remain[j]/p[j])
  
  DFS:
  1. Enter level j: Update max_x[j] and rem[j]
  2. If leaf (j = N-1): Emit in tight loop all x_j\in\N_{\le max_x[j]}
  3. If x_j \le max_x[j] descend
  4. Else: Reset, backtrack, increment parent

  optimization:
  - incremental capacity instead of recomputing sums
  - on emit, cost = C - rem[j] (no separate cost computation)
  - no infeasible nodes entered (x_j \le max_x[j])
  - leaf is tight loop (better for compiler)
  - iterative dfs

  O(#feasible prefixes + #configs)
*/
template <class Emit> inline void S7(const ProblemInstance &inst, Emit &&emit) {
  const int C = std::min(inst.t[inst.num_small_machines - 1],
                         (int)inst.avg_makespan + inst.p_max);
  const int *const __restrict p = inst.p.data();
  const int *const __restrict u = inst.n.data();
  const auto& N = inst.N;

  Eigen::VectorXi conf = Eigen::VectorXi::Zero(N);
  Eigen::VectorXi remain = Eigen::VectorXi::Zero(N + 1);
  Eigen::VectorXi max_x = Eigen::VectorXi::Zero(N);

  // full capacity at root
  remain[0] = C;

  // compute maximal conf[j]
  auto compute_max_x = [&](int j) -> int {
    const int cap = u[j];
    if (cap <= 0)
      return 0;
    const int r = remain[j];
    if (r <= 0)
      return 0;
    return std::min(cap, r / p[j]);
  };

  /*
    Compute max_x of fresh level and
    update remain according to already set conf[j] for level j.
  */
  auto enter_level = [&](int j) {
    max_x[j] = compute_max_x(j);
    remain[j + 1] = (remain[j] - conf[j] * p[j]);
  };

  int j = 0;
  enter_level(0);

  /*
    Add one job of type j and update remain with its cost.
  */
  auto advance_parent = [&]() {
    ++conf[j];
    remain[j + 1] -= p[j];
  };

  // DFS loop
  while (j >= 0) {
    // leaf path
    if (j == N - 1) {
      const int w = p[j];
      const int mx = max_x[j];

      // emit all feasible conf
      int rem1 = remain[j + 1];
      for (int x = conf[j]; x <= mx; ++x) {
        conf[j] = x;
        emit(conf, C - rem1);
        rem1 -= w;
      }

      // reset leaf level, backtrack
      conf[j] = 0;
      if (--j >= 0)
        advance_parent();
      continue;
    }

    // descend
    if (conf[j] <= max_x[j]) [[likely]] {
      ++j;
      conf[j] = 0;
      enter_level(j);
      continue;
    }

    // exhausted, so backtrack
    conf[j] = 0;
    if (--j >= 0)
      advance_parent();
  }
}

template <class Emit>
inline void optimal(const ProblemInstance &inst, Emit &&emit) {
  if (inst.num_small_machines == 0)
    return;
  return S7(inst, std::forward<Emit>(emit));
}
} // namespace enumerate_small
