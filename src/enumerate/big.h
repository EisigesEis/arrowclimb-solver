#pragma once

#include "instance/types.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <omp.h>

using Eigen::VectorXi;
using Scal = int;

namespace enumerate_big {

/*
Challenges:
- r = {1 + min(a-1, n_j) | j\in[N]};  r[j*] = 1
- we emit(x, c) for:
  - config x\in{k \le r_j | j\in[N]}
  - and its cost c = p^T x

goal: until all configs emitted:
1. find first j with x_j < r_j
2. increase x_j
3. reset all x_i with i < j to 0
4. emit(x, c)

optimizations:
- optimistic add easing branch prediction
- revert (3) by cached c -= p_times_r[i]

\prod_j r_j emits, each in armortized O(1)
*/
template <class Emit> void B1_opt(const ProblemInstance &inst, Emit &&emit) {
  const Scal *const __restrict p = inst.p.data();
  const Scal *const __restrict n = inst.n.data();

  VectorXi r(inst.N), x = VectorXi::Zero(inst.N);
  VectorXi p_times_r(inst.N);
#pragma omp simd
  for (size_t j = 0; j < inst.N; ++j) {
    const Scal rj = 1 + ((j == inst.idx_a) ? 0 : std::min(inst.a - 1, n[j]));
    r[j] = rj;
    p_times_r[j] = static_cast<uint64_t>(p[j]) * rj;
  }

  Scal cost = 0;
  emit(x, 0);

  while (true) {
    bool advanced = false;
    for (size_t j = 0; j < inst.N; ++j) {
      if (r[j] == 1)
        continue;

      Scal next = x[j] + 1;
      cost += p[j];

      if (next < r[j]) [[likely]] {
        x[j] = next;
        emit(x, cost);
        advanced = true;
        break;
      } else {
        x[j] = 0;
        cost -= p_times_r[j];
      }
    }
    if (!advanced)
      break;
  }
}

template <class Emit> void optimal(const ProblemInstance &inst, Emit &&emit) {
  B1_opt(inst, emit);
}

} // namespace enumerate_big
