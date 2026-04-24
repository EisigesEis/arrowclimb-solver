#pragma once

#include "instance/types.h"
#include <Eigen/Dense>
#include <algorithm>
#include <execution>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

using Eigen::VectorXi;
using Scal = int;

namespace enumerate_big {

// B0: plain odometer baseline
template <class Emit> void B0(const ProblemInstance &inst, Emit &&emit) {
  VectorXi r(inst.N), x = VectorXi::Zero(inst.N);
  for (size_t j = 0; j < inst.N; ++j) {
    r[j] = 1 + ((j == inst.idx_a) ? 0 : std::min(inst.a - 1, inst.n[j]));
  }

  auto recompute_cost = [&]() -> int {
    return static_cast<int>(x.cast<long long>().dot(inst.p.cast<long long>()));
  };

  emit(x, 0);

  while (true) {
    bool advanced = false;
    for (size_t j = 0; j < inst.N; ++j) {
      if (r[j] == 1)
        continue;

      u32 next = x[j] + 1;
      if (next < r[j]) [[likely]] {
        x[j] = next;
        emit(x, recompute_cost());
        advanced = true;
        break;
      } else {
        x[j] = 0;
      }
    }
    if (!advanced)
      break;
  }
}

// B1: incremental full cost
template <class Emit> void B1(const ProblemInstance &inst, Emit &&emit) {
  VectorXi r(inst.N), x = VectorXi::Zero(inst.N);
  for (size_t j = 0; j < inst.N; ++j) {
    r[j] = 1 + ((j == inst.idx_a) ? 0 : std::min(inst.a - 1, inst.n[j]));
  }

  u32 cost = 0;
  emit(x, 0);

  while (true) {
    bool advanced = false;
    for (size_t j = 0; j < inst.N; ++j) {
      if (r[j] == 1)
        continue;

      Scal next = x[j] + 1;
      cost += inst.p[j];

      if (next < r[j]) [[likely]] {
        x[j] = next;
        emit(x, cost);
        advanced = true;
        break;
      } else {
        x[j] = 0;
        cost -= inst.p[j] * r[j];
      }
    }
    if (!advanced)
      break;
  }
}

// Helper: build LSB..MSB permutation by descending radix (tie-break by p)
template <class IdxSize>
inline void build_radix_permutation(const VectorXi &p, const VectorXi &r,
                                    std::vector<IdxSize> &idx,
                                    std::vector<IdxSize> &inv) {
  const size_t N = r.size();
  idx.resize(N);
  std::iota(idx.begin(), idx.end(), 0);

  // Sort by radix descending; stable for determinism on ties.
  std::stable_sort(std::execution::unseq, idx.begin(), idx.end(),
                   [&](IdxSize i, IdxSize j) {
                     if (r[i] != r[j])
                       return r[i] > r[j];
                     return p[i] < p[j];
                   });

  inv.resize(N);
  for (int k = 0; k < N; ++k)
    inv[idx[k]] = k;
}

/**
 * - Radix-permuted odometer
 * - residue-cost-only
 * - incremental emit in original order
 */
template <class Emit> void B2(const ProblemInstance &inst, Emit &&emit) {
  VectorXi r(inst.N);
  for (size_t j = 0; j < inst.N; ++j) {
    r[j] = 1 + ((j == inst.idx_a) ? 0 : std::min(inst.a - 1, inst.n[j]));
  }

  std::vector<size_t> idx, inv;
  build_radix_permutation(inst.p, r, idx, inv);

  VectorXi pp(inst.N), rp(inst.N);
  VectorXi p_mod(inst.N), overflow_mod(inst.N);
  for (size_t k = 0; k < inst.N; ++k) {
    const size_t s = idx[k];
    const Scal ps = inst.p[s];
    const Scal rs = r[s];

    pp[k] = ps;
    rp[k] = rs;

    const Scal pm = ps % inst.a;
    p_mod[k] = pm;
    overflow_mod[k] = static_cast<Scal>((pm * rs) % inst.a);
  }

  VectorXi xo, xp;
  xo.setZero(inst.N);
  xp.setZero(inst.N);

  Scal residue = 0;
  emit(xo, 0);

  while (true) {
    bool advanced = false;
    for (size_t jp = 0; jp < inst.N; ++jp) {
      if (rp[jp] == 1)
        continue;

      const Scal radix = rp[jp];
      const Scal next = xp[jp] + 1;

      const Scal tmp = residue + p_mod[jp];
      residue = (tmp >= inst.a) ? (tmp - inst.a) : tmp;

      const auto jo = idx[jp];

      if (next < radix) [[likely]] {
        xp[jp] = next;
        xo[jo] = next;
        emit(xo, residue);
        advanced = true;
        break;
      } else {
        xp[jp] = 0;
        xo[jo] = 0;
        residue = (residue >= overflow_mod[jp])
                      ? (residue - overflow_mod[jp])
                      : (residue + (inst.a - overflow_mod[jp]));
      }
    }
    if (!advanced)
      break;
  }
}

// permute r in place
template <class Emit> void B2_rip(const ProblemInstance &inst, Emit &&emit);

// use smaller data types when radix allows
template <class Emit> void B2_small(const ProblemInstance &inst, Emit &&emit);

/**
 * B2 but emit permuted vector and transform in one pass
 */
template <class Emit> void B3(const ProblemInstance &inst, Emit &&emit) {
  VectorXi r(inst.N);
  for (size_t j = 0; j < inst.N; ++j) {
    r[j] = 1 + ((j == inst.idx_a) ? 0u : std::min(inst.a - 1, inst.n[j]));
  }

  std::vector<size_t> idx, inv;
  build_radix_permutation(inst.p, r, idx, inv);

  VectorXi pp(inst.N), rp(inst.N);
  VectorXi p_mod(inst.N), overflow_mod(inst.N);
  for (size_t k = 0; k < inst.N; ++k) {
    const size_t s = idx[k];
    const u32 ps = inst.p[s];
    const u32 rs = r[s];

    pp[k] = ps;
    rp[k] = rs;

    const u32 pm = ps % inst.a;
    p_mod[k] = pm;
    overflow_mod[k] = static_cast<u32>((static_cast<u64>(pm) * rs) % inst.a);
  }

  VectorXi xo, xp;
  xo.setZero(inst.N);
  xp.setZero(inst.N);

  u32 residue = 0;

  auto emit_perm = [&]() {
    for (size_t jp = 0; jp < inst.N; ++jp) {
      auto jo = idx[jp];
      xo[jo] = xp[jp];
    }
    emit(xo, residue);
  };

  emit_perm();

  while (true) {
    bool advanced = false;
    for (size_t jp = 0; jp < inst.N; ++jp) {
      if (rp[jp] == 1)
        continue;

      const auto radix = rp[jp];
      const auto next = xp[jp] + 1;

      const auto tmp = residue + p_mod[jp];
      residue = (tmp >= inst.a) ? (tmp - inst.a) : tmp;

      if (next < radix) [[likely]] {
        xp[jp] = next;
        emit_perm();
        advanced = true;
        break;
      } else {
        xp[jp] = 0;
        residue = (residue >= overflow_mod[jp])
                      ? (residue - overflow_mod[jp])
                      : (residue + (inst.a - overflow_mod[jp]));
      }
    }
    if (!advanced)
      break;
  }
}

// only map back changed values in emit_perm
template <class Emit> void B3_delta(const ProblemInstance &inst, Emit &&emit);

template <class... Arrays>
auto permute_pack(const std::vector<size_t> &idx, const Arrays &...in)
    -> std::tuple<std::vector<typename Arrays::value_type>...> {
  const size_t N = idx.size();
  std::tuple<std::vector<typename Arrays::value_type>...> out{
      std::vector<typename Arrays::value_type>(N)...};

  for (size_t k = 0; k < N; ++k) {
    const size_t s = idx[k];
    ((std::get<std::vector<typename Arrays::value_type>>(out)[k] = in[s]), ...);
  }
  return out;
}

template <class T, class F>
static inline void apply_cycle(std::vector<T> &a, const std::vector<size_t> &cycle,
                               F &&move_cb) {
  if (cycle.size() <= 1)
    return;
  T tmp = std::move(a[cycle[0]]);
  for (size_t i = 0; i + 1 < cycle.size(); ++i) {
    a[cycle[i]] = std::move(a[cycle[i + 1]]);
    move_cb(cycle[i], cycle[i + 1]);
  }
  a[cycle.back()] = std::move(tmp);
  move_cb(cycle.back(), cycle[0]);
}

template <class... Arrays>
void permute_inplace_multi(const std::vector<size_t> &inv, Arrays &...arrs) {
  const size_t N = inv.size();
  std::vector<char> seen(N, 0);

  auto noop = [](size_t, size_t) {};

  for (size_t start = 0; start < N; ++start) {
    if (seen[start])
      continue;
    std::vector<size_t> cyc;
    size_t cur = start;
    while (!seen[cur]) {
      seen[cur] = 1;
      cyc.push_back(cur);
      cur = inv[cur];
    }
    if (cyc.size() <= 1)
      continue;

    (apply_cycle(arrs, cyc, noop), ...);
  }
}

} // namespace enumerate_big
