#pragma once
#include "instance/types.h"
#include <Eigen/Core>
#include <algorithm>

namespace {
// We work with Eigen::VectorXi for p, u, conf.
// Counts are stored as 64-bit to avoid overflow.
using Eigen::VectorXi;
using i64 = long long;

// S0: Baseline DP (O(N * C * cap_j)) with Eigen matrix.

// dp(j, L) = #ways to use items j..N-1 to get exact load L
static inline Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>
build_suffix_dp_counts_S0(const VectorXi &p, const VectorXi &u, int C) {
  const int N = static_cast<int>(p.size());
  Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dp(N + 1,
                                                                         C + 1);
  dp.setZero();
  dp(N, 0) = 1; // empty suffix achieves load 0 in exactly 1 way

  for (int j = N - 1; j >= 0; --j) {
    const int w = p(j);
    const int cap = u(j);
    for (int L = 0; L <= C; ++L) {
      i64 sum = 0;
      const int max_x = std::min(cap, L / w); // assume w > 0
      for (int x = 0; x <= max_x; ++x) {
        sum += dp(j + 1, L - x * w);
      }
      dp(j, L) = sum;
    }
  }
  return dp;
}

// dp_le(j, L) = sum_{ell <= L} dp(j, ell)
static inline Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>
build_suffix_dp_counts_le_S0(const VectorXi &p, const VectorXi &u, int C) {
  auto dp = build_suffix_dp_counts_S0(p, u, C);
  const int N = static_cast<int>(p.size());
  for (int j = 0; j <= N; ++j) {
    i64 running = 0;
    for (int L = 0; L <= C; ++L) {
      running += dp(j, L);
      dp(j, L) = running;
    }
  }
  return dp;
}

template <class Emit>
inline void enumerate_small_all_S0(const VectorXi &p, const VectorXi &u, int C,
                                   Emit &&emit) {
  const int N = static_cast<int>(p.size());
  auto dp_le = build_suffix_dp_counts_le_S0(p, u, C);

  VectorXi conf = VectorXi::Zero(N);

  auto rec = [&](auto &&self, int j, int remain, int cost) -> void {
    if (j == N) {
      // total load is cost (≤ C by construction)
      emit(conf, cost);
      return;
    }

    const int w = p(j);
    const int cap = u(j);
    const int max_x = std::min(cap, remain / w);

    for (int x = 0; x <= max_x; ++x) {
      const int rem2 = remain - x * w;
      // Is there any completion on suffix j+1 with load ≤ rem2?
      if (dp_le(j + 1, rem2) == 0)
        continue;

      conf(j) = x;
      self(self, j + 1, rem2, cost + x * w);
    }
    conf(j) = 0;
  };

  rec(rec, 0, C, 0);
}

// S1: O(N * C) DP using sliding windows per residue class.

static inline Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>
build_suffix_dp_counts_S1(const VectorXi &p, const VectorXi &u, int C) {
  const int N = static_cast<int>(p.size());
  Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dp(N + 1,
                                                                         C + 1);
  dp.setZero();
  dp(N, 0) = 1;

  for (int j = N - 1; j >= 0; --j) {
    const int w = p(j);
    const int cap = u(j);

    if (w == 0) {
      // Degenerate: weight 0, behaves like choose 0..cap without affecting
      // load.
      const i64 factor = static_cast<i64>(cap) + 1;
      for (int L = 0; L <= C; ++L) {
        dp(j, L) = dp(j + 1, L) * factor;
      }
      continue;
    }

    // For each residue class modulo w, slide a window of size cap+1
    for (int r = 0; r < w && r <= C; ++r) {
      i64 window_sum = 0;
      int k = 0; // number of multiples for this residue
      for (int L = r; L <= C; L += w, ++k) {
        window_sum += dp(j + 1, L);
        if (k > cap) {
          const int L_out = L - (cap + 1) * w;
          if (L_out >= 0)
            window_sum -= dp(j + 1, L_out);
        }
        dp(j, L) = window_sum;
      }
    }
  }
  return dp;
}

static inline Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>
build_suffix_dp_counts_le_S1(const VectorXi &p, const VectorXi &u, int C) {
  auto dp = build_suffix_dp_counts_S1(p, u, C);
  const int N = static_cast<int>(p.size());
  for (int j = 0; j <= N; ++j) {
    i64 running = 0;
    for (int L = 0; L <= C; ++L) {
      running += dp(j, L);
      dp(j, L) = running;
    }
  }
  return dp;
}

template <class Emit>
inline void enumerate_small_all_S1(const VectorXi &p, const VectorXi &u, int C,
                                   Emit &&emit) {
  const int N = static_cast<int>(p.size());
  auto dp_le = build_suffix_dp_counts_le_S1(p, u, C);

  VectorXi conf = VectorXi::Zero(N);

  auto rec = [&](auto &&self, int j, int remain, int cost) -> void {
    if (j == N) {
      emit(conf, cost);
      return;
    }

    const int w = p(j);
    const int cap = u(j);
    const int max_x = std::min(cap, remain / w);

    for (int x = 0; x <= max_x; ++x) {
      const int rem2 = remain - x * w;
      if (dp_le(j + 1, rem2) == 0)
        continue;
      conf(j) = x;
      self(self, j + 1, rem2, cost + x * w);
    }
    conf(j) = 0;
  };

  rec(rec, 0, C, 0);
}

// S2: S0 recurrence with row-major storage.

static inline Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>
build_suffix_dp_counts_le_S2(const VectorXi &p, const VectorXi &u, int C) {
  const int N = static_cast<int>(p.size());
  const int stride = C + 1;

  Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dp(N + 1,
                                                                         C + 1);
  dp.setZero();
  dp(N, 0) = 1;

  // exact counts
  for (int j = N - 1; j >= 0; --j) {
    const int w = p(j);
    const int cap = u(j);
    for (int L = 0; L <= C; ++L) {
      i64 sum = 0;
      const int max_x = std::min(cap, L / w);
      for (int x = 0; x <= max_x; ++x) {
        sum += dp(j + 1, L - x * w);
      }
      dp(j, L) = sum;
    }
  }

  // cumulative <=
  for (int j = 0; j <= N; ++j) {
    i64 running = 0;
    i64 *row = dp.data() + j * stride;
    for (int L = 0; L <= C; ++L) {
      running += row[L];
      row[L] = running;
    }
  }

  return dp;
}

template <class Emit>
inline void enumerate_small_all_S2(const VectorXi &p, const VectorXi &u, int C,
                                   Emit &&emit) {
  const int N = static_cast<int>(p.size());
  const int stride = C + 1;

  auto dp_le = build_suffix_dp_counts_le_S2(p, u, C);
  VectorXi conf = VectorXi::Zero(N);

  auto rec = [&](auto &&self, int j, int remain, int cost) -> void {
    if (j == N) {
      emit(conf, cost);
      return;
    }

    const int w = p(j);
    const int cap = u(j);

    const int max_x = std::min(cap, remain / w);
    const i64 *row_suffix = dp_le.data() + (j + 1) * stride;

    for (int x = 0; x <= max_x; ++x) {
      const int rem2 = remain - x * w;
      if (row_suffix[rem2] == 0)
        continue;
      conf(j) = x;
      self(self, j + 1, rem2, cost + x * w);
    }
    conf(j) = 0;
  };

  rec(rec, 0, C, 0);
}

// S3: Existence-only DP for pruning.

static inline Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>
build_suffix_dp_exists_S3(const VectorXi &p, const VectorXi &u, int C) {
  const int N = static_cast<int>(p.size());
  using MatB = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>;
  MatB dp(N + 1, C + 1);
  dp.setZero();
  dp(N, 0) = 1;

  for (int j = N - 1; j >= 0; --j) {
    const int w = p(j);
    const int cap = u(j);
    for (int L = 0; L <= C; ++L) {
      bool exists = false;
      const int max_x = std::min(cap, L / w);
      for (int x = 0; x <= max_x; ++x) {
        if (dp(j + 1, L - x * w)) {
          exists = true;
          break;
        }
      }
      dp(j, L) = exists ? 1 : 0;
    }
  }
  return dp;
}

static inline Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>
build_suffix_dp_exists_le_S3(const VectorXi &p, const VectorXi &u, int C) {
  auto dp = build_suffix_dp_exists_S3(p, u, C);
  const int N = static_cast<int>(p.size());
  for (int j = 0; j <= N; ++j) {
    unsigned char running = 0;
    for (int L = 0; L <= C; ++L) {
      running = static_cast<unsigned char>(running | dp(j, L));
      dp(j, L) = running;
    }
  }
  return dp;
}

template <class Emit>
inline void enumerate_small_all_S3(const VectorXi &p, const VectorXi &u, int C,
                                   Emit &&emit) {
  const int N = static_cast<int>(p.size());
  auto dp_le = build_suffix_dp_exists_le_S3(p, u, C);

  VectorXi conf = VectorXi::Zero(N);

  auto rec = [&](auto &&self, int j, int remain, int cost) -> void {
    if (j == N) {
      emit(conf, cost);
      return;
    }

    const int w = p(j);
    const int cap = u(j);
    const int max_x = std::min(cap, remain / w);

    for (int x = 0; x <= max_x; ++x) {
      const int rem2 = remain - x * w;
      if (!dp_le(j + 1, rem2))
        continue;
      conf(j) = x;
      self(self, j + 1, rem2, cost + x * w);
    }
    conf(j) = 0;
  };

  rec(rec, 0, C, 0);
}

// S4: Cached S0 DP for identical (p, u, C).

struct SmallDP_Cache_S4 {
  VectorXi last_p;
  VectorXi last_u;
  int last_C = -1;
  bool valid = false;
  Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dp_le;

  void ensure(const VectorXi &p, const VectorXi &u, int C) {
    if (!valid || C != last_C || p.size() != last_p.size() ||
        u.size() != last_u.size() || !((p - last_p).isZero(0)) ||
        !((u - last_u).isZero(0))) {
      last_p = p;
      last_u = u;
      last_C = C;
      dp_le = build_suffix_dp_counts_le_S0(p, u, C);
      valid = true;
    }
  }
};

template <class Emit>
inline void enumerate_small_all_S4(const VectorXi &p, const VectorXi &u, int C,
                                   Emit &&emit) {
  static SmallDP_Cache_S4 cache;
  cache.ensure(p, u, C);

  const int N = static_cast<int>(p.size());
  auto &dp_le = cache.dp_le;

  VectorXi conf = VectorXi::Zero(N);

  auto rec = [&](auto &&self, int j, int remain, int cost) -> void {
    if (j == N) {
      emit(conf, cost);
      return;
    }

    const int w = p(j);
    const int cap = u(j);
    const int max_x = std::min(cap, remain / w);

    for (int x = 0; x <= max_x; ++x) {
      const int rem2 = remain - x * w;
      if (dp_le(j + 1, rem2) == 0)
        continue;
      conf(j) = x;
      self(self, j + 1, rem2, cost + x * w);
    }
    conf(j) = 0;
  };

  rec(rec, 0, C, 0);
}

} // namespace

namespace enumerate_small_old {
template <class Emit> inline void S0(const ProblemInstance &inst, Emit &&emit) {
  if (inst.num_small_machines == 0)
    return;
  const VectorXi &p = inst.p;
  const VectorXi &u = inst.n;
  const int C = inst.t[inst.num_small_machines - 1];
  enumerate_small_all_S0(p, u, C, std::forward<Emit>(emit));
}

template <class Emit> inline void S1(const ProblemInstance &inst, Emit &&emit) {
  if (inst.num_small_machines == 0)
    return;
  const VectorXi &p = inst.p;
  const VectorXi &u = inst.n;
  const int C = inst.t[inst.num_small_machines - 1];
  enumerate_small_all_S1(p, u, C, std::forward<Emit>(emit));
}

template <class Emit> inline void S2(const ProblemInstance &inst, Emit &&emit) {
  if (inst.num_small_machines == 0)
    return;
  const VectorXi &p = inst.p;
  const VectorXi &u = inst.n;
  const int C = inst.t[inst.num_small_machines - 1];
  enumerate_small_all_S2(p, u, C, std::forward<Emit>(emit));
}

template <class Emit> inline void S3(const ProblemInstance &inst, Emit &&emit) {
  if (inst.num_small_machines == 0)
    return;
  const VectorXi &p = inst.p;
  const VectorXi &u = inst.n;
  const int C = inst.t[inst.num_small_machines - 1];
  enumerate_small_all_S3(p, u, C, std::forward<Emit>(emit));
}

template <class Emit> inline void S4(const ProblemInstance &inst, Emit &&emit) {
  if (inst.num_small_machines == 0)
    return;
  const VectorXi &p = inst.p;
  const VectorXi &u = inst.n;
  const int C = inst.t[inst.num_small_machines - 1];
  enumerate_small_all_S4(p, u, C, std::forward<Emit>(emit));
}
} // namespace enumerate_small

namespace enumerate_small {

static inline Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>
build_suffix_dp_exists_le_optimized(const VectorXi &p, const VectorXi &u,
                                    int C) {
  using MatB = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>;

  const int N = static_cast<int>(p.size());
  const int rows = N + 1;
  const int cols = C + 1;

  MatB dp(rows, cols);
  unsigned char *const data = dp.data();

  {
    unsigned char *rowN = data + N * cols;
    std::memset(rowN, 1, static_cast<size_t>(cols));
  }

  const int *const __restrict__ pw = p.data();
  const int *const __restrict__ cap_arr = u.data();

  for (int j = N - 1; j >= 0; --j) {
    const int w = pw[j];
    const int cap = cap_arr[j];

    unsigned char *const __restrict__ row_cur = data + j * cols;
    const unsigned char *const __restrict__ row_next = data + (j + 1) * cols;

    if (w == 0 || cap == 0 || w > C) {
      std::memcpy(row_cur, row_next, static_cast<size_t>(cols));
      continue;
    }

    const int rmax = std::min(w - 1, C);
    const int max_window = cap + 1;

    for (int r = 0; r <= rmax; ++r) {
      int count = 0;
      int window_len = 0;
      int L_out = r;

      for (int L = r; L <= C; L += w) {
        count += row_next[L];
        ++window_len;

        if (window_len > max_window) {
          count -= row_next[L_out];
          L_out += w;
        }

        if (count > 0) {
          row_cur[L] = 1;
          for (int L2 = L + w; L2 <= C; L2 += w) {
            row_cur[L2] = 1;
          }
          break;
        } else {
          row_cur[L] = 0;
        }
      }
    }
  }

  return dp;
}

static inline void extend_suffix_dp_exists_le_optimized(
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic,
                  Eigen::RowMajor> &dp_le,
    const Eigen::VectorXi &p, const Eigen::VectorXi &u, const int old_C,
    const int new_C) {
  const int N = static_cast<int>(p.size());

  const int old_cols = old_C + 1;
  const int new_cols = new_C + 1;
  dp_le.conservativeResize(Eigen::NoChange, new_C + 1);
  unsigned char *const data = dp_le.data();
  const int *const __restrict__ pw = p.data();
  const int *const __restrict__ cap_arr = u.data();

  {
    unsigned char *rowN = data + N * new_cols;
    std::memset(rowN + old_cols, 1, static_cast<size_t>(new_cols - old_cols));
  }

  for (int j = N - 1; j >= 0; --j) {
    const int w = pw[j];
    const int cap = cap_arr[j];

    unsigned char *const __restrict__ row_cur = data + j * new_cols;
    const unsigned char *const __restrict__ row_next =
        data + (j + 1) * new_cols;

    if (w == 0 || cap == 0 || w > new_C) {
      std::memcpy(row_cur + old_cols, row_next + old_cols,
                  static_cast<size_t>(new_cols - old_cols));
      continue;
    }

    if (row_cur[old_C]) {
      std::memset(row_cur + old_cols, 1,
                  static_cast<size_t>(new_cols - old_cols));
      continue;
    }

    const int max_window = cap + 1;
    const int rmax = std::min(w - 1, new_C);

    for (int r = 0; r <= rmax; ++r) {
      int L = r;
      if (L <= old_C) {
        const int delta = (old_C + 1) - L;
        if (delta > 0) {
          const int steps = (delta + w - 1) / w;
          L += steps * w;
        }
      }
      if (L > new_C)
        continue;

      int count = 0;
      int window_len = 0;
      int L_oldest = L;

      for (int t = 0; t < max_window; ++t) {
        const int L_t = L - t * w;
        if (L_t < 0)
          break;
        count += row_next[L_t];
        ++window_len;
        L_oldest = L_t;
      }

      for (;;) {
        const bool any = (count > 0);
        row_cur[L] = any ? 1 : 0;

        if (any) {
          for (int L2 = L + w; L2 <= new_C; L2 += w) {
            row_cur[L2] = 1;
          }
          break;
        }

        if (L + w > new_C)
          break;
        const int L_next = L + w;

        if (window_len == max_window) {
          count -= row_next[L_oldest];
          L_oldest += w;
        } else {
          ++window_len;
        }
        count += row_next[L_next];

        L = L_next;
      }
    }
  }
}

struct SmallDP_Cache_S5 {
  int last_C = -1;
  bool valid = false;
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dp_le;

  void ensure(const VectorXi &p, const VectorXi &u, int C) {
    if (!valid) {
      last_C = C;
      dp_le = build_suffix_dp_exists_le_optimized(p, u, C);
      valid = true;
      return;
    }

    if (C <= last_C) {
      return;
    }

    extend_suffix_dp_exists_le_optimized(dp_le, p, u, last_C, C);
    last_C = C;
  }

  constexpr void invalidate() { this->valid = false; }
};

namespace {

SmallDP_Cache_S5 &small_dp_cache() {
  static SmallDP_Cache_S5 cache;
  return cache;
}

} // namespace

inline void invalidate_small_dp_cache() { small_dp_cache().invalidate(); }

template <class Emit>
inline void enumerate_small_all_S5(const VectorXi &p, const VectorXi &u, int C,
                                   Emit &&emit) {
  auto &cache = small_dp_cache();
  cache.ensure(p, u, C);

  const int N = static_cast<int>(p.size());
  const int *const __restrict__ pw = p.data();
  const int *const __restrict__ cap = u.data();
  const auto &dp_le = cache.dp_le;

  VectorXi conf = VectorXi::Zero(N);

  auto rec = [&](auto &&self, int j, int remain) -> void {
    if (j == N) {
      emit(conf, C - remain);
      return;
    }

    const int w = pw[j];
    const int max_x = std::min(cap[j], remain / w);

    for (int x = 0, rem2 = remain; x <= max_x; ++x, rem2 -= w) {
      if (dp_le(j + 1, rem2) == 0)
        continue;
      conf(j) = x;
      self(self, j + 1, rem2);
    }
    conf(j) = 0;
  };

  rec(rec, 0, C);
}

template <class Emit>
inline void enumerate_small_all_S6(const VectorXi &p, const VectorXi &u, int C,
                                   Emit &&emit) {
  auto &cache = small_dp_cache();
  cache.ensure(p, u, C);

  const int N = static_cast<int>(p.size());
  const int *const __restrict__ pw = p.data();
  const int *const __restrict__ cap = u.data();
  const auto &dp_le = cache.dp_le;

  VectorXi conf = VectorXi::Zero(N);
  VectorXi remain = VectorXi::Zero(N + 1);
  remain[0] = C;

  Eigen::VectorXi max_x = Eigen::VectorXi::Zero(N);
  {
    const int w0 = pw[0];
    max_x(0) = std::min(cap[0], remain[0] / w0);
  }

  int j = 0;
  auto backtrack = [&]() {
    if (--j >= 0)
      ++conf(j);
  };

  while (j >= 0) {
    if (j == N) {
      emit(conf, C - remain(N));
      backtrack();
      continue;
    }

    const int w = pw[j];
    const int mx = max_x(j);
    const int x = conf(j);

    if (x > mx) {
      conf(j) = 0;
      backtrack();
      continue;
    }

    const int rem2 = remain[j] - x * w;

    if (dp_le(j + 1, rem2) == 0) {
      conf(j) = 0;
      backtrack();
      continue;
    }

    remain[j + 1] = rem2;

    if (++j < N) {
      const int w_next = pw[j];
      max_x(j) = std::min(cap[j], remain[j] / w_next);
    }
  }
}

template <class Emit> inline void S5(const ProblemInstance &inst, Emit &&emit) {
  if (inst.num_small_machines == 0)
    return;
  const VectorXi &p = inst.p;
  const VectorXi &u = inst.n;
  const int C = std::min(inst.t[inst.num_small_machines - 1],
                         (int)inst.avg_makespan + inst.p_max);
  enumerate_small_all_S5(p, u, C, std::forward<Emit>(emit));
}

template <class Emit> inline void S6(const ProblemInstance &inst, Emit &&emit) {
  if (inst.num_small_machines == 0)
    return;
  const VectorXi &p = inst.p;
  const VectorXi &u = inst.n;
  const int C = std::min(inst.t[inst.num_small_machines - 1],
                         (int)inst.avg_makespan + inst.p_max);
  enumerate_small_all_S6(p, u, C, std::forward<Emit>(emit));
}

} // namespace enumerate_small
