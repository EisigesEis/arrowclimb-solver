#pragma once
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
#include <Eigen/Dense>

static constexpr int MAX_M = 16;
static constexpr size_t SMALL_THRESHOLD = 8192; // vec -> hash

struct alignas(64) Vec {
  std::array<int, MAX_M> x{};
};
static_assert(sizeof(Vec) == sizeof(int) * MAX_M,
              "Vec layout must stay tightly packed for fast compare/hash.");

struct VecHash {
  size_t operator()(Vec const &v) const noexcept {
    return absl::HashOf(absl::Span<const int>(v.x.data(), v.x.size()));
  }
};
struct VecEq {
  bool operator()(Vec const &a, Vec const &b) const noexcept {
    return std::memcmp(a.x.data(), b.x.data(), sizeof(int) * MAX_M) == 0;
  }
};
struct VecLess {
  bool operator()(Vec const &a, Vec const &b) const noexcept {
    return std::memcmp(a.x.data(), b.x.data(), sizeof(int) * MAX_M) < 0;
  }
};

static inline long long ceil_div_ll(long long a, long long b) {
  return (a + (b - 1)) / b;
}

static inline long long floor_div_ll(long long a, long long b) {
  assert(b > 0);
  if (a >= 0)
    return a / b;
  return -(((-a) + (b - 1)) / b);
}

struct Bounds {
  std::array<long long, MAX_M> lo{};
  std::array<long long, MAX_M> hi{};
  bool lo_active = false;
  bool hi_active = false;
  bool vacuous = false;
};

struct SumSet {
  int len = 0;
  bool hashed = false;
  std::vector<Vec> vec;
  absl::flat_hash_set<Vec, VecHash, VecEq> hset;

  size_t size() const { return hashed ? hset.size() : vec.size(); }

  void clear() {
    vec.clear();
    hset.clear();
  }

  void sort_unique() {
    if (hashed)
      return;
    std::sort(vec.begin(), vec.end(), VecLess{});
    vec.erase(std::unique(vec.begin(), vec.end(), VecEq{}), vec.end());
  }

  void ensure_hashed() {
    if (hashed)
      return;
    if (vec.size() <= SMALL_THRESHOLD)
      return;
    hset.reserve(vec.size() * 2 + 8);
    for (auto &v : vec)
      hset.insert(std::move(v));
    vec.clear();
    hashed = true;
  }

  template <class F> void for_each(F &&f) const {
    if (hashed)
      for (auto const &v : hset)
        f(v);
    else
      for (auto const &v : vec)
        f(v);
  }

  bool contains(const Vec &t) const {
    if (hashed)
      return hset.find(t) != hset.end();
    return std::binary_search(vec.begin(), vec.end(), t, VecLess{});
  }
};

inline void reset_unit_sumset(SumSet &s) {
  s.clear();
  s.len = 0;
  s.hashed = false;
  s.vec.reserve(1);
  s.vec.push_back(Vec{});
}

template <typename BoundsVec>
static inline void merge_sets_into(const SumSet &A, const SumSet &B, int m_dim,
                                   const BoundsVec &b_up, const Bounds &bounds,
                                   SumSet &Out) {
  Out.len = A.len + B.len;

  const size_t asz = A.size();
  const size_t bsz = B.size();

  const size_t threshold_prod = SMALL_THRESHOLD * 8;
  const bool want_hash = (asz != 0 && bsz != 0 && asz > threshold_prod / bsz);
  Out.hashed = want_hash;

  if (Out.hashed) {
    Out.hset.clear();
    Out.vec.clear();
    const size_t ub = asz * bsz;
    const size_t hint = std::max(asz, bsz) * 64 + 64;
    Out.hset.reserve(std::min(ub, hint));
  } else {
    Out.vec.clear();
    Out.hset.clear();
    Out.vec.reserve(std::min(asz * bsz, threshold_prod));
  }

  auto emit = [&](const Vec &w) {
    if (Out.hashed)
      Out.hset.insert(w);
    else
      Out.vec.push_back(w);
  };

  if (bounds.vacuous) {
    A.for_each([&](const Vec &a) {
      B.for_each([&](const Vec &b) {
        Vec w{};
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        int *wx = w.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k])
            return;
          wx[k] = s;
        }
        emit(w);
      });
    });
  } else if (bounds.hi_active && bounds.lo_active) {
    A.for_each([&](const Vec &a) {
      B.for_each([&](const Vec &b) {
        Vec w{};
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        int *wx = w.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k] || (long long)s > bounds.hi[(size_t)k] ||
              (long long)s < bounds.lo[(size_t)k]) {
            return;
          }
          wx[k] = s;
        }
        emit(w);
      });
    });
  } else if (bounds.hi_active) {
    A.for_each([&](const Vec &a) {
      B.for_each([&](const Vec &b) {
        Vec w{};
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        int *wx = w.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k] || (long long)s > bounds.hi[(size_t)k])
            return;
          wx[k] = s;
        }
        emit(w);
      });
    });
  } else if (bounds.lo_active) {
    A.for_each([&](const Vec &a) {
      B.for_each([&](const Vec &b) {
        Vec w{};
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        int *wx = w.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k] || (long long)s < bounds.lo[(size_t)k])
            return;
          wx[k] = s;
        }
        emit(w);
      });
    });
  } else {
    A.for_each([&](const Vec &a) {
      B.for_each([&](const Vec &b) {
        Vec w{};
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        int *wx = w.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k])
            return;
          wx[k] = s;
        }
        emit(w);
      });
    });
  }

  if (!Out.hashed)
    Out.sort_unique();
  Out.ensure_hashed();
}

template <typename BoundsVec>
static inline void merge_sets_scale2_into(const SumSet &A, const SumSet &B,
                                          int m_dim, const BoundsVec &b_up,
                                          const Bounds &bounds, SumSet &Out) {
  Out.len = A.len + B.len;

  const size_t asz = A.size();
  const size_t bsz = B.size();

  const size_t threshold_prod = SMALL_THRESHOLD * 8;
  const bool want_hash = (asz != 0 && bsz != 0 && asz > threshold_prod / bsz);
  Out.hashed = want_hash;

  if (Out.hashed) {
    Out.hset.clear();
    Out.vec.clear();
    const size_t ub = asz * bsz;
    const size_t hint = std::max(asz, bsz) * 64 + 64;
    Out.hset.reserve(std::min(ub, hint));
  } else {
    Out.vec.clear();
    Out.hset.clear();
    Out.vec.reserve(std::min(asz * bsz, threshold_prod));
  }

  auto emit = [&](const Vec &w) {
    if (Out.hashed)
      Out.hset.insert(w);
    else
      Out.vec.push_back(w);
  };

  if (bounds.vacuous) {
    A.for_each([&](const Vec &a) {
      B.for_each([&](const Vec &b) {
        Vec w{};
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        int *wx = w.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k])
            return;
          wx[k] = s + s;
        }
        emit(w);
      });
    });
  } else if (bounds.hi_active && bounds.lo_active) {
    A.for_each([&](const Vec &a) {
      B.for_each([&](const Vec &b) {
        Vec w{};
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        int *wx = w.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k] || (long long)s > bounds.hi[(size_t)k] ||
              (long long)s < bounds.lo[(size_t)k]) {
            return;
          }
          wx[k] = s + s;
        }
        emit(w);
      });
    });
  } else if (bounds.hi_active) {
    A.for_each([&](const Vec &a) {
      B.for_each([&](const Vec &b) {
        Vec w{};
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        int *wx = w.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k] || (long long)s > bounds.hi[(size_t)k])
            return;
          wx[k] = s + s;
        }
        emit(w);
      });
    });
  } else if (bounds.lo_active) {
    A.for_each([&](const Vec &a) {
      B.for_each([&](const Vec &b) {
        Vec w{};
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        int *wx = w.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k] || (long long)s < bounds.lo[(size_t)k])
            return;
          wx[k] = s + s;
        }
        emit(w);
      });
    });
  } else {
    A.for_each([&](const Vec &a) {
      B.for_each([&](const Vec &b) {
        Vec w{};
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        int *wx = w.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k])
            return;
          wx[k] = s + s;
        }
        emit(w);
      });
    });
  }

  if (!Out.hashed)
    Out.sort_unique();
  Out.ensure_hashed();
}

template <typename BoundsVec>
static inline bool merge_sets_hits_target(const SumSet &A, const SumSet &B,
                                          int m_dim, const BoundsVec &b_up,
                                          const Bounds &bounds,
                                          const Vec &target) {
  bool hit = false;
  if (bounds.vacuous) {
    A.for_each([&](const Vec &a) {
      if (hit)
        return;
      B.for_each([&](const Vec &b) {
        if (hit)
          return;
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        const int *tx = target.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k] || s != tx[k])
            return;
        }
        hit = true;
      });
    });
  } else if (bounds.hi_active && bounds.lo_active) {
    A.for_each([&](const Vec &a) {
      if (hit)
        return;
      B.for_each([&](const Vec &b) {
        if (hit)
          return;
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        const int *tx = target.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k] || (long long)s > bounds.hi[(size_t)k] ||
              (long long)s < bounds.lo[(size_t)k] || s != tx[k]) {
            return;
          }
        }
        hit = true;
      });
    });
  } else if (bounds.hi_active) {
    A.for_each([&](const Vec &a) {
      if (hit)
        return;
      B.for_each([&](const Vec &b) {
        if (hit)
          return;
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        const int *tx = target.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k] || (long long)s > bounds.hi[(size_t)k] || s != tx[k])
            return;
        }
        hit = true;
      });
    });
  } else if (bounds.lo_active) {
    A.for_each([&](const Vec &a) {
      if (hit)
        return;
      B.for_each([&](const Vec &b) {
        if (hit)
          return;
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        const int *tx = target.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k] || (long long)s < bounds.lo[(size_t)k] || s != tx[k])
            return;
        }
        hit = true;
      });
    });
  } else {
    A.for_each([&](const Vec &a) {
      if (hit)
        return;
      B.for_each([&](const Vec &b) {
        if (hit)
          return;
        const int *ax = a.x.data();
        const int *bx = b.x.data();
        const int *tx = target.x.data();
        for (int k = 0; k < m_dim; ++k) {
          const int s = ax[k] + bx[k];
          if (s > b_up[k] || s != tx[k])
            return;
        }
        hit = true;
      });
    });
  }
  return hit;
}

struct PowTable {
  std::vector<SumSet> p;  // p[t] = sums of 2^t columns from this block
  std::vector<Bounds> pb; // cached tube bounds for len=2^t (optional)
  bool inited = false;
  int digit_pick_count = -1;
  int digit_target_dim = 0;
  Vec digit_target{};

  void clear_cache() {
    p.clear();
    pb.clear();
    inited = false;
    digit_pick_count = -1;
    digit_target_dim = 0;
    digit_target = Vec{};
  }

  template <typename VecLike>
  bool matches_digit_key(const VecLike &target, int picks) const {
    if (!inited || digit_pick_count != picks ||
        digit_target_dim != static_cast<int>(target.size())) {
      return false;
    }
    for (int i = 0; i < digit_target_dim; ++i) {
      if (digit_target.x[(size_t)i] != target[i])
        return false;
    }
    return true;
  }

  template <typename VecLike>
  void reset_for_digit_key(const VecLike &target, int picks) {
    clear_cache();
    digit_pick_count = picks;
    digit_target_dim = static_cast<int>(target.size());
    for (int i = 0; i < digit_target_dim; ++i)
      digit_target.x[(size_t)i] = target[i];
  }
};

// Ensure pow table has level `bit` built (lazy). Uses exact self-Minkowski
// sums. Tube bounds are not applied during pow construction; only when
// composing into cur.
static inline void
ensure_pow_level(PowTable &T, int bit, const SumSet &base, int m_dim,
                 const Eigen::Ref<const Eigen::VectorXi> &b_up, int q,
                 long long lo_const, long long hi_const, SumSet &tmp) {
  if (!T.inited) {
    T.p.clear();
    T.pb.clear();
    T.p.push_back(base);
    T.pb.push_back(Bounds{});
    T.inited = true;
  }

  while ((int)T.p.size() <= bit) {
    const int prev = (int)T.p.size() - 1;
    const int next_len = 1 << (int)T.p.size();
    (void)next_len;

    // No tube when constructing pow itself; use vacuous bounds (only b_up
    // positivity bound will apply).
    Bounds vac;
    vac.vacuous = true;
    vac.lo_active = false;
    vac.hi_active = false;

    merge_sets_into(T.p[(size_t)prev], T.p[(size_t)prev], m_dim, b_up, vac,
                    tmp);
    T.p.push_back(std::move(tmp));
    T.pb.push_back(Bounds{});
  tmp = SumSet{};
  }
}

namespace convolve {

template <typename BoundsVec>
inline bool vec_in_bounds(const Vec &v, int m_dim, const Bounds &bounds,
                          const BoundsVec &ub) {
  for (int k = 0; k < m_dim; ++k) {
    const long long x = (long long)v.x[k];
    if (x < 0 || x > (long long)ub[k])
      return false;
    if (!bounds.vacuous && bounds.lo_active && x < bounds.lo[k])
      return false;
    if (!bounds.vacuous && bounds.hi_active && x > bounds.hi[k])
      return false;
  }
  return true;
}

template <typename BoundsVec>
inline std::pair<int, int> pick_narrow_pivots(const Bounds &bounds,
                                              const BoundsVec &ub, int m_dim) {
  int p0 = 0;
  int p1 = (m_dim > 1) ? 1 : 0;
  long long w0 = std::numeric_limits<long long>::max();
  long long w1 = std::numeric_limits<long long>::max();
  for (int k = 0; k < m_dim; ++k) {
    long long lo = 0;
    long long hi = (long long)ub[k];
    if (!bounds.vacuous && bounds.lo_active)
      lo = std::max(lo, bounds.lo[k]);
    if (!bounds.vacuous && bounds.hi_active)
      hi = std::min(hi, bounds.hi[k]);
    const long long w =
        (hi >= lo) ? (hi - lo) : std::numeric_limits<long long>::max();
    if (w < w0) {
      w1 = w0;
      p1 = p0;
      w0 = w;
      p0 = k;
    } else if (w < w1) {
      w1 = w;
      p1 = k;
    }
  }
  return {p0, p1};
}

template <typename BoundsVec>
inline void self_merge_sparse_bounded_into(const SumSet &in, int m_dim,
                                           const BoundsVec &ub,
                                           const Bounds &bounds, SumSet &out,
                                           std::vector<Vec> &scratch) {
  scratch.clear();
  scratch.reserve(in.size());
  in.for_each([&](const Vec &v) { scratch.push_back(v); });
  if (scratch.empty()) {
    out.clear();
    out.len = in.len * 2;
    out.hashed = false;
    return;
  }

  const auto [pivot0, pivot1] = pick_narrow_pivots(bounds, ub, m_dim);
  std::sort(scratch.begin(), scratch.end(), [pivot0](const Vec &a, const Vec &b) {
    if (a.x[pivot0] != b.x[pivot0])
      return a.x[pivot0] < b.x[pivot0];
    return VecLess{}(a, b);
  });
  scratch.erase(std::unique(scratch.begin(), scratch.end(), VecEq{}),
                scratch.end());

  out.clear();
  out.len = in.len * 2;
  const bool use_hash_out = (scratch.size() > SMALL_THRESHOLD);
  out.hashed = use_hash_out;
  const std::size_t s = scratch.size();
  if (use_hash_out) {
    out.hset.clear();
    out.hset.reserve(std::max<std::size_t>(64, s * 4));
  } else {
    out.vec.reserve(std::min<std::size_t>(s * 8, 2 * SMALL_THRESHOLD));
  }

  auto lower_by_pivot = [&](long long x) {
    return std::lower_bound(scratch.begin(), scratch.end(), x,
                            [pivot0](const Vec &v, long long t) {
                              return (long long)v.x[pivot0] < t;
                            });
  };
  auto upper_by_pivot = [&](long long x) {
    return std::upper_bound(scratch.begin(), scratch.end(), x,
                            [pivot0](long long t, const Vec &v) {
                              return t < (long long)v.x[pivot0];
                            });
  };

  for (std::size_t i = 0; i < scratch.size(); ++i) {
    const Vec &a = scratch[i];

    long long lo_p = 0;
    long long hi_p = (long long)ub[pivot0];
    if (!bounds.vacuous && bounds.lo_active)
      lo_p = std::max(lo_p, bounds.lo[pivot0] - (long long)a.x[pivot0]);
    if (!bounds.vacuous && bounds.hi_active)
      hi_p = std::min(hi_p, bounds.hi[pivot0] - (long long)a.x[pivot0]);
    hi_p = std::min(hi_p, (long long)ub[pivot0] - (long long)a.x[pivot0]);
    if (lo_p > hi_p)
      continue;

    long long lo_p2 = 0;
    long long hi_p2 = (long long)ub[pivot1];
    if (pivot1 != pivot0) {
      if (!bounds.vacuous && bounds.lo_active)
        lo_p2 = std::max(lo_p2, bounds.lo[pivot1] - (long long)a.x[pivot1]);
      if (!bounds.vacuous && bounds.hi_active)
        hi_p2 = std::min(hi_p2, bounds.hi[pivot1] - (long long)a.x[pivot1]);
      hi_p2 = std::min(hi_p2, (long long)ub[pivot1] - (long long)a.x[pivot1]);
      if (lo_p2 > hi_p2)
        continue;
    }

    auto it_lo = lower_by_pivot(lo_p);
    auto it_hi = upper_by_pivot(hi_p);
    std::size_t j0 = (std::size_t)std::distance(scratch.begin(), it_lo);
    std::size_t j1 = (std::size_t)std::distance(scratch.begin(), it_hi);
    if (j1 <= j0)
      continue;

    if (j0 < i)
      j0 = i;
    for (std::size_t j = j0; j < j1; ++j) {
      const Vec &b = scratch[j];
      if (pivot1 != pivot0) {
        const long long sb2 = (long long)b.x[pivot1];
        if (sb2 < lo_p2 || sb2 > hi_p2)
          continue;
      }
      Vec w{};
      bool ok = true;
      for (int k = 0; k < m_dim; ++k) {
        const long long sum = (long long)a.x[k] + (long long)b.x[k];
        if (sum < 0 || sum > (long long)ub[k]) {
          ok = false;
          break;
        }
        if (!bounds.vacuous && bounds.lo_active && sum < bounds.lo[k]) {
          ok = false;
          break;
        }
        if (!bounds.vacuous && bounds.hi_active && sum > bounds.hi[k]) {
          ok = false;
          break;
        }
        w.x[k] = (int)sum;
      }
      if (!ok)
        continue;
      if (use_hash_out)
        out.hset.insert(w);
      else
        out.vec.push_back(w);
    }
  }

  if (use_hash_out) {
    out.vec.clear();
  } else {
    out.sort_unique();
    out.ensure_hashed();
  }
}

} // namespace convolve
