#pragma once

#include "convolve/sumset.h" // Vec / Bounds / MAX_M
#include "model/matrix/packed.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace oracle::discrep::detail {

// Discrepancy-specific search geometry and embedding helpers.
// Generic base construction and generic boolean convolutions live in convolve/.
// This layer keeps the tube bounds, parameter choices, and scatter logic that
// are tied to the discrepancy DP itself.

// --- Parameter choices -------------------------------------------------------

inline int ceil_log_base_6_5(long long K) {
  if (K <= 1)
    return 0;
  const double ell = std::log((double)K) / std::log(6.0 / 5.0);
  return (int)std::ceil(ell - 1e-12);
}

inline long long choose_K_default(int r, int Delta_global) {
  return std::max(1LL, (long long)std::floor(2.0 * (r + 1) *
                                              std::log2(4.0 * (r + 1) *
                                                        double(Delta_global))));
}

inline long long choose_H_up_default(int r, int Delta) {
  constexpr long long C = 6;
  const long long root =
      (long long)std::ceil(std::sqrt((double)std::max(1, r)));
  return C * root * (long long)std::max(1, Delta);
}

inline long long choose_H_block_default(int r, int Delta_local) {
  return choose_H_up_default(r, Delta_local);
}

struct UpMatrixStats {
  std::vector<int> up_row_abs_max;
  int max_col_support = 0;
  bool up_binary = true;
};

template <class MatrixLike>
inline UpMatrixStats analyze_up_matrix(const MatrixLike &A, int r) {
  UpMatrixStats stats;
  stats.up_row_abs_max.assign((size_t)r, A.maxCoeff());

#if defined(DISCREP_USE_BECK_FIALA)
  std::fill(stats.up_row_abs_max.begin(), stats.up_row_abs_max.end(), 0);
  const int C = A.cols();
  for (int j = 0; j < C; ++j) {
    int supp = 0;
    for (int k = 0; k < r; ++k) {
      const int a = A(k, j);
      const int aa = std::abs(a);
      supp += (aa != 0);
      if (aa > stats.up_row_abs_max[(size_t)k])
        stats.up_row_abs_max[(size_t)k] = aa;
      if (aa > 1)
        stats.up_binary = false;
    }
    if (supp > stats.max_col_support)
      stats.max_col_support = supp;
  }
#endif

  return stats;
}

inline std::vector<long long> choose_H_up_per_row(const UpMatrixStats &stats,
                                                   int r) {
  std::vector<long long> H_up((size_t)r, 1);
  for (int k = 0; k < r; ++k)
    H_up[(size_t)k] = choose_H_up_default(r, stats.up_row_abs_max[(size_t)k]);

#if defined(DISCREP_USE_BECK_FIALA)
  if (!stats.up_binary)
    return H_up;
  const long long bf_cap = std::max(1LL, 2LL * stats.max_col_support - 1LL);
  for (std::size_t k = 0; k < H_up.size(); ++k)
    H_up[k] = std::min(H_up[k], bf_cap);
#endif

  return H_up;
}

template <class MatrixLike>
inline std::vector<long long> choose_H_block_per_block(const MatrixLike &A,
                                                       int r) {
  const int n = static_cast<int>(A.num_blocks());
  std::vector<long long> H_block((size_t)n, 1);
  for (int e = 0; e < n; ++e) {
    const int Delta_local =
        std::max(1, A.block_maxCoeff(static_cast<std::size_t>(e)));
    H_block[(size_t)e] = choose_H_block_default(r, Delta_local);
  }
  return H_block;
}

// --- Tube bounds -------------------------------------------------------------

// Tube box for augmented system, with monotone clamping:
//   up dims:   [0 .. b_up]
//   count dims:[0 .. b_down]
inline Bounds make_tube_box(const Eigen::Ref<const Eigen::VectorXi> &b_aug,
                            const Eigen::Ref<const Eigen::VectorXi> &b_down,
                            int r, int m_dim, int i, int ell,
                            const std::vector<long long> &H_up,
                            const std::vector<long long> &H_block) {
  Bounds B;
  B.vacuous = false;
  B.lo_active = true;
  B.hi_active = true;

  const long double scale = std::ldexp(1.0L, i - ell); // 2^(i-ell)

  for (int k = 0; k < m_dim; ++k) {
    const long long Hk =
        (k < r) ? H_up[(size_t)k] : H_block[(size_t)(k - r)];
    const long long rad = 4LL * Hk;

    const long double c = scale * (long double)b_aug[k];
    long double lo = std::floor(c - (long double)rad);
    long double hi = std::ceil(c + (long double)rad);

    lo = std::max(lo, 0.0L);
    hi = std::min(hi, (long double)b_aug[k]);

    if (k >= r) {
      const int e = k - r;
      hi = std::min(hi, (long double)b_down[e]);
      // At layer i, a count coordinate cannot exceed the number of base picks
      // representable by self-sum depth 2^i.
      const long long len_cap =
          (i >= 62) ? std::numeric_limits<long long>::max() : (1LL << i);
      hi = std::min(hi, (long double)len_cap);
    }

    const long double LLMIN =
        (long double)std::numeric_limits<long long>::min();
    const long double LLMAX =
        (long double)std::numeric_limits<long long>::max();
    const long long lok = (long long)std::max(LLMIN, std::min(LLMAX, lo));
    const long long hik = (long long)std::max(LLMIN, std::min(LLMAX, hi));

    B.lo[k] = lok;
    B.hi[k] = hik;
  }
  return B;
}

inline bool box_empty(const Bounds &B, int m_dim) {
  if (B.vacuous)
    return false;
  if (!B.lo_active || !B.hi_active)
    return false;
  for (int k = 0; k < m_dim; ++k)
    if (B.lo[k] > B.hi[k])
      return true;
  return false;
}

// --- Mixed-radix embedding ---------------------------------------------------

// Mixed radix embedding (with padding): index addition corresponds to
// coordinate-wise addition for sums.
struct Embed {
  std::size_t N = 0;
  std::vector<std::size_t> radix;
  std::vector<std::size_t> stride;
  std::vector<long long> lo;
  std::vector<long long> span;
};

inline bool build_embed(const Bounds &box, int m_dim, Embed &E) {
  E.N = 1;
  E.radix.assign((size_t)m_dim, 0);
  E.stride.assign((size_t)m_dim, 0);
  E.lo.assign((size_t)m_dim, 0);
  E.span.assign((size_t)m_dim, 0);

  for (int k = 0; k < m_dim; ++k) {
    const long long span_ll = box.hi[k] - box.lo[k] + 1;
    if (span_ll <= 0)
      return false;
    const std::size_t span = (std::size_t)span_ll;
    const std::size_t radk = (span == 1) ? 1 : (2 * span);

    E.radix[(size_t)k] = radk;
    E.stride[(size_t)k] = E.N;
    E.lo[(size_t)k] = box.lo[k];
    E.span[(size_t)k] = span_ll;

    if (radk != 0 && E.N > std::numeric_limits<std::size_t>::max() / radk)
      return false;
    E.N *= radk;
  }
  return true;
}

inline bool encode_in_box(const Embed &E, const Vec &v, int m_dim,
                          std::size_t &idx_out) {
  std::size_t idx = 0;
  for (int k = 0; k < m_dim; ++k) {
    const long long d = (long long)v.x[k] - E.lo[(size_t)k];
    if (d < 0)
      return false;
    if (d >= E.span[(size_t)k])
      return false;
    idx += (std::size_t)d * E.stride[(size_t)k];
  }
  idx_out = idx;
  return true;
}

inline constexpr std::size_t INVALID_EMBED_IDX =
    std::numeric_limits<std::size_t>::max();

// --- Scatter mapping ---------------------------------------------------------

inline void build_scatter_map_to_next(const Embed &Prev, const Bounds &box_cur,
                                      const Embed &Cur, int m_dim,
                                      std::vector<std::size_t> &map_out) {
  map_out.assign(Prev.N, INVALID_EMBED_IDX);

  std::vector<std::size_t> digits((size_t)m_dim, 0);
  for (std::size_t idx = 0; idx < Prev.N; ++idx) {
    std::size_t idx_cur = 0;
    bool ok = true;

    for (int k = 0; k < m_dim; ++k) {
      const long long coord =
          2LL * Prev.lo[(size_t)k] + (long long)digits[(size_t)k];
      if (coord < box_cur.lo[k] || coord > box_cur.hi[k]) {
        ok = false;
        break;
      }

      const long long dcur = coord - Cur.lo[(size_t)k];
      if (dcur < 0 || dcur >= Cur.span[(size_t)k]) {
        ok = false;
        break;
      }
      idx_cur += (std::size_t)dcur * Cur.stride[(size_t)k];
    }
    if (ok)
      map_out[idx] = idx_cur;

    for (int k = 0; k < m_dim; ++k) {
      const std::size_t radk = Prev.radix[(size_t)k];
      const std::size_t d = digits[(size_t)k] + 1;
      if (d < radk) {
        digits[(size_t)k] = d;
        break;
      }
      digits[(size_t)k] = 0;
    }
  }
}

inline bool scatter_conv_to_next_mapped_or_any(
    const std::vector<std::uint8_t> &g, const std::vector<std::size_t> &map,
    std::vector<std::uint8_t> &fcur) {
  bool any = false;
  const std::size_t n = std::min(g.size(), map.size());
  for (std::size_t idx = 0; idx < n; ++idx) {
    if (!g[idx])
      continue;
    const std::size_t idx_cur = map[idx];
    if (idx_cur == INVALID_EMBED_IDX)
      continue;
    fcur[idx_cur] = 1;
    any = true;
  }
  return any;
}

inline void scatter_conv_to_next(const std::vector<std::uint8_t> &g,
                                 const Embed &Prev, const Bounds &box_cur,
                                 const Embed &Cur, int m_dim,
                                 std::vector<std::uint8_t> &fcur) {
  std::fill(fcur.begin(), fcur.end(), (std::uint8_t)0);
  std::vector<std::size_t> digits((size_t)m_dim, 0);
  for (std::size_t idx = 0; idx < Prev.N; ++idx) {
    if (g[idx]) {
      std::size_t idx_cur = 0;
      bool ok = true;
      for (int k = 0; k < m_dim; ++k) {
        const long long coord =
            2LL * Prev.lo[(size_t)k] + (long long)digits[(size_t)k];
        if (coord < box_cur.lo[k] || coord > box_cur.hi[k]) {
          ok = false;
          break;
        }
        const long long dcur = coord - Cur.lo[(size_t)k];
        if (dcur < 0 || dcur >= Cur.span[(size_t)k]) {
          ok = false;
          break;
        }
        idx_cur += (std::size_t)dcur * Cur.stride[(size_t)k];
      }
      if (ok)
        fcur[idx_cur] = 1;
    }
    for (int k = 0; k < m_dim; ++k) {
      const std::size_t radk = Prev.radix[(size_t)k];
      const std::size_t d = digits[(size_t)k] + 1;
      if (d < radk) {
        digits[(size_t)k] = d;
        break;
      }
      digits[(size_t)k] = 0;
    }
  }
}

} // namespace oracle::discrep::detail
