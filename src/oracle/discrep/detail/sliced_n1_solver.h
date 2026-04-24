#pragma once

#include "bench.h"
#include "tube_embed.h"
#include "profile_helpers.h"

#include "model/matrix/model.h"
#include "model/matrix/packed.h"

#include "convolve/block_cols.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <fftw3.h>

namespace oracle::discrep::detail {

// r2c/c2r 1D FFT wrapper (power-of-two padded).
struct FFT1D {
  std::size_t N = 0;     // logical time length
  std::size_t n_fft = 0; // FFT length (power of two)
  double *t = nullptr;   // real time buffer
  fftw_complex *f = nullptr;
  fftw_plan plan_fwd = nullptr;
  fftw_plan plan_inv = nullptr;

  static inline std::size_t next_pow2(std::size_t n) {
    if (n <= 2)
      return 2;
    std::size_t m = 1;
    while (m < n)
      m <<= 1;
    return m;
  }

  explicit FFT1D(std::size_t N_, unsigned flags) : N(N_) {
    n_fft = next_pow2(N);
    t = (double *)fftw_alloc_real(n_fft);
    const std::size_t nf = (n_fft / 2) + 1;
    f = (fftw_complex *)fftw_alloc_complex(nf);
    if (!t || !f)
      throw std::bad_alloc();
    plan_fwd = fftw_plan_dft_r2c_1d((int)n_fft, t, f, flags);
    plan_inv = fftw_plan_dft_c2r_1d((int)n_fft, f, t, flags);
    if (!plan_fwd || !plan_inv)
      throw std::runtime_error("FFTW plan failed");
  }

  FFT1D(const FFT1D &) = delete;
  FFT1D &operator=(const FFT1D &) = delete;

  ~FFT1D() {
    if (plan_fwd)
      fftw_destroy_plan(plan_fwd);
    if (plan_inv)
      fftw_destroy_plan(plan_inv);
    if (f)
      fftw_free(f);
    if (t)
      fftw_free(t);
  }
};

struct SlicedCache {
  std::unique_ptr<FFT1D> fft;
  std::size_t nf = 0; // complex bins
  int C_slices = 0;
  std::vector<std::vector<std::complex<double>>> Fhat;
  std::vector<std::vector<std::complex<double>>> Ghat;
  std::vector<std::uint8_t> g_time;
  std::vector<std::uint8_t> active_prev;
  std::vector<std::uint8_t> active_cur;
  std::vector<std::uint8_t> touched_c;
  std::vector<int> active_idx;
  std::vector<int> touched_idx;

  // Batched slice FFT workspace/plans (howmany == C_slices)
  int many_how = 0;
  std::size_t many_n_fft = 0;
  std::size_t many_nf = 0;
  double *many_t = nullptr;       // [many_how][many_n_fft]
  fftw_complex *many_f = nullptr; // [many_how][many_nf]
  fftw_plan many_fwd = nullptr;   // many r2c
  fftw_plan many_inv = nullptr;   // many c2r

  ~SlicedCache() { clear_many(); }

  void clear_many() {
    if (many_fwd)
      fftw_destroy_plan(many_fwd);
    if (many_inv)
      fftw_destroy_plan(many_inv);
    if (many_f)
      fftw_free(many_f);
    if (many_t)
      fftw_free(many_t);
    many_fwd = nullptr;
    many_inv = nullptr;
    many_f = nullptr;
    many_t = nullptr;
    many_how = 0;
    many_n_fft = 0;
    many_nf = 0;
  }

  void ensure_many(std::size_t n_fft, std::size_t nf_in, int how,
                   unsigned flags) {
    if (how <= 0)
      return;
    if (many_t && many_f && many_fwd && many_inv && many_how == how &&
        many_n_fft == n_fft && many_nf == nf_in) {
      return;
    }

    clear_many();

    many_how = how;
    many_n_fft = n_fft;
    many_nf = nf_in;

    many_t = (double *)fftw_alloc_real((std::size_t)many_how * many_n_fft);
    many_f =
        (fftw_complex *)fftw_alloc_complex((std::size_t)many_how * many_nf);
    if (!many_t || !many_f)
      throw std::bad_alloc();

    {
      int rank = 1;
      int n[1] = {(int)many_n_fft};
      int inembed[1] = {(int)many_n_fft};
      int onembed[1] = {(int)many_nf};
      int istride = 1, ostride = 1;
      int idist = (int)many_n_fft, odist = (int)many_nf;
      many_fwd = fftw_plan_many_dft_r2c(rank, n, many_how, many_t, inembed,
                                        istride, idist, many_f, onembed,
                                        ostride, odist, flags);
      many_inv = fftw_plan_many_dft_c2r(rank, n, many_how, many_f, onembed,
                                        ostride, odist, many_t, inembed,
                                        istride, idist, flags);
    }
    if (!many_fwd || !many_inv)
      throw std::runtime_error("FFTW many-plan failed");
  }
};

// Profile counters
#if defined(DISCREP_FFT_PROFILE)
struct Profile {
  long long t_fwd_us = 0;
  long long t_inv_us = 0;
  long long t_combine_us = 0;
  long long t_scatter_us = 0;
  int max_c_hi = 0;
};
#endif

inline Bounds make_tube_box_up(const Eigen::Ref<const Eigen::VectorXi> &b_up,
                               const std::vector<long long> &H_up, int r,
                               int i, int ell) {
  Bounds B;
  B.vacuous = false;
  B.lo_active = true;
  B.hi_active = true;
  const long double scale = std::ldexp(1.0L, i - ell);
  for (int k = 0; k < r; ++k) {
    const long long rad = 4LL * H_up[(size_t)k];
    const long double c = scale * (long double)b_up[k];
    long double lo = std::floor(c - (long double)rad);
    long double hi = std::ceil(c + (long double)rad);
    // monotone clamp
    lo = std::max(lo, 0.0L);
    hi = std::min(hi, (long double)b_up[k]);
    B.lo[k] = (long long)lo;
    B.hi[k] = (long long)hi;
  }
  return B;
}

inline void collect_active_slices(const std::vector<std::uint8_t> &active_prev,
                                  int c_lo, int c_hi,
                                  std::vector<int> &active_idx) {
  active_idx.clear();
  for (int c = c_lo; c <= c_hi; ++c) {
    if (active_prev[(size_t)c])
      active_idx.push_back(c);
  }
}

inline void reset_touched_slices(SlicedCache &cache) {
  cache.touched_idx.clear();
  std::fill(cache.touched_c.begin(), cache.touched_c.end(), std::uint8_t{0});
}

inline void touch_spectrum_slice(SlicedCache &cache, int c) {
  if (cache.touched_c[(size_t)c])
    return;
  cache.touched_c[(size_t)c] = 1;
  cache.touched_idx.push_back(c);
  std::fill(cache.Ghat[(size_t)c].begin(), cache.Ghat[(size_t)c].end(),
            std::complex<double>(0.0, 0.0));
}

inline void combine_active_count_spectra(SlicedCache &cache, int c_lo,
                                         int c_hi) {
  reset_touched_slices(cache);

  const std::size_t active_count = cache.active_idx.size();
  const std::size_t dense_budget = (std::size_t)(c_hi - c_lo + 1);
  const bool sparse_count_conv = (active_count * active_count <= dense_budget * 2);

  if (sparse_count_conv) {
    for (int a : cache.active_idx) {
      for (int b : cache.active_idx) {
        const int c = a + b;
        if (c < c_lo || c > c_hi)
          continue;
        touch_spectrum_slice(cache, c);
        for (std::size_t k = 0; k < cache.nf; ++k)
          cache.Ghat[(size_t)c][k] +=
              cache.Fhat[(size_t)a][k] * cache.Fhat[(size_t)b][k];
      }
    }
    return;
  }

  for (int c = c_lo; c <= c_hi; ++c) {
    const int a0 = std::max(c_lo, c - c_hi);
    const int a1 = std::min(c_hi, c - c_lo);
    bool touched = false;
    for (std::size_t k = 0; k < cache.nf; ++k) {
      std::complex<double> acc(0.0, 0.0);
      for (int a = a0; a <= a1; ++a) {
        if (!cache.active_prev[(size_t)a])
          continue;
        const int b = c - a;
        if (b < c_lo || b > c_hi || !cache.active_prev[(size_t)b])
          continue;
        acc += cache.Fhat[(size_t)a][k] * cache.Fhat[(size_t)b][k];
      }
      cache.Ghat[(size_t)c][k] = acc;
      touched = touched || (acc.real() != 0.0 || acc.imag() != 0.0);
    }
    if (touched) {
      cache.touched_c[(size_t)c] = 1;
      cache.touched_idx.push_back(c);
    }
  }
}

// n==1 only: DP indexed by count c=0..q, each slice is a bitmap over up-grid.
// Empty-slice skip: only convolve slices up to c_hi=min(q, 2^i).
inline bool solve_impl(const LPModel<PackedA> &m,
                       const Eigen::Ref<const Eigen::VectorXi> &b_up,
                       const Eigen::Ref<const Eigen::VectorXi> &b_down, int r,
                       long long q, const std::vector<long long> &H_up,
                       int ell) {
#if defined(DISCREP_PROFILE)
  const auto t_total0 = oracle::discrep::profile::clock_t::now();
  oracle::discrep::profile::SlicedProf prof;
  auto emit_profile = [&]() {
    prof.t_total = oracle::discrep::profile::ns_since(t_total0);
    oracle::discrep::profile::emit_sliced_profile(prof);
  };
#endif
  auto reject_solver_run = [&](std::string_view reason) {
    bench_detail::reject_current_result(reason);
    spdlog::info(
        "discrepancy::solve timing invalidated: rejected solver result ({})",
        reason);
#if defined(DISCREP_PROFILE)
    emit_profile();
#endif
    return false;
  };
  assert(b_down.size() == 1);
  const PackedA &A = m.A;

  Bounds box0_up = make_tube_box_up(b_up, H_up, r, 0, ell);
  if (box_empty(box0_up, r))
    return false;
  Embed E0;
  if (!build_embed(box0_up, r, E0))
    return reject_solver_run("embed_build_failed");

  constexpr std::size_t U_CAP = 20'000'000;
  if (E0.N > U_CAP)
    return reject_solver_run("embed_size_cap_sliced_base");

  const int q_i = (int)q;
  const int C_slices = q_i + 1;

  std::vector<std::vector<std::uint8_t>> fprev(
      (size_t)C_slices, std::vector<std::uint8_t>(E0.N, 0));
  std::vector<std::vector<std::uint8_t>> fcur;

#if defined(DISCREP_PROFILE)
  const auto t_b0 = oracle::discrep::profile::clock_t::now();
#endif
  auto set_bit = [&](int c, const Vec &y_up) {
    std::size_t idx = 0;
    if (encode_in_box(E0, y_up, r, idx))
      fprev[(size_t)c][idx] = 1;
  };

  set_bit(0, Vec{});

  // base columns: block range(0) => count +1; uncovered => count +0
  const int C = A.cols();
  std::vector<std::size_t> idx_buf;
  std::vector<std::uint8_t> active_prev((size_t)C_slices, std::uint8_t{0});
  auto commit_unique = [&](int c) {
    const bool had_any = !idx_buf.empty();
    std::sort(idx_buf.begin(), idx_buf.end());
    idx_buf.erase(std::unique(idx_buf.begin(), idx_buf.end()), idx_buf.end());
    for (std::size_t idx : idx_buf)
      fprev[(size_t)c][idx] = 1;
    if (had_any)
      active_prev[(size_t)c] = 1;
    idx_buf.clear();
  };
  convolve::for_each_block_and_uncovered_col(
      A, /*n=*/1,
      [&](int, int j) {
        Vec y{};
        for (int k = 0; k < r; ++k)
          y.x[k] = A(k, j);
        std::size_t idx = 0;
        if (encode_in_box(E0, y, r, idx))
          idx_buf.push_back(idx);
      },
      [&](int) { commit_unique(1); },
      [&](int j) {
        Vec y{};
        for (int k = 0; k < r; ++k)
          y.x[k] = A(k, j);
        std::size_t idx = 0;
        if (encode_in_box(E0, y, r, idx))
          idx_buf.push_back(idx);
      });
  idx_buf.reserve((size_t)C);
  commit_unique(0);
  active_prev[0] = 1;
#if defined(DISCREP_PROFILE)
  prof.t_base += oracle::discrep::profile::ns_since(t_b0);
#endif

  Vec target_up{};
  for (int k = 0; k < r; ++k)
    target_up.x[k] = b_up[k];

  {
    std::size_t tidx = 0;
    if (encode_in_box(E0, target_up, r, tidx) && fprev[(size_t)q_i][tidx])
      return true;
  }
  if (ell == 0) {
#if defined(DISCREP_PROFILE)
    emit_profile();
#endif
    return false;
  }

  Embed PrevUp = std::move(E0);

  static thread_local SlicedCache cache;
#if defined(DISCREP_FFT_PROFILE)
  static thread_local Profile prof;
#endif

  auto ensure_fft = [&]() {
    constexpr unsigned flags = FFTW_ESTIMATE;
    if (!cache.fft || cache.fft->N != PrevUp.N) {
      cache.fft.reset(new FFT1D(PrevUp.N, flags));
      cache.nf = (std::size_t(cache.fft->n_fft) / 2) + 1;
    }
    if (cache.C_slices != C_slices || cache.Fhat.size() != (size_t)C_slices ||
        cache.Fhat.empty() || cache.Fhat[0].size() != cache.nf) {
      cache.C_slices = C_slices;
      cache.Fhat.assign((size_t)C_slices,
                        std::vector<std::complex<double>>(cache.nf));
      cache.Ghat.assign((size_t)C_slices,
                        std::vector<std::complex<double>>(cache.nf));
      cache.active_prev.assign((size_t)C_slices, std::uint8_t{0});
      cache.active_cur.assign((size_t)C_slices, std::uint8_t{0});
      cache.touched_c.assign((size_t)C_slices, std::uint8_t{0});
      cache.active_idx.clear();
      cache.touched_idx.clear();
    }
    if (cache.g_time.size() != PrevUp.N)
      cache.g_time.assign(PrevUp.N, 0);
    cache.ensure_many(cache.fft->n_fft, cache.nf, C_slices, flags);
  };
  ensure_fft();
  std::vector<std::size_t> scatter_map;
  cache.active_prev = active_prev;

#if defined(DISCREP_PROFILE)
  const auto t_l0 = oracle::discrep::profile::clock_t::now();
#endif
  for (int i = 1; i <= ell; ++i) {
    Bounds box_cur_up = make_tube_box_up(b_up, H_up, r, i, ell);
    if (box_empty(box_cur_up, r))
      return false;
    Embed CurUp;
    if (!build_embed(box_cur_up, r, CurUp))
      return reject_solver_run("embed_build_failed_layer");
    if (PrevUp.N > U_CAP || CurUp.N > U_CAP)
      return reject_solver_run("embed_size_cap_sliced_layer");

    // Empty slice skip: base has counts 0/1 so after i doublings max count is
    // 2^i.
    const int c_lo = 0;
    const int c_hi = std::min(q_i, 1 << std::min(i, 30));
#if defined(DISCREP_FFT_PROFILE)
    prof.max_c_hi = std::max(prof.max_c_hi, c_hi);
#endif

    ensure_fft();
    build_scatter_map_to_next(PrevUp, box_cur_up, CurUp, r, scatter_map);

    collect_active_slices(cache.active_prev, c_lo, c_hi, cache.active_idx);
#if defined(DISCREP_PROFILE)
    prof.active_slices_max =
        std::max(prof.active_slices_max, (int)cache.active_idx.size());
#endif
    if (cache.active_idx.empty())
      return false;

    // Forward spectra: batched R2C over all count slices.
    const std::size_t n_fft = cache.fft->n_fft;
#if defined(DISCREP_PROFILE)
    const auto t_f0 = oracle::discrep::profile::clock_t::now();
#endif
    for (int c = 0; c < C_slices; ++c) {
      double *row = cache.many_t + (std::size_t)c * n_fft;
      if (c <= c_hi && cache.active_prev[(size_t)c]) {
        for (std::size_t t = 0; t < PrevUp.N; ++t)
          row[t] = fprev[(size_t)c][t] ? 1.0 : 0.0;
      } else {
        std::memset(row, 0, PrevUp.N * sizeof(double));
      }
      if (n_fft > PrevUp.N) {
        std::memset(row + PrevUp.N, 0, (n_fft - PrevUp.N) * sizeof(double));
      }
    }
    fftw_execute(cache.many_fwd);
    for (int c : cache.active_idx) {
      const fftw_complex *frow = cache.many_f + (std::size_t)c * cache.nf;
      for (std::size_t k = 0; k < cache.nf; ++k) {
        cache.Fhat[(size_t)c][k] =
            std::complex<double>(frow[k][0], frow[k][1]);
      }
    }
#if defined(DISCREP_PROFILE)
    prof.t_fwd += oracle::discrep::profile::ns_since(t_f0);
#endif

#if defined(DISCREP_PROFILE)
    const auto t_c0 = oracle::discrep::profile::clock_t::now();
#endif
    combine_active_count_spectra(cache, c_lo, c_hi);
#if defined(DISCREP_PROFILE)
    prof.t_combine += oracle::discrep::profile::ns_since(t_c0);
    prof.touched_slices_max =
        std::max(prof.touched_slices_max, (int)cache.touched_idx.size());
#endif

    if (fcur.size() != (size_t)C_slices || fcur.empty() ||
        fcur[0].size() != CurUp.N) {
      fcur.assign((size_t)C_slices, std::vector<std::uint8_t>(CurUp.N, 0));
    } else {
      for (auto &slice : fcur)
        std::fill(slice.begin(), slice.end(), std::uint8_t{0});
    }
    std::fill(cache.active_cur.begin(), cache.active_cur.end(), std::uint8_t{0});

    // Inverse spectra: batched C2R over all count slices, then threshold
    // touched ones. (IFFT is unnormalized by FFTW.)
    const double thr = 0.5 * (double)cache.fft->n_fft;
#if defined(DISCREP_PROFILE)
    const auto t_i0 = oracle::discrep::profile::clock_t::now();
#endif
    for (int c = 0; c < C_slices; ++c) {
      fftw_complex *frow = cache.many_f + (std::size_t)c * cache.nf;
      if (c <= c_hi && cache.touched_c[(size_t)c]) {
        for (std::size_t k = 0; k < cache.nf; ++k) {
          frow[k][0] = cache.Ghat[(size_t)c][k].real();
          frow[k][1] = cache.Ghat[(size_t)c][k].imag();
        }
      } else {
        std::memset(frow, 0, cache.nf * sizeof(fftw_complex));
      }
    }
    fftw_execute(cache.many_inv);
#if defined(DISCREP_PROFILE)
    prof.t_inv += oracle::discrep::profile::ns_since(t_i0);
    const auto t_s0 = oracle::discrep::profile::clock_t::now();
#endif
    for (int c : cache.touched_idx) {
      const double *row = cache.many_t + (std::size_t)c * n_fft;
      for (std::size_t t = 0; t < PrevUp.N; ++t)
        cache.g_time[t] = (row[t] > thr) ? std::uint8_t{1} : std::uint8_t{0};
      cache.active_cur[(size_t)c] =
          scatter_conv_to_next_mapped_or_any(cache.g_time, scatter_map,
                                             fcur[(size_t)c])
              ? std::uint8_t{1}
              : std::uint8_t{0};
    }
#if defined(DISCREP_PROFILE)
    prof.t_scatter += oracle::discrep::profile::ns_since(t_s0);
    prof.layers++;
#endif

    if (i == ell) {
      std::size_t tidx = 0;
      if (encode_in_box(CurUp, target_up, r, tidx) && fcur[(size_t)q_i][tidx]) {
#if defined(DISCREP_PROFILE)
        prof.t_layers += oracle::discrep::profile::ns_since(t_l0);
        emit_profile();
#endif
        return true;
      }
    }

    fprev.swap(fcur);
    cache.active_prev.swap(cache.active_cur);
    PrevUp = std::move(CurUp);
  }
#if defined(DISCREP_PROFILE)
  prof.t_layers += oracle::discrep::profile::ns_since(t_l0);
  emit_profile();
#endif

  return false;
}

// Warm up FFT plans & buffers for a given instance (n==1 fast path).
inline void warmup_impl(const LPModel<PackedA> &m) {
  const PackedA &A = m.A;
  const int n = A.num_blocks();
  if (n != 1)
    return;
  if (m.b.size() < n)
    return;
  const int r = (int)m.b.size() - n;
  const Eigen::Ref<const Eigen::VectorXi> b_up(m.b.head(r));
  const Eigen::Ref<const Eigen::VectorXi> b_down(m.b.tail(n));
  const long long q = (long long)b_down.sum();
  if (q <= 0 || q > 128)
    return;
  const UpMatrixStats up_stats = analyze_up_matrix(A, r);
  const std::vector<long long> H_up = choose_H_up_per_row(up_stats, r);
  const int Delta_global = m.A.maxCoeff();
  const long long K = choose_K_default(r, Delta_global);
  const int ell = ceil_log_base_6_5(K);

  // One cheap attempt to force plan creation.
  (void)solve_impl(m, b_up, b_down, r, q, H_up, ell);
}

namespace sliced_n1 {

template <class Prepared>
inline bool can_use(const Prepared &ctx, const LPModel<PackedA> *packed_model) {
  return packed_model != nullptr && ctx.n == 1 && ctx.q <= 128;
}

template <class Prepared>
inline bool solve(const Prepared &ctx, const LPModel<PackedA> &m) {
  return solve_impl(m, ctx.b_up, ctx.b_down, ctx.r, ctx.q, ctx.H_up, ctx.ell);
}

inline void warmup(const LPModel<PackedA> &m) { warmup_impl(m); }

} // namespace sliced_n1

} // namespace oracle::discrep::detail
