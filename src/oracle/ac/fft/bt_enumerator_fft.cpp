#include <Eigen/Core>
#include <fftw3.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "convolve/fft_bool_r2c.h"
#include "convolve/sumset.h"
#include "model/matrix/packed.h"
#include "oracle/ac/common/profile.h"
#include "oracle/ac/naive/bt_enumerator.h"

// #define DBG_D_BT 1     // high-level enter/exit + shapes
// #define DBG_D_CAPS 1   // show repCap and caps
// #define DBG_D_COLS 1   // show usable column filtering summary
// #define DBG_D_LEVELS 1 // per-level sizes and a few samples
// #define DBG_D_S0 1     // s==0 path decisions

namespace {
using Eigen::ArrayXi;
using Eigen::MatrixXi;
using Eigen::Ref;
using Eigen::VectorXi;

#if (DBG_D_BT || DBG_D_CAPS || DBG_D_COLS || DBG_D_LEVELS || DBG_D_S0)
static inline std::string vec_str(const VectorXi &v) {
  std::ostringstream os;
  os << "[" << v.transpose() << "]";
  return os.str();
}
static inline std::string mat_shape_str(const MatrixXi &A) {
  std::ostringstream os;
  os << "(" << A.rows() << "x" << A.cols() << ")";
  return os.str();
}
static inline void dump_mat_head(const char *name, const MatrixXi &A,
                                 int rows = 6, int cols = 8) {
  std::cerr << name << " " << mat_shape_str(A) << "\n";
  const int r = std::min(rows, (int)A.rows());
  const int c = std::min(cols, (int)A.cols());
  for (int i = 0; i < r; ++i) {
    std::cerr << "  ";
    for (int j = 0; j < c; ++j) {
      std::cerr << A(i, j) << (j + 1 == c ? "" : " ");
    }
    if (c < A.cols())
      std::cerr << " ...";
    std::cerr << "\n";
  }
  if (r < A.rows())
    std::cerr << "  ...\n";
}
static inline void dump_vec_labeled(const char *name, const VectorXi &v) {
  std::cerr << name << " (n=" << v.size() << ") = " << vec_str(v) << "\n";
}
#endif

static thread_local std::unique_ptr<FFTBoolConvolver> g_fft_conv;
static thread_local std::size_t g_fft_conv_N = 0;

inline std::size_t next_pow2_local(std::size_t n) {
  if (n <= 2)
    return 2;
  std::size_t m = 1;
  while (m < n)
    m <<= 1;
  return m;
}
} // namespace

// Polynomial length N beyond which we fall
// back. N=2^20 means ~1M complex numbers in FFT buffers.
static constexpr size_t FFT_N_THRESHOLD = (1u << 20);

namespace {

struct BlockContext {
  int k = 0;
  int r = 0;
  int c0 = 0;
  int c1 = 0;
  int t = 0;
};

inline BlockContext make_block_context(const PackedA &A, int k) {
  BlockContext ctx;
  ctx.k = k;
  ctx.r = static_cast<int>(A.rows());
  ctx.c0 = static_cast<int>(A.get_block_offset(static_cast<std::size_t>(k)));
  ctx.c1 = static_cast<int>(A.get_block_offset(static_cast<std::size_t>(k + 1)));
  ctx.t = ctx.c1 - ctx.c0;
  return ctx;
}

inline VectorXi compute_caps_for_block(const PackedA &A, const BlockContext &ctx,
                                       const Eigen::Ref<const Eigen::VectorXi> target, int s) {
  VectorXi caps(ctx.r);
  const int Delta_local = A.block_maxCoeff(static_cast<std::size_t>(ctx.k));
  for (int i = 0; i < ctx.r; ++i) {
    long long tmp = 1LL * s * Delta_local;
    if (tmp < 0)
      tmp = 0;
    if (tmp > std::numeric_limits<int>::max())
      tmp = std::numeric_limits<int>::max();
    caps[i] = std::min(target[i], static_cast<int>(tmp));
  }
  return caps;
}

inline bool build_usable_columns(const PackedA &A, const BlockContext &ctx,
                                 const VectorXi &caps, std::vector<int> &usable_cols,
                                 VectorXi &row_max_u) {
  row_max_u = VectorXi::Zero(ctx.r);
  usable_cols.clear();
  usable_cols.reserve(static_cast<std::size_t>(std::max(0, ctx.t)));

  for (int j = ctx.c0; j < ctx.c1; ++j) {
    bool ok = true;
    for (int i = 0; i < ctx.r; ++i) {
      const int v = A(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
      if (v > caps[i]) {
        ok = false;
        break;
      }
    }
    if (!ok)
      continue;
    usable_cols.push_back(j);
    for (int i = 0; i < ctx.r; ++i) {
      row_max_u[i] = std::max(row_max_u[i],
                              A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)));
    }
  }
  return !usable_cols.empty();
}

inline bool build_embedding(const VectorXi &row_max_u, int r, int s,
                            std::vector<size_t> &base, std::vector<size_t> &stride,
                            size_t &N) {
  base.assign(static_cast<std::size_t>(r), 0);
  stride.assign(static_cast<std::size_t>(r), 0);
  N = 1;
  for (int i = 0; i < r; ++i) {
    long long bi = 1LL * s * row_max_u[i] + 1LL;
    if (bi <= 1)
      bi = 2;
    base[static_cast<std::size_t>(i)] = static_cast<size_t>(bi);
    stride[static_cast<std::size_t>(i)] = N;
    if (N > std::numeric_limits<size_t>::max() / base[static_cast<std::size_t>(i)])
      return false;
    N *= base[static_cast<std::size_t>(i)];
    if (N > FFT_N_THRESHOLD)
      return false;
  }
  return true;
}

inline SumSet decode_sparse_s1(const PackedA &A, const std::vector<int> &usable_cols, int r) {
  SumSet out;
  out.len = 0;
  out.hashed = false;
  out.vec.reserve(usable_cols.size());
  for (int j : usable_cols) {
    Vec v{};
    for (int i = 0; i < r; ++i)
      v.x[static_cast<std::size_t>(i)] =
          A(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
    out.vec.push_back(v);
  }
  out.sort_unique();
  out.ensure_hashed();
  return out;
}

inline void run_fft_power_exact_s(std::vector<uint8_t> &acc, const std::vector<uint8_t> &P,
                                  size_t N, int s) {
  std::vector<uint8_t> base = P;
  std::vector<uint8_t> tmp;
  acc.assign(N, 0);
  acc[0] = 1;

  if (!g_fft_conv || g_fft_conv_N != N) {
    g_fft_conv = std::make_unique<FFTBoolConvolver>(N);
    g_fft_conv_N = N;
  }

  int e = s;
  while (e > 0) {
    if (e & 1) {
      g_fft_conv->mul_bool(tmp, acc, base);
      acc.swap(tmp);
    }
    e >>= 1;
    if (!e)
      continue;
    g_fft_conv->mul_bool(tmp, base, base);
    base.swap(tmp);
  }
}

inline SumSet decode_fft_result(const std::vector<uint8_t> &acc, size_t N, int r,
                                const std::vector<size_t> &base,
                                const std::vector<size_t> &stride,
                                const VectorXi &caps) {
  SumSet out;
  out.len = 0;
  out.hashed = false;
  out.vec.reserve(64);

  for (size_t idx = 0; idx < N; ++idx) {
    if (!acc[idx])
      continue;
    size_t rem = idx;
    bool ok = true;
    Vec v{};
    for (int i = r - 1; i >= 0; --i) {
      const size_t st = stride[static_cast<std::size_t>(i)];
      size_t xi = rem / st;
      xi %= base[static_cast<std::size_t>(i)];
      const int xii = static_cast<int>(xi);
      if (xii > caps[i]) {
        ok = false;
        break;
      }
      v.x[static_cast<std::size_t>(i)] = xii;
      rem -= xi * st;
    }
    if (ok)
      out.vec.push_back(v);
  }

  out.sort_unique();
  out.ensure_hashed();
  return out;
}

} // namespace

SumSet compute_base_sumset_for_block_fft(const PackedA &A, int k,
                                         Eigen::Ref<const Eigen::VectorXi> target,
                                         int s, PowTable & /*pow_table*/) {
  SumSet out;
  out.len = 0;
  out.hashed = false;

  const BlockContext ctx = make_block_context(A, k);

#if DBG_D_BT
  std::cerr << "\n[BT-FFT] enter block=" << k << " r=" << ctx.r << " cols=[" << ctx.c0
            << "," << ctx.c1 << ")"
            << " s=" << s << " target.size=" << target.size() << "\n";
#endif

  if (target.size() != ctx.r) {
#if DBG_D_BT
    std::cerr << "[BT-FFT][ERROR] target.size()=" << target.size()
              << " vs r=" << ctx.r << "\n";
#endif
    assert(false &&
           "compute_base_table_for_block: target.size() must equal A.rows()");
  }

  if (ctx.r == 0 || ctx.t == 0) {
    if (s == 0)
      reset_unit_sumset(out);
    return out;
  }
  if (s == 0) {
#if DBG_D_S0
    std::cerr << "[BT-FFT][s==0] return zero vec, r=" << ctx.r << "\n";
#endif
    reset_unit_sumset(out);
    return out;
  }

  const VectorXi caps = compute_caps_for_block(A, ctx, target, s);

#if DBG_D_CAPS
  dump_vec_labeled("caps", caps);
#endif

  VectorXi row_max_u(ctx.r);
  std::vector<int> usable_cols;
  if (!build_usable_columns(A, ctx, caps, usable_cols, row_max_u))
    return out;

  if (s == 1)
    return decode_sparse_s1(A, usable_cols, ctx.r);

  std::vector<size_t> base;
  std::vector<size_t> stride;
  size_t N = 0;
  if (!build_embedding(row_max_u, ctx.r, s, base, stride, N)) {
    oracle::ac_profile::note_fft_base_call(false, 0.0, 0.0, false, 0.0);
#if DBG_D_BT
    std::cerr << "[BT-FFT] packed size too big (N=" << N
              << "), fallback to naive.\n";
#endif
    PowTable dummy_pow_table;
    return oracle::ac_naive::compute_base_sumset_for_block(A, k, target, s,
                                                           dummy_pow_table);
  }

#if DBG_D_BT
  std::cerr << "[BT-FFT] packed N=" << N << " (n_fft=" << next_pow2_local(N)
            << ")\n";
#endif

  const std::size_t n_fft = next_pow2_local(N);
  const double padding_efficiency =
      n_fft > 0 ? static_cast<double>(N) / static_cast<double>(n_fft) : 0.0;
  const double padding_waste_share =
      n_fft > 0 ? static_cast<double>(n_fft - N) / static_cast<double>(n_fft)
                : 0.0;
  const bool has_dummy_metric = A.has_slack_block() && !base.empty();
  const double dummy_work_share =
      has_dummy_metric
          ? 1.0 - (1.0 / static_cast<double>(base.back()))
          : 0.0;

  std::vector<uint8_t> P(N, 0); // boolean polynomial
  std::size_t active_coeffs = 0;
  for (int j : usable_cols) {
    size_t idx = 0;
    for (int i = 0; i < ctx.r; ++i) {
      const int v = A(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
      idx += static_cast<size_t>(v) * stride[static_cast<std::size_t>(i)];
    }
    if (!P[idx])
      ++active_coeffs;
    P[idx] = 1;
  }
  const double useful_grid_share =
      n_fft > 0 ? static_cast<double>(active_coeffs) / static_cast<double>(n_fft)
                : 0.0;
  (void)padding_efficiency;
  oracle::ac_profile::note_fft_base_call(true, useful_grid_share,
                                         padding_waste_share, has_dummy_metric,
                                         dummy_work_share);

  std::vector<uint8_t> acc;
  run_fft_power_exact_s(acc, P, N, s);

#if DBG_D_BT
  std::size_t out_sz = 0;
  for (const uint8_t x : acc)
    out_sz += (x != 0);
  std::cerr << "[BT-FFT] exit size~=" << out_sz << "\n";
#endif

  return decode_fft_result(acc, N, ctx.r, base, stride, caps);
}
