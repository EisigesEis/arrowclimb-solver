#pragma once

#include <fftw3.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// FFTW boolean convolution (single precision): out = boolconv(a,b)
// where a,b in {0,1}^N and (logical) sum-indices stay < N.
//
// This is a float/fftwf sibling of fft_bool_r2c.h, tuned for repeated calls:
// - single batched forward plan (2x R2C) via plan_many
// - cached nf / threshold
// - vector overload avoids repeated reallocations when out already sized
struct FFTBoolConvolver {
  using u8 = std::uint8_t;

  // Public: original logical size (time domain length N)
  std::size_t N = 0;

  // Internal: FFT length (>= N), chosen as "fast" length
  std::size_t n_fft = 0;

  // Internal: R2C spectrum length
  std::size_t nf = 0;

  // Threshold for unnormalized inverse (FFTW's c2r returns values scaled by
  // n_fft)
  float thr = 0.0f;

  // Real buffers (time domain): stored contiguously as [2][n_fft]
  float *t01 = nullptr;
  float *t0 = nullptr; // t01 + 0*n_fft
  float *t1 = nullptr; // t01 + 1*n_fft

  // Complex spectra (freq domain): stored contiguously as [2][nf]
  fftwf_complex *f01 = nullptr;
  fftwf_complex *f0 = nullptr;    // f01 + 0*nf
  fftwf_complex *f1 = nullptr;    // f01 + 1*nf
  fftwf_complex *fprod = nullptr; // length nf

  // Plans (reused)
  fftwf_plan plan_fwd2 = nullptr; // 2x r2c: t01 -> f01
  fftwf_plan plan_inv1 = nullptr; // c2r: fprod -> t0

  // Optional: configure planning rigor
  unsigned plan_flags = FFTW_ESTIMATE;

  // FFT length helper
  static inline bool is_fast_len(std::size_t n) {
    while ((n & 1) == 0)
      n >>= 1;
    constexpr std::size_t primes[] = {3, 5, 7};
    for (std::size_t p : primes)
      while (n % p == 0)
        n /= p;
    return n == 1;
  }

  static inline std::size_t next_fast_len(std::size_t n) {
#ifdef FFT_LEN_PRIMEFACTOR
    if (n <= 2)
      return 2;
    std::size_t m = 1;
    while (m < n)
      m <<= 1;
    std::size_t best = m;
    constexpr std::size_t MAX_FACTOR = 7;
    for (std::size_t k = 1; k <= MAX_FACTOR; ++k) {
      std::size_t cand = m * k;
      if (cand >= n && cand < best && is_fast_len(cand))
        best = cand;
    }
    return best;
#else
    if (n <= 2)
      return 2;
    std::size_t m = 1;
    while (m < n)
      m <<= 1;
    return m;
#endif
  }

  // Optional thread support
  static inline void maybe_init_threads(int nthreads) {
#if defined(FFTW3_THREADS) || defined(FFTW_THREADS) ||                         \
    defined(FFTW_ENABLE_THREADS)
    if (nthreads > 1) {
      static bool inited = false;
      if (!inited) {
        fftwf_init_threads();
        inited = true;
      }
      fftwf_plan_with_nthreads(nthreads);
    }
#else
    (void)nthreads;
#endif
  }

  explicit FFTBoolConvolver(std::size_t N_, unsigned flags = FFTW_ESTIMATE,
                             int nthreads = 1)
      : N(N_), plan_flags(flags) {
    assert(N > 0);

    // The solver only consumes the first N logical coefficients.
    n_fft = next_fast_len(N);
    nf = (n_fft / 2) + 1;
    thr = 0.5f * static_cast<float>(n_fft);

    // maybe_init_threads(nthreads);
    (void)nthreads;

    t01 = (float *)fftwf_alloc_real(2 * n_fft);
    f01 = (fftwf_complex *)fftwf_alloc_complex(2 * nf);
    fprod = (fftwf_complex *)fftwf_alloc_complex(nf);

    if (!t01 || !f01 || !fprod) {
      cleanup();
      throw std::bad_alloc();
    }

    t0 = t01;
    t1 = t01 + n_fft;
    f0 = f01;
    f1 = f01 + nf;

    // Plan 2 forward transforms at once: [2][n_fft] -> [2][nf]
    {
      int rank = 1;
      int n[1] = {static_cast<int>(n_fft)};
      int howmany = 2;

      int inembed[1] = {static_cast<int>(n_fft)};
      int onembed[1] = {static_cast<int>(nf)};

      int istride = 1;
      int ostride = 1;
      int idist = static_cast<int>(n_fft);
      int odist = static_cast<int>(nf);

      plan_fwd2 = fftwf_plan_many_dft_r2c(rank, n, howmany, t01, inembed,
                                          istride, idist, f01, onembed, ostride,
                                          odist, plan_flags);
    }

    plan_inv1 =
        fftwf_plan_dft_c2r_1d(static_cast<int>(n_fft), fprod, t0, plan_flags);

    if (!plan_fwd2 || !plan_inv1) {
      cleanup();
      throw std::runtime_error("FFTW (float) plan creation failed");
    }
  }

  FFTBoolConvolver(const FFTBoolConvolver &) = delete;
  FFTBoolConvolver &operator=(const FFTBoolConvolver &) = delete;

  FFTBoolConvolver(FFTBoolConvolver &&o) noexcept { move_from(std::move(o)); }
  FFTBoolConvolver &operator=(FFTBoolConvolver &&o) noexcept {
    if (this != &o) {
      cleanup();
      move_from(std::move(o));
    }
    return *this;
  }

  ~FFTBoolConvolver() { cleanup(); }

  // Multiply two boolean vectors (0/1) and return boolean convolution (0/1):
  // out[i] = 1 iff sum_j a[j]*b[i-j] >= 1.
  //
  // - out is sized to N and fully written.
  // - inputs must have size N.
  inline void mul_bool(std::vector<u8> &out, const std::vector<u8> &a,
                       const std::vector<u8> &b) {
    assert(a.size() == N && b.size() == N);
    if (out.size() != N)
      out.resize(N);
    mul_bool(out.data(), a.data(), b.data());
  }

  inline void mul_bool(u8 *out, const u8 *a, const u8 *b) {
    // Load + pad
    for (std::size_t i = 0; i < N; ++i) {
      t0[i] = (a[i] != 0) ? 1.0f : 0.0f;
      t1[i] = (b[i] != 0) ? 1.0f : 0.0f;
    }
    if (n_fft > N) {
      std::memset(t0 + N, 0, (n_fft - N) * sizeof(float));
      std::memset(t1 + N, 0, (n_fft - N) * sizeof(float));
    }

    // 2x forward FFT in one execution
    fftwf_execute(plan_fwd2);

    // Pointwise multiply spectra into fprod
    for (std::size_t k = 0; k < nf; ++k) {
      const float ar = f0[k][0], ai = f0[k][1];
      const float br = f1[k][0], bi = f1[k][1];
      fprod[k][0] = ar * br - ai * bi;
      fprod[k][1] = ar * bi + ai * br;
    }

    // Inverse FFT into t0 (unnormalized)
    fftwf_execute(plan_inv1);

    for (std::size_t i = 0; i < N; ++i)
      out[i] = (t0[i] > thr) ? u8{1} : u8{0};
  }

private:
  inline void cleanup() noexcept {
    if (plan_fwd2)
      fftwf_destroy_plan(plan_fwd2);
    if (plan_inv1)
      fftwf_destroy_plan(plan_inv1);
    plan_fwd2 = nullptr;
    plan_inv1 = nullptr;

    if (fprod)
      fftwf_free(fprod);
    if (f01)
      fftwf_free(f01);
    if (t01)
      fftwf_free(t01);

    fprod = nullptr;
    f01 = nullptr;
    t01 = nullptr;
    t0 = t1 = nullptr;
    f0 = f1 = nullptr;
  }

  inline void move_from(FFTBoolConvolver &&o) noexcept {
    N = o.N;
    n_fft = o.n_fft;
    nf = o.nf;
    thr = o.thr;

    t01 = o.t01;
    o.t01 = nullptr;
    f01 = o.f01;
    o.f01 = nullptr;
    fprod = o.fprod;
    o.fprod = nullptr;

    t0 = o.t0;
    o.t0 = nullptr;
    t1 = o.t1;
    o.t1 = nullptr;
    f0 = o.f0;
    o.f0 = nullptr;
    f1 = o.f1;
    o.f1 = nullptr;

    plan_fwd2 = o.plan_fwd2;
    o.plan_fwd2 = nullptr;
    plan_inv1 = o.plan_inv1;
    o.plan_inv1 = nullptr;

    plan_flags = o.plan_flags;
  }
};
