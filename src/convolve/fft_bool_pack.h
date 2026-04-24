#pragma once
#include <fftw3.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

/*
  Packed single-FFT boolean convolution (float / fftwf)

  Pack a and b into one complex signal c = a + i*b, do 1 forward complex FFT,
  recover spectra A=FFT(a), B=FFT(b) using conjugate symmetry, form product
  spectrum P = A*B (which is Hermitian because a,b are real), then 1 inverse
  FFT. Threshold real part to get boolean convolution (>=1).

  Implementation notes:
  - float (fftwf) instead of double
  - reuse Fc as the product spectrum (no separate Fp allocation)
  - compute only k=0..n_fft/2 and fill the Hermitian mirror (≈2x cheaper
  spectrum loop)
  - cached nf/thr and no unconditional out.resize in hot path
*/
struct FFTBoolConvolver {
  using u8 = std::uint8_t;

  std::size_t N = 0;     // logical length (we only output first N)
  std::size_t n_fft = 0; // FFT length (>= N), chosen by next_fast_len
  std::size_t nh = 0;    // n_fft/2 (cached)
  float thr = 0.0f;      // threshold in unnormalized iFFT domain (=0.5*n_fft)

  // Time-domain packed complex buffer (length n_fft)
  fftwf_complex *tc = nullptr;

  // Frequency-domain packed spectrum C (length n_fft)
  // Reused as product spectrum P in-place.
  fftwf_complex *Fc = nullptr;

  fftwf_plan plan_fwd = nullptr; // c2c forward: tc -> Fc
  fftwf_plan plan_inv = nullptr; // c2c backward: Fc -> tc

  unsigned plan_flags = FFTW_ESTIMATE;

  // FFT length helper
  static inline bool is_fast_len(std::size_t n) {
    while ((n & 1u) == 0u)
      n >>= 1u;
    constexpr std::size_t primes[] = {3u, 5u, 7u};
    for (std::size_t p : primes) {
      while (n % p == 0u)
        n /= p;
    }
    return n == 1u;
  }

  static inline std::size_t next_fast_len(std::size_t n) {
#ifdef FFT_LEN_PRIMEFACTOR
    if (n <= 2u)
      return 2u;
    std::size_t m = 1u;
    while (m < n)
      m <<= 1u;
    std::size_t best = m;
    constexpr std::size_t MAX_FACTOR = 7u;
    for (std::size_t k = 1u; k <= MAX_FACTOR; ++k) {
      std::size_t cand = m * k;
      if (cand >= n && cand < best && is_fast_len(cand))
        best = cand;
    }
    return best;
#else
    if (n <= 2u)
      return 2u;
    std::size_t m = 1u;
    while (m < n)
      m <<= 1u;
    return m;
#endif
  }

  explicit FFTBoolConvolver(std::size_t N_, unsigned flags = FFTW_ESTIMATE,
                                 int /*nthreads*/ = 1)
      : N(N_), plan_flags(flags) {
    assert(N > 0);

    // The solver only consumes the first N logical coefficients.
    n_fft = next_fast_len(N);
    nh = n_fft / 2u;
    thr = 0.5f * static_cast<float>(n_fft);

    tc = (fftwf_complex *)fftwf_alloc_complex(n_fft);
    Fc = (fftwf_complex *)fftwf_alloc_complex(n_fft);

    if (!tc || !Fc) {
      cleanup();
      throw std::bad_alloc();
    }

    plan_fwd = fftwf_plan_dft_1d((int)n_fft, tc, Fc, FFTW_FORWARD, plan_flags);
    plan_inv = fftwf_plan_dft_1d((int)n_fft, Fc, tc, FFTW_BACKWARD, plan_flags);

    if (!plan_fwd || !plan_inv) {
      cleanup();
      throw std::runtime_error("FFTW plan creation failed (packed/float)");
    }
  }

  FFTBoolConvolver(const FFTBoolConvolver &) = delete;
  FFTBoolConvolver &operator=(const FFTBoolConvolver &) = delete;

  FFTBoolConvolver(FFTBoolConvolver &&o) noexcept {
    move_from(std::move(o));
  }
  FFTBoolConvolver &operator=(FFTBoolConvolver &&o) noexcept {
    if (this != &o) {
      cleanup();
      move_from(std::move(o));
    }
    return *this;
  }

  ~FFTBoolConvolver() { cleanup(); }

  // out[i] = 1 iff sum_j a[j]*b[i-j] >= 1, truncated to i < N.
  inline void mul_bool(std::vector<u8> &out, const std::vector<u8> &a,
                       const std::vector<u8> &b) {
    assert(a.size() == N && b.size() == N);
    if (out.size() != N)
      out.resize(N);
    mul_bool(out.data(), a.data(), b.data());
  }

  inline void mul_bool(u8 *out, const u8 *a, const u8 *b) {
    // Pack into tc: tc[i] = a[i] + i*b[i]
    for (std::size_t i = 0; i < N; ++i) {
      tc[i][0] = (a[i] != 0) ? 1.0f : 0.0f; // real
      tc[i][1] = (b[i] != 0) ? 1.0f : 0.0f; // imag
    }
    for (std::size_t i = N; i < n_fft; ++i) {
      tc[i][0] = 0.0f;
      tc[i][1] = 0.0f;
    }

    // Fc = FFT(tc)
    fftwf_execute(plan_fwd);

    // Recover A=FFT(a), B=FFT(b) from Fc using symmetry:
    // Let x = Fc[k], y = conj(Fc[(n_fft - k) % n_fft]).
    // Then:
    //   A = (x + y)/2
    //   B = (x - y)/(2i)
    //
    // Because a,b are real, P = A*B is Hermitian: P[kr] = conj(P[k]).
    // So compute only k=0..n_fft/2 and fill the mirror.
    for (std::size_t k = 0; k <= nh; ++k) {
      const std::size_t kr = (k == 0) ? 0 : (n_fft - k);

      const float xr = Fc[k][0];
      const float xi = Fc[k][1];

      // y = conj(Fc[kr]) = (Fc[kr].re, -Fc[kr].im)
      const float yr = Fc[kr][0];
      const float yi = -Fc[kr][1];

      // A = (x + y)/2
      const float Ar = 0.5f * (xr + yr);
      const float Ai = 0.5f * (xi + yi);

      // B = (x - y)/(2i)
      // If d = x - y = (xr-yr) + i(xi-yi), then d/(2i) has:
      //   Br =  0.5*(xi - yi)
      //   Bi = -0.5*(xr - yr) = 0.5*(yr - xr)
      const float Br = 0.5f * (xi - yi);
      const float Bi = 0.5f * (yr - xr);

      // P = A * B
      const float Pr = Ar * Br - Ai * Bi;
      const float Pi = Ar * Bi + Ai * Br;

      // Store P in-place into Fc.
      Fc[k][0] = Pr;
      Fc[k][1] = Pi;

      // Fill Hermitian mirror (skip fixed points).
      if (kr != k) {
        Fc[kr][0] = Pr;
        Fc[kr][1] = -Pi;
      }
    }

    // tc = IFFT(Fc) (unnormalized: true conv coeff * n_fft in real part)
    fftwf_execute(plan_inv);

    // boolean threshold at 0.5*n_fft (since coeff are integers)
    for (std::size_t i = 0; i < N; ++i) {
      out[i] = (tc[i][0] > thr) ? u8{1} : u8{0};
    }
  }

private:
  inline void cleanup() noexcept {
    if (plan_fwd)
      fftwf_destroy_plan(plan_fwd);
    if (plan_inv)
      fftwf_destroy_plan(plan_inv);
    plan_fwd = nullptr;
    plan_inv = nullptr;

    if (Fc)
      fftwf_free(Fc);
    if (tc)
      fftwf_free(tc);
    Fc = nullptr;
    tc = nullptr;
  }

  inline void move_from(FFTBoolConvolver &&o) noexcept {
    N = o.N;
    n_fft = o.n_fft;
    nh = o.nh;
    thr = o.thr;
    tc = o.tc;
    o.tc = nullptr;
    Fc = o.Fc;
    o.Fc = nullptr;
    plan_fwd = o.plan_fwd;
    o.plan_fwd = nullptr;
    plan_inv = o.plan_inv;
    o.plan_inv = nullptr;
    plan_flags = o.plan_flags;
  }
};
