#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "oracle/discrep_pow/discr_helper.h"

// Benchmark goal:
// Compare dedup strategies on std::vector<Vec>:
//  (1) sort + unique (memcmp comparator)
//  (2) absl::flat_hash_set insert (VecHash/VecEq)
// across sizes and duplicate rates, for varying effective dimension m.

static inline double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

template <class Fn>
static double time_ms(Fn&& fn, int warmup = 1, int runs = 3) {
  for (int i = 0; i < warmup; ++i) fn();
  double total = 0.0;
  for (int i = 0; i < runs; ++i) {
    const double t0 = now_ms();
    fn();
    const double t1 = now_ms();
    total += (t1 - t0);
  }
  return total / runs;
}

static std::vector<Vec> make_random_vecs(size_t n, int m, uint32_t seed,
                                        double duplicate_rate,
                                        int value_lo, int value_hi) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> valdist(value_lo, value_hi);
  std::uniform_real_distribution<double> prob(0.0, 1.0);

  std::vector<Vec> out;
  out.reserve(n);

  auto pick_index = [&](size_t hi) -> size_t {
    if (hi == 0) return 0;
    std::uniform_int_distribution<size_t> pick(0, hi - 1);
    return pick(rng);
  };

  for (size_t i = 0; i < n; ++i) {
    Vec v{}; // zero-init tail
    if (i > 0 && prob(rng) < duplicate_rate) {
      v = out[pick_index(out.size())];
    } else {
      for (int k = 0; k < m; ++k) v.x[k] = valdist(rng);
    }
    out.push_back(v);
  }
  return out;
}

static void dedupe_sort_unique(std::vector<Vec>& a) {
  std::sort(a.begin(), a.end(), VecLess{});
  a.erase(std::unique(a.begin(), a.end(), VecEq{}), a.end());
}

static void dedupe_absl_hash(std::vector<Vec>& a) {
  absl::flat_hash_set<Vec, VecHash, VecEq> s;
  s.reserve(a.size() * 2 + 8);

  std::vector<Vec> out;
  out.reserve(a.size());
  for (auto& v : a) {
    if (s.insert(v).second) out.push_back(std::move(v));
  }
  a.swap(out);
}

struct Row {
  int m;
  size_t n;
  double dup;
  std::string mode;
  double ms;
  size_t out_n;
};

static Row bench_one(int m, size_t n, double dup, uint32_t seed,
                    const std::string& mode,
                    int value_lo = 0, int value_hi = 1000) {
  auto data = make_random_vecs(n, m, seed, dup, value_lo, value_hi);
  size_t out_n = 0;
  double ms = 0.0;

  if (mode == "sort") {
    auto fn = [&]() {
      auto a = data;
      dedupe_sort_unique(a);
      out_n = a.size();
    };
    ms = time_ms(fn);
  } else if (mode == "absl") {
    auto fn = [&]() {
      auto a = data;
      dedupe_absl_hash(a);
      out_n = a.size();
    };
    ms = time_ms(fn);
  } else {
    throw std::runtime_error("unknown mode");
  }

  return Row{m, n, dup, mode, ms, out_n};
}

static void print_header() {
  std::cout << "m,n,dup,mode,ms,out_n\n";
}

int main(int argc, char** argv) {
  // Defaults for MAX_M=16 and m in {8,12,16}.
  std::vector<int> ms = {8, 12, 16};
  std::vector<size_t> ns = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
  std::vector<double> dups = {0.0, 0.05, 0.20, 0.50, 0.80};

  // Optional CLI overrides (very lightweight):
  //   argv[1]=seed
  //   argv[2]=max_n_index (0..ns.size-1)
  uint32_t seed = 12345;
  if (argc >= 2) seed = static_cast<uint32_t>(std::stoul(argv[1]));
  size_t max_n_idx = ns.size();
  if (argc >= 3) {
    size_t idx = static_cast<size_t>(std::stoul(argv[2]));
    max_n_idx = std::min(idx + 1, ns.size());
  }

  print_header();

  // Fixed value range for generated A entries.
  const int value_lo = 0;
  const int value_hi = 200; // smaller range increases collisions -> stresses dedupe

  for (int m : ms) {
    for (double dup : dups) {
      for (size_t i = 0; i < max_n_idx; ++i) {
        const size_t n = ns[i];
        // Different seeds per point, reproducible.
        const uint32_t s0 = seed + static_cast<uint32_t>(m * 100000 + int(dup * 1000) * 100 + i);

        auto r_sort = bench_one(m, n, dup, s0, "sort", value_lo, value_hi);
        auto r_absl = bench_one(m, n, dup, s0, "absl", value_lo, value_hi);

        auto emit = [&](const Row& r) {
          std::cout << r.m << ',' << r.n << ',' << std::fixed << std::setprecision(2)
                    << r.dup << ',' << r.mode << ','
                    << std::setprecision(3) << r.ms << ',' << r.out_n << '\n';
        };
        emit(r_sort);
        emit(r_absl);
      }
    }
  }

  return 0;
}
