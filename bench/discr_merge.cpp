// bench_merge_sets_fast.cpp
// Simplified, application-like benchmark for discrepancy-style merge:
// compares vector+sort_unique vs absl::flat_hash_set insertion.
//
// Output CSV columns:
// m,asz,bsz,range,cap_tightness,mode,ms,out_sz,prod
//
// Usage:
//   ./bench_merge_sets_fast > out.csv
//   ./bench_merge_sets_fast --cases 200 --maxn 4096 --m 16 --seed 1
//
// Notes:
// - This intentionally keeps the sweep small and predictable.
// - "cap_tightness" controls pruning probability (higher => tighter caps => fewer survivors).

#include <absl/container/flat_hash_set.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef MAX_M
#define MAX_M 16
#endif

struct Vec {
  std::array<int, MAX_M> x;
};

struct VecHash {
  size_t operator()(const Vec& v) const noexcept {
    // FNV-1a over bytes (fast and decent for small fixed blobs)
    const uint8_t* p = reinterpret_cast<const uint8_t*>(v.x.data());
    size_t h = 1469598103934665603ull;
    for (size_t i = 0; i < sizeof(int) * MAX_M; ++i) {
      h ^= size_t(p[i]);
      h *= 1099511628211ull;
    }
    return h;
  }
};

struct VecEq {
  bool operator()(const Vec& a, const Vec& b) const noexcept {
    return std::memcmp(a.x.data(), b.x.data(), sizeof(int) * MAX_M) == 0;
  }
};

struct VecLess {
  bool operator()(const Vec& a, const Vec& b) const noexcept {
    return std::memcmp(a.x.data(), b.x.data(), sizeof(int) * MAX_M) < 0;
  }
};

static inline void sort_unique(std::vector<Vec>& v) {
  std::sort(v.begin(), v.end(), VecLess{});
  v.erase(std::unique(v.begin(), v.end(), VecEq{}), v.end());
}

static inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Args {
  int m = 16;
  uint32_t seed = 1;
  int cases = 120;        // number of random instances per parameter point (kept small)
  int warmup = 5;
  int repeat = 3;         // repeat per point; report min
  int maxn = 4096;        // max asz/bsz
  int range = 200;        // values in [0, range)
};

static inline bool starts_with(const std::string& s, const std::string& p) {
  return s.rfind(p, 0) == 0;
}

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s(argv[i]);
    auto get = [&](int& dst) {
      if (i + 1 < argc) dst = std::stoi(argv[++i]);
    };
    auto getu32 = [&](uint32_t& dst) {
      if (i + 1 < argc) dst = (uint32_t)std::stoul(argv[++i]);
    };
    if (s == "--m") get(a.m);
    else if (s == "--seed") getu32(a.seed);
    else if (s == "--cases") get(a.cases);
    else if (s == "--warmup") get(a.warmup);
    else if (s == "--repeat") get(a.repeat);
    else if (s == "--maxn") get(a.maxn);
    else if (s == "--range") get(a.range);
  }
  if (a.m < 1) a.m = 1;
  if (a.m > MAX_M) a.m = MAX_M;
  if (a.cases < 1) a.cases = 1;
  if (a.repeat < 1) a.repeat = 1;
  return a;
}

// cap_tightness: 0..100. Higher => tighter caps => more pruning.
static inline void make_caps(std::array<int, MAX_M>& b_up, int m, int range, int cap_tightness,
                             std::mt19937& rng) {
  // Base cap around 2*range (loose), tighten towards ~range/4
  const double t = std::clamp(cap_tightness / 100.0, 0.0, 1.0);
  const int loose = 2 * range;
  const int tight = std::max(1, range / 4);
  const int cap_center = int((1.0 - t) * loose + t * tight);
  std::uniform_int_distribution<int> jitter(std::max(1, cap_center / 10), std::max(2, cap_center / 4));
  for (int i = 0; i < m; ++i) b_up[i] = cap_center + jitter(rng);
  for (int i = m; i < MAX_M; ++i) b_up[i] = 0;
}

static inline Vec rand_vec(int m, int range, std::mt19937& rng) {
  Vec v{};
  std::uniform_int_distribution<int> dist(0, range - 1);
  for (int i = 0; i < m; ++i) v.x[i] = dist(rng);
  // tail is already zero
  return v;
}

static inline void make_set(std::vector<Vec>& out, int m, int n, int range, std::mt19937& rng) {
  out.clear();
  out.reserve(size_t(n));
  for (int i = 0; i < n; ++i) out.push_back(rand_vec(m, range, rng));
  // In real discrepancy, sets are usually unique; emulate that.
  sort_unique(out);
}

static inline size_t merge_sort_unique(const std::vector<Vec>& A, const std::vector<Vec>& B,
                                       int m, const std::array<int, MAX_M>& b_up,
                                       std::vector<Vec>& out) {
  out.clear();
  // Reserve a bounded amount: we don't want to allocate asz*bsz.
  const size_t asz = A.size(), bsz = B.size();
  out.reserve(std::min<size_t>(asz * bsz, 1u << 22)); // cap reserve ~4M entries

  for (const auto& a : A) {
    for (const auto& b : B) {
      Vec w{};
      // prune by caps
      for (int k = 0; k < m; ++k) {
        const int s = a.x[k] + b.x[k];
        if (s > b_up[k]) goto next_pair;
        w.x[k] = s;
      }
      out.push_back(w);
    next_pair:
      ;
    }
  }
  sort_unique(out);
  return out.size();
}

static inline size_t merge_hash(const std::vector<Vec>& A, const std::vector<Vec>& B,
                                int m, const std::array<int, MAX_M>& b_up,
                                absl::flat_hash_set<Vec, VecHash, VecEq>& out) {
  out.clear();
  const size_t asz = A.size(), bsz = B.size();
  // Reserve heuristic: not too high.
  out.reserve(std::min<size_t>(asz * bsz, 1u << 22));

  for (const auto& a : A) {
    for (const auto& b : B) {
      Vec w{};
      for (int k = 0; k < m; ++k) {
        const int s = a.x[k] + b.x[k];
        if (s > b_up[k]) goto next_pair;
        w.x[k] = s;
      }
      out.insert(w);
    next_pair:
      ;
    }
  }
  return out.size();
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  std::mt19937 rng(args.seed);

  // Parameter points kept small to avoid long runs.
  const std::vector<int> Ns = {64, 128, 256, 512, 1024, 2048, 4096};
  std::vector<int> Ns_used;
  for (int n : Ns) if (n <= args.maxn) Ns_used.push_back(n);

  const std::vector<int> tightness = {0, 25, 50, 75}; // loose -> tight
  const int m = args.m;

  std::vector<Vec> A, B, tmp;
  absl::flat_hash_set<Vec, VecHash, VecEq> htmp;

  std::cout << "m,asz,bsz,range,cap_tightness,mode,ms,out_sz,prod\n";

  auto bench_one = [&](int asz_target, int bsz_target, int cap_t) {
    // Make unique-ish inputs each time to reduce weird artifacts.
    make_set(A, m, asz_target, args.range, rng);
    make_set(B, m, bsz_target, args.range, rng);

    std::array<int, MAX_M> b_up{};
    make_caps(b_up, m, args.range, cap_t, rng);

    // warmup
    for (int w = 0; w < args.warmup; ++w) {
      merge_sort_unique(A, B, m, b_up, tmp);
      merge_hash(A, B, m, b_up, htmp);
    }

    // measure vector+sort
    double best_sort_ms = 1e300;
    size_t out_sort = 0;
    for (int r = 0; r < args.repeat; ++r) {
      const uint64_t t0 = now_ns();
      for (int c = 0; c < args.cases; ++c) out_sort = merge_sort_unique(A, B, m, b_up, tmp);
      const uint64_t t1 = now_ns();
      const double ms = double(t1 - t0) / 1e6;
      best_sort_ms = std::min(best_sort_ms, ms);
    }

    // measure hash
    double best_hash_ms = 1e300;
    size_t out_hash = 0;
    for (int r = 0; r < args.repeat; ++r) {
      const uint64_t t0 = now_ns();
      for (int c = 0; c < args.cases; ++c) out_hash = merge_hash(A, B, m, b_up, htmp);
      const uint64_t t1 = now_ns();
      const double ms = double(t1 - t0) / 1e6;
      best_hash_ms = std::min(best_hash_ms, ms);
    }

    const size_t prod = size_t(A.size()) * size_t(B.size());
    std::cout << m << "," << A.size() << "," << B.size() << "," << args.range << ","
              << cap_t << ",sort," << best_sort_ms << "," << out_sort << "," << prod << "\n";
    std::cout << m << "," << A.size() << "," << B.size() << "," << args.range << ","
              << cap_t << ",hash," << best_hash_ms << "," << out_hash << "," << prod << "\n";
  };

  for (int n : Ns_used) {
    for (int cap_t : tightness) {
      bench_one(n, n, cap_t);
    }
  }
  return 0;
}
