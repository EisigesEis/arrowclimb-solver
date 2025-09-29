#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <execution>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include "absl/container/flat_hash_set.h"

// -------------------- Utilities --------------------
static void dbg(const std::string& s) {
  std::cout << s << std::flush;
}

// -------------------- Hash / Eq / Compare --------------------
struct VecHash {
  size_t operator()(const Eigen::VectorXi& v) const noexcept {
    uint64_t h = 1469598103934665603ull;
    const int n = v.size();
    for (int i = 0; i < n; ++i) {
      uint64_t x = static_cast<uint64_t>(static_cast<int64_t>(v(i)));
      h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    }
    return static_cast<size_t>(h);
  }
};
struct VecEq {
  bool operator()(const Eigen::VectorXi& a, const Eigen::VectorXi& b) const noexcept {
    return (a - b).squaredNorm() == 0;
  }
};
struct LexLess {
  bool operator()(const Eigen::VectorXi& a, const Eigen::VectorXi& b) const noexcept {
    const int n = a.size();
    for (int i = 0; i < n; ++i) {
      const int ai = a(i), bi = b(i);
      if (ai < bi) return true;
      if (ai > bi) return false;
    }
    return false;
  }
};

using StdSet  = std::unordered_set<Eigen::VectorXi, VecHash, VecEq>;
using AbslSet = absl::flat_hash_set<Eigen::VectorXi, VecHash, VecEq>;

// -------------------- Random generation --------------------
static std::vector<Eigen::VectorXi> make_random_vectors(
    size_t n, int r, uint32_t seed, double duplicate_rate = 0.20,
    int value_lo = 0, int value_hi = 1000)
{
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> valdist(value_lo, value_hi);
  std::uniform_real_distribution<double> prob(0.0, 1.0);

  std::vector<Eigen::VectorXi> out;
  out.reserve(n);

  // Avoid uniform over empty range
  auto pick_index = [&](size_t hi, std::mt19937& rr)->size_t {
    if (hi == 0) return 0;
    std::uniform_int_distribution<size_t> pick(0, hi - 1);
    return pick(rr);
  };

  for (size_t i = 0; i < n; ++i) {
    Eigen::VectorXi v(r);
    if (i > 0 && prob(rng) < duplicate_rate) {
      v = out[pick_index(out.size(), rng)];
    } else {
      for (int j = 0; j < r; ++j) v(j) = valdist(rng);
    }
    out.emplace_back(std::move(v));
  }
  return out;
}

// -------------------- Strategies --------------------
static void dedupe_sort_unique(std::vector<Eigen::VectorXi>& a) {
  std::sort(std::execution::unseq, a.begin(), a.end(), LexLess{});
  a.erase(std::unique(std::execution::unseq, a.begin(), a.end(),
                      [](const Eigen::VectorXi& x, const Eigen::VectorXi& y) {
                        return (x - y).squaredNorm() == 0;
                      }),
          a.end());
}

template <class Set>
static void dedupe_set(std::vector<Eigen::VectorXi>& a) {
  Set s;
  s.reserve(static_cast<size_t>(a.size() * 1.3));
  std::vector<Eigen::VectorXi> out;
  out.reserve(a.size());
  for (auto& v : a) {
    if (s.insert(v).second) out.push_back(std::move(v));
  }
  a.swap(out);
}

template <class Set>
static void dedupe_mixed(std::vector<Eigen::VectorXi>& a, size_t threshold) {
  if (a.size() >= threshold) dedupe_sort_unique(a);
  else                       dedupe_set<Set>(a);
}

// -------------------- Timing helper --------------------
template <class Fn>
static double time_ms(Fn&& fn, int warmup = 1, int runs = 2) {
  for (int i = 0; i < warmup; ++i) fn();
  double total_ms = 0.0;
  for (int i = 0; i < runs; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  return total_ms / runs;
}

// -------------------- One run --------------------
struct ResultRow {
  size_t n;
  int r;
  std::string mode;
  size_t threshold;
  double ms;
  size_t out_size;
};

static ResultRow bench_once(size_t n, int r, uint32_t seed,
                            const std::string& mode, size_t threshold = 0) {
  // dbg("  bench_once: n=" + std::to_string(n) + " r=" + std::to_string(r) + " mode=" + mode + " T=" + std::to_string(threshold) + "\n");
  auto data = make_random_vectors(n, r, seed);
  size_t out_sz = 0;
  double ms = 0.0;

  if (mode == "sort") {
    auto fn = [&]() { auto a = data; dedupe_sort_unique(a); out_sz = a.size(); };
    ms = time_ms(fn);
  } else if (mode == "stdset") {
    auto fn = [&]() { auto a = data; dedupe_set<StdSet>(a); out_sz = a.size(); };
    ms = time_ms(fn);
  } else if (mode == "abslset") {
    auto fn = [&]() { auto a = data; dedupe_set<AbslSet>(a); out_sz = a.size(); };
    ms = time_ms(fn);
  } else if (mode == "mixed(std)") {
    auto fn = [&]() { auto a = data; dedupe_mixed<StdSet>(a, threshold); out_sz = a.size(); };
    ms = time_ms(fn);
  } else if (mode == "mixed(absl)") {
    auto fn = [&]() { auto a = data; dedupe_mixed<AbslSet>(a, threshold); out_sz = a.size(); };
    ms = time_ms(fn);
  } else {
    throw std::runtime_error("unknown mode: " + mode);
  }
  return ResultRow{n, r, mode, threshold, ms, out_sz};
}

// -------------------- Threshold tuning --------------------
static std::vector<size_t> default_threshold_grid() {
  return {0, 256, 512, 1000, 2000, 5000, 10000, 20000, 50000};
}

struct BestThreshold {
  int r;
  std::string variant; // "std" or "absl"
  size_t best_threshold;
  double best_avg_ms;
};

static BestThreshold tune_threshold_for_r(
    int r, const std::vector<size_t>& n_values, uint32_t seed_base,
    const std::vector<size_t>& candidates, bool use_absl) {

  dbg(" tune_threshold_for_r r=" + std::to_string(r) + " variant=" + (use_absl ? "absl" : "std") + "\n");
  size_t best_T = candidates.front();
  double best_avg = std::numeric_limits<double>::infinity();

  for (size_t T : candidates) {
    dbg("  candidate T=" + std::to_string(T) + "\n");
    double sum = 0.0;
    for (size_t i = 0; i < n_values.size(); ++i) {
      size_t n = n_values[i];
      const std::string mode = use_absl ? "mixed(absl)" : "mixed(std)";
      auto row = bench_once(n, r, seed_base + static_cast<uint32_t>(i * 101 + T), mode, T);
      sum += row.ms;
    }
    double avg = sum / n_values.size();
    if (avg < best_avg) { best_avg = avg; best_T = T; }
  }
  return BestThreshold{r, use_absl ? "absl" : "std", best_T, best_avg};
}

// -------------------- CLI --------------------
static void usage(const char* argv0) {
  std::cerr
    << "Usage: " << argv0 << " [--nmin N] [--nmax N] [--nstep N] "
       "[--rmin R] [--rmax R] [--rstep R] [--trials K] [--seed S] "
       "[--csv path] [--threshold-grid CSV_INTs]\n";
}

int main(int argc, char** argv) try {
  size_t nmin = 2000, nmax = 100000, nstep = 8000;
  int rmin = 4, rmax = 64, rstep = 8;
  int trials = 3;
  uint32_t seed = 1337;
  std::string csv_path;

  std::vector<size_t> thr_grid = default_threshold_grid();

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](int more) {
      if (i + more >= argc) { usage(argv[0]); std::exit(1); }
    };
    if (a == "--nmin")   { need(1); nmin = std::stoull(argv[++i]); }
    else if (a == "--nmax")  { need(1); nmax = std::stoull(argv[++i]); }
    else if (a == "--nstep") { need(1); nstep = std::stoull(argv[++i]); }
    else if (a == "--rmin")  { need(1); rmin = std::stoi(argv[++i]); }
    else if (a == "--rmax")  { need(1); rmax = std::stoi(argv[++i]); }
    else if (a == "--rstep") { need(1); rstep = std::stoi(argv[++i]); }
    else if (a == "--trials"){ need(1); trials = std::stoi(argv[++i]); }
    else if (a == "--seed")  { need(1); seed = static_cast<uint32_t>(std::stoull(argv[++i])); }
    else if (a == "--csv")   { need(1); csv_path = argv[++i]; }
    else if (a == "--threshold-grid") {
      need(1);
      thr_grid.clear();
      std::string s = argv[++i];
      size_t pos = 0;
      while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        std::string tok = s.substr(pos, (comma == std::string::npos) ? std::string::npos : comma - pos);
        if (!tok.empty()) thr_grid.push_back(static_cast<size_t>(std::stoull(tok)));
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
      if (thr_grid.empty()) throw std::runtime_error("empty threshold grid");
    }
    else {
      usage(argv[0]); return 1;
    }
  }

  std::vector<size_t> n_values;
  for (size_t n = nmin; n <= nmax; n += std::max<size_t>(1, nstep)) n_values.push_back(n);
  std::vector<int> r_values;
  for (int r = rmin; r <= rmax; r += std::max(1, rstep)) r_values.push_back(r);

  std::ofstream csv;
  if (!csv_path.empty()) {
    csv.open(csv_path, std::ios::out | std::ios::trunc);
    if (!csv.is_open()) throw std::runtime_error("failed to open CSV path: " + csv_path);
    csv << "n,r,mode,threshold,ms,out_size\n";
    csv.flush();
  }

  std::cout << "Benchmark dedup: n in [" << nmin << "," << nmax << "] step " << nstep
            << ", r in [" << rmin << "," << rmax << "] step " << rstep
            << ", trials=" << trials << ", seed=" << seed << "\n";

  // Tuning & benchmarking
  for (int r : r_values) {
    dbg("\n== r = " + std::to_string(r) + " ==\n");
    auto best_std  = tune_threshold_for_r(r, n_values, seed + 1000, thr_grid, /*use_absl=*/false);
    auto best_absl = tune_threshold_for_r(r, n_values, seed + 2000, thr_grid, /*use_absl=*/true);
    std::cout << "Best thresholds for r=" << r
              << " | std: T=" << best_std.best_threshold << " avg_ms=" << best_std.best_avg_ms
              << " | absl: T=" << best_absl.best_threshold << " avg_ms=" << best_absl.best_avg_ms
              << "\n";

    for (int t = 0; t < trials; ++t) {
      for (size_t i = 0; i < n_values.size(); ++i) {
        size_t n = n_values[i];
        uint32_t s = seed + static_cast<uint32_t>(t * 997 + i * 131 + r * 17);
        dbg(" r=" + std::to_string(r) + " trial=" + std::to_string(t) + " n=" + std::to_string(n) + "\n");

        for (auto mode : {std::string("sort"), std::string("stdset"), std::string("abslset")}) {
          auto row = bench_once(n, r, s, mode);
          if (csv.is_open()) {
            csv << row.n << "," << row.r << "," << row.mode << "," << row.threshold
                << "," << std::fixed << std::setprecision(3) << row.ms
                << "," << row.out_size << "\n";
            csv.flush();
          }
        }
        {
          auto row = bench_once(n, r, s, "mixed(std)",  best_std.best_threshold);
          if (csv.is_open()) {
            csv << row.n << "," << row.r << "," << row.mode << "," << row.threshold
                << "," << std::fixed << std::setprecision(3) << row.ms
                << "," << row.out_size << "\n";
            csv.flush();
          }
        }
        {
          auto row = bench_once(n, r, s, "mixed(absl)", best_absl.best_threshold);
          if (csv.is_open()) {
            csv << row.n << "," << row.r << "," << row.mode << "," << row.threshold
                << "," << std::fixed << std::setprecision(3) << row.ms
                << "," << row.out_size << "\n";
            csv.flush();
          }
        }
      }
    }
  }

  std::cout << "\nDONE.\n";
  if (csv.is_open()) std::cout << "CSV: " << csv_path << "\n";
  return 0;
}
catch (const std::bad_alloc&) {
  std::cerr << "FATAL: std::bad_alloc (out of memory). Try smaller ranges.\n";
  return 2;
}
catch (const std::exception& e) {
  std::cerr << "FATAL: " << e.what() << "\n";
  return 1;
}
