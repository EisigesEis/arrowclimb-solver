// bench/bench_small_dp.cpp
#include "instance/types.h"
#include "enumerate/small.h"
#include "small_old.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using Scal = int;
using Clock = std::chrono::steady_clock;

// CLI

struct CLI {
  int reps = 200;
  int warmup = 1;
  int sleep_ms = 0;
  size_t max_emits = 0; // 0 = no cap
};

static CLI parse_cli(int argc, char **argv) {
  CLI c;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](int &i) -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
    };
    if (a == "--reps")
      c.reps = std::max(1, std::stoi(next(i)));
    else if (a == "--warmup")
      c.warmup = std::max(0, std::stoi(next(i)));
    else if (a == "--sleep")
      c.sleep_ms = std::max(0, std::stoi(next(i)));
    else if (a == "--max-emits")
      c.max_emits = (size_t)std::stoull(next(i));
  }
  return c;
}

// Hashing and stats

static inline uint64_t rotl64(uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

struct Hash64 {
  uint64_t h = 1469598103934665603ull;
  inline void mix_Scal(Scal v) {
    h ^= (uint64_t)v + 0x9e3779b97f4a7c15ull;
    h = rotl64(h, 13) * 0xff51afd7ed558ccdull;
  }
  inline void mix_vec(const VectorXi &x) {
    for (int i = 0; i < x.size(); ++i)
      mix_Scal(x(i));
  }
};

struct InstStats {
  double prod_log2 = 0.0;
  Scal rmin = 0, rmax = 0, rmed = 0, rp25 = 0, rp75 = 0;
  size_t N_live = 0;
};

static InstStats compute_stats(const ProblemInstance &inst) {
  // For small-DP, "radix" per dimension is n_j + 1 (0..n_j).
  std::vector<Scal> r;
  r.reserve(inst.N);
  for (size_t j = 0; j < static_cast<size_t>(inst.N); ++j) {
    Scal u = inst.n[j];
    r.push_back(u + 1);
  }

  std::sort(r.begin(), r.end());
  InstStats s;

  if (!r.empty()) {
    s.N_live = std::count_if(r.begin(), r.end(), [](Scal v) { return v > 1; });
    s.rmin = r.front();
    s.rmax = r.back();
    s.rmed = r[r.size() / 2];
    s.rp25 = r[(r.size() * 25) / 100];
    s.rp75 = r[(r.size() * 75) / 100];

    double sumlog2 = 0.0;
    for (Scal v : r) {
      sumlog2 += std::log2(static_cast<double>(v));
    }
    s.prod_log2 = sumlog2;
  }

  return s;
}

// Small-DP defaults

// Fill the fields that small DP uses: num_small_machines, t, avg_makespan,
// p_max.
static inline void derive_small_defaults(ProblemInstance &inst) {
  // Total work if run on a single small machine.
  Scal total_work = 0;
  for (int j = 0; j < inst.p.size(); ++j)
    total_work += inst.p[j] * inst.n[j];

  inst.num_small_machines = 1;
  inst.t.resize(inst.num_small_machines);
  inst.t[0] = total_work;

  inst.avg_makespan = static_cast<double>(total_work); // m=1
  inst.p_max = 0;
  for (int j = 0; j < inst.p.size(); ++j)
    inst.p_max = std::max(inst.p_max, inst.p[j]);
}

// Instances

std::vector<ProblemInstance> make_inst_for_bench() {
  std::vector<ProblemInstance> insts;

  auto add_inst = [&](std::initializer_list<Scal> P,
                      std::initializer_list<Scal> Nvals) {
    ProblemInstance inst;
    inst.M = 0;
    inst.M_B = 0;
    inst.N = static_cast<Scal>(P.size());

    if (P.size() != Nvals.size()) {
      throw std::runtime_error("p and n sizes do not match");
    }

    inst.p = Eigen::Map<const Eigen::Matrix<Scal, Eigen::Dynamic, 1>>(P.begin(),
                                                                      inst.N);
    inst.n = Eigen::Map<const Eigen::Matrix<Scal, Eigen::Dynamic, 1>>(
        Nvals.begin(), inst.N);

    derive_small_defaults(inst);
    insts.push_back(std::move(inst));
  };

  // BIG/M25_N100_U1_1000_001.dat style instance.

  add_inst({776, 972, 484, 301, 614, 120, 273, 479},
           {1, 3, 4, 13, 3, 3, 1, 200});

  return insts;
}

// Bench harness

// Early-exit trick for micro mode
struct EarlyExit {};

// One run of a given variant on one instance; returns (secs, emits, hash)
template <class F>
static std::tuple<double, size_t, uint64_t>
run_once(ProblemInstance &inst, size_t max_emits, F &&call_variant) {
  Hash64 H;
  size_t emits = 0;

  auto start = Clock::now();
  try {
    call_variant([&](const VectorXi &x, Scal c) {
      (void)c;
      ++emits;
      H.mix_vec(x);
      H.mix_Scal(c);
      if (max_emits && emits >= max_emits)
        throw EarlyExit{};
    });
  } catch (const EarlyExit &) {
    // intentional early stop
  }
  auto end = Clock::now();
  double secs = std::chrono::duration<double>(end - start).count();
  return {secs, emits, H.h};
}

// API adapters so we keep one shape
namespace runners {
using enumerate_small::S5;
using enumerate_small::S6;
using enumerate_small::S7;
using enumerate_small_old::S0;
using enumerate_small_old::S1;
using enumerate_small_old::S2;
using enumerate_small_old::S3;
using enumerate_small_old::S4;
} // namespace runners

static std::string compiler_str() {
#if defined(_MSC_VER)
  std::ostringstream oss;
  oss << "MSVC_" << _MSC_VER;
  return oss.str();
#elif defined(__clang__)
  std::ostringstream oss;
  oss << "Clang_" << __clang_major__ << "." << __clang_minor__;
  return oss.str();
#elif defined(__GNUC__)
  std::ostringstream oss;
  oss << "GCC_" << __GNUC__ << "." << __GNUC_MINOR__;
  return oss.str();
#else
  return "unknown";
#endif
}

// Main

int main(int argc, char **argv) {
  try {
    CLI cli = parse_cli(argc, argv);
    auto insts = make_inst_for_bench();

#ifdef NDEBUG
  const std::string build = "Release";
#else
  const std::string build = "Debug";
#endif

  const std::string comp = compiler_str();

    int inst_id = 0;
    for (auto &inst : insts) {
    const InstStats stats = compute_stats(inst);
    std::cout << "\n== Small-DP Instance S" << inst_id << " | N=" << inst.N
              << " | N_live=" << stats.N_live << " | prod_log2≈" << std::fixed
              << std::setprecision(3) << stats.prod_log2 << " ==\n";

    std::cout << "Compiler: " << comp << " | Build: " << build << "\n";

    // Warm up with representative variant S2.
    for (int w = 0; w < cli.warmup; ++w) {
      (void)run_once(inst, /*max_emits=*/0,
                     [&](auto &&emit) { runners::S2(inst, emit); });
    }

    struct VDesc {
      const char *name;
    };
    std::vector<VDesc> V = {{"S0"}, {"S1"}, {"S2"}, {"S3"},
                            {"S4"}, {"S5"}, {"S6"}, {"S7"}};

    for (auto v : V) {
      std::vector<double> times;
      times.reserve(cli.reps);
      size_t last_emits = 0;
      uint64_t last_hash = 0;

      for (int r = 0; r < cli.reps; ++r) {
        double secs = 0;
        size_t emits = 0;
        uint64_t h = 0;

        switch (v.name[1]) {
        case '0':
          std::tie(secs, emits, h) =
              run_once(inst, cli.max_emits,
                       [&](auto &&emit) { runners::S0(inst, emit); });
          break;
        case '1':
          std::tie(secs, emits, h) =
              run_once(inst, cli.max_emits,
                       [&](auto &&emit) { runners::S1(inst, emit); });
          break;
        case '2':
          std::tie(secs, emits, h) =
              run_once(inst, cli.max_emits,
                       [&](auto &&emit) { runners::S2(inst, emit); });
          break;
        case '3':
          std::tie(secs, emits, h) =
              run_once(inst, cli.max_emits,
                       [&](auto &&emit) { runners::S3(inst, emit); });
          break;
        case '4':
          std::tie(secs, emits, h) =
              run_once(inst, cli.max_emits,
                       [&](auto &&emit) { runners::S4(inst, emit); });
          break;
        case '5':
          std::tie(secs, emits, h) =
              run_once(inst, cli.max_emits,
                       [&](auto &&emit) { runners::S5(inst, emit); });
          enumerate_small::invalidate_small_dp_cache();
          break;
        case '6':
          std::tie(secs, emits, h) =
              run_once(inst, cli.max_emits,
                       [&](auto &&emit) { runners::S6(inst, emit); });
          enumerate_small::invalidate_small_dp_cache();
          break;
        case '7':
          std::tie(secs, emits, h) =
              run_once(inst, cli.max_emits,
                       [&](auto &&emit) { runners::S7(inst, emit); });
          break;
        default:
          throw std::runtime_error("Unknown small benchmark variant");
        }

        times.push_back(secs);
        last_emits = emits;
        last_hash = h;
      }

      std::sort(times.begin(), times.end());
      const double p50 = times[times.size() / 2];
      const double p90 = times[(times.size() * 9) / 10];
      const double cfgs_per_s = (p50 > 0) ? (last_emits / p50) : 0.0;

      std::cout << std::left << std::setw(4) << v.name
                << "  emits=" << last_emits << "  hash=0x" << std::hex
                << last_hash << std::dec << "  p50=" << std::fixed
                << std::setprecision(6) << p50 << "s"
                << "  p90=" << p90 << "s"
                << "  cfg/s=" << std::fixed << std::setprecision(1)
                << cfgs_per_s << "\n";

      if (cli.sleep_ms > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(cli.sleep_ms));
      }
    }

      ++inst_id;
    }
  } catch (const std::exception &e) {
    std::cerr << "bench_small_dp failed: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
