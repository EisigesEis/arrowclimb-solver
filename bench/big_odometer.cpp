// bench/bench_big_odometer.cpp
#include "big_old.h"
#include "enumerate/big.h"
#include "instance/types.h"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

using Scal = int;
using Clock = std::chrono::steady_clock;

// instances

static inline void derive_big_defaults(ProblemInstance &inst) {
  inst.sort();
  int idx = 0;
  inst.n.maxCoeff(&idx);
  inst.idx_a = static_cast<size_t>(idx);
  inst.a = inst.p[inst.idx_a];
}

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
    derive_big_defaults(inst);
    insts.push_back(std::move(inst));
  };

  // I0 - tiny sanity
  /*
  B0    emits=1600  p50=0.000039s  p90=0.000047s  cfg/s=41343669.3
  B1    emits=1600  p50=0.000026s  p90=0.000027s  cfg/s=62015503.9
  B2    emits=1600  p50=0.000033s  p90=0.000141s  cfg/s=48929663.6
  B3    emits=1600  p50=0.000160s  p90=0.001980s  cfg/s=9981285.1

  B1 wins by large margin
  */
  // add_inst({1, 10, 40}, {50, 50, 60});

  // I1 - skinny radices (many small/fixed caps)
  /*
  == Instance I1 | N=16 | N_live=15 | a=2 | idx_a=0 | prod_log2Ôëê15.000 ==
  B0    emits=32768  p50=0.001025s  p90=0.003915s  cfg/s=31978139.9
  B1    emits=32768  p50=0.000983s  p90=0.001880s  cfg/s=33324519.5
  B2    emits=32768  p50=0.001313s  p90=0.003392s  cfg/s=24948987.4
  B3    emits=32768  p50=0.001251s  p90=0.002755s  cfg/s=26191351.6

  B1 wins
  */
  // add_inst({4, 9, 6, 7, 3, 8, 5, 2, 11, 13, 17, 19, 23, 29, 31, 37},
  //          {1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3});

  // I2 - alternating very large / very small caps
  /*
  == Instance I2 | N=6 | N_live=5 | a=2 | idx_a=0 | prod_log2Ôëê5.000 ==
  B0    emits=32  p50=0.000001s  p90=0.000001s  cfg/s=40000000.0
  B1    emits=32  p50=0.000001s  p90=0.000002s  cfg/s=45714285.7
  B2    emits=32  p50=0.000002s  p90=0.000013s  cfg/s=20000000.0
  B3    emits=32  p50=0.000002s  p90=0.000003s  cfg/s=20000000.0

  B1 wins
  */
  // add_inst({5, 14, 2, 21, 3, 34},//, 4, 55, 6, 89, 7, 144},
  //          {30, 1, 30, 2, 30, 3});//, 60, 4, 60, 5, 60, 6});

  // I3 - equal radices (tie-break must rely on p ascending), deterministic
  // order
  /*
  == Instance I2 | N=6 | N_live=5 | a=2 | idx_a=0 | prod_log2Ôëê5.000 ==
  B0    emits=32  p50=0.000001s  p90=0.000001s  cfg/s=40000000.0
  B1    emits=32  p50=0.000001s  p90=0.000002s  cfg/s=45714285.7
  B2    emits=32  p50=0.000002s  p90=0.000013s  cfg/s=20000000.0
  B3    emits=32  p50=0.000002s  p90=0.000003s  cfg/s=20000000.0

  B0, B1 wins
  */
  // add_inst({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
  //          {63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63});

  // I4 - equal weights = 1 (residue edge; cost = sum digits)
  /*
  == Instance I4 | N=14 | N_live=0 | a=1 | idx_a=12 | prod_log2Ôëê0.000 ==
  B0    emits=1  p50=0.000000s  p90=0.000013s  cfg/s=2000000.0
  B1    emits=1  p50=0.000000s  p90=0.000001s  cfg/s=5000000.0
  B2    emits=1  p50=0.000001s  p90=0.000003s  cfg/s=1111111.1
  B3    emits=1  p50=0.000002s  p90=0.000005s  cfg/s=588235.3

  B1 wins
  */
  // add_inst({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  //          {8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3});

  // I5 - mid-size mixed N=32 (radices and weights interleaved)
  /*
  == Instance I5 | N=16 | N_live=15 | a=7 | idx_a=6 | prod_log2Ôëê23.107 ==
  B0    emits=9031680  p50=0.249802s  p90=0.320827s  cfg/s=36155427.4
  B1    emits=9031680  p50=0.264846s  p90=0.286666s  cfg/s=34101693.3
  B2    emits=9031680  p50=0.282351s  p90=0.306797s  cfg/s=31987442.6
  B3    emits=9031680  p50=0.275157s  p90=0.290327s  cfg/s=32823685.6

  B0 wins
  */
  // {
  //   ProblemInstance inst;
  //   inst.p = {3, 5,  7,  11, 4, 6,  9,  12, 5,  10, 15, 20, 7,  14, 21,
  //   28};//,
  //             //8, 16, 24, 32, 9, 18, 27, 36, 11, 22, 33, 44, 13, 26, 39,
  //             52};
  //   inst.N = inst.p.size();
  //   inst.n = {4, 1, 1, 2, 7, 3, 1, 1, 8, 2, 1, 1, 9, 3, 1, 1};//,
  //           //  6, 2, 1, 0, 5, 1, 0, 0, 7, 2, 1, 0, 8, 2, 1, 0};
  //   derive_big_defaults(inst);
  //   insts.push_back(std::move(inst));
  // }

  // I6 - big p, low n. (BIG/M25_N100_U1_1000_001.dat)
  /*
  == Instance I6 | N=8 | N_live=7 | a=479 | idx_a=3 | prod_log2Ôëê25.425 ==
  B0    emits=45064320  p50=0.652577s  p90=0.691459s  cfg/s=69055995.7
  B1    emits=45064320  p50=0.629313s  p90=0.654648s  cfg/s=71608731.6
  B2    emits=45064320  p50=0.721372s  p90=0.778277s  cfg/s=62470275.4
  B3    emits=45064320  p50=0.728976s  p90=0.762209s  cfg/s=61818669.2

  B1 wins
  */
  // add_inst({776, 972, 484, 301, 614, 120, 273, 479}, {41, 3, 4, 13, 3, 3,
  // 1});

  add_inst({776, 972, 484, 301, 614, 120, 273, 479},
           {1, 3, 4, 13, 3, 3, 1, 200});

  return insts;
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

  template <class Vec> inline void mix_vec(const Vec &x) {
    const auto n = static_cast<int>(x.size());
    for (int i = 0; i < n; ++i)
      mix_Scal(static_cast<Scal>(x[i]));
  }
};

struct InstStats {
  double prod_log2 = 0.0;
  Scal rmin = 0, rmax = 0, rmed = 0, rp25 = 0, rp75 = 0;
  size_t N_live = 0;
  size_t expected_emits_fit = 0; // 0 if overflow
};

static InstStats compute_stats(const ProblemInstance &inst) {
  std::vector<Scal> r;
  r.reserve(inst.N);
  r.clear();
  for (size_t j = 0; j < inst.N; ++j) {
    Scal u = (j == inst.idx_a) ? 0 : std::min<Scal>(inst.a - 1, inst.n[j]);
    r.push_back(u + 1);
  }
  std::sort(r.begin(), r.end());
  InstStats s;
  s.N_live = std::count_if(r.begin(), r.end(), [](Scal v) { return v > 1; });
  s.rmin = r.front();
  s.rmax = r.back();
  s.rmed = r[r.size() / 2];
  s.rp25 = r[(r.size() * 25) / 100];
  s.rp75 = r[(r.size() * 75) / 100];
  double sumlog2 = 0.0;
  for (Scal v : r)
    sumlog2 += std::log2((double)v);
  s.prod_log2 = sumlog2;

  // Compute exact expected emits when the product fits in uint64_t.
  uint64_t prod = 1;
  bool overflow = false;
  for (Scal v : r) {
    if (v == 0)
      continue;
    uint64_t u = static_cast<uint64_t>(v);
    // overflow check: prod * u > max => prod > max / u
    if (u != 0 && prod > std::numeric_limits<uint64_t>::max() / u) {
      overflow = true;
      break;
    }
    prod *= u;
  }
  s.expected_emits_fit = overflow ? 0 : static_cast<size_t>(prod);
  return s;
}

// CLI

struct CLI {
  int reps = 200;
  int warmup = 1;
  int sleep_ms = 0;
  size_t max_emits = 0;     // 0 = no cap
  std::string out_csv = ""; // empty = no csv
};

static CLI parse_cli(int argc, char **argv) {
  CLI c;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](int &i) -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
    };
    if (a == "--reps")
      c.reps = std::stoi(next(i));
    else if (a == "--warmup")
      c.warmup = std::stoi(next(i));
    else if (a == "--sleep")
      c.sleep_ms = std::max(0, std::stoi(next(i)));
    else if (a == "--max-emits")
      c.max_emits = (size_t)std::stoull(next(i));
    else if (a == "--csv")
      c.out_csv = next(i);
  }
  return c;
}

// CSV logger

struct CsvLogger {
  std::ofstream f;
  bool header_written = false;
  explicit CsvLogger(const std::string &path) : f(path, std::ios::app) {}
  void write_header_if_needed() {
    if (header_written || !f)
      return;
    f << "ts,compiler,build,inst_id,N,N_live,a,idx_a,prod_log2,rmin,rp25,rmed,"
         "rp75,rmax,"
         "variant,rep,emits,enum_ms,hash64,expected_fit\n";
    header_written = true;
  }
  void row(const std::string &ts, const std::string &comp,
           const std::string &build, int inst_id, size_t N, size_t N_live,
           Scal a, size_t idx_a, double prod_log2, Scal rmin, Scal rp25,
           Scal rmed, Scal rp75, Scal rmax, const std::string &variant, int rep,
           size_t emits, double enum_ms, uint64_t hash64, size_t expected_fit) {
    if (!f)
      return;
    f << ts << ',' << comp << ',' << build << ',' << inst_id << ',' << N << ','
      << N_live << ',' << a << ',' << idx_a << ',' << std::fixed
      << std::setprecision(3) << prod_log2 << ',' << rmin << ',' << rp25 << ','
      << rmed << ',' << rp75 << ',' << rmax << ',' << variant << ',' << rep
      << ',' << emits << ',' << std::fixed << std::setprecision(6) << enum_ms
      << ',' << std::hex << hash64 << std::dec << ',' << expected_fit << '\n';
  }
};

// Bench harness

struct EarlyExit {};

template <class F>
static std::tuple<double, size_t, uint64_t>
run_once(ProblemInstance &inst, size_t max_emits, F &&call_variant) {
  Hash64 H;
  size_t emits = 0;

  auto start = Clock::now();
  try {
    call_variant([&](const auto &x, Scal c) {
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
template <class Emit> void B0(ProblemInstance &inst, Emit &&emit) {
  enumerate_big::B0(inst, std::forward<Emit>(emit));
}
template <class Emit> void B1(ProblemInstance &inst, Emit &&emit) {
  enumerate_big::B1(inst, std::forward<Emit>(emit));
}
template <class Emit> void B2(ProblemInstance &inst, Emit &&emit) {
  enumerate_big::B2(inst, std::forward<Emit>(emit));
}
template <class Emit> void B3(ProblemInstance &inst, Emit &&emit) {
  enumerate_big::B3(inst, std::forward<Emit>(emit));
}
template <class Emit> void B1_opt(ProblemInstance &inst, Emit &&emit) {
  enumerate_big::B1_opt(inst, std::forward<Emit>(emit));
}
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

int main(int argc, char **argv) {
  try {
    CLI cli = parse_cli(argc, argv);
    auto insts = make_inst_for_bench();

    CsvLogger csv(cli.out_csv);
    if (!cli.out_csv.empty())
      csv.write_header_if_needed();

    const std::string comp = compiler_str();
#ifdef NDEBUG
    const std::string build = "Release";
#else
    const std::string build = "Debug";
#endif

    auto now_ts = [] {
      auto t = std::chrono::system_clock::now();
      std::time_t tt = std::chrono::system_clock::to_time_t(t);
      std::tm tm{};
#if defined(_WIN32)
      localtime_s(&tm, &tt);
#else
      localtime_r(&tt, &tm);
#endif
      char buf[32];
      std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
      return std::string(buf);
    };

    int inst_id = 0;
    for (auto &inst : insts) {
      const InstStats stats = compute_stats(inst);
      std::cout << "\n== Instance I" << inst_id << " | N=" << inst.N
                << " | N_live=" << stats.N_live << " | a=" << inst.a
                << " | idx_a=" << inst.idx_a << " | prod_log2≈" << std::fixed
                << std::setprecision(3) << stats.prod_log2 << " ==\n";

      // Warmups
      for (int w = 0; w < cli.warmup; ++w) {
        (void)run_once(inst, /*max_emits=*/0,
                       [&](auto &&emit) { runners::B2(inst, emit); });
      }

      struct VDesc {
        const char *name;
      };
      std::vector<VDesc> V = {{"B0"}, {"B1"}, {"B2"}, {"B3"}, {"B1_opt"}};

      for (auto v : V) {
        std::vector<double> times;
        times.reserve(cli.reps);
        size_t last_emits = 0;
        uint64_t last_hash = 0;

        for (int r = 0; r < cli.reps; ++r) {
          double secs = 0;
          size_t emits = 0;
          uint64_t h = 0;
          const std::string_view variant(v.name);
          if (variant == "B0") {
            std::tie(secs, emits, h) =
                run_once(inst, cli.max_emits,
                         [&](auto &&emit) { runners::B0(inst, emit); });
          } else if (variant == "B1") {
            std::tie(secs, emits, h) =
                run_once(inst, cli.max_emits,
                         [&](auto &&emit) { runners::B1(inst, emit); });
          } else if (variant == "B2") {
            std::tie(secs, emits, h) =
                run_once(inst, cli.max_emits,
                         [&](auto &&emit) { runners::B2(inst, emit); });
          } else if (variant == "B3") {
            std::tie(secs, emits, h) =
                run_once(inst, cli.max_emits,
                         [&](auto &&emit) { runners::B3(inst, emit); });
          } else {
            std::tie(secs, emits, h) =
                run_once(inst, cli.max_emits,
                         [&](auto &&emit) { runners::B1_opt(inst, emit); });
          }
          times.push_back(secs);
          last_emits = emits;
          last_hash = h;

          if (!cli.out_csv.empty()) {
            csv.row(now_ts(), comp, build, inst_id, inst.N, stats.N_live,
                    inst.a, inst.idx_a, stats.prod_log2, stats.rmin, stats.rp25,
                    stats.rmed, stats.rp75, stats.rmax, v.name, r, emits,
                    secs * 1000.0, h, stats.expected_emits_fit);
          }
        }

        std::sort(times.begin(), times.end());
        const double p50 = times[times.size() / 2];
        const double p90 = times[(times.size() * 9) / 10];
        const double cfgs_per_s = (p50 > 0) ? (last_emits / p50) : 0.0;

        std::cout << std::left << std::setw(4) << v.name
                  << "  emits=" << last_emits << "  p50=" << std::fixed
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
    spdlog::error("Failed: {}", e.what());
  } catch (...) {
    spdlog::error("Failed: unknown exception");
  }

  return 0;
}
