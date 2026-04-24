#include "instance/types.h"
#include "enumerate/small.h"
#include "small_old.h"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

using Eigen::MatrixXi, Eigen::VectorXi;
using Scal = int;

using enumerate_small::build_suffix_dp_exists_le_optimized;
using enumerate_small::extend_suffix_dp_exists_le_optimized; // NEW

void bench_dp(const VectorXi &p, const VectorXi &u, int C, int iters) {
  using clock = std::chrono::steady_clock;
  std::vector<double> times;
  times.reserve(iters);

  for (int i = 0; i < iters; ++i) {
    auto t0 = clock::now();
    auto dp = build_suffix_dp_exists_le_optimized(p, u, C);
    auto t1 = clock::now();
    times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  std::sort(times.begin(), times.end());
  double p50 = times[times.size() / 2];
  double p90 = times[(times.size() * 9) / 10];
  std::cerr << "build_suffix: p50=" << p50 << "us, p90=" << p90 << "us\n";
}

// Micro-benchmark for extend_suffix_dp_exists_le_optimized.
// We build a base DP up to C_base once, then in each iteration copy it and
// measure only the extension C_base -> C.
void bench_dp_extend(const VectorXi &p, const VectorXi &u, int C_base, int C,
                     int iters) {
  using clock = std::chrono::steady_clock;

  using MatB = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>;

  // Build base DP once (not in the timed section).
  MatB base = build_suffix_dp_exists_le_optimized(p, u, C_base);

  std::vector<double> times;
  times.reserve(iters);

  for (int i = 0; i < iters; ++i) {
    MatB dp = base; // Copy cost is included; for a pure extend cost,
                    // you could benchmark this separately and subtract.
    auto t0 = clock::now();
    extend_suffix_dp_exists_le_optimized(dp, p, u, C_base, C);
    auto t1 = clock::now();
    times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  std::sort(times.begin(), times.end());
  double p50 = times[times.size() / 2];
  double p90 = times[(times.size() * 9) / 10];
  std::cerr << "extend_suffix (from C_base=" << C_base << " to C=" << C
            << "): p50=" << p50 << "us, p90=" << p90 << "us\n";
}

static inline void derive_small_defaults(ProblemInstance &inst) {
  // Total work if run on a single small machine.
  Scal total_work = inst.p.dot(inst.n);

  inst.num_small_machines = 1;
  inst.t.resize(inst.num_small_machines);
  inst.t[0] = total_work / 2;

  inst.avg_makespan = static_cast<double>(total_work); // m=1
  inst.p_max = 0;
  for (int j = 0; j < inst.p.size(); ++j)
    inst.p_max = std::max(inst.p_max, inst.p[j]);
}

std::vector<ProblemInstance> make_inst_for_bench() {
  std::vector<ProblemInstance> insts;

  auto add_inst = [&](std::initializer_list<Scal> P,
                      std::initializer_list<Scal> Nvals) {
    ProblemInstance inst;
    inst.N = static_cast<Scal>(P.size());

    if (P.size() != Nvals.size()) {
      std::cout << "p and n sizes don't match: p.size()=" << P.size()
                << " != " << Nvals.size() << "=n.size()" << std::endl;
      std::exit(1);
    }

    inst.p = Eigen::Map<const Eigen::Matrix<Scal, Eigen::Dynamic, 1>>(P.begin(),
                                                                      inst.N);
    inst.n = Eigen::Map<const Eigen::Matrix<Scal, Eigen::Dynamic, 1>>(
        Nvals.begin(), inst.N);

    derive_small_defaults(inst);
    insts.push_back(std::move(inst));
  };

  add_inst({776, 972, 484, 301, 614, 120, 273, 479},
           {1, 3, 4, 13, 3, 3, 1, 200});

  return insts;
}

int main(int argc, char **argv) {
  // Number of repetitions per (p, u, C) triple
  int iters = 1000;
  if (argc > 1) {
    iters = std::max(1, std::atoi(argv[1]));
  }

  auto insts = make_inst_for_bench();
  if (insts.empty()) {
    std::cerr << "No benchmark instances constructed.\n";
    return 1;
  }

  std::cout << "Running DP micro-benchmark with iters=" << iters << "\n";

  for (std::size_t idx = 0; idx < insts.size(); ++idx) {
    const ProblemInstance &inst = insts[idx];

    const VectorXi &p = inst.p;
    const VectorXi &u = inst.n;

    int C = static_cast<int>(inst.t[0]);

    std::cout << "Instance #" << idx << "  N=" << inst.N << "  C=" << C
              << "  iters=" << iters << "\n";

    bench_dp(p, u, C, iters);

    // Choose a base capacity to extend from, e.g. half the final C.
    int C_base = std::max(1, C / 2);
    bench_dp_extend(p, u, C_base, C, iters);
  }

  return 0;
}
