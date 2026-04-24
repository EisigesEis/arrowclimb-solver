#include "instance/types.h"
#include "io/csv_main.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <execution>
#include <iostream>
#include <limits>
#include <spdlog/spdlog.h>

// #define DBG_DER

void derive_inital(ProblemInstance &inst) {
  // max elements
  inst.s_min = inst.s(0);
  inst.s_max = inst.s(inst.s.size() - 1);
  const auto s_max = inst.s_max;
  inst.p_min = inst.p(0);
  inst.p_max = inst.p(inst.p.size() - 1);
  inst.p_max2 = inst.p_max * inst.p_max;
  inst.p_max4 = inst.p_max2 * inst.p_max2;

  // pivot element
  int idx_a = 0;
  inst.n.maxCoeff(&idx_a);
  inst.idx_a = static_cast<size_t>(idx_a);
  inst.a = inst.p[idx_a];
  inst.big_residue.init(inst.a);

  // map ops on machines
  long long total_capacity = 0;
  {
    Eigen::VectorX<long long> m_ll = inst.m.cast<long long>();
    Eigen::VectorX<long long> s_ll = inst.s.cast<long long>();
    total_capacity = m_ll.dot(s_ll);
  }

  inst.s_lcm = std::reduce(
      std::execution::par_unseq, inst.s.data() + 1,
      inst.s.data() + inst.s.size(), static_cast<long long>(inst.s(0)),
      [](long long a, int b) { return std::lcm(a, (long long)b); });
#ifdef DBG_DER
  std::cout << "s_lcm=" << inst.s_lcm << std::endl;
#endif

  // map ops on jobs
  inst.total_load = inst.n.dot(inst.p);

  // binary search bounds
  // long long lb_jobsize = (inst.p_max * inst.s_lcm + s_max - 1) / s_max;
  inst.scaled_lb = static_cast<long long>(
      std::ceil(inst.s_lcm *
                std::max(static_cast<double>(inst.total_load) / total_capacity,
                         static_cast<double>(inst.p_max) / s_max)));
  inst.scaled_ub = static_cast<long long>(
      std::ceil(inst.s_lcm * (static_cast<double>(inst.total_load) / s_max)));

  // safe gate where k cannot be inherently infeasible (guessed_capacity <
  // total_load)
  inst.k_gate =
      ((inst.total_load + inst.m.sum()) * inst.s_lcm + total_capacity - 1) /
      total_capacity;

  // OPT(I)
  inst.avg_makespan = inst.total_load * 1.0 / total_capacity;

};

void log_main_instance_fields(const ProblemInstance &inst) {
#if UNIFORMSCHED_ENABLE_CSV_LOGGING
  static constexpr std::string_view kPersistFields[] = {
      "nmax",           "mmax",        "machine_types", "job_types",
      "machines_total", "jobs_total",  "load_total",    "capacity_total",
      "avg_makespan",   "p_min",       "p_max",         "s_min",
      "s_max",          "speed_ratio",
  };
  for (const auto field : kPersistFields) {
    io::csv::main_persist(field);
  }

  const long long machines_total = inst.m.cast<long long>().sum();
  const long long jobs_total = inst.n.cast<long long>().sum();
  const long long capacity_total =
      inst.m.cast<long long>().dot(inst.s.cast<long long>());
  const long long nmax = (inst.n.size() > 0) ? inst.n.maxCoeff() : 0;
  const long long mmax = (inst.m.size() > 0) ? inst.m.maxCoeff() : 0;
  const double speed_ratio =
      (inst.s_min > 0) ? static_cast<double>(inst.s_max) / inst.s_min : 0.0;

  io::csv::main_set("nmax", nmax);
  io::csv::main_set("mmax", mmax);
  io::csv::main_set("machine_types", static_cast<long long>(inst.M));
  io::csv::main_set("job_types", static_cast<long long>(inst.N));
  io::csv::main_set("machines_total", machines_total);
  io::csv::main_set("jobs_total", jobs_total);
  io::csv::main_set("load_total", inst.total_load);
  io::csv::main_set("capacity_total", capacity_total);
  io::csv::main_set("avg_makespan", inst.avg_makespan);
  io::csv::main_set("p_min", inst.p_min);
  io::csv::main_set("p_max", inst.p_max);
  io::csv::main_set("s_min", inst.s_min);
  io::csv::main_set("s_max", inst.s_max);
  io::csv::main_set("speed_ratio", speed_ratio);
#else
  (void)inst;
#endif
}

bool derive_for_guess(ProblemInstance &inst, long long k_mid) {
#ifdef DBG_DER
  std::cout << "k_mid/inst.s_lcm = " << k_mid << "/" << inst.s_lcm << " = "
            << k_mid / inst.s_lcm << std::endl;
#endif

  // inherent infeasible check is needed below k_gate
  const bool do_cover_check = k_mid < inst.k_gate;
  bool covers = !do_cover_check;
  long long covered_sum = 0;

  // distribute guessed load
  for (int i = 0; i < inst.M; ++i) {
    inst.t[i] = k_mid * inst.s[i] / inst.s_lcm;
#ifdef DBG_DER
    std::cout << "machine k=" << i << " has s=" << inst.s[i]
              << " and guessed load t=" << inst.t[i] << std::endl;
#endif

    // C_guess = m.dot(t) must cover total_load
    if (do_cover_check && !covers) {
      if (inst.m[i] > 0 && inst.t[i] > 0) {
        const long long rem = inst.total_load - covered_sum;
        const long long need_t = (rem + inst.m[i] - 1) / inst.m[i];
        if (inst.t[i] >= need_t) {
          covers = true;
        } else {
          covered_sum += static_cast<long long>(inst.m[i]) * inst.t[i];
        }
      }
    }
  }
#ifdef DBG_DER
  std::cout << "t.transpose=" << inst.t.transpose() << std::endl;
#endif

  // split machines into small/big
  auto is_small = [&](const int t) { return t < inst.p_max4; };
  auto it_first_big =
      std::partition_point(inst.t.begin(), inst.t.end(), is_small);
  inst.num_small_machines = std::distance(inst.t.begin(), it_first_big);

#ifdef DBG_DER
  std::cout << "found " << inst.num_small_machines << " small and "
            << inst.m.size() - inst.num_small_machines << " big machines"
            << std::endl;
#endif

  // count of big machines
  inst.M_B = inst.m.tail(inst.M - inst.num_small_machines).sum();

  // big residue grouping
  inst.big_residue.init_next_iter();
  for (int i = inst.num_small_machines; i < inst.M; ++i) {
    const Scal r = inst.t[i] % inst.a;
    inst.big_residue.add(r, inst.m[i]);
#ifdef DBG_DER
    spdlog::info(
        "[big residue]: added r={} with added={} so bucket_cnt={} (size={})", r,
        inst.m[i], inst.big_residue.get_cnt_for_residue(r),
        inst.big_residue.size());
#endif
  }
  inst.big_residue.finalize();

  return !covers;
}
