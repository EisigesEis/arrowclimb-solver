#pragma once
#include "bench.h"
#include "instance/derive.h"
#include "instance/types.h"
#include "model/matrix/model.h"
#include "model/matrix/packed.h"
#include "oracle/ac/fft/solve.h"
#include "oracle/ac/batch/solve.h"
#include "oracle/ac/discrepancy/solve.h"
#include "oracle/ac/legacy/solve.h"
#include "oracle/ac/naive/solve.h"
#include "oracle/discrep/solve.h"
#include "oracle/gupta/solve.h"
#include "oracle/gupta/solve_batch.h"
#include "oracle/gur/solve.h"
#include "io/csv_main.h"
#include <spdlog/spdlog.h>

#ifndef BSEARCH_CHECK_DISCREP_VARIANTS
#define BSEARCH_CHECK_DISCREP_VARIANTS 0
#endif

namespace proc {

inline double bsearch_c_max(ProblemInstance &inst,
                            RunMode mode = RunMode::Main) {
  LPModel<PackedA> lpm(inst);
  io::csv::main_set("name", inst.name);

  spdlog::info(
      "Beginning makespan bsearch in scaled interval [{}, {}]  (lb={}, ub={})",
      inst.scaled_lb, inst.scaled_ub,
      static_cast<double>(inst.scaled_lb) / inst.s_lcm,
      static_cast<double>(inst.scaled_ub) / inst.s_lcm);

  while (inst.scaled_lb < inst.scaled_ub) {
    const long long k_mid =
        inst.scaled_lb + (inst.scaled_ub - inst.scaled_lb) / 2;
    const double makespan = (double)k_mid * 1.0 / inst.s_lcm;
    io::csv::main_set("C_guess", makespan);

    // prepare instance
    bool inherent_infeasible = derive_for_guess(inst, k_mid);

    if (inherent_infeasible) {
      spdlog::warn("makespan {} found inherently infeasible", makespan);
      inst.scaled_lb = k_mid + 1;
      io::csv::main_set("status", -1);
      io::csv::main_flush();

      continue;
    }

    // construct configuration matrix
    lpm.update();

    bool res = false;
    if (mode == RunMode::AcDc) {
      res = BENCH_RUN("ac_batch", oracle::ac_batch::solve(lpm));
      BENCH_RUN_CHECK("ac_batch", res, "ac_discrepancy",
                      oracle::ac_discrepancy::solve(lpm));
    } else {
      // compute trusted result
      res = BENCH_RUN("gur", oracle::gur::solve(lpm));

      BENCH_RUN_CHECK("gur", res, "ac_legacy",
                      oracle::ac_legacy::solve(lpm));
      BENCH_RUN_CHECK("gur", res, "ac_naive",
                      oracle::ac_naive::solve(lpm));
      BENCH_RUN_CHECK("gur", res, "ac_batch",
                      oracle::ac_batch::solve(lpm));
      BENCH_RUN_CHECK("gur", res, "ac_fft",
                      oracle::ac_fft::solve(lpm));
      BENCH_RUN_CHECK("gur", res, "gupta",
                      oracle::gupta::solve(lpm));
      BENCH_RUN_CHECK("gur", res, "gupta_batch",
                      oracle::gupta_batch::solve(lpm));
      BENCH_RUN_CHECK("gur", res, "discrepancy",
                      oracle::discrep::solve(lpm));
    }

    log_mem("");

    if (res) {
      spdlog::info("makespan {} found feasible", makespan);
      inst.scaled_ub = k_mid;
      io::csv::main_set("status", 1);
    } else {
      spdlog::info("makespan {} found infeasible", makespan);
      inst.scaled_lb = k_mid + 1;
      io::csv::main_set("status", 0);
    }
    io::csv::main_flush();
  }

  return (double)inst.scaled_lb * 1.0 / inst.s_lcm;
}

} // namespace proc
