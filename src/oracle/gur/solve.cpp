#include "gurobi_c.h"
#include "model/matrix/model.h"
#include "model/matrix/packed.h"
#include "gurobi_env.h"
#include <spdlog/spdlog.h>

using std::cerr, std::endl;

// #define KEEP_RESULT
// #define DBG_GUR
// #define BENCH_SOLVE

#ifdef BENCH_SOLVE
#include <chrono>
#endif

namespace oracle::gur {

bool solve(const LPModel<PackedA> &m) {
#ifdef BENCH_SOLVE
  const auto t0 = std::chrono::system_clock::now();
#endif
  try {
    const auto &A = m.A;
    const auto nrows = A.rows();
    const auto ncols = A.cols();
    const auto nblocks = A.num_blocks();
    const auto &b = m.b;

    const auto max_choices = b.tail(b.size() - nrows).maxCoeff();
    const auto Delta = A.maxCoeff();
    // const auto Delta_n = 0 * ncols;
    // const int RHS_elem_bound = max_choices > Delta_n ? max_choices : Delta_n;

    GRBEnv &env = grb_global_env();
    GRBModel model = GRBModel(env);
    model.set(GRB_IntParam_MIPFocus, 1); // feasibility focus

// Debug logging for this instance.
#ifdef DBG_GUR
    spdlog::info("Building model with nrows = {}, ncols = {}, nblocks = {}",
                 nrows, ncols, nblocks);

    spdlog::info("Matrix A ({} x {}):", nrows, ncols);
    for (int i = 0; i < nrows; ++i) {
      std::string row_str;
      row_str.reserve(4 * ncols);
      for (int j = 0; j < ncols; ++j) {
        row_str += std::to_string(A(i, j));
        if (j + 1 < ncols)
          row_str += " ";
      }
      spdlog::info("  row {}: {}", i, row_str);
    }

    spdlog::info("RHS b (size = {}):", b.size());
    {
      std::string b_str;
      b_str.reserve(4 * b.size());
      for (int i = 0; i < static_cast<int>(b.size()); ++i) {
        b_str += std::to_string(b[i]);
        if (i + 1 < static_cast<int>(b.size()))
          b_str += " ";
      }
      spdlog::info("  b^T = {}", b_str);
    }
#endif

    // Build constraints.
    std::vector<GRBLinExpr> row_lhs(nrows);

#ifdef DBG_GUR
    for (int k = 0; k <= nblocks; ++k) {
      auto off = A.get_block_offset(k);
      spdlog::info("Block offset {}: {}", k, off);
    }
#endif

    size_t block_start = 0;
    for (int k = 0; k < nblocks; ++k) {
      const auto block_end = A.get_block_offset(k + 1);
#ifdef DBG_GUR
      spdlog::info("Block {}: columns [{} .. {}), RHS = b[{}] = {}", k,
                   block_start, block_end, nrows + k, b[nrows + k]);
#endif

      GRBLinExpr blk_expr; // sum of x_j in this block

      for (int j = static_cast<int>(block_start);
           j < static_cast<int>(block_end); ++j) {
        const auto ub = GRB_INFINITY;
        GRBVar xj = model.addVar(0.0, ub, 0.0, GRB_INTEGER);
#ifdef DBG_GUR
        spdlog::debug("  Created var x_{} with bounds [0, +inf) in block {}", j,
                      k);
#endif

#ifdef KEEP_RESULT
        x.push_back(xj);
#endif

        for (int i = 0; i < nrows; ++i) {
          const auto aij = A(i, j);
          if (aij != 0) {
#ifdef DBG_GUR
            spdlog::debug("    A({}, {}) = {} -> row {} gets {} * x_{}", i, j,
                          aij, i, aij, j);
#endif
            row_lhs[i] += static_cast<double>(aij) * xj;
          }
        }
        blk_expr += xj;
#ifdef DBG_GUR
        spdlog::debug("    x_{} contributes to block {} sum", j, k);
#endif
      }
#ifdef DBG_GUR
      spdlog::info("  Adding block constraint: sum_{{j in [{}..{})}} x_j == {}",
                   block_start, block_end, b[nrows + k]);
#endif

      model.addConstr(blk_expr == b[nrows + k]);

      block_start = block_end;
    }

    // Add row constraints A * x = b.
    for (int i = 0; i < nrows; ++i) {
#ifdef DBG_GUR
      spdlog::info("Adding row constraint {}: (A-row {} * x) == b[{}] = {}", i,
                   i, i, b[i]);
#endif
      model.addConstr(row_lhs[i] == b[i]);
    }

    model.optimize();

    auto status = model.get(GRB_IntAttr_Status);

    if (status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL) {

#ifdef BENCH_SOLVE
      const auto t1 = std::chrono::system_clock::now();
      std::cout << "gur solve took "
                << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 -
                                                                         t0)
                << std::endl;
#endif
      return true;
    }

  } catch (GRBException &e) {
    spdlog::error("Gurobi error: {}", e.getMessage());
  } catch (std::runtime_error &e) {
    spdlog::error("Runtime error: {}", e.what());
  } catch (...) {
    spdlog::error("Unknown error during optimization.");
  }

#ifdef BENCH_SOLVE
  const auto t1 = std::chrono::system_clock::now();
  std::cout << "gur solve took "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
            << std::endl;
#endif
  return false;
}

} // namespace oracle::gur
