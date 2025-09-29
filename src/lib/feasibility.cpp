#include "feasibility.h"
// #include "arrow_climb.h"
#include "BlockedMatrix.h"
#include "CsvLogger.h"
#include "Util.h"
#include "gurobi_c++.h"
#include <spdlog/spdlog.h>
#include <unordered_set>

#include "arrow_climb.h"
#include "Config.h"
#include "gurobi_env.h"

bool solve_direct(const int Delta_global, const BlockedMatrix &BM,
                  const VectorXi &b) {
  try {
    const auto &A = BM.mat_;
    const auto &r = A.rows();
    const auto &n = A.cols();

    const auto &max_choices = b.tail(b.size() - r).maxCoeff();
    const int RHS_elem_bound = max_choices > Delta_global * n ? max_choices : Delta_global * n;

    // Gurobi environment and model
    GRBEnv &env = grb_global_env();
    GRBModel model = GRBModel(env);
    model.set(GRB_IntParam_MIPFocus, 1); // feasibility focus
    // model.set(GRB_DoubleParam_IntFeasTol, 1e-9);      // integer feasibility tolerance
    // model.set(GRB_DoubleParam_FeasibilityTol, 1e-9);  // constraint feasibility tolerance
    // model.set(GRB_DoubleParam_OptimalityTol, 1e-9);   // reduced cost / optimality
    // model.set(GRB_DoubleParam_MIPGap, 0.0);           // require exact optimality
    // model.set(GRB_IntParam_NumericFocus, 3);          // most conservative numeric handling

    // disable cut generation (TODO: Test if this is worth)
    // model.set(GRB_IntParam_Cuts, 0);
    // model.set(GRB_IntParam_GomoryPasses, 0);

    // add variables: x \in \N^n
    std::vector<GRBVar> x(n);
    for (int j = 0; j < n; ++j) {
      x[j] = model.addVar(0.0, RHS_elem_bound, 0.0, GRB_INTEGER);
    }

    // add r constraints: A * x = b
    std::vector<GRBConstr> constrs(r + BM.num_blocks());
    for (int i = 0; i < r; ++i) {
      GRBLinExpr lhs = 0;
      for (int j = 0; j < n; ++j) {
        if (A(i, j) != 0)
          lhs += A(i, j) * x[j];
      }
      constrs[i] = model.addConstr(lhs == b[i]);
    }

    // add choice constraints
    int offset = 0;
    for (int k = 0; k < BM.num_blocks(); ++k) {
      GRBLinExpr num_choices_blk = 0;
      const auto &Ak = BM.get_block(k);
      const auto &n_k = Ak.cols();

      for (int j = 0; j < n_k; ++j) {
        num_choices_blk += x[offset + j];
      }
      offset += n_k;

      constrs[r + k] = model.addConstr(num_choices_blk == b(r + k));
    }

    model.optimize();

    if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
      return true;
    }

  } catch (GRBException &e) {
    spdlog::error("Gurobi error: {}", e.getMessage());
    cerr << "Gurobi error: " << e.getMessage() << endl;
  } catch (...) {
    spdlog::error("Unknown error during optimization.");
    cerr << "Unknown error during optimization." << endl;
  }
  return false;
}

bool solve_arrow_climb(
    const int Delta_global, const BlockedMatrix &BM,
    const VectorXi &b_full, const BaseTableFn &bt) {
  if (BM.num_blocks() == 0) {
    return (b_full.array() == 0).all();
  }

  const auto &A = BM.mat_;
  const auto &r = A.rows();
  const auto &n = BM.num_blocks();
  // if (!(BM.num_blocks() == (b_full.size() - BM.mat_.rows()))) {
  //   std::cerr << "BM.num_blocks()=" << BM.num_blocks() << " != " << (b_full.size() - BM.mat_.rows()) << "=(b_full.size() - BM.mat_.rows())" << std::endl;
  // }
  assert(BM.num_blocks() == (b_full.size() - BM.mat_.rows()) && "Block rows must equal r");
  const auto &h = A.cols();

  const auto &b_up = b_full.head(r);
  const auto &b_down = b_full.tail(BM.num_blocks());
  // spdlog::debug("b_up={} b_down={}", b_up.transpose(), b_down.transpose());
  assert(b_full.size() == r + n && "b_full = [b_up(r) | b_down(n)]");

  // 1) Determine global parameters
  const int K = max(1, int(floor(2.0 * (r + 1) * log2(4.0 * (r + 1) * double(Delta_global)))));
  const size_t D = n * K * Delta_global;
  const auto b_down_max = (n > 0) ? b_down.maxCoeff() : 0;
  const int I = max(1, int(ceil(log2(double(b_down_max + K) / (2*K + 1)))) + 1);
  assert(K >= 1 && I >= 1 && D >= 1);

  // 2) Direct solve on base case
  spdlog::debug("Computed params: K={} Delta={} I={}", K, Delta_global, I);
  csv_logger.seti("K", K);
  csv_logger.seti("I", I);
  csv_logger.seti("D", D);

#ifdef AC_USE_DIRECT_SOLVE
  if (I == 1) {
    spdlog::debug("Calling solve direct.");
    return solve_direct(Delta_global, BM, b_full);
  }
#endif

  // 3) Alg.2 get_num_machines to obtain problem split
  MatrixXi b_down_tilde = MatrixXi::Zero(BM.num_blocks(), I);
#ifdef AC_DEBUG
  std::cerr << "b_down=" << b_down.transpose() << endl;
#endif
  get_num_machines(I, BM.num_blocks(), K, b_down, b_down_tilde);
#ifdef AC_DEBUG
  std::cerr << "b_down_tilde=" << b_down_tilde.transpose() << endl;
#endif

  return arrow_climb_combine(BM, K, D, b_up, b_down_tilde);
}