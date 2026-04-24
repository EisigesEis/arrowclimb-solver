#include "enumerate_helper.h"
#include "enumerate/small.h"

// dp(j, L) = #ways to use items j..N-1 to get exact load L
static inline Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>
build_suffix_dp_counts_S0(const VectorXi &p, const VectorXi &u, int C) {
  const int N = static_cast<int>(p.size());
  Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dp(N + 1,
                                                                         C + 1);
  dp.setZero();
  dp(N, 0) = 1; // empty suffix achieves load 0 in exactly 1 way

  for (int j = N - 1; j >= 0; --j) {
    const int w = p(j);
    const int cap = u(j);
    for (int L = 0; L <= C; ++L) {
      i64 sum = 0;
      const int max_x = std::min(cap, L / w); // assume w > 0
      for (int x = 0; x <= max_x; ++x) {
        sum += dp(j + 1, L - x * w);
      }
      dp(j, L) = sum;
    }
  }
  return dp;
}

static inline Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>
build_suffix_dp_counts_le_S0(const VectorXi &p, const VectorXi &u, int C) {
  auto dp = build_suffix_dp_counts_S0(p, u, C);
  const int N = static_cast<int>(p.size());
  for (int j = 0; j <= N; ++j) {
    i64 running = 0;
    for (int L = 0; L <= C; ++L) {
      running += dp(j, L);
      dp(j, L) = running;
    }
  }
  return dp;
}

TEST(SmallEnum, Manual) {
  ProblemInstance inst;
  inst.M = 1;
  inst.N = 3;
  inst.m = (VectorXi(1) << 1).finished();
  inst.s = (VectorXi(1) << 8).finished();
  inst.t = (VectorXi(1) << 8).finished();
  inst.num_small_machines = 1;
  inst.n = (VectorXi(3) << 2, 3, 4).finished();
  inst.p = (VectorXi(3) << 2, 3, 4).finished();
  inst.avg_makespan = (inst.n.dot(inst.p)) * 1.0 / (inst.m.dot(inst.s));
  inst.p_max = inst.p.maxCoeff();

  int idx = 0;
  inst.n.maxCoeff(&idx);
  inst.idx_a = static_cast<size_t>(idx);
  inst.a = inst.p[inst.idx_a];

  size_t expected_max = 13;
  MatrixXi m_confs;
  m_confs.resize(inst.N, expected_max);
  VectorXi v_cost;
  v_cost.resize(expected_max);

  size_t emitted = 0;
  auto emit = [&](const VectorXi &conf, Scal cost) {
    // Print immediately when emitted
    std::cout << "conf = [";
    for (int i = 0; i < conf.size(); ++i) {
      std::cout << conf(i);
      if (i + 1 != conf.size())
        std::cout << ", ";
    }
    std::cout << "], cost = " << cost << "\n";

    EXPECT_TRUE(emitted < expected_max);

    // account results
    m_confs.col(emitted) = conf;
    v_cost(emitted) = cost;

    // count results
    ++emitted;
  };

  enumerate_small::optimal(inst, emit);

  EXPECT_LE(emitted, expected_max);

  Eigen::VectorXi u = inst.n;
  const auto C = std::min(inst.t[inst.num_small_machines - 1],
                          (int)inst.avg_makespan + inst.p_max);

  EXPECT_UNIQUE(m_confs, emitted);

  auto computed = m_confs.leftCols(emitted).transpose() * inst.p;

  for (int i = 0; i < emitted; ++i) {
    EXPECT_EQ(v_cost(i), computed(i))
        << " where i=" << i << " and computed=" << m_confs.col(i)
        << ".dot(inst.p(i)) != " << " reported=" << v_cost(i);
    EXPECT_LE(v_cost(i), C);
  }
}
