#include "enumerate/big.h"
#include "enumerate_helper.h"

TEST(BigEnum, Manual) {
  ProblemInstance inst;
  inst.M = 1;
  inst.N = 3;
  inst.m = (VectorXi(1) << 1).finished();
  inst.s = (VectorXi(1) << 200).finished();
  inst.t = (VectorXi(1) << 200).finished();
  inst.num_small_machines = 0;
  inst.n = (VectorXi(3) << 2, 3, 4).finished();
  inst.p = (VectorXi(3) << 2, 3, 4).finished();

  int idx = 0;
  inst.n.maxCoeff(&idx);
  inst.idx_a = static_cast<size_t>(idx);
  inst.a = inst.p[inst.idx_a];

  size_t expected = get_hatB_width(inst);
  MatrixXi m_confs;
  m_confs.resize(inst.N, expected);
  VectorXi v_cost;
  v_cost.resize(expected);

  size_t emitted = 0;
  auto emit = [&](const VectorXi &conf, Scal cost) {
    EXPECT_TRUE(emitted < expected);

    m_confs.col(emitted) = conf;
    v_cost(emitted) = cost;

    ++emitted;
  };

  enumerate_big::optimal(inst, emit);

  EXPECT_EQ(emitted, expected);
  EXPECT_UNIQUE(m_confs);

  EXPECT_TRUE(m_confs.row(inst.idx_a).isZero());

  auto computed = m_confs.transpose() * inst.p;
  // const auto computed = (inst.p.transpose() * m_confs).transpose();
  // EXPECT_TRUE((computed - v_cost).isZero());

  for (int i = 0; i < v_cost.size(); ++i)
    EXPECT_EQ(v_cost(i), computed(i))
        << " where i=" << i << " and computed=" << m_confs.col(i)
        << ".dot(inst.p(i)) != " << " reported=" << v_cost(i);
}

TEST(BigEnum, Random) {
  ProblemInstance inst = random_instance();

  size_t expected = get_hatB_width(inst);
  MatrixXi m_confs;
  m_confs.resize(inst.N, expected);
  VectorXi v_cost;
  v_cost.resize(expected);

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

    EXPECT_TRUE(emitted < expected);

    m_confs.col(emitted) = conf;
    v_cost(emitted) = cost;

    ++emitted;
  };

  enumerate_big::optimal(inst, emit);

  EXPECT_EQ(emitted, expected);
  EXPECT_UNIQUE(m_confs);

  EXPECT_TRUE(m_confs.row(inst.idx_a).isZero());

  auto computed = m_confs.transpose() * inst.p;
  // const auto computed = (inst.p.transpose() * m_confs).transpose();
  // EXPECT_TRUE((computed - v_cost).isZero());

  for (int i = 0; i < v_cost.size(); ++i)
    EXPECT_EQ(v_cost(i), computed(i))
        << " where i=" << i << " and computed=" << m_confs.col(i)
        << ".dot(inst.p(i)) != " << " reported=" << v_cost(i);
}
