#include <gtest/gtest.h>

#include "instance/derive.h"
#include "instance/parse.h"
#include "model/matrix/model.h"
#include "model/matrix/packed.h"
#include "test_instances.h"
#include "test_paths.h"

#include <filesystem>
#include <optional>

namespace {

std::optional<std::filesystem::path> find_big_machine_instance() {
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(
           test_support::test_repo_root() / "instances")) {
    if (!entry.is_regular_file() || entry.path().extension() != ".dat")
      continue;

    const ParseResult parsed = parse_file(entry.path());
    if (!parsed.ok)
      continue;

    ProblemInstance inst = parsed.inst;
    const long long k_mid =
        inst.scaled_lb + (inst.scaled_ub - inst.scaled_lb) / 2;
    if (derive_for_guess(inst, k_mid))
      continue;

    LPModel<PackedA> lpm(inst);
    lpm.update();
    if (lpm.A.num_big_blocks() > 0)
      return entry.path();
  }
  return std::nullopt;
}

int compute_num_slack(const ProblemInstance &inst, int ell) {
  int d_idx_a = 0;
  int sum_ceils = 0;
  for (int j = 0; j < inst.n.size(); ++j) {
    int h_j = 0;
    for (int k = 0; k < inst.num_small_machines; ++k) {
      h_j += inst.m[k] * (inst.t[k] / inst.p[j]);
    }
    h_j += (j == inst.idx_a) ? 0 : (inst.a - 1) * inst.M_B;

    const int d_j = std::max(0, inst.n[j] - h_j);
    if (j == inst.idx_a) {
      d_idx_a = d_j;
    } else {
      sum_ceils += (d_j + inst.a - 1) / inst.a;
    }
  }

  const int dummy_bundles = (ell + inst.a - 1) / inst.a;
  return d_idx_a + std::max(dummy_bundles, sum_ceils);
}

TEST(Rhs, SmallOnly) {
  ProblemInstance inst =
      test_support::make_midpoint_instance(test_support::test_repo_root() /
                                           "instances" / "single_small.dat");
  LPModel<PackedA> lpm(inst);
  lpm.update();

  ASSERT_EQ(lpm.A.num_big_blocks(), 0U);
  ASSERT_FALSE(lpm.A.has_slack_block());
  ASSERT_EQ(lpm.b.size(),
            static_cast<Eigen::Index>(lpm.A.rows() + lpm.A.num_blocks()));
  ASSERT_EQ(lpm.A.num_blocks(), static_cast<std::size_t>(inst.num_small_machines));
  EXPECT_TRUE(lpm.b.head(inst.n.size()).isApprox(inst.n));
  EXPECT_TRUE(lpm.b.tail(lpm.A.num_blocks())
                  .isApprox(inst.m.head(inst.num_small_machines)));
}

TEST(Rhs, Mixed) {
  const auto instance_path = find_big_machine_instance();
  ASSERT_TRUE(instance_path.has_value()) << "No midpoint instance with big blocks found.";

  ProblemInstance inst = test_support::make_midpoint_instance(*instance_path);
  LPModel<PackedA> lpm(inst);
  lpm.update();

  ASSERT_GT(lpm.A.num_big_blocks(), 0U) << instance_path->string();
  ASSERT_TRUE(lpm.A.has_slack_block()) << instance_path->string();
  ASSERT_EQ(lpm.b.size(),
            static_cast<Eigen::Index>(lpm.A.rows() + lpm.A.num_blocks()));

  const Eigen::Index ell_offset = inst.p.size();
  const Eigen::Index small_offset = ell_offset + 1;
  const Eigen::Index big_offset = small_offset + inst.num_small_machines;
  const Eigen::Index caln_offset = big_offset + inst.big_residue.size();

  const int ell = inst.t.dot(inst.m) - inst.total_load;
  const int cal_n = compute_num_slack(inst, ell);
  ASSERT_GE(ell, 0) << instance_path->string();

  EXPECT_TRUE(lpm.b.head(inst.n.size()).isApprox(inst.n));
  EXPECT_EQ(lpm.b[ell_offset], ell);
  EXPECT_TRUE(lpm.b.segment(small_offset, inst.num_small_machines)
                  .isApprox(inst.m.head(inst.num_small_machines)));
  for (int i = 0; i < inst.big_residue.size(); ++i) {
    EXPECT_EQ(lpm.b[big_offset + i], inst.big_residue.get_cnt_for_block(i));
  }
  EXPECT_EQ(lpm.b[caln_offset], cal_n);
}

} // namespace
