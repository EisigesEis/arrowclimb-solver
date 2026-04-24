#include <gtest/gtest.h>

#include "model/matrix/model.h"
#include "model/matrix/packed.h"
#include "oracle/discrep/solve.h"
#include "oracle/gupta/solve_batch.h"
#include "test_instances.h"
#include "test_paths.h"

#include <filesystem>
#include <vector>

TEST(Discrepancy, MatchesGupta) {
  const std::vector<std::filesystem::path> inputs = {
      test_support::test_repo_root() / "instances" / "single_small.dat",
      test_support::test_repo_root() / "instances" / "E1" /
          "M3_N15_U1_20_001.dat",
  };

  for (const auto &input : inputs) {
    ProblemInstance inst = test_support::make_midpoint_instance(input);
    LPModel<PackedA> lpm(inst);
    lpm.update();

    EXPECT_EQ(oracle::discrep::solve(lpm), oracle::gupta_batch::solve(lpm))
        << input.string();

    ASSERT_GT(lpm.b.size(), 0);
    lpm.b[0] += 100000;
    EXPECT_EQ(oracle::discrep::solve(lpm), oracle::gupta_batch::solve(lpm))
        << input.string();
  }
}
