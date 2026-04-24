#include <gtest/gtest.h>

#include "bench.h"
#include "io/csv_main.h"
#include "model/matrix/model.h"
#include "model/matrix/packed.h"
#include "oracle/ac/batch/solve.h"
#include "oracle/ac/discrepancy/bt_enumerator.h"
#include "oracle/ac/discrepancy/solve.h"
#include "oracle/discrep/detail/fft_solver_core.h"
#include "oracle/discrep/solve.h"
#include "run.h"
#include "test_csv.h"
#include "test_files.h"
#include "test_instances.h"
#include "test_paths.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string_view>
#include <utility>

namespace {

int compute_ac_K(int r, int Delta) {
  const int normalized_delta = std::max(1, Delta);
  return std::max(
      1,
      int(std::floor(2.0 * (r + 1) *
                     std::log2(4.0 * (r + 1) * double(normalized_delta)))));
}

struct HugeEmbeddingSingleBlock {
  std::size_t rows() const { return 4; }
  std::size_t cols() const { return 1; }
  std::size_t num_blocks() const { return 1; }
  int block_maxCoeff(std::size_t) const { return 1000; }
  int maxCoeff() const { return 1000; }
  std::pair<std::size_t, std::size_t> block_col_range(std::size_t k) const {
    return k == 0 ? std::pair<std::size_t, std::size_t>{0, 1}
                  : std::pair<std::size_t, std::size_t>{1, 1};
  }
  int operator()(std::size_t, std::size_t) const { return 1000; }
};

} // namespace

TEST(AcDc, MatchesBatch) {
  ProblemInstance inst =
      test_support::make_midpoint_instance(test_support::test_repo_root() /
                                           "instances" / "single_small.dat");
  LPModel<PackedA> lpm(inst);
  lpm.update();

  EXPECT_EQ(oracle::ac_discrepancy::solve(lpm), oracle::ac_batch::solve(lpm));

  lpm.b[0] += 100000;
  EXPECT_FALSE(oracle::ac_batch::solve(lpm));
  EXPECT_EQ(oracle::ac_discrepancy::solve(lpm), oracle::ac_batch::solve(lpm));
}

TEST(AcDc, LocalCaps) {
  constexpr std::string_view instance = R"(
2
24,1
24,5
1
144,10
)";
  ProblemInstance inst =
      test_support::make_instance_at_guess(instance, 50, "uneven_small_blocks");
  LPModel<PackedA> lpm(inst);
  lpm.update();

  ASSERT_EQ(lpm.A.num_blocks(), 2U);
  ASSERT_LT(lpm.A.block_maxCoeff(0), lpm.A.maxCoeff());

  const Eigen::VectorXi target =
      Eigen::VectorXi::Constant(static_cast<int>(lpm.A.rows()), 100000);
  const Eigen::VectorXi caps =
      oracle::ac_discrepancy::compute_block_caps(lpm.A, 0, target);
  const int local_delta = lpm.A.block_maxCoeff(0);
  const int expected_cap =
      compute_ac_K(static_cast<int>(lpm.A.rows()), local_delta) * local_delta;
  const int global_cap =
      compute_ac_K(static_cast<int>(lpm.A.rows()), lpm.A.maxCoeff()) *
      lpm.A.maxCoeff();
  for (int i = 0; i < caps.size(); ++i) {
    EXPECT_EQ(caps[i], expected_cap);
    EXPECT_LT(caps[i], global_cap);
  }
}

TEST(AcDc, WritesCsv) {
  const auto root = test_support::test_repo_root();
  const auto source_path = root / "instances" / "single_small.dat";
  const auto work_dir = root / "build";
  std::filesystem::create_directories(work_dir);

  const auto input_path = work_dir / "acdc_single_small.dat";
  const auto acdc_csv = work_dir / "acdc.csv";
  const auto main_csv = work_dir / "main.csv";
  io::csv::reset_main();
  test_support::remove_if_exists(input_path);
  test_support::remove_if_exists(acdc_csv);
  test_support::remove_if_exists(main_csv);
  std::filesystem::copy_file(source_path, input_path,
                             std::filesystem::copy_options::overwrite_existing);

  const auto [processed, succeeded] = proc::run(input_path, proc::RunMode::AcDc);
  EXPECT_EQ(processed, 1);
  EXPECT_EQ(succeeded, 1);
  EXPECT_TRUE(std::filesystem::exists(acdc_csv));
  EXPECT_FALSE(std::filesystem::exists(main_csv));

  const auto row =
      test_support::read_csv_first_row_with_value(acdc_csv, "ns_ac_batch");
  ASSERT_TRUE(row.has_value());
  EXPECT_NE(row->at("ns_ac_batch"), "");
  EXPECT_NE(row->at("ns_ac_discrepancy"), "");
  EXPECT_NE(row->at("ns_ac_batch_base_block_totals"), "");
  EXPECT_NE(row->at("ns_ac_discrepancy_base_block_totals"), "");
  EXPECT_TRUE(row->count("ac_discrepancy_useful_grid_share_ns_weighted") > 0);
  EXPECT_TRUE(row->count("ac_discrepancy_grid_points_total") > 0);
  EXPECT_TRUE(row->count("ac_discrepancy_feasible_vectors_total") > 0);
  EXPECT_TRUE(row->count("valid_ac_discrepancy") > 0);
  EXPECT_TRUE(row->count("fallback_reason_ac_discrepancy") > 0);

  io::csv::reset_main();
  test_support::remove_if_exists(input_path);
  test_support::remove_if_exists(acdc_csv);
}

TEST(AcDc, ProfileCsv) {
  ProblemInstance inst =
      test_support::make_midpoint_instance(test_support::test_repo_root() /
                                           "instances" / "single_small.dat");
  LPModel<PackedA> lpm(inst);
  lpm.update();

  const auto csv_path =
      test_support::test_repo_root() / "build" / "ac_discrepancy_profile.csv";
  io::csv::reset_main();
  test_support::remove_if_exists(csv_path);
  io::csv::init_main(csv_path, io::csv::MainSchema::AcDc);
  (void)oracle::ac_discrepancy::solve(lpm);
  io::csv::main_flush();
  io::csv::reset_main();

  const auto row = test_support::read_csv_first_row(csv_path);
  EXPECT_EQ(test_support::split_semicolon_row(
                row.at("ns_ac_discrepancy_base_block_totals"))
                .size(),
            lpm.A.num_blocks());
  if (!row.at("ac_discrepancy_useful_grid_share_ns_weighted").empty()) {
    const double useful =
        std::stod(row.at("ac_discrepancy_useful_grid_share_ns_weighted"));
    EXPECT_GE(useful, 0.0);
    EXPECT_LE(useful, 1.0);
  }

  io::csv::reset_main();
  test_support::remove_if_exists(csv_path);
}

TEST(AcDc, RejectsCap) {
  const auto csv_path =
      test_support::test_repo_root() / "build" / "ac_discrepancy_invalid.csv";
  io::csv::reset_main();
  test_support::remove_if_exists(csv_path);
  io::csv::init_main(csv_path, io::csv::MainSchema::AcDc);

  HugeEmbeddingSingleBlock A;
  Eigen::VectorXi b(static_cast<int>(A.rows() + A.num_blocks()));
  b.head(static_cast<int>(A.rows())) =
      Eigen::VectorXi::Constant(static_cast<int>(A.rows()), 1000000);
  b[static_cast<int>(A.rows())] = 1;
  oracle::discrep::detail::FFTWorkspace ws;

  BENCH_RUN_CHECK("baseline", false, "ac_discrepancy",
                  oracle::discrep::detail::run_solver(
                      A, b, nullptr, ws, "ac_discrepancy::single_block"));
  io::csv::main_flush();
  io::csv::reset_main();

  const auto row = test_support::read_csv_first_row(csv_path);
  EXPECT_EQ(row.at("ns_ac_discrepancy"), "");
  EXPECT_EQ(row.at("valid_ac_discrepancy"), "0");
  EXPECT_NE(row.at("fallback_reason_ac_discrepancy"), "");

  io::csv::reset_main();
  test_support::remove_if_exists(csv_path);
}
