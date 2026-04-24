#include <gtest/gtest.h>

#include "io/csv_file.h"
#include "io/csv_main.h"
#include "model/matrix/model.h"
#include "model/matrix/packed.h"
#include "oracle/ac/batch/solve.h"
#include "oracle/ac/fft/solve.h"
#include "test_csv.h"
#include "test_files.h"
#include "test_instances.h"
#include "test_paths.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

ProblemInstance make_first_existing_profile_instance() {
  const std::vector<std::filesystem::path> candidates = {
      test_support::test_repo_root() / "instances" / "E1" /
          "M3_N15_U1_20_001.dat",
      test_support::test_repo_root() / "instances" / "E1" /
          "M3_N15_U20_50_001.dat",
      test_support::test_repo_root() / "instances" / "M2_N10_U1_20_001.dat",
      test_support::test_repo_root() / "instances" / "single_small.dat",
  };
  for (const auto &candidate : candidates) {
    if (auto inst = test_support::try_make_midpoint_instance(candidate))
      return *inst;
  }
  ADD_FAILURE() << "No parseable profile fixture found";
  return test_support::make_midpoint_instance(
      test_support::test_repo_root() / "instances" / "single_small.dat");
}

std::optional<double> parse_fraction_field(
    const std::map<std::string, std::string> &row, std::string_view key) {
  const auto it = row.find(std::string(key));
  if (it == row.end() || it->second.empty())
    return std::nullopt;
  return std::stod(it->second);
}

void expect_fraction_between_zero_and_one(
    const std::map<std::string, std::string> &row, std::string_view key) {
  const auto value = parse_fraction_field(row, key);
  ASSERT_TRUE(value.has_value()) << key;
  EXPECT_GE(*value, 0.0) << key;
  EXPECT_LE(*value, 1.0) << key;
}

int compute_digit_count_I(const LPModel<PackedA> &lpm) {
  const PackedA &A = lpm.A;
  if (A.num_blocks() == 0)
    return 1;

  const int r = A.rows();
  const int n = A.num_blocks();
  const int Delta_global = A.maxCoeff();
  const auto &b = lpm.b;
  const auto b_down = b.tail(n);
  const int K = std::max(
      1,
      int(std::floor(2.0 * (r + 1) * std::log2(4.0 * (r + 1) *
                                                double(Delta_global)))));
  const int b_down_max = b_down.maxCoeff();
  return std::max(
      1, int(std::ceil(std::log2(double(b_down_max + K) / (2 * K + 1)))) + 1);
}

int compute_actual_block_delta(const PackedA &A, std::size_t block_index) {
  const auto [c0, c1] = A.block_col_range(block_index);
  int delta = 0;
  for (std::size_t row = 0; row < A.rows(); ++row) {
    for (std::size_t col = c0; col < c1; ++col)
      delta = std::max(delta, A(row, col));
  }
  return delta;
}

TEST(AcCsv, Vectors) {
  const auto csv_path =
      test_support::test_repo_root() / "build" / "csv_vector_serialization.csv";
  test_support::remove_if_exists(csv_path);

  {
    io::csv::File file(csv_path, {"name", "vec"}, false);
    file.set("name", "demo");
    file.set("vec", std::vector<double>{1.0, 2.5, 3.0});
    file.flush();
  }

  const auto row = test_support::read_csv_first_row(csv_path);
  EXPECT_EQ(row.at("vec"), "1;2.5;3");

  test_support::remove_if_exists(csv_path);
}

TEST(AcCsv, BlockCounts) {
  ProblemInstance inst = make_first_existing_profile_instance();
  LPModel<PackedA> lpm(inst);
  lpm.update();

  const auto csv_path =
      test_support::test_repo_root() / "build" / "ac_profile_vectors.csv";
  io::csv::reset_main();
  test_support::remove_if_exists(csv_path);

  io::csv::init_main(csv_path);
  (void)oracle::ac_batch::solve(lpm);
  (void)oracle::ac_fft::solve(lpm);
  io::csv::main_flush();
  io::csv::reset_main();

  const auto row = test_support::read_csv_first_row(csv_path);
  EXPECT_EQ(std::stoll(row.at("num_small_blocks")),
            static_cast<long long>(lpm.A.num_small_blocks()));
  EXPECT_EQ(std::stoll(row.at("num_big_blocks")),
            static_cast<long long>(lpm.A.num_big_blocks()));
  EXPECT_EQ(test_support::split_semicolon_row(
                row.at("ns_ac_batch_base_block_totals"))
                .size(),
            lpm.A.num_blocks());
  EXPECT_EQ(test_support::split_semicolon_row(
                row.at("ns_ac_fft_base_block_totals"))
                .size(),
            lpm.A.num_blocks());
  expect_fraction_between_zero_and_one(row, "ac_fft_base_runtime_share");
  const auto useful =
      parse_fraction_field(row, "ac_fft_useful_grid_share_ns_weighted");
  const auto padding =
      parse_fraction_field(row, "ac_fft_padding_waste_share_ns_weighted");
  if (useful.has_value() || padding.has_value()) {
    ASSERT_TRUE(useful.has_value());
    ASSERT_TRUE(padding.has_value());
    EXPECT_GE(*useful, 0.0);
    EXPECT_LE(*useful, 1.0);
    EXPECT_GE(*padding, 0.0);
    EXPECT_LE(*padding, 1.0);
    EXPECT_LE(*useful + *padding, 1.0 + 1e-9);
  }

  io::csv::reset_main();
  test_support::remove_if_exists(csv_path);
}

TEST(AcCsv, BlockDelta) {
  ProblemInstance inst = make_first_existing_profile_instance();
  LPModel<PackedA> lpm(inst);
  lpm.update();

  ASSERT_GT(lpm.A.num_blocks(), 0U);
  for (std::size_t block_index = 0; block_index < lpm.A.num_blocks();
       ++block_index) {
    EXPECT_EQ(lpm.A.block_maxCoeff(block_index),
              compute_actual_block_delta(lpm.A, block_index))
        << "block_index=" << block_index;
  }
}

TEST(AcCsv, DigitBlocks) {
  constexpr std::string_view instance = R"(
2
24,1
24,5
1
144,10
)";
  ProblemInstance inst =
      test_support::make_instance_at_guess(instance, 50, "multi_digit_blocks");
  LPModel<PackedA> lpm(inst);
  lpm.update();
  ASSERT_GT(compute_digit_count_I(lpm), 1);

  const auto csv_path =
      test_support::test_repo_root() / "build" / "ac_profile_i_gt_1.csv";
  io::csv::reset_main();
  test_support::remove_if_exists(csv_path);

  io::csv::init_main(csv_path);
  (void)oracle::ac_batch::solve(lpm);
  (void)oracle::ac_fft::solve(lpm);
  io::csv::main_flush();
  io::csv::reset_main();

  const auto row = test_support::read_csv_first_row(csv_path);
  EXPECT_EQ(test_support::split_semicolon_row(
                row.at("ns_ac_batch_base_block_totals"))
                .size(),
            lpm.A.num_blocks());
  EXPECT_EQ(test_support::split_semicolon_row(
                row.at("ns_ac_fft_base_block_totals"))
                .size(),
            lpm.A.num_blocks());

  io::csv::reset_main();
  test_support::remove_if_exists(csv_path);
}

TEST(AcCsv, NoDummy) {
  ProblemInstance inst =
      test_support::make_midpoint_instance(test_support::test_repo_root() /
                                           "instances" / "single_small.dat");
  LPModel<PackedA> lpm(inst);
  lpm.update();
  ASSERT_FALSE(lpm.A.has_slack_block());

  const auto csv_path =
      test_support::test_repo_root() / "build" / "ac_profile_small_only.csv";
  io::csv::reset_main();
  test_support::remove_if_exists(csv_path);

  io::csv::init_main(csv_path);
  (void)oracle::ac_fft::solve(lpm);
  io::csv::main_flush();
  io::csv::reset_main();

  const auto row = test_support::read_csv_first_row(csv_path);
  EXPECT_EQ(row.at("ac_fft_dummy_work_share_avg"), "");
  const auto path_share = parse_fraction_field(row, "ac_fft_fft_path_share");
  const auto base_runtime_share =
      parse_fraction_field(row, "ac_fft_base_runtime_share");
  ASSERT_TRUE(base_runtime_share.has_value());
  EXPECT_GE(*base_runtime_share, 0.0);
  EXPECT_LE(*base_runtime_share, 1.0);
  if (path_share.has_value()) {
    EXPECT_GE(*path_share, 0.0);
    EXPECT_LE(*path_share, 1.0);
  }
  if (path_share.has_value() && *path_share == 0.0) {
    EXPECT_DOUBLE_EQ(*base_runtime_share, 0.0);
    EXPECT_EQ(row.at("ac_fft_useful_grid_share_ns_weighted"), "");
    EXPECT_EQ(row.at("ac_fft_padding_waste_share_ns_weighted"), "");
  } else {
    const auto useful =
        parse_fraction_field(row, "ac_fft_useful_grid_share_ns_weighted");
    const auto padding =
        parse_fraction_field(row, "ac_fft_padding_waste_share_ns_weighted");
    if (useful.has_value() || padding.has_value()) {
      ASSERT_TRUE(useful.has_value());
      ASSERT_TRUE(padding.has_value());
      EXPECT_GE(*useful, 0.0);
      EXPECT_LE(*useful, 1.0);
      EXPECT_GE(*padding, 0.0);
      EXPECT_LE(*padding, 1.0);
      const double empty = 1.0 - *useful - *padding;
      EXPECT_GE(empty, -1e-9);
      EXPECT_LE(empty, 1.0 + 1e-9);
    }
  }

  io::csv::reset_main();
  test_support::remove_if_exists(csv_path);
}

} // namespace
