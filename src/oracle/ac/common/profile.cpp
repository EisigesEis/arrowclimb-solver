#include "oracle/ac/common/profile.h"

#include "io/csv_main.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace oracle::ac_profile {

struct RunProfileState {
  std::string solver_name;
  std::size_t num_small_blocks = 0;
  std::size_t num_big_blocks = 0;
  bool has_slack_block = false;
  std::vector<long long> base_block_totals_ns;
  long long total_base_runtime_ns = 0;
  long long fft_base_runtime_ns = 0;

  bool pending_fft_observation = false;
  bool pending_fft_used = false;
  double pending_padding_efficiency = 0.0;
  double pending_useful_grid_share = 0.0;
  double pending_padding_waste_share = 0.0;
  bool pending_has_dummy_metric = false;
  double pending_dummy_work_share = 0.0;

  long long fft_path_numerator = 0;
  long long fft_path_denominator = 0;
  double fft_padding_efficiency_sum = 0.0;
  long long fft_padding_efficiency_count = 0;
  double fft_useful_grid_weighted_sum = 0.0;
  double fft_padding_waste_weighted_sum = 0.0;
  long long fft_grid_weight_total_ns = 0;
  double fft_dummy_weighted_sum = 0.0;
  long long fft_dummy_weight_total_ns = 0;

  bool pending_grid_observation = false;
  long long pending_grid_points_total = 0;
  long long pending_feasible_vectors_total = 0;
  double useful_grid_weighted_sum = 0.0;
  long long useful_grid_weight_total_ns = 0;
  long long grid_points_total = 0;
  long long feasible_vectors_total = 0;
};

namespace {

thread_local RunProfileState *g_active_profile = nullptr;

std::vector<double> to_double_vector(const std::vector<long long> &values) {
  std::vector<double> out;
  out.reserve(values.size());
  for (const long long value : values)
    out.push_back(static_cast<double>(value));
  return out;
}

void clear_pending_fft_observation(RunProfileState &state) {
  state.pending_fft_observation = false;
  state.pending_fft_used = false;
  state.pending_padding_efficiency = 0.0;
  state.pending_useful_grid_share = 0.0;
  state.pending_padding_waste_share = 0.0;
  state.pending_has_dummy_metric = false;
  state.pending_dummy_work_share = 0.0;
  state.pending_grid_observation = false;
  state.pending_grid_points_total = 0;
  state.pending_feasible_vectors_total = 0;
}

} // namespace

ScopedRunProfile::ScopedRunProfile(std::string_view solver_name, const PackedA &A)
    : prev_(g_active_profile), state_(new RunProfileState{}) {
  state_->solver_name = std::string(solver_name);
  state_->num_small_blocks = A.num_small_blocks();
  state_->num_big_blocks = A.num_big_blocks();
  state_->has_slack_block = A.has_slack_block();
  state_->base_block_totals_ns.assign(A.num_blocks(), 0);
  g_active_profile = state_;
}

ScopedRunProfile::~ScopedRunProfile() {
  if (state_ != nullptr) {
    io::csv::main_set("num_small_blocks",
                      static_cast<long long>(state_->num_small_blocks));
    io::csv::main_set("num_big_blocks",
                      static_cast<long long>(state_->num_big_blocks));

    if (state_->solver_name == "ac_batch") {
      io::csv::main_set("ns_ac_batch_base_block_totals",
                        to_double_vector(state_->base_block_totals_ns));
    } else if (state_->solver_name == "ac_discrepancy") {
      io::csv::main_set("ns_ac_discrepancy_base_block_totals",
                        to_double_vector(state_->base_block_totals_ns));
      io::csv::main_set("ac_discrepancy_grid_points_total",
                        state_->grid_points_total);
      io::csv::main_set("ac_discrepancy_feasible_vectors_total",
                        state_->feasible_vectors_total);
      if (state_->useful_grid_weight_total_ns > 0) {
        io::csv::main_set(
            "ac_discrepancy_useful_grid_share_ns_weighted",
            state_->useful_grid_weighted_sum /
                static_cast<double>(state_->useful_grid_weight_total_ns));
      }
    } else if (state_->solver_name == "ac_fft") {
      io::csv::main_set("ns_ac_fft_base_block_totals",
                        to_double_vector(state_->base_block_totals_ns));
      if (state_->total_base_runtime_ns > 0) {
        io::csv::main_set(
            "ac_fft_base_runtime_share",
            static_cast<double>(state_->fft_base_runtime_ns) /
                static_cast<double>(state_->total_base_runtime_ns));
      }
      if (state_->fft_grid_weight_total_ns > 0) {
        io::csv::main_set(
            "ac_fft_useful_grid_share_ns_weighted",
            state_->fft_useful_grid_weighted_sum /
                static_cast<double>(state_->fft_grid_weight_total_ns));
        io::csv::main_set(
            "ac_fft_padding_waste_share_ns_weighted",
            state_->fft_padding_waste_weighted_sum /
                static_cast<double>(state_->fft_grid_weight_total_ns));
      }
      if (state_->fft_path_denominator > 0) {
        io::csv::main_set(
            "ac_fft_fft_path_share",
            static_cast<double>(state_->fft_path_numerator) /
                static_cast<double>(state_->fft_path_denominator));
      }
      if (state_->fft_padding_efficiency_count > 0) {
        io::csv::main_set(
            "ac_fft_fft_padding_efficiency_avg",
            state_->fft_padding_efficiency_sum /
                static_cast<double>(state_->fft_padding_efficiency_count));
      }
      if (state_->fft_dummy_weight_total_ns > 0) {
        io::csv::main_set(
            "ac_fft_dummy_work_share_avg",
            state_->fft_dummy_weighted_sum /
                static_cast<double>(state_->fft_dummy_weight_total_ns));
      }
    }
  }

  delete state_;
  g_active_profile = prev_;
}

void begin_base_call() {
  if (g_active_profile == nullptr)
    return;
  clear_pending_fft_observation(*g_active_profile);
}

void finish_base_call(std::size_t block_index, long long duration_ns) {
  if (g_active_profile == nullptr)
    return;

  g_active_profile->total_base_runtime_ns += duration_ns;
  if (block_index < g_active_profile->base_block_totals_ns.size()) {
    g_active_profile->base_block_totals_ns[block_index] += duration_ns;
  }

  if (!g_active_profile->pending_fft_observation)
    goto finish_grid;

  g_active_profile->fft_path_denominator++;
  if (g_active_profile->pending_fft_used) {
    g_active_profile->fft_base_runtime_ns += duration_ns;
    g_active_profile->fft_path_numerator++;
    g_active_profile->fft_padding_efficiency_sum +=
        g_active_profile->pending_padding_efficiency;
    g_active_profile->fft_padding_efficiency_count++;
    g_active_profile->fft_useful_grid_weighted_sum +=
        g_active_profile->pending_useful_grid_share *
        static_cast<double>(duration_ns);
    g_active_profile->fft_padding_waste_weighted_sum +=
        g_active_profile->pending_padding_waste_share *
        static_cast<double>(duration_ns);
    g_active_profile->fft_grid_weight_total_ns += duration_ns;
    if (g_active_profile->pending_has_dummy_metric) {
      g_active_profile->fft_dummy_weighted_sum +=
          g_active_profile->pending_dummy_work_share *
          static_cast<double>(duration_ns);
      g_active_profile->fft_dummy_weight_total_ns += duration_ns;
    }
  }

  clear_pending_fft_observation(*g_active_profile);
  return;

finish_grid:
  if (g_active_profile->pending_grid_observation) {
    g_active_profile->grid_points_total +=
        g_active_profile->pending_grid_points_total;
    g_active_profile->feasible_vectors_total +=
        g_active_profile->pending_feasible_vectors_total;
    if (g_active_profile->pending_grid_points_total > 0) {
      g_active_profile->useful_grid_weighted_sum +=
          (static_cast<double>(g_active_profile->pending_feasible_vectors_total) /
           static_cast<double>(g_active_profile->pending_grid_points_total)) *
          static_cast<double>(duration_ns);
      g_active_profile->useful_grid_weight_total_ns += duration_ns;
    }
  }

  clear_pending_fft_observation(*g_active_profile);
}

void note_fft_base_call(bool used_fft, double useful_grid_share,
                        double padding_waste_share, bool has_dummy_metric,
                        double dummy_work_share) {
  if (g_active_profile == nullptr)
    return;
  if (g_active_profile->solver_name != "ac_fft")
    return;

  g_active_profile->pending_fft_observation = true;
  g_active_profile->pending_fft_used = used_fft;
  g_active_profile->pending_padding_efficiency =
      std::max(0.0, 1.0 - padding_waste_share);
  g_active_profile->pending_useful_grid_share = useful_grid_share;
  g_active_profile->pending_padding_waste_share = padding_waste_share;
  g_active_profile->pending_has_dummy_metric = has_dummy_metric;
  g_active_profile->pending_dummy_work_share = dummy_work_share;
}

void note_grid_enumeration_base_call(long long grid_points_total,
                                     long long feasible_vectors_total) {
  if (g_active_profile == nullptr)
    return;
  if (g_active_profile->solver_name != "ac_discrepancy")
    return;

  g_active_profile->pending_grid_observation = true;
  g_active_profile->pending_grid_points_total = std::max(0LL, grid_points_total);
  g_active_profile->pending_feasible_vectors_total =
      std::max(0LL, feasible_vectors_total);
}

} // namespace oracle::ac_profile
