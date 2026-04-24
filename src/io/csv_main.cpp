#include "io/csv_main.h"

#include <memory>
#include <string>
#include <vector>

namespace io::csv {

namespace {
std::unique_ptr<Logger> g_main_csv;
bool g_main_enabled = true;
const std::vector<std::string> k_main_fields = {
    "name",
    "d",
    "nmax",
    "mmax",
    "machine_types",
    "job_types",
    "machines_total",
    "jobs_total",
    "load_total",
    "capacity_total",
    "avg_makespan",
    "p_min",
    "p_max",
    "Delta",
    "s_min",
    "s_max",
    "speed_ratio",
    "ell",
    "calN",
    "C_guess",
    "num_small_blocks",
    "num_big_blocks",
    "status",
    "ns_mat_update",
    "ns_gur",
    "ns_ac_legacy",
    "ns_ac_naive",
    "ns_ac_batch",
    "ns_ac_fft",
    "ns_ac_batch_base_block_totals",
    "ns_ac_fft_base_block_totals",
    "ac_fft_base_runtime_share",
    "ac_fft_useful_grid_share_ns_weighted",
    "ac_fft_padding_waste_share_ns_weighted",
    "ac_fft_fft_path_share",
    "ac_fft_fft_padding_efficiency_avg",
    "ac_fft_dummy_work_share_avg",
    "ns_gupta",
    "ns_gupta_batch",
    "ns_discrepancy",
    "discrepancy_fft_conv_runtime_share",
    "discrepancy_fft_useful_grid_share_conv_weighted",
    "discrepancy_fft_layer_share",
    "discrepancy_fill_ratio_avg",
    "valid_discrepancy",
    "fallback_reason_discrepancy",
  };

const std::vector<std::string> k_acdc_fields = {
    "name",
    "d",
    "nmax",
    "mmax",
    "machine_types",
    "job_types",
    "machines_total",
    "jobs_total",
    "load_total",
    "capacity_total",
    "avg_makespan",
    "p_min",
    "p_max",
    "Delta",
    "s_min",
    "s_max",
    "speed_ratio",
    "ell",
    "calN",
    "C_guess",
    "num_small_blocks",
    "num_big_blocks",
    "status",
    "ns_mat_update",
    "ns_ac_batch",
    "ns_ac_discrepancy",
    "ns_ac_batch_base_block_totals",
    "ns_ac_discrepancy_base_block_totals",
    "ac_discrepancy_useful_grid_share_ns_weighted",
    "ac_discrepancy_grid_points_total",
    "ac_discrepancy_feasible_vectors_total",
    "valid_ac_discrepancy",
    "fallback_reason_ac_discrepancy",
};

const std::vector<std::string> &fields_for_schema(MainSchema schema) {
  return schema == MainSchema::AcDc ? k_acdc_fields : k_main_fields;
}
} // namespace

void set_main_enabled(bool enabled) { g_main_enabled = enabled; }

void init_main(const std::filesystem::path &path, MainSchema schema) {
#if UNIFORMSCHED_ENABLE_CSV_LOGGING
  if (!g_main_enabled) {
    g_main_csv.reset();
    return;
  }
  g_main_csv = std::make_unique<Logger>(path, fields_for_schema(schema), false);
  main_persist("name");
#else
  (void)path;
  (void)schema;
#endif
}

void reset_main() {
#if UNIFORMSCHED_ENABLE_CSV_LOGGING
  g_main_csv.reset();
#endif
}

Log main() {
#if UNIFORMSCHED_ENABLE_CSV_LOGGING
  if (g_main_csv) {
    return Log(*g_main_csv);
  }
#endif
  return Log{};
}

void main_persist(std::string_view key) {
#if UNIFORMSCHED_ENABLE_CSV_LOGGING
  if (g_main_csv) {
    g_main_csv->set_persist_field(std::string(key));
  }
#else
  (void)key;
#endif
}

} // namespace io::csv
