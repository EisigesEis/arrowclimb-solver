#pragma once

#include "model/matrix/packed.h"

#include <cstddef>
#include <string_view>

namespace oracle::ac_profile {

struct RunProfileState;

class ScopedRunProfile {
public:
  ScopedRunProfile(std::string_view solver_name, const PackedA &A);
  ~ScopedRunProfile();

  ScopedRunProfile(const ScopedRunProfile &) = delete;
  ScopedRunProfile &operator=(const ScopedRunProfile &) = delete;

private:
  RunProfileState *prev_ = nullptr;
  RunProfileState *state_ = nullptr;
};

void begin_base_call();
void finish_base_call(std::size_t block_index, long long duration_ns);

void note_fft_base_call(bool used_fft, double useful_grid_share,
                        double padding_waste_share, bool has_dummy_metric,
                        double dummy_work_share);
void note_grid_enumeration_base_call(long long grid_points_total,
                                     long long feasible_vectors_total);

} // namespace oracle::ac_profile
