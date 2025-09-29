#pragma once
#include <optional>

std::optional<double> read_cpu_temp_c_windows();
void cooldown_between_jobs_windows();