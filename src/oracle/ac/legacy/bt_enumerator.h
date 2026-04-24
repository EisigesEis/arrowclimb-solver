#pragma once

#include "model/matrix/packed.h"

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

namespace oracle::ac_legacy {

std::vector<Eigen::VectorXi>
compute_base_table_for_block(const PackedA &A, std::size_t k,
                             Eigen::Ref<const Eigen::VectorXi> target, int s);

} // namespace oracle::ac_legacy
