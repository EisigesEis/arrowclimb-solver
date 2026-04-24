#pragma once
#include "model/matrix/packed.h"
#include <Eigen/Dense>

struct SumSet;
struct PowTable;

namespace oracle::ac_naive {

SumSet compute_base_sumset_for_block(
    const PackedA &A, int k,
    Eigen::Ref<const Eigen::VectorXi> target, int s,
    PowTable & /*pow_table*/);

} // namespace oracle::ac_naive
