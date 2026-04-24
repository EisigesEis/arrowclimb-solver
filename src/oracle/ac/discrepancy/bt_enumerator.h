#pragma once

#include "model/matrix/packed.h"

#include <Eigen/Dense>

struct SumSet;
struct PowTable;

namespace oracle::ac_discrepancy {

Eigen::VectorXi compute_block_caps(const PackedA &A, int k,
                                   Eigen::Ref<const Eigen::VectorXi> target);

SumSet compute_base_sumset_for_block(
    const PackedA &A, int k,
    Eigen::Ref<const Eigen::VectorXi> target, int s, PowTable &pow_table);

} // namespace oracle::ac_discrepancy
