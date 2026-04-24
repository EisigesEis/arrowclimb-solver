#pragma once

#include "model/matrix/model.h"
#include "model/matrix/packed.h"

namespace oracle::discrep {

bool solve(const LPModel<PackedA> &lpm);
void warmup(const LPModel<PackedA> &lpm);
bool solve_single_block(const PackedA &A, int block_index,
                        const Eigen::Ref<const Eigen::VectorXi> &target,
                        int picks);

} // namespace oracle::discrep
