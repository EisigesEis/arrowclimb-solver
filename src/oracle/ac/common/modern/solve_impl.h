#pragma once

#include "model/matrix/model.h"
#include "model/matrix/packed.h"

struct SumSet;
struct PowTable;

namespace oracle::ac_modern {

using BaseTableFnSumSet =
    SumSet (*)(const PackedA &, int, const Eigen::Ref<const Eigen::VectorXi>,
               int, PowTable &);
bool solve_impl(const LPModel<PackedA> &m, const BaseTableFnSumSet &bt);

} // namespace oracle::ac_modern
