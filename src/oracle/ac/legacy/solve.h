#pragma once

#include "oracle/ac/common/legacy/solve_impl.h"
#include "oracle/ac/legacy/bt_enumerator.h"
#include "model/matrix/model.h"
#include "model/matrix/packed.h"

namespace oracle::ac_legacy {

inline bool solve(const LPModel<PackedA> &m) {
  return oracle::ac_common::solve_impl(m, oracle::ac_legacy::compute_base_table_for_block);
}

} // namespace oracle::ac_legacy
