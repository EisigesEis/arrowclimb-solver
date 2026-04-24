#pragma once

#include "oracle/ac/common/profile.h"
#include "oracle/ac/common/modern/solve_impl.h"
#include "bt_enumerator.h"
#include "model/matrix/model.h"
#include "model/matrix/packed.h"

namespace oracle::ac_batch {

inline bool solve(const LPModel<PackedA> &m) {
  oracle::ac_profile::ScopedRunProfile profile("ac_batch", m.A);
  return oracle::ac_modern::solve_impl(m, compute_base_sumset_for_block);
}

} // namespace oracle::ac_batch
