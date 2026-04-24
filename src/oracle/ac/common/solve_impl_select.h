#pragma once

#include "oracle/ac/common/legacy/solve_impl.h"

namespace oracle::ac_select {

inline bool solve_impl(const LPModel<PackedA> &m, const BaseTableFn &bt) {
  return oracle::ac_common::solve_impl(m, bt);
}

} // namespace oracle::ac_select
