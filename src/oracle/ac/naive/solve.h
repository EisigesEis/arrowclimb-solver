#include "oracle/ac/common/modern/solve_impl.h"
#include "bt_enumerator.h"
#include "model/matrix/model.h"
#include "model/matrix/packed.h"

namespace oracle::ac_naive {

inline bool solve(const LPModel<PackedA> &m) {
  return oracle::ac_modern::solve_impl(m, compute_base_sumset_for_block);
}

} // namespace oracle::ac_naive
