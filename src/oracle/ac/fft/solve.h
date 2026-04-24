#include "oracle/ac/common/modern/solve_impl.h"
#include "oracle/ac/common/profile.h"
#include "bt_enumerator_fft.h"
#include "model/matrix/model.h"
#include "model/matrix/packed.h"

namespace oracle::ac_fft {

inline bool solve(const LPModel<PackedA> &m) {
  oracle::ac_profile::ScopedRunProfile profile("ac_fft", m.A);
  return oracle::ac_modern::solve_impl(m, compute_base_sumset_for_block_fft);
}

} // namespace oracle::ac_fft
