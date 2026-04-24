#include "oracle/ac/common/legacy/dynamic_program.h"
#include "model/matrix/model.h"
#include "model/matrix/packed.h"

namespace oracle::ac_common {

bool solve_impl(const LPModel<PackedA> &m, const BaseTableFn &bt);

}
