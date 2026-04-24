#pragma once
#include "model/matrix/model.h"
#include "model/matrix/packed.h"

namespace oracle::gupta_batch {

bool solve(const LPModel<PackedA> &lpm);

}