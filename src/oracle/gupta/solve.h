#pragma once
#include "model/matrix/model.h"
#include "model/matrix/packed.h"

namespace oracle::gupta {

bool solve(const LPModel<PackedA> &lpm);

}