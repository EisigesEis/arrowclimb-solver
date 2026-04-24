#pragma once
#include "instance/types.h"

void derive_inital(ProblemInstance &I);

bool derive_for_guess(ProblemInstance &inst, long long k_mid);

void log_main_instance_fields(const ProblemInstance &inst);
