#pragma once
#include "Types.h"
#include "matrix.h"
#include "bt_enumerator.h"
#include "dynamic_program.h"

bool solve_direct(const int Delta_global, const BlockedMatrix &BM,
                  const VectorXi &b);

bool solve_arrow_climb(const int Delta_global,
                       const BlockedMatrix& BM,
                       const Eigen::VectorXi& b_full,
                       const BaseTableFn& bt = compute_base_table_for_block);