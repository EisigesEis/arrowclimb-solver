#pragma once
#include <Eigen/Dense>
#include <functional>
#include <vector>

#include "bt_enumerator.h"

class BlockedMatrix;
struct BTOptions;

using BaseTableFn = std::function<vector<VectorXi>(
    MatrixXi& /*A_k*/, Eigen::Ref<const VectorXi> /*target*/, int /*s*/)>;

vector<VectorXi>
dynamic_program(const BlockedMatrix &BM,
                int K,
                Eigen::Ref<const VectorXi> upper,
                Eigen::Ref<const VectorXi> target,
                const BaseTableFn &bt = compute_base_table_for_block);