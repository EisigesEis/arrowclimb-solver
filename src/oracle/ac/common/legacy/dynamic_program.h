#pragma once
#include "model/matrix/model.h"
#include "model/matrix/packed.h"
#include <Eigen/Dense>
#include <functional>
#include <vector>

class BlockedMatrix;
struct BTOptions;

using BaseTableFn = std::function<std::vector<Eigen::VectorXi>(
    const PackedA & /*A_k*/, const int /*k*/, const Eigen::Ref<const Eigen::VectorXi> /*target*/,
    int /*s*/)>;

std::vector<Eigen::VectorXi>
dynamic_program(const PackedA &BM,
                Eigen::Ref<const Eigen::VectorXi> upper,
                Eigen::Ref<const Eigen::VectorXi> target,
                const BaseTableFn &bt);