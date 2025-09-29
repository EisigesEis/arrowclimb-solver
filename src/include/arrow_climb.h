#pragma once
#include <Eigen/Dense>
class BlockedMatrix;

bool arrow_climb_combine(const BlockedMatrix &BM,
                         int K,
                         int D,
                         const Eigen::VectorXi &b_up,
                         const Eigen::MatrixXi &b_down_tilde);