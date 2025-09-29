#pragma once
#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXi;
using Eigen::VectorXi;
using std::vector;

vector<VectorXi>
compute_base_table_for_block(const MatrixXi &A_k,
                             Eigen::Ref<const VectorXi> target,
                             int s);