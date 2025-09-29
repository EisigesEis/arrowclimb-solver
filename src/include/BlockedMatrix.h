#pragma once
#include <cassert>
#ifdef _MSC_VER
#pragma warning(disable: 4068) // fix for CUDA warnings
#endif
#include <Eigen/Dense>
#ifdef _MSC_VER
#pragma warning(default: 4068)
#endif
#include <vector>

class BlockedMatrix {
  std::vector<int> start_col_, widths_;

public:
  Eigen::MatrixXi mat_;
  
  BlockedMatrix(){}
  BlockedMatrix(int rows, int cols) : mat_(rows, cols) {}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void add_block(Eigen::MatrixXi new_block) {
    const auto &w = new_block.cols();
    int total_used = widths_.empty() ? 0 : start_col_.back() + widths_.back();

    if (new_block.rows() != mat_.rows())
      throw std::runtime_error("add_block: row mismatch");
    if (new_block.cols() <= 0)
      throw std::runtime_error("add_block: empty block");
    if (total_used + new_block.cols() > mat_.cols())
      throw std::runtime_error("add_block: exceeds matrix width");

    start_col_.push_back(total_used);
    widths_.push_back(w);

    mat_.block(0, total_used, new_block.rows(), new_block.cols()) = new_block;
  }

  int num_blocks() const { return int(widths_.size()); }

  auto get_block(int i) const {
    assert(i >= 0 && i < num_blocks());
    return mat_.middleCols(start_col_[i], widths_[i]);
  }
};