#pragma once

#include <Eigen/Dense>

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;
using IntVector = Eigen::VectorXi;

using MatrixIn = Eigen::Ref<const Matrix>;
using MatrixOut = Eigen::Ref<Matrix>;
using VectorIn = Eigen::Ref<const Vector>;
using VectorOut = Eigen::Ref<Vector>;
using IntVectorIn = Eigen::Ref<const IntVector>;
