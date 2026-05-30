#pragma once

#include <Eigen/Dense>

namespace metrics {
float accuracy(const Eigen::MatrixXd& predictive_inputs,
			   const Eigen::VectorXi& target_inputs);
}
