#include "../../includes/loss/ALoss.hpp"

#include "Eigen/src/Core/Matrix.h"

#include <Eigen/Dense>

double ALoss::calculate(const Eigen::MatrixXd& predictive_inputs,
						const Eigen::VectorXi& target_inputs) {
	forward(predictive_inputs, target_inputs);
	return _outputs.mean();
}

const Eigen::VectorXd& ALoss::getOutputs() const {
	return _outputs;
}
