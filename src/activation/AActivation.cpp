#include "activation/AActivation.hpp"

#include <ostream>

#include <Eigen/Dense>

const Eigen::MatrixXd& AActivation::getOutputs() const {
	return outputs_;
}

const Eigen::MatrixXd& AActivation::getInputsGradient() const {
	return inputs_gradient_;
}

std::ostream& operator<<(std::ostream& os, const AActivation& rhs) {
	const Eigen::IOFormat mat_fmt(4, 0, ", ", "\n", "    [", "]");

	os << "Activation\n";
	os << "  outputs:\n" << rhs.getOutputs().format(mat_fmt) << "\n";
	os << "  inputs_gradient:\n" << rhs.getInputsGradient().format(mat_fmt);

	return os;
}
