#include "loss/ALoss.hpp"

#include <ostream>

#include <Eigen/Dense>

double ALoss::calculate(const Eigen::MatrixXd& predictive_inputs,
						const Eigen::VectorXi& target_inputs) {
	forward(predictive_inputs, target_inputs);
	return outputs_.mean();
}

const Eigen::VectorXd& ALoss::getOutputs() const {
	return outputs_;
}

std::ostream& operator<<(std::ostream& os, const ALoss& rhs) {
	const Eigen::IOFormat vec_fmt(4, 0, ", ", "\n", "    [", "]");

	os << "Loss\n";
	os << "  outputs:\n" << rhs.getOutputs().transpose().format(vec_fmt);

	return os;
}
