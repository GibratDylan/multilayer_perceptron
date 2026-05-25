#include "loss/ALoss.hpp"

#include <ostream>

#include <Eigen/Dense>

double ALoss::getLoss() const {
	assert(outputs_.size() > 0);
	return outputs_.mean();
}

const Eigen::VectorXd& ALoss::getOutputs() const {
	return outputs_;
}

const Eigen::MatrixXd& ALoss::getInputsGradient() const {
	return inputs_gradient_;
}

std::ostream& operator<<(std::ostream& os, const ALoss& rhs) {
	const Eigen::IOFormat mat_fmt(4, 0, ", ", "\n", "    [", "]");
	const Eigen::IOFormat vec_fmt(4, 0, ", ", "\n", "    [", "]");
	os << "Loss\n";
	os << "  outputs:\n" << rhs.outputs_.transpose().format(vec_fmt);
	os << "  inputs_gradient:\n"
	   << rhs.inputs_gradient_.format(mat_fmt) << "\n";
	os << "  loss_means:\n" << rhs.getLoss();
	return os;
}
