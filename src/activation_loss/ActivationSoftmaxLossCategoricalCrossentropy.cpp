#include "activation_loss/ActivationSoftmaxLossCategoricalCrossentropy.hpp"

#include <ostream>

#include <Eigen/Dense>

double ActivationSoftmaxLossCategoricalCrossentropy::forward(
	const Eigen::MatrixXd& inputs, const Eigen::VectorXi& target_inputs) {
	activation_.forward(inputs);
	loss_.forward(activation_.getOutputs(), target_inputs);

	return loss_.getLoss();
}

void ActivationSoftmaxLossCategoricalCrossentropy::backward(
	const Eigen::VectorXi& target_inputs) {
	inputs_gradient_ = outputs_;

	const Eigen::Index samples = target_inputs.size();

	assert(inputs_gradient_.cols() == target_inputs.rows() &&
		   target_inputs.maxCoeff() <= inputs_gradient_.rows());

	assert(inputs_gradient_.size() > 0 && target_inputs.size() > 0);

	for (Eigen::Index index = 0; index < samples; ++index) {
		inputs_gradient_(target_inputs(index), index) -= 1.0;
	}

	inputs_gradient_ /= static_cast<double>(samples);
}

const Eigen::MatrixXd&
ActivationSoftmaxLossCategoricalCrossentropy::getOutputs() const {
	return outputs_;
}

const Eigen::MatrixXd&
ActivationSoftmaxLossCategoricalCrossentropy::getInputsGradient() const {
	return inputs_gradient_;
}

std::ostream& operator<<(
	std::ostream& os, const ActivationSoftmaxLossCategoricalCrossentropy& rhs) {
	const Eigen::IOFormat mat_fmt(4, 0, ", ", "\n", "    [", "]");
	const Eigen::IOFormat vec_fmt(4, 0, ", ", "\n", "    [", "]");

	os << "ActivationSoftmaxLossCategoricalCrossentropy\n";
	os << "  outputs:\n" << rhs.outputs_.format(mat_fmt) << "\n";
	os << "  inputs_gradient:\n"
	   << rhs.inputs_gradient_.format(mat_fmt) << "\n";
	os << "  loss_outputs:\n"
	   << rhs.loss_.getOutputs().transpose().format(vec_fmt);
	os << "  loss_means:\n" << rhs.loss_.getLoss();

	return os;
}
