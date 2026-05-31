#include "activation_loss/ActivationSoftmaxLossCategoricalCrossentropy.hpp"

#include <ostream>

void ActivationSoftmaxLossCategoricalCrossentropy::Forward(
	MatrixIn input_batch, IntVectorIn targets_batch) {
	activation_.Forward(input_batch);
	loss_.Forward(activation_.GetOutputs(), targets_batch);
}

void ActivationSoftmaxLossCategoricalCrossentropy::Backward(
	IntVectorIn targets_batch) {
	inputs_gradient_ = outputs_;

	const int64_t batch_size = targets_batch.size();

	assert(inputs_gradient_.cols() == targets_batch.rows() &&
		   targets_batch.maxCoeff() <= inputs_gradient_.rows());

	assert(inputs_gradient_.size() > 0 && targets_batch.size() > 0);

	for (int64_t index = 0; index < batch_size; ++index) {
		inputs_gradient_(targets_batch(index), index) -= 1.F;
	}

	inputs_gradient_ /= static_cast<float>(batch_size);
}

MatrixIn ActivationSoftmaxLossCategoricalCrossentropy::GetOutputs() const {
	return outputs_;
}

MatrixIn ActivationSoftmaxLossCategoricalCrossentropy::GetInputsGradient()
	const {
	return inputs_gradient_;
}

float ActivationSoftmaxLossCategoricalCrossentropy::GetLoss() const {
	return loss_.GetLoss();
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
	   << rhs.loss_.GetOutputs().transpose().format(vec_fmt);
	os << "  loss_means:\n" << rhs.loss_.GetLoss();

	return os;
}
