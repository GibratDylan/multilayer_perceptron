#include "activation_loss/ActivationSoftmaxLossCategoricalCrossentropy.hpp"

#include <ostream>

LossValue ActivationSoftmaxLossCategoricalCrossentropy::Forward(
	const InputBatch& input_batch, const TargetsBatch& targets_batch) {
	activation_.Forward(input_batch);
	loss_.Forward(activation_.GetOutputs(), targets_batch);

	return loss_.GetLoss();
}

void ActivationSoftmaxLossCategoricalCrossentropy::Backward(
	const TargetsBatch& targets_batch) {
	inputs_gradient_ = outputs_;

	const Index batch_size = targets_batch.size();

	assert(inputs_gradient_.cols() == targets_batch.rows() &&
		   targets_batch.maxCoeff() <= inputs_gradient_.rows());

	assert(inputs_gradient_.size() > 0 && targets_batch.size() > 0);

	for (Index index = 0; index < batch_size; ++index) {
		inputs_gradient_(targets_batch(index), index) -= 1.0;
	}

	inputs_gradient_ /= static_cast<LossValue>(batch_size);
}

const ActivationSoftmaxLossCategoricalCrossentropy::OutputBatch&
ActivationSoftmaxLossCategoricalCrossentropy::GetOutputs() const {
	return outputs_;
}

const ActivationSoftmaxLossCategoricalCrossentropy::GradientBatch&
ActivationSoftmaxLossCategoricalCrossentropy::GetInputsGradient() const {
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
	   << rhs.loss_.GetOutputs().transpose().format(vec_fmt);
	os << "  loss_means:\n" << rhs.loss_.GetLoss();

	return os;
}
