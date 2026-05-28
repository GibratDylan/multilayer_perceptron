#include "loss/LossCategoricalCrossEntropy.hpp"

#include <Eigen/Dense>

void LossCategoricalCrossEntropy::forward(
	const Eigen::MatrixXd& predictive_inputs,
	const Eigen::VectorXi& target_inputs) {
	assert(predictive_inputs.cols() == target_inputs.rows() &&
		   target_inputs.maxCoeff() <= predictive_inputs.rows());
	assert(predictive_inputs.size() > 0 && target_inputs.size() > 0);

	inputs_ = predictive_inputs;

	Eigen::VectorXd correct_confidences(target_inputs.size());
	for (int index = 0; index < target_inputs.size(); ++index) {
		correct_confidences(index) =
			predictive_inputs(target_inputs(index), index);
	}

	assert(correct_confidences.size() > 0);

	for (auto& correct_confidence : correct_confidences)
		correct_confidence =
			std::max(1e-7, std::min(1 - 1e-7, correct_confidence));

	outputs_ = -(correct_confidences.array().log());

	assert(outputs_.rows() == target_inputs.rows());
}

void LossCategoricalCrossEntropy::backward(
	const Eigen::VectorXi& target_inputs) {
	assert(inputs_.cols() == target_inputs.rows() &&
		   target_inputs.maxCoeff() <= inputs_.rows());
	assert(inputs_.size() > 0 && target_inputs.size() > 0);

	inputs_gradient_.resizeLike(inputs_);
	for (int i = 0; i < inputs_.cols(); ++i) {
		int label = target_inputs(i);
		double predictive_input =
			std::max(1e-7, std::min(1 - 1e-7, inputs_(label, i)));
		inputs_gradient_(label, i) = -1.0 / predictive_input;
	}
	inputs_gradient_ /= static_cast<double>(inputs_.cols());

	assert(inputs_gradient_.rows() == inputs_.rows() &&
		   inputs_gradient_.cols() == inputs_.cols());
}
